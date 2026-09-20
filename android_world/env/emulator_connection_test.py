# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for existing emulator discovery and authenticated gRPC calls."""

from concurrent import futures
import os
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest
from android_env.proto import emulator_controller_pb2
from android_env.proto import emulator_controller_pb2_grpc
from android_world.env import emulator_connection
from google.protobuf import empty_pb2
import grpc


class EmulatorDiscoveryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.directory = Path(directory.name)
    self.enter_context(mock.patch.object(
        emulator_connection, '_metadata_directories',
        return_value=[self.directory],
    ))

  def _metadata(self, console=5556, port=8556, token='test-token'):
    (self.directory / f'pid_{os.getpid()}.ini').write_text(
        f'port.serial={console}\ngrpc.port={port}\ngrpc.token={token}\n'
    )

  def test_matches_console_port_and_discovers_token(self):
    self._metadata()
    connection = emulator_connection.discover(5556)
    self.assertEqual(connection.port, 8556)
    self.assertEqual(connection.token, 'test-token')
    self.assertNotIn('test-token', repr(connection))

  def test_does_not_take_another_devices_token(self):
    self._metadata()
    connection = emulator_connection.discover(5554)
    self.assertEqual(connection.port, 8554)
    self.assertIsNone(connection.token)

  def test_manual_launch_without_token(self):
    self._metadata(token='')
    connection = emulator_connection.discover(5556)
    self.assertEqual(connection.port, 8556)
    self.assertIsNone(connection.token)

  def test_explicit_port_mismatch_fails_before_connecting(self):
    self._metadata()
    with self.assertRaisesRegex(ValueError, '8556'):
      emulator_connection.discover(5556, grpc_port=8554)

  def test_explicit_port_without_metadata(self):
    connection = emulator_connection.discover(5556, grpc_port=9000)
    self.assertEqual(connection.port, 9000)
    self.assertIsNone(connection.token)

  def test_ignores_stale_process_metadata(self):
    self._metadata()
    with mock.patch.object(
        emulator_connection, '_process_exists', return_value=False
    ):
      self.assertIsNone(emulator_connection.discover(5556).token)


class _TokenRequiredController(
    emulator_controller_pb2_grpc.EmulatorControllerServicer
):

  def _check(self, context):
    if dict(context.invocation_metadata()).get('authorization') != (
        'Bearer test-token'
    ):
      context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Missing token')

  def getStatus(self, request, context):
    self._check(context)
    return emulator_controller_pb2.EmulatorStatus(booted=True)

  def streamScreenshot(self, request, context):
    self._check(context)
    yield emulator_controller_pb2.Image(image=b'frame')


class EmulatorAuthenticationTest(absltest.TestCase):

  def test_authentication_covers_unary_streaming_and_reconnections(self):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    emulator_controller_pb2_grpc.add_EmulatorControllerServicer_to_server(
        _TokenRequiredController(), server
    )
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    self.addCleanup(lambda: server.stop(0).wait())

    with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
      stub = emulator_controller_pb2_grpc.EmulatorControllerStub(channel)
      with self.assertRaises(grpc.RpcError) as error:
        stub.getStatus(empty_pb2.Empty(), timeout=5)
      self.assertEqual(error.exception.code(), grpc.StatusCode.UNAUTHENTICATED)

    simulator = emulator_connection.AuthenticatedEmulatorSimulator.__new__(
        emulator_connection.AuthenticatedEmulatorSimulator
    )
    simulator._grpc_token = 'test-token'
    simulator._channel = None
    self.addCleanup(lambda: simulator._channel.close())
    for _ in range(2):
      stub, _ = simulator._connect_to_emulator(port, timeout_sec=5)
      self.assertTrue(stub.getStatus(empty_pb2.Empty(), timeout=5).booted)
      frames = list(stub.streamScreenshot(
          emulator_controller_pb2.ImageFormat(), timeout=5
      ))
      self.assertEqual(frames[0].image, b'frame')


if __name__ == '__main__':
  absltest.main()
