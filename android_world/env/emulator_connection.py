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

"""Discovers and authenticates connections to existing local emulators."""

import dataclasses
import os
from pathlib import Path
import re
import sys

from android_env import environment
from android_env.components import config_classes
from android_env.components import coordinator as coordinator_lib
from android_env.components import device_settings
from android_env.components import task_manager as task_manager_lib
from android_env.components.simulators.emulator import emulator_simulator
from android_env.proto import emulator_controller_pb2_grpc
from android_env.proto import snapshot_service_pb2_grpc
from android_env.proto import task_pb2
from google.protobuf import text_format
import grpc


@dataclasses.dataclass(frozen=True)
class Connection:
  port: int
  token: str | None = dataclasses.field(default=None, repr=False)


def _metadata_directories() -> list[Path]:
  if sys.platform == 'darwin':
    return [Path.home() / 'Library/Caches/TemporaryItems/avd/running']
  roots = [Path.home() / '.android']
  for name in ('ANDROID_EMULATOR_HOME', 'XDG_RUNTIME_DIR'):
    if os.environ.get(name):
      roots.insert(0, Path(os.environ[name]))
  if hasattr(os, 'getuid'):
    roots.append(Path('/run/user') / str(os.getuid()))
  return [root / 'avd/running' for root in roots]


def _process_exists(pid: int) -> bool:
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False
  except PermissionError:
    return True


def discover(console_port: int, grpc_port: int | None = None) -> Connection:
  """Finds the gRPC port and token for the selected emulator's console port.

  The discovery files are created by the emulator, including when Agentsims
  or Android Studio launches it. Older manual launches can omit these files.
  """
  for directory in _metadata_directories():
    for path in sorted(directory.glob('pid_*.ini')):
      match = re.fullmatch(r'pid_(\d+)(?:_info)?\.ini', path.name)
      if not match or not _process_exists(int(match.group(1))):
        continue
      try:
        values = dict(
            line.strip().split('=', 1)
            for line in path.read_text().splitlines()
            if '=' in line and not line.lstrip().startswith('#')
        )
      except FileNotFoundError:
        # The emulator can exit between listing and reading its file.
        continue
      if values.get('port.serial') != str(console_port):
        continue
      try:
        port = int(values.get('grpc.port', '0'))
      except ValueError:
        port = 0
      if not 1 <= port <= 65535:
        raise ValueError(
            f'Emulator-{console_port} has invalid gRPC discovery metadata.'
        )
      if grpc_port is not None and grpc_port != port:
        raise ValueError(
            f'Emulator-{console_port} uses gRPC port {port}, but --grpc_port='
            f'{grpc_port} was requested. Omit --grpc_port to discover it.'
        )
      return Connection(port, values.get('grpc.token') or None)
  return Connection(grpc_port if grpc_port is not None else 8554)


class AuthenticatedEmulatorSimulator(emulator_simulator.EmulatorSimulator):
  """Adds a token to AndroidEnv's local channel, including after reconnects."""

  def __init__(self, config: config_classes.EmulatorConfig, token: str):
    self._grpc_token = token
    super().__init__(config)

  def _connect_to_emulator(self, grpc_port: int, timeout_sec: int = 100):
    if self._channel is not None:
      self._channel.close()
    credentials = grpc.composite_channel_credentials(
        grpc.local_channel_credentials(),
        grpc.access_token_call_credentials(self._grpc_token),
    )
    self._channel = grpc.secure_channel(
        f'localhost:{grpc_port}',
        credentials,
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ],
    )
    try:
      grpc.channel_ready_future(self._channel).result(timeout=timeout_sec)
    except (grpc.RpcError, grpc.FutureTimeoutError) as error:
      self._channel.close()
      raise emulator_simulator.EmulatorBootError(
          f'Failed to connect to emulator gRPC port {grpc_port}.'
      ) from error
    return (
        emulator_controller_pb2_grpc.EmulatorControllerStub(self._channel),
        snapshot_service_pb2_grpc.SnapshotServiceStub(self._channel),
    )


def load_authenticated_env(
    config: config_classes.AndroidEnvConfig, token: str
) -> environment.AndroidEnv:
  """Builds AndroidEnv with a token-aware simulator without global patches.

  AndroidEnv 1.2.3's loader has no channel-credential or simulator factory
  parameter. Compose its public components with our simulator subclass here.
  """
  if not isinstance(config.simulator, config_classes.EmulatorConfig):
    raise TypeError('Authenticated connections require an emulator config.')
  if not isinstance(config.task, config_classes.FilesystemTaskConfig):
    raise TypeError('Authenticated connections require a filesystem task.')
  task = text_format.Parse(Path(config.task.path).read_text(), task_pb2.Task())
  task_manager = task_manager_lib.TaskManager(task)
  config.simulator.adb_controller.adb_path = os.path.expanduser(
      config.simulator.adb_controller.adb_path
  )
  simulator = AuthenticatedEmulatorSimulator(config.simulator, token)
  try:
    settings = device_settings.DeviceSettings(simulator)
    coordinator = coordinator_lib.Coordinator(simulator, task_manager, settings)
    return environment.AndroidEnv(
        simulator=simulator, coordinator=coordinator, task_manager=task_manager
    )
  except BaseException:
    simulator.close()
    raise
