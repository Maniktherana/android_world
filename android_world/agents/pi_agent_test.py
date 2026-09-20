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

"""Tests for the Pi wrapper. Every test runs offline, with no tmux or device."""

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from unittest import mock

from absl.testing import absltest
from android_world.agents import pi_agent
from android_world.env import json_action


def _tempdir(test: absltest.TestCase) -> str:
  """Returns a throwaway directory. absl flags are unparsed under pytest."""
  path = tempfile.mkdtemp()
  test.addCleanup(shutil.rmtree, path, True)
  return path


def _canned(text: str | None, **overrides) -> pi_agent.PiRunResult:
  """Builds the PiRunResult a session with this final message would return."""
  result, answer = pi_agent.parse_final_lines(text)
  fields = dict(
      exit_code=0 if text is not None else None,
      timed_out=text is None,
      duration_sec=1.0,
      transcript=f'{text or ""}\n',
      log_path='/tmp/pi-session.jsonl',
      tool_calls=3,
      answer=answer,
      result=result,
  )
  fields.update(overrides)
  return pi_agent.PiRunResult(**fields)


def _trace(*, stopped: bool) -> pi_agent.PiTaskTrace:
  """Builds trace metadata returned by the Agentsims CLI."""
  return pi_agent.PiTaskTrace(
      device='android:emulator-5554',
      trace_id='android-emulator-5554-20260919-120000-task',
      directory='/tmp/agentsims-trace',
      started_at='2026-09-19T12:00:00.000Z',
      ended_at='2026-09-19T12:01:00.000Z' if stopped else None,
      calls=4 if stopped else None,
      name='turn-on-wifi-android-emulator-5554-20260919T120000Z',
  )


class FakeEnv:
  """Records the actions an agent executes."""

  def __init__(self):
    self.controller = mock.MagicMock()
    self.actions: list[json_action.JSONAction] = []

  def execute_action(self, action: json_action.JSONAction) -> None:
    self.actions.append(action)


class FakeSession:
  """Stands in for a PiTmuxSession with canned turns."""

  def __init__(self, first: pi_agent.PiRunResult, follow_up=None):
    self._first = first
    self._follow_up = follow_up
    self.prompts: list[str] = []
    self.continued: list[tuple[str, float]] = []

  @property
  def attach_command(self) -> str:
    return 'tmux attach -t fake'

  def run_task(
      self, prompt: str, timeout_sec: float, before_submit=None
  ) -> pi_agent.PiRunResult:
    del timeout_sec
    self.prompts.append(prompt)
    if before_submit is not None:
      before_submit()
    return self._first

  def continue_task(self, text: str, timeout_sec: float):
    self.continued.append((text, timeout_sec))
    return self._follow_up


class ParseFinalLinesTest(absltest.TestCase):

  def test_no_text(self):
    self.assertEqual(pi_agent.parse_final_lines(None), (None, None))
    self.assertEqual(pi_agent.parse_final_lines(''), (None, None))

  def test_last_line_wins(self):
    text = 'RESULT: done\nsome prose\nRESULT: failed'
    self.assertEqual(pi_agent.parse_final_lines(text), ('failed', None))

  def test_last_answer_wins(self):
    text = 'ANSWER: 12\nreconsidering\nANSWER: 25'
    self.assertEqual(pi_agent.parse_final_lines(text), (None, '25'))

  def test_prefixes_are_case_insensitive(self):
    text = 'result: DONE\nAnswer: 25'
    self.assertEqual(pi_agent.parse_final_lines(text), ('done', '25'))

  def test_empty_payload_is_none(self):
    self.assertEqual(
        pi_agent.parse_final_lines('RESULT:\nANSWER:   '), (None, None)
    )

  def test_answer_with_commas_passes_through(self):
    text = 'ANSWER: eggs, milk, bread'
    self.assertEqual(
        pi_agent.parse_final_lines(text), (None, 'eggs, milk, bread')
    )

  def test_result_and_answer_together(self):
    text = 'Here you go.\nRESULT: done\nANSWER: 25'
    self.assertEqual(pi_agent.parse_final_lines(text), ('done', '25'))

  def test_answer_keeps_inner_colons_and_case(self):
    text = 'ANSWER: 15:30 on Tuesday'
    self.assertEqual(pi_agent.parse_final_lines(text), (None, '15:30 on Tuesday'))


class LooksLikeQuestionTest(absltest.TestCase):

  def test_question_goals(self):
    for goal in (
        'How many messages are unread?',
        'What is the sender of the most recent message',
        'Which city has the highest temperature',
        'Answer with the name only.',
        'Express your answer in minutes',
        'When did the last event finish',
    ):
      self.assertTrue(pi_agent.looks_like_question(goal), goal)

  def test_action_goals(self):
    for goal in (
        'Turn on wifi.',
        'Create a new contact named Alice',
        'Delete the top note in Markor',
    ):
      self.assertFalse(pi_agent.looks_like_question(goal), goal)


class PasteTest(absltest.TestCase):

  def _session(self) -> pi_agent.PiTmuxSession:
    return pi_agent.PiTmuxSession(
        'test-pi',
        pi_binary='/usr/bin/pi',
        provider='provider',
        model='model',
        skill_path='/skill',
        thinking='high',
        workdir=pathlib.Path(_tempdir(self)),
    )

  def test_raises_transport_error_after_three_failures(self):
    session = self._session()
    error = subprocess.CalledProcessError(133, ['tmux', 'load-buffer'])
    with mock.patch.object(
        pi_agent.subprocess, 'run', side_effect=error
    ) as run, mock.patch.object(pi_agent.time, 'sleep') as sleep:
      with self.assertRaises(pi_agent.PiTransportError):
        session._paste('a prompt')

    self.assertEqual(run.call_count, pi_agent._PASTE_ATTEMPTS)
    self.assertEqual(sleep.call_count, pi_agent._PASTE_ATTEMPTS - 1)
    sleep.assert_called_with(0.5)

  def test_succeeds_on_second_attempt(self):
    session = self._session()
    error = subprocess.CalledProcessError(133, ['tmux', 'load-buffer'])
    with mock.patch.object(
        pi_agent.subprocess,
        'run',
        side_effect=[error, mock.Mock(), mock.Mock()],
    ) as run, mock.patch.object(pi_agent.time, 'sleep') as sleep:
      session._paste('a prompt')

    self.assertEqual(run.call_count, 3)
    self.assertEqual(sleep.call_count, 1)
    commands = [call.args[0] for call in run.call_args_list]
    self.assertEqual([c[1] for c in commands],
                     ['load-buffer', 'load-buffer', 'paste-buffer'])
    # A fresh temp file per attempt, and none of them left behind.
    first, second = commands[0][-1], commands[1][-1]
    self.assertNotEqual(first, second)
    self.assertFalse(os.path.exists(first))
    self.assertFalse(os.path.exists(second))


class RunTaskTransportTest(absltest.TestCase):

  def _session(self) -> pi_agent.PiTmuxSession:
    return pi_agent.PiTmuxSession(
        'test-pi',
        pi_binary='/usr/bin/pi',
        provider='provider',
        model='model',
        skill_path='/skill',
        thinking='high',
        workdir=pathlib.Path(_tempdir(self)),
    )

  def test_restarts_the_tui_and_retries_once(self):
    session = self._session()
    good = _canned('RESULT: done')
    with mock.patch.object(session, 'kill') as kill, mock.patch.object(
        session, 'start'
    ) as start, mock.patch.object(
        session,
        '_run_task_once',
        side_effect=[pi_agent.PiTransportError('load-buffer died'), good],
    ):
      result = session.run_task('a prompt', 5.0)

    kill.assert_called_once()
    start.assert_called_once()
    self.assertEqual(result.result, 'done')

  def test_reports_failure_instead_of_raising(self):
    session = self._session()
    with mock.patch.object(session, 'kill'), mock.patch.object(
        session, 'start'
    ), mock.patch.object(
        session,
        '_run_task_once',
        side_effect=pi_agent.PiTransportError('load-buffer died'),
    ):
      result = session.run_task('a prompt', 5.0)

    self.assertEqual(result.result, 'failed')
    self.assertFalse(result.timed_out)
    self.assertIsNone(result.exit_code)
    self.assertIsNone(result.log_path)
    self.assertIn('load-buffer died', result.transcript)

  def test_a_paste_fault_never_escapes_the_send_path(self):
    session = self._session()
    with mock.patch.object(
        session, '_paste', side_effect=pi_agent.PiTransportError('SIGTRAP')
    ), mock.patch.object(session, '_send_line'), mock.patch.object(
        session, 'kill'
    ), mock.patch.object(session, 'start'), mock.patch.object(
        pi_agent.time, 'sleep'
    ):
      result = session.run_task('a prompt', 5.0)

    self.assertEqual(result.result, 'failed')
    self.assertIn('SIGTRAP', result.transcript)

  def test_before_submit_hook_runs_once_when_transport_retries(self):
    session = self._session()
    calls = []
    attempts = 0

    def run_once(prompt, timeout_sec, before_submit):
      nonlocal attempts
      del prompt, timeout_sec
      attempts += 1
      before_submit()
      if attempts == 1:
        raise pi_agent.PiTransportError('submit failed')
      return _canned('RESULT: done')

    with mock.patch.object(
        session, '_run_task_once', side_effect=run_once
    ), mock.patch.object(session, 'kill'), mock.patch.object(session, 'start'):
      result = session.run_task(
          'a prompt', 5.0, before_submit=lambda: calls.append('start')
      )

    self.assertEqual(result.result, 'done')
    self.assertEqual(calls, ['start'])

  def test_hook_starts_trace_immediately_before_prompt_execution(self):
    session = self._session()
    events = []
    with mock.patch.object(session, '_send_line'), mock.patch.object(
        session, '_paste', side_effect=lambda prompt: events.append('paste')
    ), mock.patch.object(
        session, '_submit', side_effect=lambda: events.append('submit')
    ), mock.patch.object(
        session, '_await_new_session', return_value=None
    ), mock.patch.object(pi_agent.time, 'sleep'):
      session._run_task_once(
          'a prompt',
          5.0,
          before_submit=lambda: events.append('start'),
      )

    self.assertEqual(events, ['paste', 'start', 'submit'])


class PiAgentTest(absltest.TestCase):

  def _agent(self, **kwargs) -> tuple[pi_agent.PiAgent, FakeEnv]:
    env = FakeEnv()
    with mock.patch.object(
        pi_agent.shutil, 'which', return_value='/usr/bin/pi'
    ):
      agent = pi_agent.PiAgent(
          env,  # pytype: disable=wrong-arg-types
          'android:emulator-5554',
          skill_path=_tempdir(self),
          output_dir=_tempdir(self),
          **kwargs,
      )
    return agent, env

  def test_device_time_returns_the_decoded_clock(self):
    agent, _ = self._agent()
    response = mock.Mock()
    response.generic.output = b'Tuesday, 03 October 2023, 15:26\n'
    with mock.patch.object(
        pi_agent.adb_utils, 'issue_generic_request', return_value=response
    ):
      self.assertEqual(
          agent._device_time(), 'Tuesday, 03 October 2023, 15:26'
      )

  def test_device_time_falls_back_when_adb_raises(self):
    agent, _ = self._agent()
    with mock.patch.object(
        pi_agent.adb_utils,
        'issue_generic_request',
        side_effect=RuntimeError('adb is gone'),
    ):
      self.assertStartsWith(agent._device_time(), 'unknown;')

  def test_device_time_falls_back_on_empty_output(self):
    agent, _ = self._agent()
    response = mock.Mock()
    response.generic.output = b'  \n'
    with mock.patch.object(
        pi_agent.adb_utils, 'issue_generic_request', return_value=response
    ):
      self.assertStartsWith(agent._device_time(), 'unknown;')

  def test_step_forwards_the_answer_once(self):
    agent, env = self._agent()
    with mock.patch.object(
        agent, 'run_pi_tmux', return_value=_canned('ANSWER: 25')
    ):
      step = agent.step('How many messages are unread?')

    self.assertTrue(step.done)
    self.assertLen(env.actions, 1)
    self.assertEqual(env.actions[0].action_type, json_action.ANSWER)
    self.assertEqual(env.actions[0].action_type, 'answer')
    self.assertEqual(env.actions[0].text, '25')
    self.assertEqual(step.data['pi_answer'], '25')
    self.assertFalse(step.data['pi_followed_up'])

  def test_step_does_not_act_without_an_answer(self):
    agent, env = self._agent()
    with mock.patch.object(
        agent, 'run_pi_tmux', return_value=_canned('RESULT: done')
    ):
      step = agent.step('Turn on wifi.')

    self.assertTrue(step.done)
    self.assertEmpty(env.actions)
    self.assertEqual(step.data['pi_result'], 'done')
    self.assertIsNone(step.data['pi_answer'])

  def test_step_reports_done_after_a_transport_failure(self):
    agent, env = self._agent()
    failed = _canned(None, result='failed', timed_out=False, log_path=None,
                     transcript='Transport error: load-buffer died\n')
    with mock.patch.object(agent, 'run_pi_tmux', return_value=failed):
      step = agent.step('Turn on wifi.')

    self.assertTrue(step.done)
    self.assertEmpty(env.actions)
    self.assertEqual(step.data['pi_result'], 'failed')


class TraceLifecycleTest(absltest.TestCase):

  def _agent(self) -> tuple[pi_agent.PiAgent, FakeEnv]:
    env = FakeEnv()
    with mock.patch.object(
        pi_agent.shutil,
        'which',
        side_effect=lambda binary: f'/usr/bin/{binary}',
    ):
      agent = pi_agent.PiAgent(
          env,  # pytype: disable=wrong-arg-types
          'android:emulator-5554',
          skill_path=_tempdir(self),
          output_dir=_tempdir(self),
      )
    return agent, env

  def test_step_stops_trace_before_forwarding_answer(self):
    agent, env = self._agent()
    events = []
    session = mock.Mock()
    agent._tmux = session

    def forward_answer(action):
      events.append('answer')
      env.actions.append(action)

    def start_trace(goal):
      del goal
      events.append('start')
      return _trace(stopped=False)

    def run_task(prompt, timeout_sec, before_submit):
      del prompt, timeout_sec
      events.append('prompt-ready')
      before_submit()
      events.append('run')
      return _canned('ANSWER: 25')

    def stop_trace():
      events.append('stop')
      return _trace(stopped=True)

    env.execute_action = forward_answer
    session.run_task.side_effect = run_task
    with mock.patch.object(
        agent, '_device_time', return_value='Tuesday'
    ), mock.patch.object(
        agent, '_start_trace', side_effect=start_trace
    ), mock.patch.object(
        agent,
        '_stop_trace',
        side_effect=stop_trace,
    ):
      step = agent.step('How many messages are unread?')

    self.assertEqual(
        events, ['prompt-ready', 'start', 'run', 'stop', 'answer']
    )
    self.assertEqual(step.data['pi_trace_id'], _trace(stopped=True).trace_id)
    self.assertEqual(
        step.data['pi_trace_name'],
        'turn-on-wifi-android-emulator-5554-20260919T120000Z',
    )
    self.assertEqual(step.data['pi_trace_directory'], '/tmp/agentsims-trace')
    self.assertEqual(step.data['pi_trace_calls'], 4)

  def test_step_stops_trace_when_pi_raises(self):
    agent, _ = self._agent()
    session = mock.Mock()
    agent._tmux = session

    def run_task(prompt, timeout_sec, before_submit):
      del prompt, timeout_sec
      before_submit()
      raise RuntimeError('Pi failed')

    session.run_task.side_effect = run_task
    with mock.patch.object(
        agent, '_device_time', return_value='Tuesday'
    ), mock.patch.object(
        agent, '_start_trace', return_value=_trace(stopped=False)
    ), mock.patch.object(
        agent, '_stop_trace', return_value=_trace(stopped=True)
    ) as stop:
      with self.assertRaisesRegex(RuntimeError, 'Pi failed'):
        agent.step('Turn on wifi.')

    stop.assert_called_once_with()

  def test_trace_start_failure_prevents_pi_from_running(self):
    agent, _ = self._agent()
    session = mock.Mock()
    agent._tmux = session
    ran = []

    def run_task(prompt, timeout_sec, before_submit):
      del prompt, timeout_sec
      before_submit()
      ran.append(True)
      return _canned('RESULT: done')

    session.run_task.side_effect = run_task
    with mock.patch.object(
        agent, '_device_time', return_value='Tuesday'
    ), mock.patch.object(
        agent, '_start_trace', side_effect=pi_agent.PiTraceError('start failed')
    ), mock.patch.object(
        agent, '_stop_trace'
    ) as stop:
      with self.assertRaisesRegex(pi_agent.PiTraceError, 'start failed'):
        agent.step('Turn on wifi.')

    self.assertEmpty(ran)
    stop.assert_not_called()

  def test_trace_stop_failure_prevents_answer_forwarding(self):
    agent, env = self._agent()
    session = FakeSession(_canned('ANSWER: 25'))
    agent._tmux = session
    with mock.patch.object(
        agent, '_device_time', return_value='Tuesday'
    ), mock.patch.object(
        agent, '_start_trace', return_value=_trace(stopped=False)
    ), mock.patch.object(
        agent, '_stop_trace', side_effect=pi_agent.PiTraceError('stop failed')
    ):
      with self.assertRaisesRegex(pi_agent.PiTraceError, 'stop failed'):
        agent.step('How many messages are unread?')

    self.assertEmpty(env.actions)

  def test_trace_commands_use_the_selected_device_and_name(self):
    agent, _ = self._agent()
    agent.set_task_name('ContactsAddContact')
    started = {
        'device': 'android:emulator-5554',
        'id': _trace(stopped=False).trace_id,
        'directory': '/tmp/agentsims-trace',
        'startedAt': '2026-09-19T12:00:00.000Z',
    }
    stopped = {
        **started,
        'endedAt': '2026-09-19T12:01:00.000Z',
        'calls': 4,
    }
    with mock.patch.object(
        pi_agent.time, 'strftime', return_value='20260919T120000Z'
    ), mock.patch.object(
        pi_agent.subprocess,
        'run',
        side_effect=[
            subprocess.CompletedProcess([], 0, json.dumps(started), ''),
            subprocess.CompletedProcess([], 0, json.dumps(stopped), ''),
        ],
    ) as run:
      agent._start_trace('Turn on wifi.')
      agent._stop_trace()

    start_command = run.call_args_list[0].args[0]
    stop_command = run.call_args_list[1].args[0]
    self.assertEqual(
        start_command,
        [
            '/usr/bin/agentsims',
            'trace',
            'start',
            '-d',
            'android:emulator-5554',
            '--name',
            'contactsaddcontact-android-emulator-5554-20260919T120000Z',
            '--json',
        ],
    )
    self.assertEqual(
        stop_command,
        [
            '/usr/bin/agentsims',
            'trace',
            'stop',
            '-d',
            'android:emulator-5554',
            '--json',
        ],
    )


class FollowUpTest(absltest.TestCase):

  def _agent(self, session) -> tuple[pi_agent.PiAgent, FakeEnv]:
    env = FakeEnv()
    with mock.patch.object(
        pi_agent.shutil, 'which', return_value='/usr/bin/pi'
    ):
      agent = pi_agent.PiAgent(
          env,  # pytype: disable=wrong-arg-types
          'android:emulator-5554',
          skill_path=_tempdir(self),
          output_dir=_tempdir(self),
      )
    agent._tmux = session  # Skip the real tmux start-up.
    self.enter_context(
        mock.patch.object(agent, '_device_time', return_value='Tuesday')
    )
    self.enter_context(
        mock.patch.object(
            agent, '_start_trace', return_value=_trace(stopped=False)
        )
    )
    self.enter_context(
        mock.patch.object(
            agent, '_stop_trace', return_value=_trace(stopped=True)
        )
    )
    return agent, env

  def test_question_goal_without_an_answer_asks_once(self):
    session = FakeSession(_canned('RESULT: done'), _canned('ANSWER: 25'))
    agent, _ = self._agent(session)

    result = agent.run_pi_tmux('How many messages are unread?')

    self.assertLen(session.continued, 1)
    self.assertEqual(session.continued[0][0], 'Reply with only the ANSWER: line.')
    self.assertEqual(result.answer, '25')
    self.assertEqual(result.result, 'done')
    self.assertTrue(result.followed_up)

  def test_follow_up_is_capped_at_one(self):
    # Pi still gives no answer, so the wrapper stops asking.
    session = FakeSession(_canned('RESULT: done'), _canned('still working'))
    agent, _ = self._agent(session)

    result = agent.run_pi_tmux('How many messages are unread?')

    self.assertLen(session.continued, 1)
    self.assertIsNone(result.answer)
    self.assertTrue(result.followed_up)

  def test_session_without_a_result_line_also_asks(self):
    session = FakeSession(_canned('All done!'), _canned('ANSWER: 25'))
    agent, _ = self._agent(session)

    result = agent.run_pi_tmux('What is the sender of the newest message')

    self.assertLen(session.continued, 1)
    self.assertEqual(result.answer, '25')

  def test_action_goal_never_asks(self):
    session = FakeSession(_canned('RESULT: done'), _canned('ANSWER: 25'))
    agent, _ = self._agent(session)

    result = agent.run_pi_tmux('Turn on wifi.')

    self.assertEmpty(session.continued)
    self.assertIsNone(result.answer)
    self.assertFalse(result.followed_up)

  def test_existing_answer_never_asks(self):
    session = FakeSession(_canned('ANSWER: 25'), _canned('ANSWER: 99'))
    agent, _ = self._agent(session)

    result = agent.run_pi_tmux('How many messages are unread?')

    self.assertEmpty(session.continued)
    self.assertEqual(result.answer, '25')

  def test_failed_session_never_asks(self):
    session = FakeSession(_canned('RESULT: failed'), _canned('ANSWER: 25'))
    agent, _ = self._agent(session)

    result = agent.run_pi_tmux('How many messages are unread?')

    self.assertEmpty(session.continued)
    self.assertIsNone(result.answer)

  def test_timed_out_session_never_asks(self):
    session = FakeSession(_canned(None), _canned('ANSWER: 25'))
    agent, _ = self._agent(session)

    result = agent.run_pi_tmux('How many messages are unread?')

    self.assertEmpty(session.continued)
    self.assertIsNone(result.answer)

  def test_transport_failure_in_the_follow_up_is_swallowed(self):
    session = FakeSession(_canned('RESULT: done'))
    agent, _ = self._agent(session)
    with mock.patch.object(
        session,
        'continue_task',
        side_effect=pi_agent.PiTransportError('load-buffer died'),
    ):
      result = agent.run_pi_tmux('How many messages are unread?')

    self.assertIsNone(result.answer)
    self.assertFalse(result.followed_up)

  def test_log_records_the_answer_and_the_follow_up(self):
    session = FakeSession(_canned('RESULT: done'), _canned('ANSWER: 25'))
    agent, _ = self._agent(session)

    agent.run_pi_tmux('How many messages are unread?')

    logs = sorted(pathlib.Path(str(agent._output_dir)).glob('*.log'))
    self.assertLen(logs, 1)
    written = logs[0].read_text()
    self.assertIn('pi_result: done', written)
    self.assertIn('pi_answer: 25', written)
    self.assertIn('pi_followed_up: True', written)


if __name__ == '__main__':
  absltest.main()
