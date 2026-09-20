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

"""Offline tests for the AndroidWorld Codex CLI agent."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from android_world.agents import codex_agent
from android_world.env import json_action


def _tempdir(test: absltest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, path, True)
    return path


def _trace(*, stopped: bool) -> codex_agent.CodexTaskTrace:
    return codex_agent.CodexTaskTrace(
        device="android:emulator-5554",
        trace_id="trace-1",
        directory="/tmp/agentsims-trace",
        started_at="2026-09-19T12:00:00.000Z",
        ended_at="2026-09-19T12:01:00.000Z" if stopped else None,
        calls=4 if stopped else None,
        name="turn-on-wifi-android-emulator-5554-20260919T120000Z",
    )


class FakeEnv:
    """Records AndroidWorld actions without opening a device."""

    def __init__(self):
        self.controller = mock.MagicMock()
        self.actions: list[json_action.JSONAction] = []

    def execute_action(self, action: json_action.JSONAction) -> None:
        self.actions.append(action)

    def reset(self, go_home: bool = False) -> None:
        del go_home


class FakeSession:
    """Runs one canned Codex task and records lifecycle order."""

    def __init__(
        self,
        result: codex_agent.CodexRunResult,
        events: list[str] | None = None,
    ):
        self.result = result
        self.events = events if events is not None else []
        self.prompts: list[str] = []

    @property
    def attach_command(self) -> str:
        return "tmux attach -r -t fake"

    def run_task(
        self,
        prompt: str,
        timeout_sec: float,
        *,
        artifact_stem: Path,
        before_submit=None,
    ) -> codex_agent.CodexRunResult:
        del timeout_sec, artifact_stem
        self.prompts.append(prompt)
        self.events.append("command-loaded")
        if before_submit is not None:
            before_submit()
        self.events.append("prompt-submitted")
        self.events.append("codex-exited")
        return self.result


class ParseFinalLinesTest(absltest.TestCase):
    def test_last_values_win(self):
        text = "RESULT: failed\nANSWER: first\nRESULT: DONE\nANSWER: Final"
        self.assertEqual(codex_agent.parse_final_lines(text), ("done", "Final"))

    def test_empty_values_are_none(self):
        self.assertEqual(
            codex_agent.parse_final_lines("RESULT:\nANSWER:  "), (None, None)
        )


class AzureConfigTest(absltest.TestCase):
    def test_tmux_attach_is_read_only(self):
        session = codex_agent.CodexTmuxSession(
            "test",
            codex_binary="/opt/homebrew/bin/codex",
            workdir=_tempdir(self),
            model="gpt-5.6-sol",
            reasoning="high",
            provider="azure",
            azure_base_url="https://example.openai.azure.com/openai",
            azure_api_version="2025-04-01-preview",
            azure_api_key_env="TEST_AZURE_KEY",
            model_catalog_path=Path("/tmp/model-catalog.json"),
        )
        self.assertEqual(session.attach_command, "tmux attach -r -t test")

    def test_resource_endpoint_gets_openai_suffix(self):
        self.assertEqual(
            codex_agent._normalize_azure_base_url("https://example.openai.azure.com/"),
            "https://example.openai.azure.com/openai",
        )

    def test_codex_command_uses_key_name_not_secret(self):
        session = codex_agent.CodexTmuxSession(
            "test",
            codex_binary="/opt/homebrew/bin/codex",
            workdir=_tempdir(self),
            model="gpt-5.6-sol",
            reasoning="high",
            provider="azure",
            azure_base_url="https://example.openai.azure.com/openai",
            azure_api_version="2025-04-01-preview",
            azure_api_key_env="TEST_AZURE_KEY",
            model_catalog_path=Path("/tmp/model-catalog.json"),
        )
        with mock.patch.dict(os.environ, {"TEST_AZURE_KEY": "top-secret"}):
            command = session._codex_command(Path("/tmp/final.txt"))

        joined = " ".join(command)
        self.assertIn("TEST_AZURE_KEY", joined)
        self.assertNotIn("top-secret", joined)
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("gpt-5.6-sol", command)
        self.assertIn('approval_policy="never"', command)
        self.assertIn("model_catalog_json", joined)
        self.assertIn("skills.include_instructions=false", command)

    def test_azure_catalog_disables_unsupported_luna_modes(self):
        destination = _tempdir(self) / "catalog.json"
        bundled = {
            "models": [
                {
                    "slug": "gpt-5.6-sol",
                    "use_responses_lite": True,
                    "multi_agent_version": "v1",
                }
            ]
        }
        completed = mock.Mock(returncode=0, stdout=json.dumps(bundled), stderr="")
        with mock.patch.object(codex_agent.subprocess, "run", return_value=completed):
            codex_agent._write_azure_model_catalog(
                "/usr/local/bin/codex", "gpt-5.6-sol", destination
            )

        model = json.loads(destination.read_text())["models"][0]
        self.assertFalse(model["use_responses_lite"])
        self.assertIsNone(model["multi_agent_version"])


class CodexAgentTest(absltest.TestCase):
    def _agent(self) -> codex_agent.CodexAgent:
        root = _tempdir(self)
        skill = root / "SKILL.md"
        skill.write_text("# Test skill\n")

        def find_binary(name: str) -> str:
            return f"/usr/local/bin/{name}"

        with (
            mock.patch.dict(os.environ, {"TEST_AZURE_KEY": "top-secret"}),
            mock.patch.object(codex_agent.shutil, "which", side_effect=find_binary),
            mock.patch.object(codex_agent.CodexTmuxSession, "start"),
            mock.patch.object(codex_agent, "_write_azure_model_catalog"),
        ):
            return codex_agent.CodexAgent(
                FakeEnv(),
                "android:emulator-5554",
                azure_base_url="https://example.openai.azure.com/openai",
                azure_api_key_env="TEST_AZURE_KEY",
                skill_path=str(skill),
                output_dir=str(root / "output"),
            )

    def test_run_starts_trace_after_load_and_stops_after_exit(self):
        agent = self._agent()
        events: list[str] = []
        result = codex_agent.CodexRunResult(
            exit_code=0,
            timed_out=False,
            duration_sec=1.0,
            transcript="RESULT: done\n",
            log_path="/tmp/codex.log",
            result="done",
        )
        agent._tmux = FakeSession(result, events)

        def start_trace(goal: str) -> codex_agent.CodexTaskTrace:
            del goal
            events.append("trace-started")
            return _trace(stopped=False)

        def stop_trace() -> codex_agent.CodexTaskTrace:
            events.append("trace-stopped")
            return _trace(stopped=True)

        with (
            mock.patch.object(agent, "_device_time", return_value="today"),
            mock.patch.object(agent, "_start_trace", side_effect=start_trace),
            mock.patch.object(agent, "_stop_trace", side_effect=stop_trace),
        ):
            actual = agent.run_codex("Turn on Wi-Fi")

        self.assertEqual(
            events,
            [
                "command-loaded",
                "trace-started",
                "prompt-submitted",
                "codex-exited",
                "trace-stopped",
            ],
        )
        self.assertEqual(actual.trace, _trace(stopped=True))

    def test_step_forwards_answer_to_android_world(self):
        agent = self._agent()
        result = codex_agent.CodexRunResult(
            exit_code=0,
            timed_out=False,
            duration_sec=1.0,
            transcript="ANSWER: 42\n",
            log_path="/tmp/codex.log",
            answer="42",
            trace=_trace(stopped=True),
        )
        with mock.patch.object(agent, "run_codex", return_value=result):
            interaction = agent.step("How many?")

        self.assertTrue(interaction.done)
        self.assertEqual(
            agent.env.actions,
            [json_action.JSONAction(action_type=json_action.ANSWER, text="42")],
        )
        self.assertEqual(interaction.data["codex_trace_id"], "trace-1")

    def test_missing_key_is_rejected_before_tmux_starts(self):
        root = _tempdir(self)
        skill = root / "SKILL.md"
        skill.write_text("# Test skill\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(EnvironmentError, "MISSING_AZURE_KEY"):
                codex_agent.CodexAgent(
                    FakeEnv(),
                    "android:emulator-5554",
                    azure_base_url="https://example.openai.azure.com/openai",
                    azure_api_key_env="MISSING_AZURE_KEY",
                    skill_path=str(skill),
                )


if __name__ == "__main__":
    absltest.main()
