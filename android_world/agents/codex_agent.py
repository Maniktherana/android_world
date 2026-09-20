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

"""Runs Codex CLI as an AndroidWorld agent through Agentsims."""

import dataclasses
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from android_world.agents import base_agent, pi_render
from android_world.env import adb_utils, interface, json_action

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING = "high"
DEFAULT_AZURE_BASE_URL = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_AZURE_API_KEY_ENV = "AZURE_OPENAI_API_KEY"
DEFAULT_SKILL_PATH = "~/code/agentsims/skills/build-mobile-apps/SKILL.md"
DEFAULT_TMUX_SESSION = "androidworld-codex"

_TRACE_COMMAND_TIMEOUT_SEC = 30.0
_POLL_INTERVAL_SEC = 0.25
_INTERRUPT_GRACE_SEC = 5.0

_BENCHMARK_RULES = """Device: {device}
Pass -d {device} to every Agentsims command. Ignore every other device.
The device clock reads {device_time}. That is today. Use it for every relative
date such as today, this week, or next week.

Task: {goal}

Before acting, read and follow this mobile-device skill:
{skill_path}

Rules:
- Work only through the device UI with {agentsims}. Do not use adb, sqlite,
  content providers, broadcasts, or direct device file or database writes.
- Do not edit workspace files. Agentsims screenshots for visual inspection are
  allowed.
- Do not reboot, wipe, uninstall, or install apps unless the task asks.
- Do not change the device date, time, or timezone.
- A successful command only proves that the command ran. Inspect the resulting
  app state before deciding that the task is complete.
- There is no time limit other than the {timeout_min} minute budget. Finish
  every part of the task.

Your final response must be exactly one line. For an action task, use one of:
RESULT: done
RESULT: failed
If the task asks a question, use:
ANSWER: <the answer in exactly the format requested by the task>
"""


class CodexTransportError(RuntimeError):
    """Raised when tmux cannot start a Codex task."""


class CodexTraceError(RuntimeError):
    """Raised when Agentsims cannot manage a task trace."""


class CodexConfigurationError(RuntimeError):
    """Raised when Codex cannot prepare its Azure model configuration."""


@dataclasses.dataclass(frozen=True)
class CodexTaskTrace:
    """Agentsims trace metadata for one AndroidWorld task."""

    device: str
    trace_id: str
    directory: str
    started_at: str
    ended_at: str | None = None
    calls: int | None = None
    name: str | None = None


@dataclasses.dataclass(frozen=True)
class CodexRunResult:
    """Outcome of one Codex CLI task."""

    exit_code: int | None
    timed_out: bool
    duration_sec: float
    transcript: str
    log_path: str | None
    answer: str | None = None
    result: str | None = None
    trace: CodexTaskTrace | None = None


def parse_final_lines(text: str | None) -> tuple[str | None, str | None]:
    """Returns the final RESULT and ANSWER values from Codex output."""
    result = answer = None
    if not text:
        return result, answer
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("RESULT:"):
            result = stripped.split(":", 1)[1].strip().lower() or None
        elif stripped.upper().startswith("ANSWER:"):
            answer = stripped.split(":", 1)[1].strip() or None
    return result, answer


def _toml_string(value: str) -> str:
    """Returns a TOML-compatible quoted string."""
    return json.dumps(value)


def _normalize_azure_base_url(value: str) -> str:
    """Returns the Azure Responses API base URL expected by Codex."""
    base_url = value.strip().rstrip("/")
    if not base_url:
        raise ValueError(
            "Set --codex_azure_base_url or AZURE_OPENAI_ENDPOINT for Codex."
        )
    if not base_url.startswith("https://"):
        raise ValueError("Use an https Azure OpenAI endpoint.")
    if base_url.endswith(".openai.azure.com"):
        base_url += "/openai"
    return base_url


def _write_azure_model_catalog(
    codex_binary: str, model: str, destination: Path
) -> None:
    """Writes Codex's bundled catalog with Azure-incompatible modes disabled."""
    try:
        completed = subprocess.run(
            [codex_binary, "debug", "models", "--bundled"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CodexConfigurationError(
            f"Codex could not read its bundled model catalog: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CodexConfigurationError(
            "Codex could not read its bundled model catalog: "
            f"{detail or completed.returncode}"
        )
    try:
        catalog = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise CodexConfigurationError(
            "Codex returned an invalid bundled model catalog."
        ) from error
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list):
        raise CodexConfigurationError("Codex returned an invalid model catalog.")
    selected = next(
        (
            item
            for item in models
            if isinstance(item, dict) and item.get("slug") == model
        ),
        None,
    )
    if selected is None:
        raise CodexConfigurationError(
            f"Model {model!r} is not in this Codex CLI model catalog."
        )
    selected["use_responses_lite"] = False
    selected["multi_agent_version"] = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(catalog, indent=2) + "\n")


class CodexTmuxSession:
    """Runs fresh Codex exec processes in one visible tmux console."""

    def __init__(
        self,
        name: str,
        *,
        codex_binary: str,
        workdir: Path,
        model: str,
        reasoning: str,
        provider: str,
        azure_base_url: str,
        azure_api_version: str,
        azure_api_key_env: str,
        model_catalog_path: Path,
    ):
        self._name = name
        self._codex = codex_binary
        self._workdir = workdir
        self._model = model
        self._reasoning = reasoning
        self._provider = provider
        self._azure_base_url = azure_base_url
        self._azure_api_version = azure_api_version
        self._azure_api_key_env = azure_api_key_env
        self._model_catalog_path = model_catalog_path

    @property
    def attach_command(self) -> str:
        return f"tmux attach -r -t {self._name}"

    def _tmux(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["tmux", *args], capture_output=True, text=True, check=check
        )

    def exists(self) -> bool:
        return self._tmux("has-session", "-t", self._name, check=False).returncode == 0

    def start(self) -> None:
        """Starts a clean tmux shell that remains for the whole benchmark."""
        if shutil.which("tmux") is None:
            raise FileNotFoundError("tmux is required for the Codex live console.")
        if self.exists():
            self._tmux("kill-session", "-t", self._name, check=False)
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._tmux(
            "new-session",
            "-d",
            "-s",
            self._name,
            "-x",
            "200",
            "-y",
            "50",
            "-c",
            str(self._workdir),
        )

    def _codex_command(self, last_message_path: Path) -> list[str]:
        configs = [
            f"model_provider={_toml_string(self._provider)}",
            f"model_providers.{self._provider}.name=" + _toml_string("Azure OpenAI"),
            (
                f"model_providers.{self._provider}.base_url="
                f"{_toml_string(self._azure_base_url)}"
            ),
            (
                f"model_providers.{self._provider}.env_key="
                f"{_toml_string(self._azure_api_key_env)}"
            ),
            (
                f"model_providers.{self._provider}.query_params="
                f'{{ "api-version" = {_toml_string(self._azure_api_version)} }}'
            ),
            f'model_providers.{self._provider}.wire_api="responses"',
            f"model_reasoning_effort={_toml_string(self._reasoning)}",
            f"model_catalog_json={_toml_string(str(self._model_catalog_path))}",
            'approval_policy="never"',
            "skills.include_instructions=false",
            "agents.enabled=false",
        ]
        command = [
            self._codex,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--color",
            "always",
            "--sandbox",
            "danger-full-access",
            "--cd",
            str(self._workdir),
            "--model",
            self._model,
            "--output-last-message",
            str(last_message_path),
        ]
        for config in configs:
            command.extend(["--config", config])
        command.append("-")
        return command

    def _write_task_script(
        self,
        *,
        prompt_path: Path,
        log_path: Path,
        last_message_path: Path,
        status_path: Path,
    ) -> Path:
        """Writes a shell wrapper whose output is visible in tmux and a log."""
        script_path = status_path.with_suffix(".sh")
        temporary_status = status_path.with_suffix(".tmp")
        command = shlex.join(self._codex_command(last_message_path))
        script_path.write_text(
            "#!/usr/bin/env bash\n"
            "set -o pipefail\n"
            f"{command} < {shlex.quote(str(prompt_path))} 2>&1 | "
            f"tee {shlex.quote(str(log_path))}\n"
            "status=${PIPESTATUS[0]}\n"
            f"printf '%s\\n' \"$status\" > {shlex.quote(str(temporary_status))}\n"
            f"mv {shlex.quote(str(temporary_status))} "
            f"{shlex.quote(str(status_path))}\n"
            'exit "$status"\n'
        )
        script_path.chmod(0o700)
        return script_path

    def run_task(
        self,
        prompt: str,
        timeout_sec: float,
        *,
        artifact_stem: Path,
        before_submit: Callable[[], None] | None = None,
    ) -> CodexRunResult:
        """Runs one isolated Codex process and mirrors its live output."""
        prompt_path = artifact_stem.with_suffix(".prompt.txt")
        log_path = artifact_stem.with_suffix(".log")
        last_message_path = artifact_stem.with_suffix(".final.txt")
        status_path = artifact_stem.with_suffix(".status")
        for path in (log_path, last_message_path, status_path):
            path.unlink(missing_ok=True)
        prompt_path.write_text(prompt)
        script_path = self._write_task_script(
            prompt_path=prompt_path,
            log_path=log_path,
            last_message_path=last_message_path,
            status_path=status_path,
        )

        command = f"bash {shlex.quote(str(script_path))}"
        try:
            self._tmux("send-keys", "-t", self._name, "-l", command)
            if before_submit is not None:
                before_submit()
            self._tmux("send-keys", "-t", self._name, "Enter")
        except (OSError, subprocess.CalledProcessError) as error:
            raise CodexTransportError(
                f"tmux could not start the Codex task: {error}"
            ) from error

        start = time.time()
        deadline = start + timeout_sec
        offset = 0
        transcript: list[str] = []
        timed_out = False
        while time.time() < deadline:
            if log_path.is_file():
                with log_path.open("r", errors="replace") as log:
                    log.seek(offset)
                    chunk = log.read()
                    offset = log.tell()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    transcript.append(chunk)
            if status_path.is_file():
                break
            time.sleep(_POLL_INTERVAL_SEC)
        else:
            timed_out = True
            self._tmux("send-keys", "-t", self._name, "C-c", check=False)
            interrupt_deadline = time.time() + _INTERRUPT_GRACE_SEC
            while time.time() < interrupt_deadline and not status_path.is_file():
                time.sleep(_POLL_INTERVAL_SEC)
            if not status_path.is_file():
                self._tmux("send-keys", "-t", self._name, "C-c", check=False)
                interrupt_deadline = time.time() + _INTERRUPT_GRACE_SEC
                while time.time() < interrupt_deadline and not status_path.is_file():
                    time.sleep(_POLL_INTERVAL_SEC)
            if not status_path.is_file():
                self._tmux(
                    "respawn-pane",
                    "-k",
                    "-t",
                    self._name,
                    "-c",
                    str(self._workdir),
                    check=False,
                )

        if log_path.is_file():
            with log_path.open("r", errors="replace") as log:
                log.seek(offset)
                chunk = log.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                transcript.append(chunk)

        exit_code = None
        if status_path.is_file():
            try:
                exit_code = int(status_path.read_text().strip())
            except ValueError:
                exit_code = None
        final = (
            last_message_path.read_text().strip()
            if last_message_path.is_file()
            else None
        )
        result, answer = parse_final_lines(final)
        return CodexRunResult(
            exit_code=exit_code,
            timed_out=timed_out,
            duration_sec=time.time() - start,
            transcript="".join(transcript),
            log_path=str(log_path),
            answer=answer,
            result=result,
        )


class CodexAgent(base_agent.EnvironmentInteractingAgent):
    """Delegates a complete AndroidWorld task to an isolated Codex process."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        device_id: str,
        *,
        model: str = DEFAULT_MODEL,
        reasoning: str = DEFAULT_REASONING,
        azure_base_url: str = DEFAULT_AZURE_BASE_URL,
        azure_api_version: str = DEFAULT_AZURE_API_VERSION,
        azure_api_key_env: str = DEFAULT_AZURE_API_KEY_ENV,
        skill_path: str = DEFAULT_SKILL_PATH,
        timeout_sec: float = 900.0,
        output_dir: str | None = None,
        codex_binary: str = "codex",
        agentsims_binary: str = "agentsims",
        tmux_session: str = DEFAULT_TMUX_SESSION,
        provider: str = "azure",
        name: str = "codex",
    ):
        super().__init__(env, name)
        if timeout_sec <= 0:
            raise ValueError("Use a positive --codex_timeout_sec.")
        if not device_id:
            raise ValueError("Provide the Agentsims device ID.")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", provider):
            raise ValueError("Use a simple Codex provider ID.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", azure_api_key_env):
            raise ValueError("Use a valid API key environment variable name.")
        if not os.environ.get(azure_api_key_env):
            raise EnvironmentError(f"Set {azure_api_key_env} before running Codex.")

        resolved_codex = shutil.which(codex_binary)
        if resolved_codex is None:
            raise FileNotFoundError(
                f"The Codex binary {codex_binary!r} is not on PATH."
            )
        resolved_agentsims = shutil.which(agentsims_binary)
        if resolved_agentsims is None:
            raise FileNotFoundError(
                f"The Agentsims binary {agentsims_binary!r} is not on PATH."
            )
        resolved_skill_path = Path(skill_path).expanduser().resolve()
        if not resolved_skill_path.is_file():
            raise FileNotFoundError(
                f"The Agentsims skill was not found at {resolved_skill_path}."
            )

        self._device_id = device_id
        self._model = model
        self._reasoning = reasoning
        self._provider = provider
        self._timeout_sec = timeout_sec
        self._agentsims = resolved_agentsims
        self._skill_path = str(resolved_skill_path)
        self._output_dir = (
            Path(output_dir).expanduser().resolve() if output_dir else Path.cwd()
        )
        self._workdir = self._output_dir / "workdir"
        self._workdir.mkdir(parents=True, exist_ok=True)
        model_catalog_path = self._workdir / "azure-model-catalog.json"
        _write_azure_model_catalog(resolved_codex, model, model_catalog_path)
        self._tmux = CodexTmuxSession(
            tmux_session,
            codex_binary=resolved_codex,
            workdir=self._workdir,
            model=model,
            reasoning=reasoning,
            provider=provider,
            azure_base_url=_normalize_azure_base_url(azure_base_url),
            azure_api_version=azure_api_version,
            azure_api_key_env=azure_api_key_env,
            model_catalog_path=model_catalog_path,
        )
        self._tmux.start()
        self._episode_index = 0
        self._task_name: str | None = None
        self.transition_pause = None
        sys.stdout.write(
            f"\n{pi_render._BOLD}Codex CLI ready.{pi_render._RESET} Watch it live"
            f" with:\n    {pi_render._CYAN}{self._tmux.attach_command}"
            f"{pi_render._RESET}\n"
        )
        sys.stdout.flush()

    @property
    def device_id(self) -> str:
        return self._device_id

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self._episode_index += 1

    def set_task_name(self, task_name: str) -> None:
        self._task_name = task_name

    def _device_time(self) -> str:
        try:
            response = adb_utils.issue_generic_request(
                ["shell", "date", "+%A, %d %B %Y, %H:%M"], self.env.controller
            )
            text = response.generic.output.decode().strip()
            if text:
                return text
        except Exception as error:  # pylint: disable=broad-except
            logging.warning("Could not read the device clock: %s", error)
        return "unknown; read the status bar clock before date reasoning"

    def _slug(self, value: str) -> str:
        return "-".join(
            part
            for part in "".join(
                character.lower() if character.isalnum() else "-" for character in value
            ).split("-")
            if part
        )

    def _trace_name(self, goal: str) -> str:
        task_slug = self._slug(self._task_name or goal)
        device_slug = self._slug(self._device_id)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        device_name = device_slug or "device"
        suffix = f"{device_name}-{timestamp}"
        task_limit = 120 - len(suffix) - 1
        task_name = task_slug[:task_limit] or "task"
        return f"{task_name}-{suffix}"

    def _trace_command(self, action: str, *extra: str) -> dict[str, object]:
        command = [
            self._agentsims,
            "trace",
            action,
            "-d",
            self._device_id,
            *extra,
            "--json",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_TRACE_COMMAND_TIMEOUT_SEC,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CodexTraceError(
                f"Agentsims could not {action} the task trace: {error}"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise CodexTraceError(
                f"Agentsims could not {action} the task trace: "
                f"{detail or completed.returncode}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise CodexTraceError(
                f"Agentsims trace {action} returned invalid JSON."
            ) from error
        if not isinstance(payload, dict):
            raise CodexTraceError(
                f"Agentsims trace {action} returned an invalid result."
            )
        return payload

    def _parse_trace(
        self, payload: dict[str, object], *, stopped: bool
    ) -> CodexTaskTrace:
        device = payload.get("device")
        trace_id = payload.get("id")
        directory = payload.get("directory")
        started_at = payload.get("startedAt")
        if not all(
            isinstance(value, str) and value
            for value in (device, trace_id, directory, started_at)
        ):
            raise CodexTraceError("Agentsims returned incomplete trace metadata.")
        ended_at = payload.get("endedAt") if stopped else None
        calls = payload.get("calls") if stopped else None
        if stopped and (
            not isinstance(ended_at, str)
            or not ended_at
            or not isinstance(calls, int)
            or isinstance(calls, bool)
            or calls < 0
        ):
            raise CodexTraceError("Agentsims returned incomplete trace metadata.")
        trace = CodexTaskTrace(
            device=device,
            trace_id=trace_id,
            directory=directory,
            started_at=started_at,
            ended_at=ended_at,
            calls=calls,
        )
        if trace.device != self._device_id:
            raise CodexTraceError("Agentsims returned a trace for the wrong device.")
        return trace

    def _start_trace(self, goal: str) -> CodexTaskTrace:
        name = self._trace_name(goal)
        trace = dataclasses.replace(
            self._parse_trace(
                self._trace_command("start", "--name", name), stopped=False
            ),
            name=name,
        )
        sys.stdout.write(
            f"{pi_render._DIM}   trace started · {trace.name}{pi_render._RESET}\n"
        )
        sys.stdout.flush()
        return trace

    def _stop_trace(self) -> CodexTaskTrace:
        trace = self._parse_trace(self._trace_command("stop"), stopped=True)
        sys.stdout.write(
            f"{pi_render._DIM}   trace stopped · {trace.calls} calls ·"
            f" {trace.directory}{pi_render._RESET}\n"
        )
        sys.stdout.flush()
        return trace

    def _artifact_stem(self, goal: str) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        goal_slug = "".join(
            character if character.isalnum() else "_" for character in goal
        )[:60].strip("_")
        return self._output_dir / f"{self._episode_index:03d}_{goal_slug}"

    def run_codex(self, goal: str) -> CodexRunResult:
        """Runs one task with Codex and an automatically managed trace."""
        prompt = _BENCHMARK_RULES.format(
            device=self._device_id,
            goal=goal,
            device_time=self._device_time(),
            timeout_min=max(1, round(self._timeout_sec / 60)),
            skill_path=self._skill_path,
            agentsims=shlex.quote(self._agentsims),
        )
        sys.stdout.write(f"\n{pi_render._BOLD}── {goal}{pi_render._RESET}\n")
        sys.stdout.flush()
        started_trace: CodexTaskTrace | None = None
        stopped_trace: CodexTaskTrace | None = None

        def start_trace() -> None:
            nonlocal started_trace
            started_trace = self._start_trace(goal)

        try:
            result = self._tmux.run_task(
                prompt,
                self._timeout_sec,
                artifact_stem=self._artifact_stem(goal),
                before_submit=start_trace,
            )
        finally:
            if started_trace is not None:
                stopped_trace = self._stop_trace()
                if stopped_trace.trace_id != started_trace.trace_id:
                    raise CodexTraceError("Agentsims stopped a different task trace.")
                stopped_trace = dataclasses.replace(
                    stopped_trace, name=started_trace.name
                )
        if stopped_trace is not None:
            result = dataclasses.replace(result, trace=stopped_trace)

        status = "timeout" if result.timed_out else "ok"
        color = pi_render._RED if result.timed_out else pi_render._GREEN
        sys.stdout.write(
            f"{pi_render._DIM}   {color}{status}{pi_render._RESET}"
            f"{pi_render._DIM} · {result.duration_sec:.0f}s{pi_render._RESET}\n"
        )
        sys.stdout.flush()
        return result

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Runs Codex to completion, then hands grading back to AndroidWorld."""
        result = self.run_codex(goal)
        if result.timed_out:
            logging.warning(
                "Codex hit the %.0fs budget on: %s", self._timeout_sec, goal
            )
        elif result.exit_code:
            logging.warning("Codex exited with %s on: %s", result.exit_code, goal)
        if result.answer:
            self.env.execute_action(
                json_action.JSONAction(
                    action_type=json_action.ANSWER, text=result.answer
                )
            )
        return base_agent.AgentInteractionResult(
            done=True,
            data={
                "codex_exit_code": result.exit_code,
                "codex_timed_out": result.timed_out,
                "codex_duration_sec": result.duration_sec,
                "codex_transcript": result.transcript,
                "codex_log_path": result.log_path,
                "codex_device_id": self._device_id,
                "codex_model": f"{self._provider}/{self._model}",
                "codex_reasoning": self._reasoning,
                "codex_result": result.result,
                "codex_answer": result.answer,
                "codex_trace_id": result.trace.trace_id if result.trace else None,
                "codex_trace_name": result.trace.name if result.trace else None,
                "codex_trace_directory": (
                    result.trace.directory if result.trace else None
                ),
                "codex_trace_started_at": (
                    result.trace.started_at if result.trace else None
                ),
                "codex_trace_ended_at": (
                    result.trace.ended_at if result.trace else None
                ),
                "codex_trace_calls": result.trace.calls if result.trace else None,
            },
        )
