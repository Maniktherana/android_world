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

"""Runs the Pi coding agent as an AndroidWorld agent.

Pi is a full agentic loop of its own. It drives the emulator through the
agentsims CLI and its build-mobile-apps skill. One AndroidWorld step is one
complete Pi session, so the episode needs only a single step.

Pi's output streams to this process's stdout while it runs, so the terminal
shows each tool call as it happens.
"""

import dataclasses
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from absl import logging

from android_world.agents import base_agent, pi_render
from android_world.env import adb_utils, interface, json_action

DEFAULT_SKILL_PATH = "~/code/agentsims/skills/build-mobile-apps"
DEFAULT_PROVIDER = "azure-openai-foundry"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_THINKING = "high"

_BENCHMARK_RULES = """You control this Android device: {device}
Pass -d {device} to every agentsims command. Ignore every other device.

The device clock reads {device_time}. Treat this as the authoritative current
date and time for today, tomorrow, yesterday, this week, next week, and all
other relative date or time references. Do not change the device date, time,
or timezone.

Task:
{goal}

Follow the loaded build-mobile-apps skill.

Rules:
- Work only through the device UI with agentsims.
- Do not use adb, sqlite, content providers, broadcasts, or direct device file
  or database writes.
- Reading device state through agentsims, including device logs, is allowed.
- Do not reboot, wipe, install, or uninstall apps unless the task requires it.
- You have a {timeout_min} minute budget. This is a ceiling not target.
- Do not invent another time limit.
- Tasks are designed to be passed one way or another. Try to verify your work before prematurely marking a task failed.
- Continue after recoverable failures, but stop as soon as the task is complete.

Final reply:
- For a completed action task, reply with exactly:
  RESULT: done
- If an action task cannot be completed, reply with exactly:
  RESULT: failed
- If the task asks a question, reply with exactly:
  ANSWER: <answer in exactly the requested format>
"""
# Sent once into the same Pi session when a question task ends without an
# ANSWER line.
_ANSWER_FOLLOW_UP = "Reply with only the ANSWER: line."
_FOLLOW_UP_TIMEOUT_SEC = 120.0

# Goals that read like questions, so a missing ANSWER line is worth chasing.
_QUESTION_MARKERS = (
    "answer with",
    "express your answer",
    "what ",
    "which ",
    "how many",
    "when ",
)

# tmux occasionally dies mid-paste, so every send is retried.
_PASTE_ATTEMPTS = 3
_PASTE_RETRY_SEC = 0.5

# How long the session file must stay quiet after a final message before the
# turn is treated as over.
_SETTLE_SEC = 8.0
_TRACE_COMMAND_TIMEOUT_SEC = 30.0


class PiTransportError(RuntimeError):
    """Raised when tmux cannot carry a prompt into the Pi TUI."""


class PiTraceError(RuntimeError):
    """Raised when Agentsims cannot manage the task trace."""


def looks_like_question(goal: str) -> bool:
    """Returns whether a goal asks for an answer rather than a device change."""
    text = goal.strip()
    if text.endswith("?"):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _QUESTION_MARKERS)


@dataclasses.dataclass
class PiRunResult:
    """Outcome of one Pi session."""

    exit_code: int | None
    timed_out: bool
    duration_sec: float
    transcript: str
    log_path: str | None
    tool_calls: int = 0
    answer: str | None = None
    result: str | None = None
    followed_up: bool = False
    trace: "PiTaskTrace | None" = None


@dataclasses.dataclass(frozen=True)
class PiTaskTrace:
    """Agentsims trace metadata for one AndroidWorld task."""

    device: str
    trace_id: str
    directory: str
    started_at: str
    ended_at: str | None = None
    calls: int | None = None
    name: str | None = None


def parse_final_lines(text: str | None) -> tuple[str | None, str | None]:
    """Returns (result, answer) from Pi's final message.

    The last `RESULT:` or `ANSWER:` line wins. An answer is passed through
    unchanged, because each task states its own answer format and the grader
    parses the raw string.
    """
    if not text:
        return None, None
    result = answer = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("RESULT:"):
            result = stripped.split(":", 1)[1].strip().lower() or None
        elif stripped.upper().startswith("ANSWER:"):
            answer = stripped.split(":", 1)[1].strip() or None
    return result, answer


def _session_dir_for(workdir: Path) -> Path:
    """Returns Pi's session directory for a working directory."""
    slug = str(workdir.resolve()).lstrip("/").replace("/", "-")
    return Path("~/.pi/agent/sessions").expanduser() / f"--{slug}--"


class PiTmuxSession:
    """A long-lived Pi TUI in a detached tmux session.

    One Pi process serves the whole benchmark. Each task is sent into the same
    TUI after `/new` clears the previous context, so a single
    `tmux attach -t <name>` follows every task of a run without reattaching.
    """

    def __init__(
        self,
        name: str,
        *,
        pi_binary: str,
        provider: str,
        model: str,
        skill_path: str,
        thinking: str,
        workdir: Path,
        width: int = 200,
        height: int = 50,
    ):
        self._name = name
        self._pi = pi_binary
        self._provider = provider
        self._model = model
        self._skill_path = skill_path
        self._thinking = thinking
        self._workdir = workdir
        self._size = (width, height)
        self._session_dir = _session_dir_for(workdir)
        # The session file of the task in flight, so a follow-up can keep reading
        # the same conversation.
        self._session_file: Path | None = None
        self._session_lines = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def attach_command(self) -> str:
        return f"tmux attach -t {self._name}"

    def _tmux(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["tmux", *args], capture_output=True, text=True, check=check
        )

    def exists(self) -> bool:
        return self._tmux("has-session", "-t", self._name, check=False).returncode == 0

    def start(self) -> None:
        """Starts the TUI, replacing any session left by an earlier run."""
        if shutil.which("tmux") is None:
            raise FileNotFoundError(
                "tmux is required to run the Pi TUI. Install tmux and retry."
            )
        if self.exists():
            self._tmux("kill-session", "-t", self._name, check=False)
        self._workdir.mkdir(parents=True, exist_ok=True)
        width, height = self._size
        command = " ".join(
            shlex.quote(part)
            for part in [
                self._pi,
                "--provider",
                self._provider,
                "--model",
                self._model,
                "--thinking",
                self._thinking,
                "--skill",
                self._skill_path,
                "--no-context-files",
            ]
        )
        self._tmux(
            "new-session",
            "-d",
            "-s",
            self._name,
            "-x",
            str(width),
            "-y",
            str(height),
            "-c",
            str(self._workdir),
            command,
        )
        # Let the TUI draw before the first paste.
        deadline = time.time() + 30
        while time.time() < deadline:
            pane = self._tmux(
                "capture-pane", "-p", "-t", self._name, check=False
            ).stdout
            if "pi v" in pane or "thinking" in pane:
                return
            time.sleep(0.5)
        raise RuntimeError(f"The Pi TUI did not start in tmux session {self._name!r}.")

    def kill(self) -> None:
        self._tmux("kill-session", "-t", self._name, check=False)

    def _send_line(self, text: str) -> None:
        try:
            self._tmux("send-keys", "-t", self._name, "-l", text)
            self._tmux("send-keys", "-t", self._name, "Enter")
        except subprocess.CalledProcessError as error:
            raise PiTransportError(
                f"tmux could not send {text!r} to the Pi TUI: {error}"
            ) from error

    def _submit(self) -> None:
        """Presses Enter on the prompt already in the TUI."""
        try:
            self._tmux("send-keys", "-t", self._name, "Enter")
        except subprocess.CalledProcessError as error:
            raise PiTransportError(
                f"tmux could not submit the prompt: {error}"
            ) from error

    def _paste(self, text: str) -> None:
        """Sends a multi-line prompt without submitting it early.

        tmux sometimes dies part way through `load-buffer`, which leaves the prompt
        half-written, so each attempt starts from a fresh temp file.

        Raises:
          PiTransportError: If every attempt fails.
        """
        buffer = f"pi_{self._name}"
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(1, _PASTE_ATTEMPTS + 1):
            if attempt > 1:
                time.sleep(_PASTE_RETRY_SEC)
            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False
            ) as handle:
                handle.write(text)
                path = handle.name
            try:
                self._tmux("load-buffer", "-b", buffer, path)
                # Bracketed paste keeps newlines out of the submit handler.
                self._tmux("paste-buffer", "-b", buffer, "-t", self._name, "-p", "-d")
                return
            except subprocess.CalledProcessError as error:
                last_error = error
                logging.warning(
                    "tmux paste attempt %d of %d failed: %s",
                    attempt,
                    _PASTE_ATTEMPTS,
                    error,
                )
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        raise PiTransportError(
            f"tmux could not paste the prompt after {_PASTE_ATTEMPTS} attempts:"
            f" {last_error}"
        )

    def _existing_sessions(self) -> set[Path]:
        if not self._session_dir.is_dir():
            return set()
        return set(self._session_dir.glob("*.jsonl"))

    def _await_new_session(self, before: set[Path], deadline: float) -> Path | None:
        while time.time() < deadline:
            new = self._existing_sessions() - before
            if new:
                return max(new, key=lambda p: p.stat().st_mtime)
            time.sleep(0.3)
        return None

    def _tail_session(
        self,
        session_file: Path,
        deadline: float,
        skip_lines: int = 0,
    ) -> tuple[str | None, str, int, int]:
        """Streams one Pi turn from a session file to stdout.

        Args:
          session_file: The JSONL file Pi appends to.
          deadline: Wall-clock time at which to give up.
          skip_lines: Lines already consumed by an earlier turn.

        Returns:
          (final assistant text or None, transcript, tool calls, lines consumed).
        """
        handle = session_file.open("r")
        buffer = ""
        final: str | None = None
        calls = 0
        seen = 0
        chunks: list[str] = []
        last_change = time.time()
        try:
            while time.time() < deadline:
                chunk = handle.read()
                if not chunk:
                    # Pi has stopped writing and already gave a final message.
                    if final is not None and time.time() - last_change > _SETTLE_SEC:
                        break
                    time.sleep(0.4)
                    continue
                last_change = time.time()
                buffer += chunk
                *lines, buffer = buffer.split("\n")
                for line in lines:
                    seen += 1
                    if seen <= skip_lines:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    shown = pi_render.render_session_event(event)
                    if shown:
                        calls += shown.count("▸ ")
                        sys.stdout.write(shown)
                        sys.stdout.flush()
                        chunks.append(shown)
                    text = pi_render.final_text(event)
                    if text:
                        final = text
        finally:
            handle.close()
        return final, "".join(chunks), calls, max(seen, skip_lines)

    def run_task(
        self,
        prompt: str,
        timeout_sec: float,
        before_submit: Callable[[], None] | None = None,
    ) -> "PiRunResult":
        """Runs one task in the TUI, restarting it once on a tmux fault."""
        start = time.time()
        hook_called = False

        def call_before_submit() -> None:
            nonlocal hook_called
            if before_submit is not None and not hook_called:
                before_submit()
                hook_called = True

        try:
            return self._run_task_once(
                prompt, timeout_sec, before_submit=call_before_submit
            )
        except PiTransportError as first_error:
            logging.warning(
                "tmux lost the Pi TUI (%s). Restarting it and retrying the task.",
                first_error,
            )
            error: Exception = first_error
            try:
                self.kill()
                self.start()
                return self._run_task_once(
                    prompt, timeout_sec, before_submit=call_before_submit
                )
            except (
                PiTransportError,
                RuntimeError,
                FileNotFoundError,
                subprocess.CalledProcessError,
            ) as retry_error:
                error = retry_error
            logging.error("Pi is unreachable over tmux: %s", error)
            return PiRunResult(
                exit_code=None,
                timed_out=False,
                duration_sec=time.time() - start,
                transcript=(
                    "The Pi TUI was unreachable over tmux, so this task never ran.\n"
                    f"Transport error: {error}\n"
                ),
                log_path=None,
                result="failed",
            )

    def _run_task_once(
        self,
        prompt: str,
        timeout_sec: float,
        before_submit: Callable[[], None] | None = None,
    ) -> "PiRunResult":
        """Sends one task into a cleared TUI and follows it to its final message."""
        before = self._existing_sessions()
        self._session_file = None
        self._session_lines = 0
        self._send_line("/new")
        time.sleep(1.5)
        self._paste(prompt)
        time.sleep(0.4)
        if before_submit is not None:
            before_submit()
        self._submit()

        start = time.time()
        deadline = start + timeout_sec
        session_file = self._await_new_session(before, min(deadline, start + 90))
        if session_file is None:
            return PiRunResult(
                exit_code=None,
                timed_out=True,
                duration_sec=time.time() - start,
                transcript="Pi never opened a session file for this task.",
                log_path=None,
            )
        self._session_file = session_file

        final, transcript, calls, seen = self._tail_session(session_file, deadline)
        self._session_lines = seen
        result, answer = parse_final_lines(final)
        return PiRunResult(
            exit_code=0 if final is not None else None,
            timed_out=final is None,
            duration_sec=time.time() - start,
            transcript=transcript,
            log_path=str(session_file),
            tool_calls=calls,
            answer=answer,
            result=result,
        )

    def continue_task(self, text: str, timeout_sec: float) -> "PiRunResult":
        """Sends one more line into the current session and awaits its reply.

        Unlike `run_task` this keeps the context, so Pi can answer from what it
        already saw on the device.

        Raises:
          PiTransportError: If tmux cannot carry the line into the TUI.
        """
        start = time.time()
        session_file = self._session_file
        if session_file is None:
            return PiRunResult(
                exit_code=None,
                timed_out=True,
                duration_sec=0.0,
                transcript="There is no open Pi session to continue.\n",
                log_path=None,
            )
        self._paste(text)
        time.sleep(0.4)
        self._submit()
        final, transcript, calls, seen = self._tail_session(
            session_file, time.time() + timeout_sec, self._session_lines
        )
        self._session_lines = seen
        result, answer = parse_final_lines(final)
        return PiRunResult(
            exit_code=0 if final is not None else None,
            timed_out=final is None,
            duration_sec=time.time() - start,
            transcript=transcript,
            log_path=str(session_file),
            tool_calls=calls,
            answer=answer,
            result=result,
        )


class PiAgent(base_agent.EnvironmentInteractingAgent):
    """Delegates a whole AndroidWorld task to one Pi session."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        device_id: str,
        *,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        skill_path: str = DEFAULT_SKILL_PATH,
        thinking: str = DEFAULT_THINKING,
        timeout_sec: float = 900.0,
        output_dir: str | None = None,
        pi_binary: str = "pi",
        agentsims_binary: str = "agentsims",
        tmux_session: str = "androidworld-pi",
        name: str = "pi",
    ):
        """Initializes the agent.

        Args:
          env: The AndroidWorld environment. Used only for reset and grading.
          device_id: The agentsims device ID, such as `android:emulator-5554`.
          provider: Pi provider name from `~/.pi/agent/models.json`.
          model: Pi model ID within that provider.
          skill_path: Path to the agentsims build-mobile-apps skill.
          thinking: Pi thinking level.
          timeout_sec: Wall-clock budget for one task.
          output_dir: Directory for per-task transcripts and Pi sessions.
          pi_binary: Pi executable name or path.
          agentsims_binary: Agentsims executable name or path.
          tmux_session: tmux session name for the shared Pi TUI.
          name: Agent name.

        Raises:
          ValueError: If the timeout is not positive or the device ID is empty.
          FileNotFoundError: If a binary or the skill path is missing.
        """
        super().__init__(env, name)
        if timeout_sec <= 0:
            raise ValueError("Use a positive --pi_timeout_sec.")
        if not device_id:
            raise ValueError("Provide the agentsims device ID.")
        resolved = shutil.which(pi_binary)
        if resolved is None:
            raise FileNotFoundError(
                f"The Pi binary {pi_binary!r} is not on PATH. Install Pi first."
            )
        self._pi = resolved
        resolved_agentsims = shutil.which(agentsims_binary)
        if resolved_agentsims is None:
            raise FileNotFoundError(
                f"The Agentsims binary {agentsims_binary!r} is not on PATH."
            )
        self._agentsims = resolved_agentsims
        self._skill_path = str(Path(skill_path).expanduser())
        if not Path(self._skill_path).exists():
            raise FileNotFoundError(
                f"The agentsims skill was not found at {self._skill_path}."
            )
        self._device_id = device_id
        self._provider = provider
        self._model = model
        self._thinking = thinking
        self._timeout_sec = timeout_sec
        self._pi_binary = pi_binary
        self._output_dir = Path(output_dir).expanduser() if output_dir else None
        self._episode_index = 0
        self._task_name: str | None = None
        self._session_name = "androidworld"
        self._tmux_name = tmux_session
        self._tmux: PiTmuxSession | None = None
        self._workdir = (
            self._output_dir / "workdir"
            if self._output_dir is not None
            else Path(os.getcwd())
        )
        # Pi runs its own loop, so a step is the whole episode.
        self.transition_pause = None

    def _tmux_session(self) -> PiTmuxSession:
        """Starts the shared Pi TUI on first use."""
        if self._tmux is None:
            self._workdir.mkdir(parents=True, exist_ok=True)
            self._tmux = PiTmuxSession(
                self._tmux_name,
                pi_binary=self._pi,
                provider=self._provider,
                model=self._model,
                skill_path=self._skill_path,
                thinking=self._thinking,
                workdir=self._workdir,
            )
            self._tmux.start()
            sys.stdout.write(
                f"\n{pi_render._BOLD}Pi TUI ready.{pi_render._RESET} Watch it live"
                f" with:\n    {pi_render._CYAN}{self._tmux.attach_command}"
                f"{pi_render._RESET}\n"
            )
            sys.stdout.flush()
        return self._tmux

    @property
    def device_id(self) -> str:
        return self._device_id

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self._episode_index += 1

    def set_task_name(self, task_name: str) -> None:
        """Sets the AndroidWorld task template used in trace names."""
        self._task_name = task_name

    def _log_path(self, goal: str) -> Path | None:
        if self._output_dir is None:
            return None
        self._output_dir.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in goal)[:60].strip("_")
        return self._output_dir / f"{self._episode_index:03d}_{slug}.log"

    def _device_time(self) -> str:
        """Reads the device clock, which AndroidWorld pins per task."""
        try:
            response = adb_utils.issue_generic_request(
                ["shell", "date", "+%A, %d %B %Y, %H:%M"], self.env.controller
            )
            text = response.generic.output.decode().strip()
            if text:
                return text
        except Exception as error:  # pylint: disable=broad-except
            logging.warning("Could not read the device clock: %s", error)
        return "unknown; read the status bar clock before any date reasoning"

    def _trace_name(self, goal: str) -> str:
        """Returns <task>-<device>-<UTC timestamp> for the current trace."""
        task_label = self._task_name or goal
        task_slug = "-".join(
            part
            for part in "".join(
                character.lower() if character.isalnum() else "-"
                for character in task_label
            ).split("-")
            if part
        )
        device_slug = "-".join(
            part
            for part in "".join(
                character.lower() if character.isalnum() else "-"
                for character in self._device_id
            ).split("-")
            if part
        )
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        suffix = f"{device_slug or 'device'}-{timestamp}"
        task_limit = 120 - len(suffix) - 1
        return f"{(task_slug[:task_limit] or 'task')}-{suffix}"

    def _trace_command(self, action: str, *extra: str) -> dict[str, object]:
        """Runs one Agentsims trace command and returns its JSON object."""
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
            raise PiTraceError(
                f"Agentsims could not {action} the task trace: {error}"
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if not detail:
                detail = f"exit code {completed.returncode}"
            raise PiTraceError(f"Agentsims could not {action} the task trace: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise PiTraceError(
                f"Agentsims trace {action} returned invalid JSON."
            ) from error
        if not isinstance(payload, dict):
            raise PiTraceError(f"Agentsims trace {action} returned an invalid result.")
        return payload

    def _parse_trace(self, payload: dict[str, object], *, stopped: bool) -> PiTaskTrace:
        """Validates trace metadata returned by Agentsims."""
        device = payload.get("device")
        trace_id = payload.get("id")
        directory = payload.get("directory")
        started_at = payload.get("startedAt")
        if not (
            isinstance(device, str)
            and device
            and isinstance(trace_id, str)
            and trace_id
            and isinstance(directory, str)
            and directory
            and isinstance(started_at, str)
            and started_at
        ):
            raise PiTraceError("Agentsims returned incomplete trace metadata.")
        ended_at = payload.get("endedAt") if stopped else None
        calls = payload.get("calls") if stopped else None
        if stopped and (
            not isinstance(ended_at, str)
            or not ended_at
            or not isinstance(calls, int)
            or isinstance(calls, bool)
            or calls < 0
        ):
            raise PiTraceError("Agentsims returned incomplete trace metadata.")
        trace = PiTaskTrace(
            device=device,
            trace_id=trace_id,
            directory=directory,
            started_at=started_at,
            ended_at=ended_at,
            calls=calls,
        )
        if trace.device != self._device_id:
            raise PiTraceError("Agentsims returned a trace for the wrong device.")
        return trace

    def _start_trace(self, goal: str) -> PiTaskTrace:
        """Starts the required trace before Pi can begin the task."""
        name = self._trace_name(goal)
        payload = self._trace_command("start", "--name", name)
        trace = dataclasses.replace(
            self._parse_trace(payload, stopped=False), name=name
        )
        sys.stdout.write(
            f"{pi_render._DIM}   trace started · {trace.name}{pi_render._RESET}\n"
        )
        sys.stdout.flush()
        return trace

    def _stop_trace(self) -> PiTaskTrace:
        """Stops the task trace before AndroidWorld receives the result."""
        trace = self._parse_trace(self._trace_command("stop"), stopped=True)
        sys.stdout.write(
            f"{pi_render._DIM}   trace stopped · {trace.calls} calls ·"
            f" {trace.directory}{pi_render._RESET}\n"
        )
        sys.stdout.flush()
        return trace

    def run_pi_tmux(self, goal: str) -> PiRunResult:
        """Runs one task inside the shared Pi TUI."""
        session = self._tmux_session()
        prompt = _BENCHMARK_RULES.format(
            device=self._device_id,
            goal=goal,
            device_time=self._device_time(),
            timeout_min=max(1, round(self._timeout_sec / 60)),
        )
        sys.stdout.write(f"\n{pi_render._BOLD}── {goal}{pi_render._RESET}\n")
        sys.stdout.flush()
        started_trace: PiTaskTrace | None = None
        stopped_trace: PiTaskTrace | None = None

        def start_trace() -> None:
            nonlocal started_trace
            started_trace = self._start_trace(goal)

        try:
            result = session.run_task(
                prompt, self._timeout_sec, before_submit=start_trace
            )
            result = self._chase_missing_answer(session, goal, result)
        finally:
            if started_trace is not None:
                stopped_trace = self._stop_trace()
                if stopped_trace.trace_id != started_trace.trace_id:
                    raise PiTraceError("Agentsims stopped a different task trace.")
                stopped_trace = dataclasses.replace(
                    stopped_trace, name=started_trace.name
                )
        if stopped_trace is not None:
            result = dataclasses.replace(result, trace=stopped_trace)
        verdict = (
            f"{pi_render._RED}timeout{pi_render._RESET}"
            if result.timed_out
            else f"{pi_render._GREEN}ok{pi_render._RESET}"
        )
        sys.stdout.write(
            f"{pi_render._DIM}   {verdict}{pi_render._DIM} ·"
            f" {result.tool_calls} calls · {result.duration_sec:.0f}s"
            f"{pi_render._RESET}\n"
        )
        sys.stdout.flush()
        log = self._log_path(goal)
        if log is not None:
            log.write_text(
                f"{result.transcript}\n"
                f"pi_result: {result.result}\n"
                f"pi_answer: {result.answer}\n"
                f"pi_followed_up: {result.followed_up}\n"
            )
        return result

    def _chase_missing_answer(
        self,
        session: PiTmuxSession,
        goal: str,
        result: PiRunResult,
    ) -> PiRunResult:
        """Asks once more for the ANSWER line a question task is missing.

        Information-retrieval tasks are graded only on the answer, so a session
        that finished its work but dropped the line is worth one more turn in the
        same context.
        """
        if (
            result.answer is not None
            or result.timed_out
            or result.result not in (None, "done")
            or not looks_like_question(goal)
        ):
            return result
        sys.stdout.write(
            f"{pi_render._DIM}   no ANSWER line; asking once more{pi_render._RESET}\n"
        )
        sys.stdout.flush()
        try:
            extra = session.continue_task(
                _ANSWER_FOLLOW_UP, min(self._timeout_sec, _FOLLOW_UP_TIMEOUT_SEC)
            )
        except PiTransportError as error:
            logging.warning("The answer follow-up never reached Pi: %s", error)
            return result
        return dataclasses.replace(
            result,
            answer=extra.answer,
            result=result.result or extra.result,
            transcript=result.transcript + extra.transcript,
            tool_calls=result.tool_calls + extra.tool_calls,
            duration_sec=result.duration_sec + extra.duration_sec,
            followed_up=True,
        )

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Runs Pi to completion, then hands grading back to AndroidWorld."""
        result = self.run_pi_tmux(goal)
        if result.timed_out:
            logging.warning("Pi hit the %.0fs budget on: %s", self._timeout_sec, goal)
        elif result.exit_code:
            logging.warning("Pi exited with %s on: %s", result.exit_code, goal)
        if result.answer:
            # Information-retrieval tasks are graded only on this answer action.
            self.env.execute_action(
                json_action.JSONAction(
                    action_type=json_action.ANSWER, text=result.answer
                )
            )
        # Always report done. The task evaluator decides success from device state.
        return base_agent.AgentInteractionResult(
            done=True,
            data={
                "pi_exit_code": result.exit_code,
                "pi_timed_out": result.timed_out,
                "pi_duration_sec": result.duration_sec,
                "pi_transcript": result.transcript,
                "pi_log_path": result.log_path,
                "pi_device_id": self._device_id,
                "pi_model": f"{self._provider}/{self._model}",
                "pi_tool_calls": result.tool_calls,
                "pi_result": result.result,
                "pi_answer": result.answer,
                "pi_followed_up": result.followed_up,
                "pi_trace_id": result.trace.trace_id if result.trace else None,
                "pi_trace_name": result.trace.name if result.trace else None,
                "pi_trace_directory": (
                    result.trace.directory if result.trace else None
                ),
                "pi_trace_started_at": (
                    result.trace.started_at if result.trace else None
                ),
                "pi_trace_ended_at": (result.trace.ended_at if result.trace else None),
                "pi_trace_calls": result.trace.calls if result.trace else None,
            },
        )
