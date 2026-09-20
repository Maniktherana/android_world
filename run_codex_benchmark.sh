#!/usr/bin/env bash
# Runs AndroidWorld with a fresh Azure-backed Codex CLI process for each task.
# Live Codex output is available in one tmux session for the full benchmark.
#
#   ./run_codex_benchmark.sh --tasks=ContactsAddContact
#   ./run_codex_benchmark.sh --tasks=ContactsAddContact,ClockStopWatchRunning
#   ./run_codex_benchmark.sh
#
# Any extra argument is passed straight to run.py.
set -euo pipefail

PYTHON="${PYTHON:-$HOME/anaconda3/envs/android_world/bin/python}"
CODEX_BIN="${CODEX_BIN:-codex}"
AGENTSIMS_BIN="${AGENTSIMS_BIN:-agentsims}"
CONSOLE_PORT="${CONSOLE_PORT:-5554}"
DEVICE_ID="${DEVICE_ID:-android:emulator-${CONSOLE_PORT}}"
OUTPUT_PATH="${OUTPUT_PATH:-$HOME/android_world/runs/codex}"
SKILL_PATH="${SKILL_PATH:-$HOME/code/agentsims/skills/build-mobile-apps/SKILL.md}"
CODEX_TIMEOUT_SEC="${CODEX_TIMEOUT_SEC:-900}"
CODEX_TMUX_SESSION="${CODEX_TMUX_SESSION:-androidworld-codex}"

CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
CODEX_REASONING="${CODEX_REASONING:-high}"
AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}"
CODEX_AZURE_API_VERSION="${CODEX_AZURE_API_VERSION:-2025-04-01-preview}"
CODEX_AZURE_API_KEY_ENV="${CODEX_AZURE_API_KEY_ENV:-AZURE_OPENAI_API_KEY}"

export GRPC_VERBOSITY=ERROR
export GRPC_TRACE=none
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

say() { printf '\033[1;36m==> %s\033[0m\n' "$1"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

command -v "$CODEX_BIN" >/dev/null 2>&1 \
  || die "${CODEX_BIN} is not on PATH."
[[ -n "$AZURE_OPENAI_ENDPOINT" ]] \
  || die "Set AZURE_OPENAI_ENDPOINT before starting the benchmark."
[[ -n "${!CODEX_AZURE_API_KEY_ENV:-}" ]] \
  || die "Set ${CODEX_AZURE_API_KEY_ENV} before starting the benchmark."
[[ -f "$SKILL_PATH" ]] \
  || die "Agentsims skill not found at ${SKILL_PATH}."

say "Checking the emulator on console port ${CONSOLE_PORT}"
adb devices | grep -q "emulator-${CONSOLE_PORT}[[:space:]]*device" \
  || die "emulator-${CONSOLE_PORT} is not attached. Start AndroidWorldAvd first."

say "Checking the agentsims workspace"
if ! "$AGENTSIMS_BIN" status >/dev/null 2>&1; then
  say "Starting agentsims"
  "$AGENTSIMS_BIN" start --detach >/dev/null
  sleep 3
fi
"$AGENTSIMS_BIN" devices list | grep -q "$DEVICE_ID" \
  || die "agentsims does not see ${DEVICE_ID}. Run '${AGENTSIMS_BIN} devices list'."

say "Device ${DEVICE_ID} | model azure/${CODEX_MODEL}"
printf '\033[1;36m==> Watch Codex live from another terminal:\033[0m\n'
printf '      tmux attach -r -t %s\n\n' "$CODEX_TMUX_SESSION"

exec "$PYTHON" -u run.py \
  --agent_name=codex \
  --console_port="$CONSOLE_PORT" \
  --codex_device_id="$DEVICE_ID" \
  --codex_model="$CODEX_MODEL" \
  --codex_reasoning="$CODEX_REASONING" \
  --codex_azure_base_url="$AZURE_OPENAI_ENDPOINT" \
  --codex_azure_api_version="$CODEX_AZURE_API_VERSION" \
  --codex_azure_api_key_env="$CODEX_AZURE_API_KEY_ENV" \
  --codex_skill_path="$SKILL_PATH" \
  --codex_timeout_sec="$CODEX_TIMEOUT_SEC" \
  --codex_tmux_session="$CODEX_TMUX_SESSION" \
  --codex_binary="$CODEX_BIN" \
  --agentsims_binary="$AGENTSIMS_BIN" \
  --output_path="$OUTPUT_PATH" \
  "$@"
