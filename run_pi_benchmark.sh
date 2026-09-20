#!/usr/bin/env bash
# Runs the AndroidWorld benchmark with the Pi coding agent driving the
# emulator through the agentsims CLI. Pi's output streams to this terminal.
#
#   ./run_pi_benchmark.sh --tasks=ContactsAddContact
#   ./run_pi_benchmark.sh --tasks=ContactsAddContact,ClockStopWatchRunning
#   ./run_pi_benchmark.sh                      # whole suite
#
# Any extra argument is passed straight to run.py.
set -euo pipefail

PYTHON="${PYTHON:-$HOME/anaconda3/envs/android_world/bin/python}"
AGENTSIMS_BIN="${AGENTSIMS_BIN:-agentsims}"
CONSOLE_PORT="${CONSOLE_PORT:-5554}"
DEVICE_ID="${DEVICE_ID:-android:emulator-${CONSOLE_PORT}}"
OUTPUT_PATH="${OUTPUT_PATH:-$HOME/android_world/runs/pi}"
SKILL_PATH="${SKILL_PATH:-$HOME/code/agentsims/skills/build-mobile-apps}"
PI_TIMEOUT_SEC="${PI_TIMEOUT_SEC:-900}"

PI_PROVIDER="${PI_PROVIDER:-azure-openai-foundry}"
PI_MODEL="${PI_MODEL:-gpt-5.6-sol}"
PI_THINKING="${PI_THINKING:-high}"

export GRPC_VERBOSITY=ERROR
export GRPC_TRACE=none
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

say() { printf '\033[1;36m==> %s\033[0m\n' "$1"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

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

say "Device ${DEVICE_ID} | model ${PI_PROVIDER}/${PI_MODEL}"
printf '\033[1;36m==> Watch pi native TUI live from another terminal:\033[0m\n'
printf '      tmux attach -t %s\n' "${PI_TMUX_SESSION:-androidworld-pi}"
echo

exec "$PYTHON" -u run.py \
  --agent_name=pi \
  --console_port="$CONSOLE_PORT" \
  --pi_device_id="$DEVICE_ID" \
  --pi_agent_provider="$PI_PROVIDER" \
  --pi_agent_model="$PI_MODEL" \
  --agentsims_binary="$AGENTSIMS_BIN" \
  --pi_thinking="$PI_THINKING" \
  --pi_skill_path="$SKILL_PATH" \
  --pi_timeout_sec="$PI_TIMEOUT_SEC" \
  --pi_tmux_session="${PI_TMUX_SESSION:-androidworld-pi}" \
  --output_path="$OUTPUT_PATH" \
  "$@"
