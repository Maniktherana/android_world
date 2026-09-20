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

"""Run eval suite.

The run.py module is used to run a suite of tasks, with configurable task
combinations, environment setups, and agent configurations. You can run specific
tasks or all tasks in the suite and customize various settings using the
command-line flags.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from android_world import checkpointer as checkpointer_lib
from android_world import registry
from android_world import suite_utils
from android_world.agents import base_agent
from android_world.agents import codex_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import m3a
from android_world.agents import pi_agent
from android_world.agents import random_agent
from android_world.agents import seeact
from android_world.agents import t3a
from android_world.env import env_launcher
from android_world.env import interface

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing


def _find_adb_directory() -> str:
  """Returns the directory where adb is located."""
  potential_paths = [
      os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
      os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
  ]
  for path in potential_paths:
    if os.path.isfile(path):
      return path
  raise EnvironmentError(
      'adb not found in the common Android SDK paths. Please install Android'
      " SDK and ensure adb is in one of the expected directories. If it's"
      ' already installed, point to the installed location.'
  )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    False,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_SUITE_FAMILY = flags.DEFINE_enum(
    'suite_family',
    registry.TaskRegistry.ANDROID_WORLD_FAMILY,
    [
        # Families from the paper.
        registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        registry.TaskRegistry.MINIWOB_FAMILY_SUBSET,
        # Other families for more testing.
        registry.TaskRegistry.MINIWOB_FAMILY,
        registry.TaskRegistry.ANDROID_FAMILY,
        registry.TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ],
    'Suite family to run. See registry.py for more information.',
)
_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port',
    None,
    'Emulator gRPC port. By default, discover it from --console_port.',
    lower_bound=1,
    upper_bound=65535,
)
_TASK_RANDOM_SEED = flags.DEFINE_integer(
    'task_random_seed', 30, 'Random seed for task randomness.'
)

_TASKS = flags.DEFINE_list(
    'tasks',
    None,
    'List of specific tasks to run in the given suite family. If None, run all'
    ' tasks in the suite family.',
)
_N_TASK_COMBINATIONS = flags.DEFINE_integer(
    'n_task_combinations',
    1,
    'Number of task instances to run for each task template.',
)

_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    '',
    'The directory to save checkpoints and resume evaluation from. If the'
    ' directory contains existing checkpoint files, evaluation will resume from'
    ' the latest checkpoint. If the directory is empty or does not exist, a new'
    ' directory will be created.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    os.path.expanduser('~/android_world/runs'),
    'The path to save results to if not resuming from a checkpoint is not'
    ' provided.',
)

# Agent specific.
_AGENT_NAME = flags.DEFINE_string('agent_name', 'm3a_gpt4v', help='Agent name.')

# Pi coding agent, driving the device through the agentsims CLI.
_PI_PROVIDER_NAME = flags.DEFINE_string(
    'pi_agent_provider',
    pi_agent.DEFAULT_PROVIDER,
    'Pi provider name from ~/.pi/agent/models.json.',
)
_PI_MODEL = flags.DEFINE_string(
    'pi_agent_model', pi_agent.DEFAULT_MODEL, 'Pi model ID for that provider.'
)
_PI_SKILL_PATH = flags.DEFINE_string(
    'pi_skill_path',
    pi_agent.DEFAULT_SKILL_PATH,
    'Path to the agentsims build-mobile-apps skill.',
)
_PI_DEVICE_ID = flags.DEFINE_string(
    'pi_device_id',
    None,
    'Agentsims device ID. Defaults to android:emulator-<console_port>.',
)
_AGENTSIMS_BINARY = flags.DEFINE_string(
    'agentsims_binary', 'agentsims', 'Agentsims executable name or path.'
)
_PI_THINKING = flags.DEFINE_string(
    'pi_thinking', pi_agent.DEFAULT_THINKING, 'Pi thinking level.'
)
_PI_TIMEOUT_SEC = flags.DEFINE_float(
    'pi_timeout_sec', 900.0, 'Wall-clock budget for one Pi task.'
)
_PI_TMUX_SESSION = flags.DEFINE_string(
    'pi_tmux_session', 'androidworld-pi', 'tmux session name for the Pi TUI.'
)

# Codex CLI agent, driving the device through the agentsims CLI.
_CODEX_MODEL = flags.DEFINE_string(
    'codex_model', codex_agent.DEFAULT_MODEL, 'Codex model deployment name.'
)
_CODEX_REASONING = flags.DEFINE_string(
    'codex_reasoning',
    codex_agent.DEFAULT_REASONING,
    'Codex reasoning effort.',
)
_CODEX_AZURE_BASE_URL = flags.DEFINE_string(
    'codex_azure_base_url',
    codex_agent.DEFAULT_AZURE_BASE_URL,
    'Azure OpenAI endpoint. The /openai suffix is optional.',
)
_CODEX_AZURE_API_VERSION = flags.DEFINE_string(
    'codex_azure_api_version',
    codex_agent.DEFAULT_AZURE_API_VERSION,
    'Azure OpenAI API version for Codex.',
)
_CODEX_AZURE_API_KEY_ENV = flags.DEFINE_string(
    'codex_azure_api_key_env',
    codex_agent.DEFAULT_AZURE_API_KEY_ENV,
    'Environment variable that contains the Azure OpenAI key.',
)
_CODEX_SKILL_PATH = flags.DEFINE_string(
    'codex_skill_path',
    codex_agent.DEFAULT_SKILL_PATH,
    'Path to the agentsims build-mobile-apps SKILL.md file.',
)
_CODEX_DEVICE_ID = flags.DEFINE_string(
    'codex_device_id',
    None,
    'Agentsims device ID. Defaults to android:emulator-<console_port>.',
)
_CODEX_TIMEOUT_SEC = flags.DEFINE_float(
    'codex_timeout_sec', 900.0, 'Wall-clock budget for one Codex task.'
)
_CODEX_TMUX_SESSION = flags.DEFINE_string(
    'codex_tmux_session',
    codex_agent.DEFAULT_TMUX_SESSION,
    'tmux session name for the Codex live console.',
)
_CODEX_BINARY = flags.DEFINE_string(
    'codex_binary', 'codex', 'Codex executable name or path.'
)

_FIXED_TASK_SEED = flags.DEFINE_boolean(
    'fixed_task_seed',
    False,
    'Whether to use the same task seed when running multiple task combinations'
    ' (n_task_combinations > 1).',
)


# MiniWoB is very lightweight and new screens/View Hierarchy load quickly.
_MINIWOB_TRANSITION_PAUSE = 0.2

# Additional guidelines for the MiniWob tasks.
_MINIWOB_ADDITIONAL_GUIDELINES = [
    (
        'This task is running in a mock app, you must stay in this app and'
        ' DO NOT use the `navigate_home` action.'
    ),
]


def _get_agent(
    env: interface.AsyncEnv,
    run_dir: str,
    family: str | None = None,
) -> base_agent.EnvironmentInteractingAgent:
  """Gets agent."""
  print('Initializing agent...')
  agent = None
  if _AGENT_NAME.value == 'human_agent':
    agent = human_agent.HumanAgent(env)
  elif _AGENT_NAME.value == 'random_agent':
    agent = random_agent.RandomAgent(env)
  # Gemini.
  elif _AGENT_NAME.value == 'm3a_gemini_gcp':
    agent = m3a.M3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  elif _AGENT_NAME.value == 't3a_gemini_gcp':
    agent = t3a.T3A(
        env, infer.GeminiGcpWrapper(model_name='gemini-1.5-pro-latest')
    )
  # GPT.
  elif _AGENT_NAME.value == 't3a_gpt4':
    agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  elif _AGENT_NAME.value == 'm3a_gpt4v':
    agent = m3a.M3A(env, infer.Gpt4Wrapper('gpt-4-turbo-2024-04-09'))
  # Pi coding agent driving agentsims.
  elif _AGENT_NAME.value == 'pi':
    agent = pi_agent.PiAgent(
        env,
        _PI_DEVICE_ID.value
        or f'android:emulator-{_DEVICE_CONSOLE_PORT.value}',
        provider=_PI_PROVIDER_NAME.value,
        model=_PI_MODEL.value,
        skill_path=_PI_SKILL_PATH.value,
        thinking=_PI_THINKING.value,
        timeout_sec=_PI_TIMEOUT_SEC.value,
        agentsims_binary=_AGENTSIMS_BINARY.value,
        tmux_session=_PI_TMUX_SESSION.value,
        output_dir=os.path.join(run_dir, 'pi_logs'),
    )
  # Codex CLI agent driving agentsims.
  elif _AGENT_NAME.value == 'codex':
    agent = codex_agent.CodexAgent(
        env,
        _CODEX_DEVICE_ID.value
        or f'android:emulator-{_DEVICE_CONSOLE_PORT.value}',
        model=_CODEX_MODEL.value,
        reasoning=_CODEX_REASONING.value,
        azure_base_url=_CODEX_AZURE_BASE_URL.value,
        azure_api_version=_CODEX_AZURE_API_VERSION.value,
        azure_api_key_env=_CODEX_AZURE_API_KEY_ENV.value,
        skill_path=_CODEX_SKILL_PATH.value,
        timeout_sec=_CODEX_TIMEOUT_SEC.value,
        codex_binary=_CODEX_BINARY.value,
        agentsims_binary=_AGENTSIMS_BINARY.value,
        tmux_session=_CODEX_TMUX_SESSION.value,
        output_dir=os.path.join(run_dir, 'codex_logs'),
    )
  # SeeAct.
  elif _AGENT_NAME.value == 'seeact':
    agent = seeact.SeeAct(env)

  if not agent:
    raise ValueError(f'Unknown agent: {_AGENT_NAME.value}')

  if (
      agent.name in ['M3A', 'T3A', 'SeeAct']
      and family
      and family.startswith('miniwob')
      and hasattr(agent, 'set_task_guidelines')
  ):
    agent.set_task_guidelines(_MINIWOB_ADDITIONAL_GUIDELINES)
  agent.name = _AGENT_NAME.value

  return agent


def _main() -> None:
  """Runs eval suite and gets rewards back."""
  env = env_launcher.load_and_setup_env(
      console_port=_DEVICE_CONSOLE_PORT.value,
      emulator_setup=_EMULATOR_SETUP.value,
      adb_path=_ADB_PATH.value,
      grpc_port=_GRPC_PORT.value,
  )

  n_task_combinations = _N_TASK_COMBINATIONS.value
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family=_SUITE_FAMILY.value),
      n_task_combinations=n_task_combinations,
      seed=_TASK_RANDOM_SEED.value,
      tasks=_TASKS.value,
      use_identical_params=_FIXED_TASK_SEED.value,
  )
  suite.suite_family = _SUITE_FAMILY.value

  if _CHECKPOINT_DIR.value:
    checkpoint_dir = _CHECKPOINT_DIR.value
  else:
    checkpoint_dir = checkpointer_lib.create_run_directory(_OUTPUT_PATH.value)

  agent = _get_agent(env, checkpoint_dir, _SUITE_FAMILY.value)

  if _SUITE_FAMILY.value.startswith('miniwob'):
    # MiniWoB pages change quickly, don't need to wait for screen to stabilize.
    agent.transition_pause = _MINIWOB_TRANSITION_PAUSE
  else:
    agent.transition_pause = None

  print(
      f'Starting eval with agent {_AGENT_NAME.value} and writing to'
      f' {checkpoint_dir}'
  )
  suite_utils.run(
      suite,
      agent,
      checkpointer=checkpointer_lib.IncrementalCheckpointer(checkpoint_dir),
      demo_mode=False,
  )
  print(
      f'Finished running agent {_AGENT_NAME.value} on {_SUITE_FAMILY.value}'
      f' family. Wrote to {checkpoint_dir}.'
  )
  env.close()


def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
