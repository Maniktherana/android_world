# AndroidWorld

<!-- mdlint off(WHITESPACE_LINE_LENGTH) -->

[![Unittests](https://github.com/google-research/android_world/actions/workflows/pytest.yml/badge.svg)](https://github.com/google-research/android_world/actions/workflows/pytest.yml)

<p align="center">
<a href="https://google-research.github.io/android_world/">Website</a> •
<a href="https://arxiv.org/pdf/2405.14573">Paper</a> •
<a href="https://google-research.github.io/android_world/task_list.html">Tasks</a> •
<a href="https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?gid=0#gid=0">Leaderboard</a>
</p>

![Overview](assets/overview.png)

**AndroidWorld** is an environment for building and benchmarking autonomous
computer control agents.

It runs on a live Android emulator and contains a highly reproducible benchmark
of 116 hand-crafted tasks across 20 apps, which are dynamically instantiated
with randomly-generated parameters to create millions of unique task variations.

In addition to the built-in tasks, AndroidWorld also supports the popular web benchmark, MiniWoB++ from [Liu et al.](http://arxiv.org/abs/1802.08802).

Key features of AndroidWorld include:

* 📝 **116 diverse tasks** across 20 real-world apps
* 🎲 **Dynamic task instantiation** for millions of unique variations
* 🏆 **Durable reward signals** for reliable evaluation
* 🐳 **Experimental Docker Support** for simplified setup and consistent environments (as of 06/02/2025)
* 🌐 **Open environment** with access to millions of Android apps and websites
* 💾 **Lightweight footprint** (2 GB memory, 8 GB disk)
* 🔧 **Extensible design** to easily add new tasks and benchmarks
* 🖥️ **Integration with MiniWoB++** web-based tasks

See demo videos on our [website](https://google-research.github.io/android_world/).
o

## Installation

1. Set up the Android Emulator
   1. Download Android Studio [here](https://developer.android.com/studio?gad_source=1&gclid=Cj0KCQjw3ZayBhDRARIsAPWzx8oLcadBD0vAq8xmUutaunLGSzhgEtLz4xVZ_SpV4G0xJazS7LxQkDsaAuveEALw_wcB&gclsrc=aw.ds)
   2. Create an Android Virtual Device (AVD) by following these instructions. For hardware select **Pixel 6**, for System Image select **Tiramisu, API Level 33**, and choose AVD name as **AndroidWorldAvd**. [Watch the setup video.](https://github.com/google-research/android_world/assets/162379927/efc33980-8b36-44be-bb2b-a92d4c334a50)

1. Launch the Android Emulator

    Launch `AndroidWorldAvd` through Agentsims or Android Studio, or use the
    following command. AndroidWorld connects to the running emulator through
    ADB and gRPC.

    ```bash
    # Typically it's located in ~/Android/Sdk/emulator/emulator or
    # ~/Library/Android/sdk/emulator/emulator
    EMULATOR_NAME=AndroidWorldAvd # From previous step
    ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
    ```

    AndroidWorld reads the emulator's gRPC port and authentication token from its
    local discovery file on macOS and Linux. This supports the connection that
    Agentsims uses for its preview.

    Check the running device with `adb devices`. Set `--console_port` in either
    runner to the number after `emulator-`. This number can change after a restart.
    For example, `emulator-5554` requires `--console_port=5554`.
    The console port and gRPC port are separate. Both runners discover the gRPC
    port automatically. Use `--grpc_port` only when automatic discovery is
    unavailable. Manual launches without discovery metadata default to gRPC port
    8554. Each running emulator needs a separate gRPC port.

1. [Optional] It's recommended to use `conda`, which you can download [here](https://docs.anaconda.com/free/miniconda/miniconda-install/).

    ```
    conda create -n android_world python=3.11.8
    conda activate android_world
    ```

1. Install AndroidWorld. *Note: Python 3.11 or above is required.*

    ```python
    git clone https://github.com/google-research/android_world.git
    cd ./android_world
    pip install -r requirements.txt
    python setup.py install
    ```

1. Add model provider APIs as environment variables.

    ```bash
    # Add to .bashrc.
    export OPENAI_API_KEY=your-key
    export GCP_API_KEY=your-key
    ```

1. Install `ffmpeg`, if not already installed.

    ```bash
    # Linux (Ubuntu/Debian)
    # sudo apt update && sudo apt install ffmpeg

    # macOS
    brew install ffmpeg
    ```

## Quickstart

Run the `minimal_task_runner.py` script to see the basic mechanics of
AndroidWorld components. It initializes the environment, sets up a task, and
runs the default agent, M3A, on it.
```bash
python minimal_task_runner.py --task=ContactsAddContact
```

If you don't specify a task, a random task will be selected. *NOTE: If you want
to try open-source apps, i.e. not included with Android OS, please run
`--perform_emulator_setup` in the script below.*

**Note on Model Cost:** The `minimal_task_runner.py` script uses a legacy model `gpt-4-turbo-2024-04-09` by default. This model can be expensive. For serious usage, you can switch to a more cost-effective model, by modifying the `model_name` in the script.

### Run the benchmark with the Pi coding agent

`run_pi_benchmark.sh` scores the [Pi](https://github.com/badlogic/pi-mono) coding
agent on AndroidWorld. Pi drives the emulator through the
[agentsims](https://github.com/Maniktherana/agentsims) CLI and its
`build-mobile-apps` skill, the same way a person uses the phone. One task is one
Pi session. AndroidWorld grades device state after Pi stops.

Requirements:

- `agentsims` on `PATH`, with a workspace that lists the emulator.
- `pi` on `PATH`, with a provider in `~/.pi/agent/models.json`.

```bash
./run_pi_benchmark.sh --tasks=ContactsAddContact   # one task
./run_pi_benchmark.sh                              # the whole suite
```

The script checks the emulator and the agentsims workspace, then starts the run.
Each task starts an Agentsims trace before Pi runs. The trace stops before
AndroidWorld receives Pi's result, including when Pi fails or times out. Traces
are in `~/.agentsims/traces/`. Their names use
`<task>-<device>-<UTC timestamp>`. Pi's output streams to the terminal. Each
session is also written to `<output_path>/pi_logs/`.

Override the defaults with environment variables: `PI_PROVIDER`, `PI_MODEL`,
`PI_TIMEOUT_SEC`, `CONSOLE_PORT`, `DEVICE_ID`, `AGENTSIMS_BIN`, `SKILL_PATH`,
`OUTPUT_PATH`, and `PYTHON`. The equivalent `run.py` flags are
`--agent_name=pi` with `--pi_agent_provider`, `--pi_agent_model`,
`--pi_device_id`, `--pi_skill_path`, `--pi_thinking`, `--pi_timeout_sec`, and
`--agentsims_binary`.

The task prompt tells Pi to reach the end state through the UI and not through
`adb` or direct database writes. That is an instruction, not a sandbox. Read the
transcripts in `pi_logs/` to audit a run.

### Run the benchmark with Codex CLI

`run_codex_benchmark.sh` runs a new Codex CLI process for every AndroidWorld
task. Codex uses `gpt-5.6-luna` through Azure OpenAI and drives the emulator
through the agentsims CLI.

Requirements:

- `codex`, `agentsims`, and `tmux` on `PATH`.
- `AZURE_OPENAI_API_KEY` set in the environment.
- The Azure deployment available as `gpt-5.6-luna`.

```bash
export AZURE_OPENAI_API_KEY='...'
./run_codex_benchmark.sh --tasks=ContactsAddContact
./run_codex_benchmark.sh
```

Watch the live Codex output from another terminal:

```bash
tmux attach -r -t androidworld-codex
```

The tmux window stays available for the full run. The `-r` option makes the
attached client read-only, so terminal input cannot affect the benchmark. Each
task starts with an empty Codex context. The runner ignores the normal Codex
user configuration and uses only the Azure provider that it supplies for the
task. The key remains in the environment and is not written into the command or
log.

Each task starts an Agentsims trace immediately before its prompt is submitted.
The trace stops after Codex exits and before AndroidWorld grades the task. Trace
names use `<task>-<device>-<UTC timestamp>`. Task files are written to
`<output_path>/codex_logs/`.

Override the defaults with `CODEX_MODEL`, `CODEX_REASONING`,
`AZURE_OPENAI_ENDPOINT`, `CODEX_AZURE_API_VERSION`, `CODEX_TIMEOUT_SEC`,
`CODEX_TMUX_SESSION`, `CONSOLE_PORT`, `DEVICE_ID`, `AGENTSIMS_BIN`,
`CODEX_BIN`, `SKILL_PATH`, `OUTPUT_PATH`, and `PYTHON`.

## Docker Support (Experimental)

AndroidWorld now offers Docker support. This allows you to run the Android
environment and server within a Docker container, which can simplify setup and
ensure a consistent environment.

**Note:** This feature is experimental and has not been extensively tested.

1.  **Build the Docker image:**

    Navigate to the root directory of the `android_world` repository and run:
    ```bash
    docker build -t android_world:latest .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run --privileged -p 5000:5000 -it android_world:latest
    ```
    This will start the Android emulator and the FastAPI server inside the
    container. The server will be accessible on `http://localhost:5000`.

3.  **Interact with the environment:**
    You can see the `scripts/run_suite_on_docker.py` script as an example client
    to interact with the Android environment server running in Docker.

### Note for Apple Silicon users

There are known [issues](https://github.com/amrsa1/Android-Emulator-image/issues/10) with installing the required package `emulator` on ARM chips (Apple Silicon). To get around this, if building images locally, you should build images for the AMD64/x86_64 instruction set, by running:
```bash
docker buildx build --platform linux/amd64 -t android-emulator:latest .
```

Note, running in a Docker container like this, on an Apple Silicon device will run quite slowly compared to running the Android
Device and Emulator natively (because you end up running an Android Emulator inside a Linux Emulator...).

## Run the benchmark

Note: **Task Step Limits Update**
As of 11/18/2024, the max_steps/step_budget for each task in AndroidWorld have been updated to approximately **2x the human average completion time**. This adjustment ensures agents have sufficient time to complete tasks, while also reducing overhead of running thebenchmark. [Here](https://docs.google.com/spreadsheets/d/1KF-vY0Uy47o0mnursvs-HmS6hreU6U3rPrAjgEfjMK4/edit?usp=sharing) are the per-task updates.

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=t3a_gpt4 \
  --perform_emulator_setup \
  --tasks=ContactsAddContact,ClockStopWatchRunning \  # Optional: Just run on a subset.
```

The first time you run this script, you must install the necessary apps and set
permissions by specifying `--perform_emulator_setup`. This is a one-time setup.
It may take several minutes depending on the connection speed.

Above we specify the optional `--tasks` flag to run on a subset of tasks. Leave
it empty to run on the entire AndroidWorld suite.

The `n_task_combinations` argument specifies how many parameter permutations to
use for each task. For example, for an SMS task, it would correspond to
different phone number/message combinations for each run.

If a run fails part-way through, you can resume it by re-running the script with
the `--checkpoint_dir` flag pointing to the output directory from the original
run.

## Running MiniWoB++ tasks

To run the MiniWoB++ web-based tasks in AndroidWorld, simply set
`--suite_family=miniwob` and `--perform_emulator_setup` in the command above.

A key advantage of running MiniWoB++ tasks is that common input elements are
rendered as native, commonly used Android UI widgets, rather than as HTML. Thus
agents must learn to use universal widgets such as time- and date-pickers:

<p align="center">
   <img src="assets/miniwob.png" style="width:30%">
</p>

## Create your own agent

In addition to the agents we provide [here](https://github.com/google-research/android_world/tree/main/android_world/agents), you can also easily create your own agent and run the benchmark with it as follows.

1. Create an agent class that inherits from [EnvironmentInteractingAgent](https://github.com/google-research/android_world/blob/6e4feb00702735c9a7485f4ae714528a058cb2b7/android_world/agents/base_agent.py#L39C1-L39C44) and implement the [step](https://github.com/google-research/android_world/blob/6e4feb00702735c9a7485f4ae714528a058cb2b7/android_world/agents/base_agent.py#L116) method.
In the current workflow, the agent tries to complete a task in a for loop. In each round, the [step](https://github.com/google-research/android_world/blob/6e4feb00702735c9a7485f4ae714528a058cb2b7/android_world/agents/base_agent.py#L116) method will be called and this is where you implement your agent's logic. A typical approach involves first gathering information like the current screenshot, the UI elements (like buttons, icons) through the AndroidEnv instance within the agent, selecting one of the [supported actions](https://github.com/google-research/android_world/blob/main/android_world/env/json_action.py), executing it through the AndroidEnv and returning an [AgentInteractionResult](https://github.com/google-research/android_world/blob/6e4feb00702735c9a7485f4ae714528a058cb2b7/android_world/agents/base_agent.py#L26). The `done` property on AgentInteractionResult should be set to true to indicate that the task is finished.

2. Import your agent in [run.py](https://github.com/google-research/android_world/blob/main/run.py) and also add it into the [_get_agent](https://github.com/google-research/android_world/blob/15471441ac306ff08bca87454b1b546ae81db7af/run.py#L147) method which takes in your agent's name and return an instance of it.

3. Now you can run the benchmark with your new agent using the command above with the `agent_name` flag changed to your agent's name.

## Adding new tasks

Please see [the guide](https://github.com/google-research/android_world/blob/main/docs/tasks_guide.md) on adding new tasks to AndroidWorld.

## Citation

If you use our environment or data, please cite our paper:

```
@misc{rawles2024androidworlddynamicbenchmarkingenvironment,
      title={AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents},
      author={Christopher Rawles and Sarah Clinckemaillie and Yifan Chang and Jonathan Waltz and Gabrielle Lau and Marybeth Fair and Alice Li and William Bishop and Wei Li and Folawiyo Campbell-Ajala and Daniel Toyama and Robert Berry and Divya Tyamagundlu and Timothy Lillicrap and Oriana Riva},
      year={2024},
      eprint={2405.14573},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.14573},
}
```

*This is not an officially supported Google product.*
