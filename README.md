# Real-time Adaptive Traffic Signal Control (SUMO + RL)

Bachelor’s thesis project: an end-to-end pipeline to train, evaluate, and analyze a reinforcement learning (RL) policy that controls a single intersection in the SUMO traffic simulator via TraCI.

## What this project does

- Provides a Gymnasium-compatible environment (`SumoEnv`) around SUMO for a single traffic light.
- Trains a discrete-action RL agent (e.g., PPO) and compares it to a fixed baseline.
- Logs performance and safety metrics (min-green enforcement), and produces plots/reports.
- Includes profiling tools to measure real-time step latency on Windows.

## Prerequisites (Windows)

- SUMO (sumo and sumo-gui) installed
  - Example: `C:\Program Files (x86)\Eclipse\Sumo` or `C:\Program Files\Eclipse\Sumo`
- Python 3.10/3.11 with pip
  - Option A: Use your system Python
  - Option B: Use the bundled runtime at `RealSumo\python.exe`

Set environment variables (new terminal):

```cmd
set SUMO_HOME=C:\Program Files\Eclipse\Sumo
set PATH=%SUMO_HOME%\bin;%PATH%
```

## Setup

Install Python dependencies from the project root:

- System Python:

```cmd
python -m pip install -r requirements.txt
```

- Bundled Python (optional):

```cmd
RealSumo\python.exe -m pip install -r requirements.txt
```

## Quickstart

- Sanity check: run a short environment rollout without training

```cmd
python test_env.py
```

- Train PPO (example flags, adjust to taste)

```cmd
python train_ppo.py --total-steps 100000 --gui false --fast-forward true
```

- Evaluate a saved model and generate a short report

```cmd
python eval_agent_report.py --model runs\ppo_model.zip --episodes 10 --gui false
```

- Policy analysis (plots/figures)

```cmd
python policy_analysis.py --model runs\ppo_model.zip
```

- Real-time latency profiling (checks if you can run faster than real time)

```cmd
python profile_realtime_latency.py --model runs\ppo_model.zip --steps 1000 --warmup 50
```

## Key components

- `SumoEnv.py`: Gymnasium environment wrapping TraCI. 54-dim observation, 4 discrete actions.
- `TrafficLightActionSpace.py` + `TrafficLightPhaseSwitcher.py`: Action mapping and safe phase switching with min-green enforcement.
- `reward_calculator.py` and `reward_delta_calculator.py`: Reward formulations (raw and delta variants). `reward_normalizer.py` bounds rewards.
- `train_ppo.py`: Training entry point (SB3 PPO).
- `eval_agent_report.py`: Evaluates agent vs. baseline and summarizes results.
- `policy_analysis.py`: Produces detailed plots and a safety report.
- `profile_realtime_latency.py` and `performance_benchmark.py`: Timing and performance tools.

## Scenario files (SUMO)

- `validatednet.sumocfg`, `validated_net.net.xml`, `vehicles.rou.xml`, `pedestrians.rou.xml`, `Ampel.xml`
  - Define the network, flows, and traffic light program used by the environment.

## Tips and troubleshooting

- “SUMO not found” or TraCI errors: recheck `SUMO_HOME` and `%SUMO_HOME%\bin` on `PATH`.
- Port-in-use errors: fully close previous SUMO instances (or restart the terminal) before re-running.
- Very slow GUI: set `--gui false` for headless speed, or increase SUMO GUI speed in its toolbar.
- Determinism: pass a fixed `--seed` where available.

## Purpose

This repository supports a Bachelor’s thesis on adaptive traffic signal control. It is designed to be reproducible, minimal, and easy to extend for further experiments (reward shaping, additional sensors, or multi-intersection setups).
