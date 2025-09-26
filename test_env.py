"""Random-action sanity test with observation CSV logging.

What it does:
- Starts the environment (GUI optional via env var GUI=1)
- Runs N random steps (default 300 via SANITY_STEPS)
- Logs phase/FSM to console and writes the full observation vector to CSV

CSV columns: step, sim_time, action, phase_name, fsm_state, then all feature names.
Feature names are taken from ObservationBuilder.feature_names(), or fall back to f0..fN.
"""

import csv
import os
import random
from datetime import datetime

import traci

from SumoEnv import SumoEnv


def _get_phase(env: SumoEnv):
    raw = traci.trafficlight.getPhase(env.tl_id)  # type: ignore[arg-type]
    if isinstance(raw, (list, tuple)):
        idx = int(raw[0])
    else:
        idx = int(raw)
    name = env.tl_action_space.INDEX_TO_NAME.get(idx, str(idx))
    return idx, name


if __name__ == "__main__":
    seed = 123
    random.seed(seed)

    # Sanity run: keep it short
    STEPS = int(os.environ.get("SANITY_STEPS", 300))
    EPISODE_DURATION = max(STEPS + 5, 30)
    TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_NAME = os.environ.get("OBS_LOG_CSV", f"obs_log_{TS}.csv")
    csv_path = os.path.abspath(CSV_NAME)
    print(f"Schreibe Observation CSV nach: {csv_path}")

    # Disable GUI for a fast sanity run (override with env GUI=1)
    use_gui = os.environ.get("GUI", "0") == "1"
    env = SumoEnv(
        start_time=30000,
        episode_duration=EPISODE_DURATION,
        gui=use_gui,
        reward_mode="delta",
    )
    # Always enable reward logging alongside the obs CSV
    REWARD_CSV = os.environ.get("REWARD_LOG_CSV", f"reward_log_{TS}.csv")
    try:
        env.enable_reward_logging(REWARD_CSV)
        print(f"Schreibe Reward CSV nach: {os.path.abspath(REWARD_CSV)}")
    except Exception as e:
        print(f"Konnte Reward-Logging nicht aktivieren: {e}")
    obs, info = env.reset()
    # Disable observation normalization for debugging if available
    if getattr(env, "obs_builder", None) is not None:
        try:
            env.obs_builder.normalize = False  # type: ignore[attr-defined]
        except Exception:
            pass
    print("Episode Start:", info)

    try:
        feature_names = (
            env.obs_builder.feature_names()  # type: ignore[union-attr]
            if getattr(env, "obs_builder", None) is not None
            else [f"f{i}" for i in range(len(obs))]
        )
    except Exception:
        feature_names = [f"f{i}" for i in range(len(obs))]

    header = ["step", "sim_time", "action", "phase_name", "fsm_state"] + feature_names
    # buffering=1 for line-buffered (if possible); additionally force flush()
    with open(CSV_NAME, "w", newline="", encoding="utf-8", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        # Action space is Discrete(4)
        try:
            num_actions = int(getattr(env.action_space, "n"))
        except Exception:
            num_actions = 4
        group_map = env.tl_action_space.ACTION_TO_GROUP
        switcher = env.tl_action_space.switcher

        for step in range(1, STEPS + 1):  # Sanity Loop
            action = random.randrange(num_actions)
            before_idx, before_name = _get_phase(env)
            obs, reward, terminated, truncated, info = env.step(action)

            sim_time = info.get("sim_time", step)
            cur_phase_name = info.get("phase_name", before_name)
            fsm_state = getattr(switcher, "current_state", lambda: "?")()

            # Compact console (optional)
            # print(f"Step {step:03d} | act={action}->{group_map.get(action,'?')} | sim={sim_time:.0f} | phase={cur_phase_name} | FSM={fsm_state} | reward={reward:.3f}")

            # CSV row
            row = [step, sim_time, action, cur_phase_name, fsm_state] + [
                float(x) for x in obs
            ]
            writer.writerow(row)
            # Make immediately visible (Windows: flush + os.fsync for hard persistence)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
            if terminated or truncated:
                print("Episode beendet (terminated/truncated).")
                break

    env.close()
    print("Fertig. CSV geschrieben.")
