from __future__ import annotations

"""Run a trained PPO agent in SUMO GUI at morning rush hour.

loads model, builds a GUI env, fast-forwards
to the desired start_time, runs deterministically, and closes cleanly.
"""

from typing import Tuple

from stable_baselines3 import PPO

from SumoEnv import SumoEnv


def build_env(start_time: int, episode_duration: int) -> SumoEnv:
    """Create a single SUMO GUI environment configured for viewing.

    - SUMO starts at t=0 under the hood; the env fast-forwards to start_time.
    - GUI is enabled; agent_control=True so the policy controls the TLS.
    """
    return SumoEnv(
        start_time=start_time,
        episode_duration=episode_duration,
        step_length=1.0,
        seed=42,
        gui=True,
        sumo_config="validatednet.sumocfg",
        fast_forward=True,
        # keep reward details consistent, but irrelevant for viewing
        reward_mode="delta",
        agent_control=True,
    )


def run_episode(model: PPO, env: SumoEnv) -> Tuple[float, int]:
    """Run one deterministic episode and return (sum_reward, steps)."""
    obs, _info = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return total_reward, steps


def main() -> None:
    # Point to your trained model zip
    model_path = r"C:\Users\yilma\Desktop\RealSumo\runs\RUN_4_ppo_6_000_000_20k_accreward\ppo_sumo_final.zip"

    # Morning rush hour start (07:30) and a viewing window (e.g., 1800s = 30min)
    start_time = 7 * 3600 + 30 * 60
    episode_duration = 1800

    # Load model and build GUI env
    model = PPO.load(model_path)
    env = build_env(start_time=start_time, episode_duration=episode_duration)

    try:
        ret, steps = run_episode(model, env)
        print(f"Finished viewing episode: steps={steps}, return={ret:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
