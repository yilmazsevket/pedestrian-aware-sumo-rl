from __future__ import annotations

"""Interactive PPO training script for SUMO.

Prompts for key hyperparameters, builds vectorized environments via SubprocVecEnv
(spawn-safe), configures PPO with linear LR decay, and registers callbacks for
checkpoints, baseline-vs-agent evaluation, and optional entropy scheduling.
TensorBoard logs are written under runs/ppo_<timestamp> and the final model is saved.
"""

import csv
import os
from datetime import datetime

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import callbacks as cb
from env_factory import EnvConfig, make_env
from SumoEnv import SumoEnv
from tensor_metrics import SB3MetricsCallback


def make_vec_env(n_envs: int, config: EnvConfig) -> VecMonitor:
    env_fns = [make_env(config, rank=i) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns, start_method="spawn")
    vec = VecMonitor(vec, filename=None)
    return vec


def _prompt_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "":
        return default
    try:
        return int(s)
    except ValueError:
        print(f"Invalid input, using default {default}.")
        return default


def _prompt_str(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s or default


def main() -> None:
    print("Interactive PPO training setup. Press Enter to accept defaults.")
    # Interactive parameters
    n_envs = _prompt_int("Number of parallel envs (divisors of 4096 recommended)", 4)
    total_timesteps = _prompt_int("Total timesteps", 2_000_000)
    seed = _prompt_int("Random seed", 42)
    episode_duration = _prompt_int("Episode duration (steps)", 10_000)
    sumo_cfg = _prompt_str("SUMO config file", "validatednet.sumocfg")
    eval_freq_target = _prompt_int("Eval frequency in steps (global)", 200_000)
    ckpt_freq_target = _prompt_int("Checkpoint frequency in steps (global)", 50_000)
    schedule_choice = (
        _prompt_str("Entropy schedule (none/linear_long/two_phase)", "linear_long")
        .strip()
        .lower()
    )

    # Compute n_steps so that one PPO rollout ~ 4096 steps total
    # per_update = n_envs * n_steps; aim for 4096
    if n_envs <= 0:
        n_envs = 1
    n_steps = max(1, 4096 // n_envs)
    per_update = n_steps * n_envs
    if per_update != 4096:
        print(
            f"Note: Using n_steps={n_steps} with n_envs={n_envs} -> {per_update} steps/update (closest to 4096)."
        )

    # Env config: randomized start within 0..86400s
    cfg = EnvConfig(
        episode_duration=episode_duration,
        step_length=1.0,
        start_time=0,
        max_sim_time=86_400,
        seed=seed,
        gui=False,
        fast_forward=True,
        sumo_config=sumo_cfg,
    )

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(f"runs/ppo_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"TensorBoard log dir: {run_dir}")

    # Vectorized envs
    train_env = make_vec_env(n_envs, cfg)
    # Define fixed rush-hour windows (in seconds) – only two peaks
    eval_windows = [7 * 3600 + 30 * 60, 17 * 3600]

    # PPO model
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128, 64]),
        activation_fn=nn.ReLU,
        ortho_init=True,
    )

    # Linear LR decay: lr = lr_start * progress_remaining (1 -> 0)
    lr_start = 3e-4
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=run_dir,
        seed=seed,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=lambda progress_remaining: lr_start * progress_remaining,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=8,
    )

    # Callbacks
    # scale by n_envs because vec.step collects n_envs transitions per call
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, ckpt_freq_target // max(1, n_envs)),
        save_path=run_dir,
        name_prefix="ppo_sumo",
    )
    eval_cb = cb.BaselineEvalCallback(
        config=cfg,
        eval_windows=eval_windows,
        episode_duration=3000,  # evaluate for 3000 steps per run
        n_episodes=1,  # one episode per window
        deterministic=True,
        first=50_000,
        second=100_000,
        every=150_000,
    )

    # Learn
    # Log every rollout (≈ per_update steps) and show a progress bar.
    callbacks = [ckpt_cb, eval_cb, SB3MetricsCallback()]
    if schedule_choice == "linear_long":
        callbacks.append(
            cb.EntropyCoefScheduler(
                total_timesteps=total_timesteps,
                start=0.01,
                end=0.0005,
                start_frac=0.0,
                end_frac=0.8,
            )
        )
    elif schedule_choice == "two_phase":
        callbacks.append(
            cb.EntropyCoefTwoPhaseScheduler(
                total_timesteps=total_timesteps,
                start1=0.01,
                end1=0.003,
                f0=0.0,
                f1=0.5,
                start2=0.003,
                end2=0.001,
                f2=0.5,
                f3=0.9,
            )
        )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=1,
        progress_bar=True,
    )
    model.save(os.path.join(run_dir, "ppo_sumo_final"))

    # Cleanup
    train_env.close()


if __name__ == "__main__":
    main()
    main()
