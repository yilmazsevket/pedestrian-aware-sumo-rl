from __future__ import annotations

"""Environment factory for SumoEnv.

Defines:
- EnvConfig: configuration for building SumoEnv instances
- RandomStartWrapper: randomizes the episode start time on each reset
- make_env: returns a Windows spawn-safe thunk for vectorized training
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
from gymnasium.wrappers import TimeLimit  # noqa: F401  (kept for optional use)

from SumoEnv import SumoEnv


@dataclass
class EnvConfig:
    """Configuration for building SumoEnv instances.

    episode_duration: RL episode length in seconds (sim steps if step_length=1.0).
    max_sim_time: maximum available simulation time window (e.g., 86400 seconds).
    """

    # Simulation/episode
    episode_duration: int = 10_000
    step_length: float = 1.0
    start_time: int = 0  # initial seed start, will be overridden by wrapper
    max_sim_time: int = 86_400

    # SUMO/Env
    seed: int = 42
    gui: bool = False
    fast_forward: bool = True
    sumo_config: str = "validatednet.sumocfg"

    # Reward parameters
    reward_alpha: float = 1.0
    reward_beta: float = 0.01
    reward_clip_min: float = -10.0
    reward_clip_max: float = 0.0
    reward_pressure_scale: float = 10.0
    # Reward mode: 'raw' (legacy) or 'delta' (incremental)
    reward_mode: str = "delta"
    queue_weight: float = 0.1
    ped_wait_weight: float = 1.0
    ped_queue_weight: float = 1.0
    # Reward normalization (applied inside SumoEnv before PPO sees rewards)
    reward_norm_scale: float = 1000.0
    reward_norm_clip_min: float = -3.0
    reward_norm_clip_max: float = 2.0
    # Vehicle wait source for training (snapshot recommended for delta reward)
    use_accumulated_vehicle_waits: bool = False


class RandomStartWrapper(gym.Wrapper):
    """Wrapper that randomizes start_time on every reset.

    Picks a random window [start_time, start_time + episode_duration] within [0, max_sim_time].
    """

    def __init__(
        self,
        env: SumoEnv,
        base_seed: int,
        rank: int,
        episode_duration: int,
        max_sim_time: int,
    ) -> None:
        super().__init__(env)
        self.episode_duration = int(episode_duration)
        self.max_sim_time = int(max_sim_time)
        self._rng = random.Random(base_seed + rank)

    def reset(self, **kwargs):  # type: ignore[override]
        max_start = max(0, self.max_sim_time - self.episode_duration)
        start_time = self._rng.randint(0, max_start)
        # Configure underlying env window before reset
        try:
            set_start = getattr(self.env, "set_start_time", None)
            if callable(set_start):
                set_start(start_time)
            set_ep = getattr(self.env, "set_episode_duration", None)
            if callable(set_ep):
                set_ep(self.episode_duration)
        except Exception:
            pass
        return self.env.reset(**kwargs)


def make_env(config: EnvConfig, rank: int = 0) -> Callable[[], Any]:
    """Factory returning a thunk for SubprocVecEnv (Windows spawn-safe).

    The thunk creates a SumoEnv and applies RandomStartWrapper.
    """

    def _init():
        env = SumoEnv(
            start_time=config.start_time,
            episode_duration=config.episode_duration,
            step_length=config.step_length,
            seed=config.seed + rank,
            gui=config.gui,
            sumo_config=config.sumo_config,
            fast_forward=config.fast_forward,
            reward_alpha=config.reward_alpha,
            reward_beta=config.reward_beta,
            reward_clip_min=config.reward_clip_min,
            reward_clip_max=config.reward_clip_max,
            reward_pressure_scale=config.reward_pressure_scale,
            reward_mode=config.reward_mode,
            queue_weight=config.queue_weight,
            ped_wait_weight=config.ped_wait_weight,
            ped_queue_weight=config.ped_queue_weight,
            reward_norm_scale=config.reward_norm_scale,
            reward_norm_clip_min=config.reward_norm_clip_min,
            reward_norm_clip_max=config.reward_norm_clip_max,
        )
        # Randomize start each episode
        env = RandomStartWrapper(
            env,
            base_seed=config.seed,
            rank=rank,
            episode_duration=config.episode_duration,
            max_sim_time=config.max_sim_time,
        )
        return env

    return _init
