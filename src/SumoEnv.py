"""SumoEnv: Gymnasium environment for controlling a single traffic light in SUMO.

- Wraps a TraCI simulation and exposes a 54-dim observation and 4 discrete actions.
- Supports raw or delta reward with optional normalization and logging.
- Designed for vectorized PPO training (e.g., SubprocVecEnv) with randomized start times.
"""

from typing import Any, Dict, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces

from observation_builder import ObservationBuilder
from pedestrian_manager import PedestrianManager
from reward_calculator import RewardCalculator
from reward_delta_calculator import DeltaRewardCalculator
from reward_logger import RewardLogger
from reward_normalizer import RewardNormalizer
from TrafficLightActionSpace import TrafficLightActionSpace
from vehicle_manager import VehicleManager


class SumoEnv(gym.Env):

    def __init__(
        self,
        start_time: int = 0,
        episode_duration: int = 1000,
        step_length: float = 1.0,
        seed: int = 42,
        gui: bool = True,
        sumo_config: str = "validatednet.sumocfg",
        fast_forward: bool = True,
        reward_alpha: float = 1.0,
        reward_beta: float = 0.01,
        reward_clip_min: float = -10.0,
        reward_clip_max: float = 0.0,
        reward_pressure_scale: float = 10.0,
        # Reward strategy
        reward_mode: str = "raw",  # "raw" | "delta"
        queue_weight: float = 0.1,
        ped_wait_weight: float = 1.0,
        ped_queue_weight: float = 1.0,
        reward_norm_scale: float = 1000.0,
        reward_norm_clip_min: float = -3.0,
        reward_norm_clip_max: float = 2.0,
        agent_control: bool = True,
        traci_label: Optional[str] = None,
    ) -> None:
        super().__init__()
        # 54 features total (30 base + 24 directional pairs) – dynamically sized if needed
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(54,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.start_time = int(start_time)
        self.episode_duration = int(episode_duration)
        self.step_length = float(step_length)
        self.seed_value = int(seed)
        self.use_gui = bool(gui)
        self.sumoConfig = sumo_config
        self.fast_forward = bool(fast_forward)
        self.agent_control = bool(agent_control)
        # Optional TraCI label for isolated connections (useful in multi-process eval)
        self.traci_label = traci_label
        # Reward strategy (raw vs delta)
        # Note: Observations use accumulated (ACC) waits, while the Δ-reward uses snapshot waits (local per-step delta).
        self.reward_mode = str(reward_mode).lower()
        if self.reward_mode == "delta":
            self.reward_calc = DeltaRewardCalculator(
                queue_weight=queue_weight,
                ped_wait_weight=ped_wait_weight,
                ped_queue_weight=ped_queue_weight,
            )
        else:
            # Default raw reward
            self.reward_calc = RewardCalculator(
                alpha=reward_alpha,
                beta=reward_beta,
                clip_min=reward_clip_min,
                clip_max=reward_clip_max,
                pressure_scale=reward_pressure_scale,
            )
        # Reward normalizer to bound rewards into [-1, 1]
        # Tip for Δ-reward: use scale ≈ 150–200 to avoid saturation.
        self.reward_norm = RewardNormalizer(
            scale=reward_norm_scale,
            clip_min=reward_norm_clip_min,
            clip_max=reward_norm_clip_max,
        )
        self._reward_logger: Optional[RewardLogger] = None

        self.episode_end_sim_time = self.start_time + self.episode_duration
        self.step_count = 0

        self.tl_action_space = TrafficLightActionSpace(step_length=self.step_length)
        self.tl_id: Optional[str] = None
        self.vehicle_manager: Optional[VehicleManager] = None
        self.pedestrian_manager: Optional[PedestrianManager] = None
        self.obs_builder: Optional[ObservationBuilder] = None
        np.random.seed(self.seed_value)

    def _build_sumo_cmd(self) -> list:
        binary = "sumo-gui" if self.use_gui else "sumo"
        return [
            binary,
            "-c",
            self.sumoConfig,
            "--begin",
            "0",
            "--step-length",
            str(self.step_length),
            "--seed",
            str(self.seed_value),
            "--no-step-log",
            "--no-warnings",
        ]

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if options:
            if "start_time" in options:
                self.start_time = int(options["start_time"])
            if "episode_duration" in options:
                self.episode_duration = int(options["episode_duration"])
            self.episode_end_sim_time = self.start_time + self.episode_duration

        if traci.isLoaded():
            try:
                traci.close()
            except Exception:
                pass

        cmd = self._build_sumo_cmd()
        label = getattr(self, "traci_label", None)
        try:
            if label:
                traci.start(cmd, label=str(label))
                # Ensure the labeled connection is the active default for global traci.* calls
                try:
                    traci.switch(str(label))
                except Exception:
                    pass
            else:
                traci.start(cmd)
        except Exception:
            # Fallback without label if labeled startup fails
            traci.start(cmd)
        self.step_count = 0
        # Reset stateful reward calculators (e.g., delta-based)
        try:
            reset_fn = getattr(self.reward_calc, "reset", None)
            if callable(reset_fn):
                reset_fn()
        except Exception:
            pass

        fast_forwarded = False
        fast_steps = 0
        if self.fast_forward and self.start_time > 0:
            target = self.start_time
            # Try direct fast jump
            try:
                traci.simulationStep(target)
                fast_forwarded = True
                fast_steps = int(target / self.step_length)
            except Exception:
                # Fallback: slow forward step-by-step
                while True:
                    current = self._get_sim_time()
                    if current >= target:
                        break
                    traci.simulationStep()
                    fast_steps += 1
                fast_forwarded = True

        # Capture first (and only) TLS id
        try:
            tls = traci.trafficlight.getIDList()
            self.tl_id = tls[0] if tls else None
        except Exception:
            self.tl_id = None

        # Build managers / builder
        if self.tl_id is not None:
            try:
                self.tl_action_space.build(traci, self.tl_id)
                # VehicleManager tracks both ACC and snapshot waits in parallel.
                # Observations consume ACC waits; DeltaRewardCalculator consumes snapshot waits.
                self.vehicle_manager = VehicleManager(
                    traci,
                    inbound_lanes=(),
                    left_lanes=ObservationBuilder.LEFT_LANES,
                    right_targets=ObservationBuilder.RIGHT_TARGETS,
                )
                self.pedestrian_manager = PedestrianManager(
                    traci,
                    approaches=ObservationBuilder.PED_WAIT_EDGES,
                    island_lanes=ObservationBuilder.PED_ISLAND_LANES,
                )
                self.obs_builder = ObservationBuilder(
                    self.tl_action_space.switcher,
                    normalize=True,
                    vehicle_manager=self.vehicle_manager,
                    pedestrian_manager=self.pedestrian_manager,
                )
            except Exception:
                self.obs_builder = None

        obs = self._build_observation()
        info = {
            "sim_time": self._get_sim_time(),
            "episode_end_time": self.episode_end_sim_time,
            "start_time": self.start_time,
            "fast_forwarded": fast_forwarded,
            "fast_forward_steps": fast_steps,
        }
        # Include TLS state if available
        if self.tl_id is not None:
            try:
                cur_idx = int(cast(Any, traci.trafficlight.getPhase(self.tl_id)))
                info["tls_id"] = self.tl_id
                info["phase_index"] = cur_idx
                info["phase_name"] = self.tl_action_space.INDEX_TO_NAME.get(
                    cur_idx, str(cur_idx)
                )
            except Exception:
                pass
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not traci.isLoaded():
            obs = self._build_observation()
            return obs, 0.0, True, False, {"reason": "traci_not_loaded"}

        # Try to apply the agent action (may do internal transition steps)
        if self.agent_control:
            try:
                self._apply_action(action)
            except Exception:
                pass

        # Advance one step for the environment step
        try:
            traci.simulationStep()
        except Exception as e:
            obs = self._build_observation()
            return obs, 0.0, True, False, {"error": str(e)}

        # Update internal switcher timers/state AFTER the simulation step so phase index reflects new state
        try:
            self.tl_action_space.tick(traci, self.tl_id)
        except Exception:
            pass

        self.step_count += 1
        sim_time = self._get_sim_time()
        obs = self._build_observation()
        reward_raw, reward_info = self.reward_calc.compute(
            self.vehicle_manager, self.pedestrian_manager
        )
        # Apply final normalization into a bounded range
        reward_norm = self.reward_norm.normalize(reward_raw)
        # Optional: reward logging
        logger = getattr(self, "_reward_logger", None)
        if logger is not None:
            comps = {
                k: float(v)
                for k, v in reward_info.items()
                if isinstance(v, (int, float))
            }
            try:
                action_int = int(action) if isinstance(action, (int, float)) else -1
            except Exception:
                action_int = -1
            logger.log(
                step=self.step_count,
                sim_time=sim_time,
                action=action_int,
                reward=float(reward_raw),
                components=comps,
            )
        terminated = sim_time >= self.episode_end_sim_time
        truncated = False
        if terminated:
            self.close()
        info: Dict[str, Any] = {
            "sim_time": sim_time,
            "episode_step": self.step_count,
            **reward_info,
            "reward_norm": float(reward_norm),
            "reward_raw": float(reward_raw),
        }
        # Enrich info with pedestrian-type-specific wait sums (e.g., vulnerable)
        try:
            if self.pedestrian_manager is not None:
                type_stats = self.pedestrian_manager.wait_stats_by_type()
                vuln = type_stats.get("vulnerable_pedestrian")
                if isinstance(vuln, tuple) and len(vuln) >= 1:
                    info["vuln_ped_wait_sum"] = float(vuln[0])
        except Exception:
            pass
        if self.tl_id is not None:
            try:
                cur_idx = int(cast(Any, traci.trafficlight.getPhase(self.tl_id)))
                info["phase_index"] = cur_idx
                info["phase_name"] = self.tl_action_space.INDEX_TO_NAME.get(
                    cur_idx, str(cur_idx)
                )
            except Exception:
                pass
        return obs, float(reward_norm), terminated, truncated, info

    def close(self) -> None:
        # Ensure reward logger is closed
        if getattr(self, "_reward_logger", None) is not None:
            try:
                self._reward_logger.close()  # type: ignore[union-attr]
            finally:
                self._reward_logger = None
        if traci.isLoaded():
            try:
                traci.close()
            except Exception:
                pass

    def _build_observation(self) -> np.ndarray:
        if self.obs_builder is None or self.tl_id is None:
            shape = getattr(self.observation_space, "shape", (54,)) or (54,)
            return np.zeros(shape, dtype=np.float32)
        try:
            # Ensure switcher timing stays in sync even when observation requested outside step()
            try:
                self.tl_action_space.tick(traci, self.tl_id)
            except Exception:
                pass
            vec = self.obs_builder.compute(traci, self.tl_id)
            # If dynamic length (feature names may be extended after first compute)
            obs_shape = getattr(self.observation_space, "shape", None)
            if not obs_shape or not isinstance(obs_shape, tuple):
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(vec.shape[0],), dtype=np.float32
                )
            elif vec.shape[0] != obs_shape[0]:
                # Resize observation_space once to match vector length
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(vec.shape[0],), dtype=np.float32
                )
            return vec
        except Exception:
            shape = getattr(self.observation_space, "shape", (54,)) or (54,)
            return np.zeros(shape, dtype=np.float32)

    # --- TLS control helpers ---
    def _apply_action(self, action: Any) -> None:
        self.tl_action_space.apply_action(traci, self.tl_id, action)

    def _get_sim_time(self) -> float:
        """Return current simulation time as float, tolerant to TraCI typing quirks."""
        t = traci.simulation.getTime()
        try:
            return float(cast(Any, t))
        except Exception:
            # Try tuple first element fallback
            try:
                return float(cast(Any, t[0]))  # type: ignore[index]
            except Exception:
                return 0.0

    def set_start_time(self, start_time: int) -> None:
        self.start_time = int(start_time)
        self.episode_end_sim_time = self.start_time + self.episode_duration

    def set_episode_duration(self, duration: int) -> None:
        self.episode_duration = int(duration)
        self.episode_end_sim_time = self.start_time + self.episode_duration

    # --- Reward parameter setters (delegated to RewardCalculator) ---
    def set_reward_alpha(self, value: float) -> None:
        if isinstance(self.reward_calc, RewardCalculator):
            self.reward_calc.alpha = value

    def set_reward_beta(self, value: float) -> None:
        if isinstance(self.reward_calc, RewardCalculator):
            self.reward_calc.beta = value

    def set_reward_clip(
        self, clip_min: float | None = None, clip_max: float | None = None
    ) -> None:
        if isinstance(self.reward_calc, RewardCalculator):
            if clip_max is not None:
                self.reward_calc.clip_max = clip_max
            if clip_min is not None:
                self.reward_calc.clip_min = clip_min

    def set_reward_pressure_scale(self, value: float) -> None:
        if isinstance(self.reward_calc, RewardCalculator):
            self.reward_calc.pressure_scale = value

    # --- Reward Normalizer controls ---
    def set_reward_normalizer(
        self,
        scale: float | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> None:
        if scale is not None:
            self.reward_norm.scale = scale
        if clip_min is not None:
            self.reward_norm.clip_min = clip_min
        if clip_max is not None:
            self.reward_norm.clip_max = clip_max

    # --- Reward Logger controls ---
    def enable_reward_logging(self, path: Optional[str] = None) -> None:
        self._reward_logger = RewardLogger(path=path)

    def disable_reward_logging(self) -> None:
        if self._reward_logger is not None:
            try:
                self._reward_logger.close()
            finally:
                self._reward_logger = None

    # (Reward logic delegated to RewardCalculator)
