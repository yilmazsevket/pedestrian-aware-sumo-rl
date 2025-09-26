from __future__ import annotations

"""Custom training callbacks for Stable-Baselines3.

Includes:
- BaselineEvalCallback: scheduled evaluation vs SUMO baseline in separate processes.
- EntropyCoefScheduler: single-phase linear decay of ent_coef over training.
- EntropyCoefTwoPhaseScheduler: two-phase linear decay with final hold.
"""

import multiprocessing as mp
import os
import time
import uuid
from typing import List, Optional, Tuple

from stable_baselines3.common.callbacks import BaseCallback

from env_factory import EnvConfig
from SumoEnv import SumoEnv


class BaselineEvalCallback(BaseCallback):
    """Evaluate agent vs. SUMO baseline (agent_control=False) on a schedule.

    Logs to TensorBoard under:
      - eval_agent/mean_reward, eval_agent/mean_ep_len
      - eval_baseline/mean_reward, eval_baseline/mean_ep_len
    """

    def __init__(
        self,
        config: EnvConfig,
        eval_windows: List[int],
        episode_duration: int,
        n_episodes: int = 3,
        deterministic: bool = True,
        first: int = 50_000,
        second: int = 100_000,
        every: int = 150_000,
        verbose: int = 0,
        timeout_sec: int = 0,
        sleep_between_runs: float = 1.0,
    ) -> None:
        super().__init__(verbose)
        self.cfg = config
        self.eval_windows = list(eval_windows)
        self.episode_duration = int(episode_duration)
        self.n_episodes = int(n_episodes)
        self.deterministic = bool(deterministic)
        self._schedule = [int(first), int(second)]
        self._every = int(every)
        self._next = self._schedule.pop(0) if self._schedule else self._every
        self.timeout_sec = int(timeout_sec)
        self.sleep_between_runs = float(sleep_between_runs)

    def _on_step(self) -> bool:
        steps = int(self.model.num_timesteps)
        if steps >= self._next:
            try:
                self._run_eval()
            finally:
                if self._schedule:
                    self._next = self._schedule.pop(0)
                else:
                    self._next += self._every
        return True

    def _build_env(
        self, start_time: int, agent_control: bool, seed_offset: int
    ) -> SumoEnv:
        # Start at the requested evaluation window; SumoEnv will start SUMO at 0 and fast-forward to this time
        return SumoEnv(
            start_time=start_time,
            episode_duration=self.episode_duration,
            step_length=self.cfg.step_length,
            seed=self.cfg.seed + seed_offset,
            gui=False,
            sumo_config=self.cfg.sumo_config,
            fast_forward=self.cfg.fast_forward,
            reward_alpha=self.cfg.reward_alpha,
            reward_beta=self.cfg.reward_beta,
            reward_clip_min=self.cfg.reward_clip_min,
            reward_clip_max=self.cfg.reward_clip_max,
            reward_pressure_scale=self.cfg.reward_pressure_scale,
            reward_mode=getattr(self.cfg, "reward_mode", "delta"),
            queue_weight=getattr(self.cfg, "queue_weight", 0.1),
            ped_wait_weight=getattr(self.cfg, "ped_wait_weight", 1.0),
            ped_queue_weight=getattr(self.cfg, "ped_queue_weight", 1.0),
            agent_control=agent_control,
        )

    def _rollout_episode(self, env: SumoEnv, use_agent: bool) -> tuple[float, int]:
        try:
            obs, _ = env.reset()
            done = False
            ep_rew = 0.0
            ep_len = 0
            while not done:
                if use_agent:
                    action, _ = self.model.predict(
                        obs, deterministic=self.deterministic
                    )
                else:
                    action = 0  # ignored by env when agent_control=False
                obs, rew, terminated, truncated, _info = env.step(action)
                ep_rew += float(rew)
                ep_len += 1
                done = bool(terminated or truncated)
            return ep_rew, ep_len
        finally:
            try:
                env.close()
            except Exception:
                pass

    # --- Separate process eval helpers ---
    @staticmethod
    def _proc_eval_entry(
        model_path: Optional[str],
        cfg_dict: dict,
        start_time: int,
        episode_duration: int,
        deterministic: bool,
        agent_control: bool,
        seed: int,
        label: str,
        out_q: mp.Queue,
    ) -> None:
        """Child process: builds its own env with a unique TraCI label and runs one episode.

        If model_path is None or agent_control=False, runs baseline (no policy calls).
        """
        try:
            # Lazy imports inside child
            from stable_baselines3 import PPO  # type: ignore

            from SumoEnv import SumoEnv  # type: ignore

            env = SumoEnv(
                start_time=start_time,
                episode_duration=episode_duration,
                step_length=cfg_dict.get("step_length", 1.0),
                seed=seed,
                gui=False,
                sumo_config=cfg_dict.get("sumo_config", "validatednet.sumocfg"),
                fast_forward=cfg_dict.get("fast_forward", True),
                reward_alpha=cfg_dict.get("reward_alpha", 1.0),
                reward_beta=cfg_dict.get("reward_beta", 0.4),
                reward_clip_min=cfg_dict.get("reward_clip_min", -10.0),
                reward_clip_max=cfg_dict.get("reward_clip_max", 0.0),
                reward_pressure_scale=cfg_dict.get("reward_pressure_scale", 10.0),
                reward_mode=cfg_dict.get("reward_mode", "delta"),
                queue_weight=cfg_dict.get("queue_weight", 0.4),
                ped_wait_weight=cfg_dict.get("ped_wait_weight", 1.0),
                ped_queue_weight=cfg_dict.get("ped_queue_weight", 1.0),
                agent_control=agent_control,
                traci_label=str(label),
            )

            # Build policy (optional)
            policy = None
            if agent_control and model_path is not None and os.path.exists(model_path):
                policy = PPO.load(model_path, device="cpu")

            obs, _ = env.reset()
            done = False
            ep_rew = 0.0
            ep_len = 0
            while not done:
                if agent_control and policy is not None:
                    action, _ = policy.predict(obs, deterministic=deterministic)
                else:
                    action = 0
                obs, rew, terminated, truncated, _info = env.step(action)
                ep_rew += float(rew)
                ep_len += 1
                done = bool(terminated or truncated)
            out_q.put((float(ep_rew), int(ep_len)))
        except Exception:
            out_q.put((float("nan"), -1))
        finally:
            try:
                env.close()  # type: ignore[name-defined]
            except Exception:
                pass

    def _run_single_eval_with_timeout(
        self,
        start_t: int,
        agent_control: bool,
        seed_offset: int,
        model_tmp_path: Optional[str],
    ) -> Tuple[float, int]:
        cfgd = dict(
            step_length=self.cfg.step_length,
            sumo_config=self.cfg.sumo_config,
            fast_forward=self.cfg.fast_forward,
            reward_alpha=self.cfg.reward_alpha,
            reward_beta=self.cfg.reward_beta,
            reward_clip_min=self.cfg.reward_clip_min,
            reward_clip_max=self.cfg.reward_clip_max,
            reward_pressure_scale=self.cfg.reward_pressure_scale,
            reward_mode=getattr(self.cfg, "reward_mode", "delta"),
            queue_weight=getattr(self.cfg, "queue_weight", 0.4),
            ped_wait_weight=getattr(self.cfg, "ped_wait_weight", 1.0),
            ped_queue_weight=getattr(self.cfg, "ped_queue_weight", 1.0),
        )
        label = f"eval_{uuid.uuid4().hex[:8]}"
        q: mp.Queue = mp.Queue()
        p = mp.Process(
            target=BaselineEvalCallback._proc_eval_entry,
            args=(
                model_tmp_path if agent_control else None,
                cfgd,
                int(start_t),
                int(self.episode_duration),
                bool(self.deterministic),
                bool(agent_control),
                int(self.cfg.seed + seed_offset),
                label,
                q,
            ),
        )
        p.start()
        # Adaptive timeout: if not provided (>0), derive from episode duration and step length
        adaptive = max(60, int(self.episode_duration * float(self.cfg.step_length) * 5))
        run_timeout = int(self.timeout_sec) if int(self.timeout_sec) > 0 else adaptive
        p.join(run_timeout)
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.kill()
            except Exception:
                pass
            return float("nan"), -1
        try:
            res = q.get_nowait()
            return res  # type: ignore[return-value]
        except Exception:
            return float("nan"), -1

    def _run_eval(self) -> None:
        agent_rewards = []
        agent_lengths = []
        base_rewards = []
        base_lengths = []
        seed_base = 10_000
        # Save current model snapshot to temp path for child processes
        model_tmp_path: Optional[str] = None
        try:
            run_dir = getattr(self.model, "tensorboard_log", None)
            if run_dir:
                model_tmp_path = os.path.join(str(run_dir), "_eval_tmp_model.zip")
                self.model.save(model_tmp_path)
        except Exception:
            model_tmp_path = None
        for win_idx, start_t in enumerate(self.eval_windows):
            for ep in range(self.n_episodes):
                # Agent-controlled (separate process with timeout)
                r_a, l_a = self._run_single_eval_with_timeout(
                    start_t,
                    agent_control=True,
                    seed_offset=seed_base + ep + win_idx * 100,
                    model_tmp_path=model_tmp_path,
                )
                agent_rewards.append(r_a)
                agent_lengths.append(l_a)
                # Baseline (SUMO default) separate process
                r_b, l_b = self._run_single_eval_with_timeout(
                    start_t,
                    agent_control=False,
                    seed_offset=seed_base + 50 + ep + win_idx * 100,
                    model_tmp_path=None,
                )
                base_rewards.append(r_b)
                base_lengths.append(l_b)
                # Ensure serialization and give SUMO some time to release resources
                time.sleep(self.sleep_between_runs)

        # Aggregate and log
        def _safe_mean(xs: list[float]) -> float:
            clean = [
                float(x)
                for x in xs
                if isinstance(x, (int, float))
                and x == x
                and x not in (float("inf"), float("-inf"))
            ]
            if not clean:
                return -1.0
            return float(sum(clean) / len(clean))

        self.logger.record("eval_agent/mean_reward", _safe_mean(agent_rewards))
        self.logger.record(
            "eval_agent/mean_ep_len", _safe_mean([float(x) for x in agent_lengths])
        )
        self.logger.record("eval_baseline/mean_reward", _safe_mean(base_rewards))
        self.logger.record(
            "eval_baseline/mean_ep_len", _safe_mean([float(x) for x in base_lengths])
        )
        # Ensure metrics are written at current step
        self.logger.dump(step=self.model.num_timesteps)


class EntropyCoefScheduler(BaseCallback):
    """Linearly decay ent_coef between two training fractions.

    Example: start=0.01, end=0.001, start_frac=0.3, end_frac=0.5
    """

    def __init__(
        self,
        total_timesteps: int,
        start: float = 0.01,
        end: float = 0.001,
        start_frac: float = 0.3,
        end_frac: float = 0.5,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.total = max(1, int(total_timesteps))
        self.start = float(start)
        self.end = float(end)
        self.f0 = max(0.0, float(start_frac))
        self.f1 = max(self.f0, float(end_frac))

    def _on_step(self) -> bool:
        frac = min(1.0, self.model.num_timesteps / self.total)
        if frac <= self.f0:
            value = self.start
        elif frac >= self.f1:
            value = self.end
        else:
            # Linear interpolation
            t = (frac - self.f0) / (self.f1 - self.f0)
            value = self.start + t * (self.end - self.start)
        try:
            # Preferred: update attribute used by loss computation if exposed
            if hasattr(self.model, "ent_coef"):
                setattr(self.model, "ent_coef", float(value))
            elif hasattr(self.model, "policy") and hasattr(
                self.model.policy, "ent_coef"
            ):
                setattr(self.model.policy, "ent_coef", float(value))
        except Exception:
            pass
        return True


class EntropyCoefTwoPhaseScheduler(BaseCallback):
    """Two-phase linear decay of ent_coef with final hold.

    Phase 1: start1 -> end1 from [f0, f1]
    Phase 2: start2 -> end2 from [f2, f3]
    After f3: hold at end2.
    """

    def __init__(
        self,
        total_timesteps: int,
        start1: float = 0.01,
        end1: float = 0.003,
        f0: float = 0.0,
        f1: float = 0.5,
        start2: float = 0.003,
        end2: float = 0.001,
        f2: float = 0.5,
        f3: float = 0.9,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.total = max(1, int(total_timesteps))
        self.s1 = float(start1)
        self.e1 = float(end1)
        self.f0 = max(0.0, float(f0))
        self.f1 = max(self.f0, float(f1))
        self.s2 = float(start2)
        self.e2 = float(end2)
        self.f2 = max(self.f1, float(f2))
        self.f3 = max(self.f2, float(f3))

    def _on_step(self) -> bool:
        frac = min(1.0, self.model.num_timesteps / self.total)
        if frac <= self.f0:
            value = self.s1
        elif frac <= self.f1:
            t = (frac - self.f0) / max(1e-9, (self.f1 - self.f0))
            value = self.s1 + t * (self.e1 - self.s1)
        elif frac <= self.f2:
            # plateau between phases if gap exists
            value = self.e1
        elif frac <= self.f3:
            t = (frac - self.f2) / max(1e-9, (self.f3 - self.f2))
            value = self.s2 + t * (self.e2 - self.s2)
        else:
            value = self.e2
        try:
            if hasattr(self.model, "ent_coef"):
                setattr(self.model, "ent_coef", float(value))
            elif hasattr(self.model, "policy") and hasattr(
                self.model.policy, "ent_coef"
            ):
                setattr(self.model.policy, "ent_coef", float(value))
        except Exception:
            pass
        return True
