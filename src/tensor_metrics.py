from __future__ import annotations

"""SB3 callback to log additional TensorBoard metrics from SumoEnv infos.

Logs per-episode means and counts:
- train/ep_veh_queue_mean, train/ep_ped_queue_mean
- train/ep_phase_switches
- train/ep_veh_wait_sum_mean, train/ep_ped_wait_sum_mean
- train/ep_veh_queue_total_mean, train/ep_ped_queue_total_mean
- train/reward_clip_at_min_rate, train/reward_clip_at_max_rate


"""

from typing import Any, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback


class SB3MetricsCallback(BaseCallback):
    """Aggregate metrics per episode across vectorized envs and log to SB3 logger."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        # Per-env running state
        self._steps: Dict[int, int] = {}
        self._sum_veh_queue: Dict[int, float] = {}
        self._sum_ped_queue: Dict[int, float] = {}
        self._sum_veh_wait_sum: Dict[int, float] = {}
        self._sum_ped_wait_sum: Dict[int, float] = {}
        self._sum_veh_queue_total: Dict[int, float] = {}
        self._sum_ped_queue_total: Dict[int, float] = {}
        self._phase_switches: Dict[int, int] = {}
        self._last_phase: Dict[int, Optional[int]] = {}
        self._clip_min_hits: Dict[int, int] = {}
        self._clip_max_hits: Dict[int, int] = {}
        self._clip_min: Optional[float] = None
        self._clip_max: Optional[float] = None

    def _on_training_start(self) -> None:
        # Try to fetch clip bounds from the first underlying env
        try:
            if self.training_env is not None:
                envs = self.training_env.envs if hasattr(self.training_env, "envs") else []  # type: ignore[attr-defined]
                if envs:
                    rn = getattr(envs[0], "reward_norm", None)
                    if rn is not None:
                        self._clip_min = float(getattr(rn, "clip_min", -3.0))
                        self._clip_max = float(getattr(rn, "clip_max", 2.0))
        except Exception:
            pass

    def _reset_env_stats(self, idx: int) -> None:
        self._steps[idx] = 0
        self._sum_veh_queue[idx] = 0.0
        self._sum_ped_queue[idx] = 0.0
        self._sum_veh_wait_sum[idx] = 0.0
        self._sum_ped_wait_sum[idx] = 0.0
        self._sum_veh_queue_total[idx] = 0.0
        self._sum_ped_queue_total[idx] = 0.0
        self._phase_switches[idx] = 0
        self._last_phase[idx] = None
        self._clip_min_hits[idx] = 0
        self._clip_max_hits[idx] = 0

    def _ensure_idx(self, idx: int) -> None:
        if idx not in self._steps:
            self._reset_env_stats(idx)

    def _maybe_count_clip(self, idx: int, info: Dict[str, Any]) -> None:
        try:
            r_norm = float(info.get("reward_norm", 0.0))
            # tolerate tiny float errors
            if self._clip_min is not None and r_norm <= self._clip_min + 1e-9:
                self._clip_min_hits[idx] += 1
            if self._clip_max is not None and r_norm >= self._clip_max - 1e-9:
                self._clip_max_hits[idx] += 1
        except Exception:
            pass

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = []
        try:
            infos = list(self.locals.get("infos", []))  # type: ignore[assignment]
        except Exception:
            pass
        if not infos:
            return True

        for idx, info in enumerate(infos):
            self._ensure_idx(idx)
            self._steps[idx] += 1

            # Phase switches
            cur_phase = None
            if "phase_index" in info:
                try:
                    val = info.get("phase_index")
                    if val is not None:
                        cur_phase = int(val)
                except Exception:
                    cur_phase = None
            last = self._last_phase.get(idx)
            if cur_phase is not None:
                if last is not None and cur_phase != last:
                    self._phase_switches[idx] += 1
                self._last_phase[idx] = cur_phase

            # Queues (veh/ped totals in info from DeltaRewardCalculator and PedestrianManager)
            try:
                veh_q_total = float(info.get("veh_queue_total", 0.0))
                ped_q_total = float(
                    info.get("ped_queue_total", info.get("wait_count", 0.0))
                )
                self._sum_veh_queue[idx] += veh_q_total
                self._sum_ped_queue[idx] += ped_q_total
                # For explicit means also mirror into *_total sums
                self._sum_veh_queue_total[idx] += veh_q_total
                self._sum_ped_queue_total[idx] += ped_q_total
            except Exception:
                pass

            # Wait sums (snapshot from delta info)
            try:
                self._sum_veh_wait_sum[idx] += float(info.get("veh_wait_sum", 0.0))
                self._sum_ped_wait_sum[idx] += float(info.get("ped_wait_sum", 0.0))
            except Exception:
                pass

            # Reward clipping detection
            self._maybe_count_clip(idx, info)

        # Handle episode dones to log per-episode means
        dones = self.locals.get("dones")
        if dones is None:
            return True
        try:
            for idx, done in enumerate(dones):
                if bool(done):
                    steps = max(1, self._steps.get(idx, 1))
                    # Means per episode per env
                    veh_queue_mean = self._sum_veh_queue.get(idx, 0.0) / steps
                    ped_queue_mean = self._sum_ped_queue.get(idx, 0.0) / steps
                    veh_wait_sum_mean = self._sum_veh_wait_sum.get(idx, 0.0) / steps
                    ped_wait_sum_mean = self._sum_ped_wait_sum.get(idx, 0.0) / steps
                    veh_queue_total_mean = (
                        self._sum_veh_queue_total.get(idx, 0.0) / steps
                    )
                    ped_queue_total_mean = (
                        self._sum_ped_queue_total.get(idx, 0.0) / steps
                    )
                    clip_min_rate = float(self._clip_min_hits.get(idx, 0)) / steps
                    clip_max_rate = float(self._clip_max_hits.get(idx, 0)) / steps

                    # Record metrics (SB3 logger -> TensorBoard)
                    self.logger.record("train/ep_veh_queue_mean", veh_queue_mean)
                    self.logger.record("train/ep_ped_queue_mean", ped_queue_mean)
                    self.logger.record(
                        "train/ep_phase_switches",
                        float(self._phase_switches.get(idx, 0)),
                    )
                    self.logger.record("train/ep_veh_wait_sum_mean", veh_wait_sum_mean)
                    self.logger.record("train/ep_ped_wait_sum_mean", ped_wait_sum_mean)
                    self.logger.record(
                        "train/ep_veh_queue_total_mean", veh_queue_total_mean
                    )
                    self.logger.record(
                        "train/ep_ped_queue_total_mean", ped_queue_total_mean
                    )
                    self.logger.record("train/reward_clip_at_min_rate", clip_min_rate)
                    self.logger.record("train/reward_clip_at_max_rate", clip_max_rate)

                    # Reset this env episode stats
                    self._reset_env_stats(idx)
        except Exception:
            # Be permissive; do not stop training due to logging
            return True

        return True
