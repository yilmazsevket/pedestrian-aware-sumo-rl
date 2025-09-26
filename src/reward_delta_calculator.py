from __future__ import annotations

"""Delta-based reward calculator for SUMO RL.

This module implements a reward that penalizes per-step increases in total waiting
time and queue length. Vehicles use accumulated waiting times for stability; pedestrians
use a snapshot-based estimate. A small level penalty discourages large absolute waits.
"""

from typing import Any, Dict, Optional, Tuple


class DeltaRewardCalculator:
    """Reward based on per-step deltas of total wait time and total queue length.

    r_t = - ( (Wait_t - Wait_{t-1}) + w_q * (Queue_t - Queue_{t-1}) )

        Where:
            - Wait_t = veh_acc_wait_sum + ped_wait_weight * ped_wait_sum
                • veh_acc_wait_sum uses accumulated vehicle waiting times
                  (vehicle.getAccumulatedWaitingTime) summed over present vehicles.
                • ped_wait_sum uses a snapshot estimate (time since wait-start this step).
            - Queue_t = veh_queue_total + ped_queue_weight * ped_queue_total
                • veh_queue_total is the sum of queued vehicles across groups/lanes.
                • ped_queue_total is the count of waiting pedestrians.

        Notes:
            - This class is stateful per episode; call reset() on env reset.
            - Switching to ACC can change scales; consider increasing reward normalizer scale
                if rewards saturate (e.g., scale ~ 600–1500 depending on traffic).
    """

    def __init__(
        self,
        queue_weight: float = 0.4,
        ped_wait_weight: float = 1.0,
        ped_queue_weight: float = 1.0,
        level_penalty_lambda: float = 0.002,
    ) -> None:
        self.queue_weight = float(queue_weight)
        self.ped_wait_weight = float(ped_wait_weight)
        self.ped_queue_weight = float(ped_queue_weight)
        # Small level term: penalizes absolute waiting per step (snapshot-based)
        self.level_penalty_lambda = max(0.0, float(level_penalty_lambda))
        self._prev_wait_total: Optional[float] = None
        self._prev_queue_total: Optional[float] = None

    def reset(self) -> None:
        self._prev_wait_total = None
        self._prev_queue_total = None

    def compute(
        self, vehicle_manager: Any, pedestrian_manager: Any
    ) -> Tuple[float, Dict[str, Any]]:
        if vehicle_manager is None or pedestrian_manager is None:
            return 0.0, {
                "veh_wait_sum": 0.0,
                "ped_wait_sum": 0.0,
                "veh_queue_total": 0.0,
                "ped_queue_total": 0.0,
                "delta_wait": 0.0,
                "delta_queue": 0.0,
            }

        # Current metrics: use ACCUMULATED waits for vehicles (stable across steps)
        # VehicleManager.wait_stats() returns ACC-based stats by design.
        veh_sum_wait, _veh_max, _veh_mean = vehicle_manager.wait_stats()
        ped_sum_wait, _ped_max, _ped_mean = pedestrian_manager.wait_stats()
        queues = vehicle_manager.group_queue_counts()
        veh_queue_total = float(sum(queues.values()))
        try:
            ped_queue_total = float(getattr(pedestrian_manager, "wait_count")())
        except Exception:
            # Fallback: sum detailed counts if available
            try:
                ped_counts = getattr(pedestrian_manager, "waiting_counts")()
                ped_queue_total = float(sum(ped_counts.values()))
            except Exception:
                ped_queue_total = 0.0

        wait_total = float(veh_sum_wait) + self.ped_wait_weight * float(ped_sum_wait)
        queue_total = float(veh_queue_total) + self.ped_queue_weight * float(
            ped_queue_total
        )

        # Deltas (0 for the very first step after reset)
        if self._prev_wait_total is None:
            delta_wait = 0.0
        else:
            delta_wait = wait_total - self._prev_wait_total
        if self._prev_queue_total is None:
            delta_queue = 0.0
        else:
            delta_queue = queue_total - self._prev_queue_total

        # Update state
        self._prev_wait_total = wait_total
        self._prev_queue_total = queue_total

        level_penalty = self.level_penalty_lambda * float(wait_total)
        reward = -(
            float(delta_wait) + self.queue_weight * float(delta_queue) + level_penalty
        )

        info: Dict[str, Any] = {
            "veh_wait_sum": float(veh_sum_wait),
            "ped_wait_sum": float(ped_sum_wait),
            "veh_queue_total": float(veh_queue_total),
            "ped_queue_total": float(ped_queue_total),
            "delta_wait": float(delta_wait),
            "delta_queue": float(delta_queue),
            "level_penalty": float(level_penalty),
            "reward_raw": float(reward),
        }
        return reward, info
