from __future__ import annotations

"""Reward calculation utilities for SumoEnv.

Provides a simple, raw reward based on mean vehicle/pedestrian waiting times and
network pressure. Clipping/normalization is handled externally.
"""

from typing import Any, Dict, Tuple


class RewardCalculator:
    """Encapsulates reward computation logic for SumoEnv.

    Default formula (Version 0):
        r_raw = -(veh_wait_mean + alpha * ped_wait_mean + beta * pressure)
        pressure = (sum queues NS + sum queues EW) / pressure_scale

    Note: This class now returns the raw reward (without clipping).
    Final bounding (e.g., mapping into [-1, 1]) is handled externally via RewardNormalizer.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.4,
        clip_min: float = -10.0,
        clip_max: float = 0.0,
        pressure_scale: float = 10.0,
    ) -> None:
        # Internal attributes (validated via property setters)
        self._alpha = 1.0
        self._beta = 0.4
        self._clip_min = -10.0
        self._clip_max = 0.0
        self._pressure_scale = 10.0
        # Initialize through setters to reuse validation logic
        self.alpha = alpha
        self.beta = beta
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.pressure_scale = pressure_scale

    # --- Properties ---
    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = max(0.0, float(value))  # disallow negative weights

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = max(0.0, float(value))

    @property
    def clip_min(self) -> float:
        return self._clip_min

    @clip_min.setter
    def clip_min(self, value: float) -> None:
        v = float(value)
        # Ensure clip_min <= clip_max
        self._clip_min = v if v <= self._clip_max else self._clip_max

    @property
    def clip_max(self) -> float:
        return self._clip_max

    @clip_max.setter
    def clip_max(self, value: float) -> None:
        v = float(value)
        # Ensure clip_min <= clip_max
        self._clip_max = v
        if self._clip_min > self._clip_max:
            self._clip_min = self._clip_max

    @property
    def pressure_scale(self) -> float:
        return self._pressure_scale

    @pressure_scale.setter
    def pressure_scale(self, value: float) -> None:
        v = float(value)
        self._pressure_scale = 1.0 if v <= 0 else v

    def compute(
        self, vehicle_manager: Any, pedestrian_manager: Any
    ) -> Tuple[float, Dict[str, Any]]:
        if vehicle_manager is None or pedestrian_manager is None:
            return 0.0, {"veh_wait_mean": 0.0, "ped_wait_mean": 0.0, "pressure": 0.0}
        try:
            _veh_sum, _veh_max, veh_mean = vehicle_manager.wait_stats()
            _ped_sum, _ped_max, ped_mean = pedestrian_manager.wait_stats()
            queues = vehicle_manager.group_queue_counts()
            pressure_ns = sum(
                queues.get(g, 0) for g in ["NS_through", "NS_right", "NS_left"]
            )
            pressure_ew = sum(
                queues.get(g, 0) for g in ["EW_through", "EW_right", "EW_left"]
            )
            pressure = (pressure_ns + pressure_ew) / self.pressure_scale
            raw = -(veh_mean + self.alpha * ped_mean + self.beta * pressure)
            # Return raw reward; keep 'reward_clipped' for compatibility with previous pipelines
            return raw, {
                "veh_wait_mean": veh_mean,
                "ped_wait_mean": ped_mean,
                "pressure": pressure,
                "reward_raw": raw,
                "reward_clipped": raw,
            }
        except Exception:
            return 0.0, {"veh_wait_mean": 0.0, "ped_wait_mean": 0.0, "pressure": 0.0}
