from __future__ import annotations

"""Reward normalization into a bounded range, kept simple and explicit.

Default behavior: r_norm = clip(r / scale, clip_min, clip_max)
- Use a fixed scale to keep rewards within [-1, 1] (or subset thereof).
- Keep it independent from reward calculation (single responsibility).
"""

from typing import Final


class RewardNormalizer:
    def __init__(
        self,
        scale: float = 1000.0,
        clip_min: float = -3.0,
        clip_max: float = 2.0,
    ) -> None:
        self._clip_min: float = -3.0
        self._clip_max: float = 2.0
        self._scale: float = 1000.0
        # initialize via setters for validation
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.scale = scale

    @property
    def clip_min(self) -> float:
        return self._clip_min

    @clip_min.setter
    def clip_min(self, v: float) -> None:
        v = float(v)
        # ensure clip_min < clip_max (strict to avoid zero-range)
        if v >= getattr(self, "_clip_max", 1.0):
            self._clip_min = getattr(self, "_clip_max", 1.0) - 1.0
        else:
            self._clip_min = v

    @property
    def clip_max(self) -> float:
        return self._clip_max

    @clip_max.setter
    def clip_max(self, v: float) -> None:
        v = float(v)
        self._clip_max = v
        if getattr(self, "_clip_min", -1.0) >= self._clip_max:
            self._clip_min = self._clip_max - 1.0

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, v: float) -> None:
        v = float(v)
        # guard against zero/negative
        self._scale = v if v > 0.0 else 1.0

    def normalize(self, r: float) -> float:
        """Scale and clip reward into [clip_min, clip_max].

        Keeps sign; with scale=10, rewards in [-10, 10] map into [-1, 1].
        """
        # fast path locals
        s: Final[float] = self._scale
        lo: Final[float] = self._clip_min
        hi: Final[float] = self._clip_max
        x = float(r) / s
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x
