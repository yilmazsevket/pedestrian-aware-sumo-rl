from __future__ import annotations

"""CSV logger for rewards and component breakdowns.

Writes rows with step, sim_time, action, reward, selected numeric components
as separate columns, and a JSON snapshot of all components per row.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, cast


class RewardLogger:
    """CSV logger for rewards and their components.

    Usage:
        rl = RewardLogger()  # path auto-generated
        rl.log(step=1, sim_time=0.0, action=0, reward=-1.23, components={"veh_wait_mean": 10})
        rl.close()
    """

    def __init__(self, path: Optional[str] = None, flush_always: bool = True) -> None:
        self.path = path or os.path.abspath(
            f"reward_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.flush_always = bool(flush_always)
        self._f = None
        self._writer = None
        self._header_written = False
        self._extra_keys: list[str] = []

    def _ensure_open(self) -> None:
        if self._f is not None:
            return
        dirpath = os.path.dirname(self.path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.writer(self._f)

    def _write_header(self, components: Dict[str, object]) -> None:
        if self._header_written or self._writer is None:
            return
        # stable, sorted component keys
        self._extra_keys = sorted(
            [k for k, v in components.items() if isinstance(v, (int, float))]
        )
        header = (
            ["step", "sim_time", "action", "reward"]
            + self._extra_keys
            + ["components_json"]
        )
        self._writer.writerow(header)
        self._header_written = True
        if self.flush_always:
            self._f.flush()  # type: ignore[union-attr]

    def log(
        self,
        step: int,
        sim_time: float,
        action: int,
        reward: float,
        components: Optional[Dict[str, object]] = None,
    ) -> None:
        self._ensure_open()
        if self._writer is None:
            return
        components = components or {}
        if not self._header_written:
            self._write_header(components)
        row = [step, sim_time, action, float(reward)]
        # write numeric columns in stable order
        for k in self._extra_keys:
            v = components.get(k, 0.0)
            try:
                row.append(float(cast(Any, v)))
            except Exception:
                row.append(0.0)
        # always append JSON snapshot of all components (captures schema changes)
        try:
            comp_json = json.dumps(
                components, ensure_ascii=False, separators=(",", ":")
            )
        except Exception:
            comp_json = "{}"
        row.append(comp_json)
        self._writer.writerow(row)
        if self.flush_always:
            self._f.flush()  # type: ignore[union-attr]

    def close(self) -> None:
        try:
            if self._f is not None:
                self._f.flush()
                self._f.close()
        finally:
            self._f = None
            self._writer = None
            self._header_written = False
