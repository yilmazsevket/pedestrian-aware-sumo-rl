"""VehicleManager: aggregate per-group vehicle queues and waiting times.

Collects both accumulated (ACC) and snapshot waits on every refresh and exposes:
    - group_queue_counts(): queued vehicles per group
    - wait_stats(): (sum, max, mean) of ACC waits (global)
    - group_mean_waits[_acc|_snapshot](): mean waits per group

Groups map inbound lanes to six logical groups (assumed suffix: _1=right, _2=through, _3=left):
    - NS_* uses inbound lanes -E0_x / -E2_x
    - EW_* uses inbound lanes -E1_x / -E3_x
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


class VehicleManager:
    def __init__(
        self,
        traci_mod,
        inbound_lanes: Sequence[str],  # deprecated/unused (kept for API compatibility)
        left_lanes: Iterable[str],  # reserved for potential future differentiation
        right_targets: Dict[str, str],  # unused; API compatibility
        speed_threshold: float = 0.1,
    ) -> None:
        self.traci = traci_mod
        self.speed_threshold = float(speed_threshold)
        # Collect both ACC and Snapshot views in parallel (no toggle)

        self.group_lanes = {
            "NS_right": ["-E0_1", "-E2_1"],
            "NS_through": ["-E0_2", "-E2_2"],
            "NS_left": ["-E0_3", "-E2_3"],
            "EW_right": ["-E1_1", "-E3_1"],
            "EW_through": ["-E1_2", "-E3_2"],
            "EW_left": ["-E1_3", "-E3_3"],
        }
        self.group_order = [
            "NS_through",
            "NS_right",
            "NS_left",
            "EW_through",
            "EW_right",
            "EW_left",
        ]
        self.group_lane_counts = {g: len(ls) for g, ls in self.group_lanes.items()}
        self._group_queue = {g: 0 for g in self.group_lanes}
        # Separate storage: ACC and SNAPSHOT
        self._veh_wait_times_acc: List[float] = []
        self._veh_wait_times_snapshot: List[float] = []
        self._group_wait_times_acc = {g: [] for g in self.group_lanes}
        self._group_wait_times_snapshot = {g: [] for g in self.group_lanes}
        # For optional delta calculations: last accumulated value per vehicle
        self._last_acc = {}

    def refresh(self) -> None:
        if self.traci is None:
            return
        self._group_queue = {g: 0 for g in self.group_lanes}
        self._veh_wait_times_acc = []
        self._veh_wait_times_snapshot = []
        self._group_wait_times_acc = {g: [] for g in self.group_lanes}
        self._group_wait_times_snapshot = {g: [] for g in self.group_lanes}
        seen_vids: List[str] = []
        for group, lanes in self.group_lanes.items():
            for lane_id in lanes:
                try:
                    veh_ids = self.traci.lane.getLastStepVehicleIDs(lane_id)
                except Exception:
                    veh_ids = []
                # Queue: halting number per lane
                try:
                    halt_num = int(self.traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception:
                    halt_num = 0
                self._group_queue[group] += halt_num
                # Collect waiting times for each vehicle
                for vid in veh_ids:
                    seen_vids.append(vid)
                    # ACC waiting time
                    try:
                        acc_wt = float(
                            self.traci.vehicle.getAccumulatedWaitingTime(vid)
                        )
                    except Exception:
                        acc_wt = 0.0
                    if acc_wt > 0:
                        self._veh_wait_times_acc.append(acc_wt)
                        self._group_wait_times_acc[group].append(acc_wt)
                    # Snapshot waiting time (currently waiting)
                    try:
                        snap_wt = float(self.traci.vehicle.getWaitingTime(vid))
                    except Exception:
                        snap_wt = 0.0
                    if snap_wt > 0:
                        self._veh_wait_times_snapshot.append(snap_wt)
                        self._group_wait_times_snapshot[group].append(snap_wt)
                    # Update last_acc
                    try:
                        self._last_acc[vid] = acc_wt if acc_wt >= 0 else 0.0
                    except Exception:
                        self._last_acc[vid] = 0.0
        # Remove departed vehicles from the last_acc map
        if self._last_acc:
            alive = set(seen_vids)
            self._last_acc = {k: v for k, v in self._last_acc.items() if k in alive}

    # Public API
    def group_queue_counts(self) -> Dict[str, int]:
        return dict(self._group_queue)

    def wait_stats(self) -> Tuple[float, float, float]:
        # Legacy: ACC for observations
        return self.wait_stats_acc()

    def wait_stats_acc(self) -> Tuple[float, float, float]:
        arr = self._veh_wait_times_acc
        if not arr:
            return 0.0, 0.0, 0.0
        s = sum(arr)
        mx = max(arr)
        mean = s / len(arr)
        return s, mx, mean

    def wait_stats_snapshot(self) -> Tuple[float, float, float]:
        arr = self._veh_wait_times_snapshot
        if not arr:
            return 0.0, 0.0, 0.0
        s = sum(arr)
        mx = max(arr)
        mean = s / len(arr)
        return s, mx, mean

    def group_mean_waits(self) -> Dict[str, float]:
        # ACC-based for observations
        return {
            g: (sum(arr) / len(arr) if arr else 0.0)
            for g, arr in self._group_wait_times_acc.items()
        }

    def group_mean_waits_acc(self) -> Dict[str, float]:
        return self.group_mean_waits()

    def group_mean_waits_snapshot(self) -> Dict[str, float]:
        return {
            g: (sum(arr) / len(arr) if arr else 0.0)
            for g, arr in self._group_wait_times_snapshot.items()
        }
