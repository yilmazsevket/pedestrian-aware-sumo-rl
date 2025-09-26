"""Observation Builder – compact feature selection matching the specification.

Important Changes from previous version:
 1. Removed vehicle volume (queue length is the relevant signal).
 2. Added mean wait per vehicle group; no per-group max wait (only a global max).
 3. Added pedestrian max wait per approach.
 4. Renamed ``ped_wait:*`` to ``ped_queue:*`` (counts of waiting pedestrians).
 5. Removed pedestrian long-wait fraction (redundant).

Base feature layout (without directional details):
 0..5   grp_queue_*                (6)
 6..11 grp_mean_wait_*            (6) – normalised by ``CAP_WAIT_TIME``
 12..15 ped_queue:*               (4)
 16..19 ped_max_wait:*            (4) – normalised by ``CAP_WAIT_TIME``
 20..23 green_onehot_*            (4)
 24     transition_remaining
 25     min_green_reached
 26     elapsed_green_time
 27     max_vehicle_wait_time (global)
 28     pressure_NS
 29     pressure_EW
 30..   directional pedestrian blocks (per allowed pair: count, max_wait, vulnerable_frac)

Directional pairs (8): each approach connects to two realistic targets (no self/UNKNOWN).
This yields an additional 8 * 3 = 24 features → total dimension 54.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # traci is only available when SUMO is running
    import traci  # type: ignore
except Exception:  # pragma: no cover
    traci = None  # type: ignore


class ObservationBuilder:
    CAP_VEH_PER_LANE = 40.0  # normalise queue length (queue / (lanes * cap))
    CAP_PED_PER_EDGE = 40.0
    CAP_WAIT_TIME = 120.0  # observations use ACC waits; raise to ~600 to avoid saturation if required

    # Per-lane features removed
    INBOUND_LANES: Sequence[str] = ()

    # Pedestrian waiting edges (pedestrian inbound lanes/edges)
    PED_WAIT_EDGES: Sequence[str] = ("-E0", "-E2", "-E1", "-E3")
    PED_LANES: Sequence[str] = ("-E0_0", "-E2_0", "-E1_0", "-E3_0")  # inbound
    # Island/crosswalk internal lanes at the junction (heuristic mapping)
    PED_ISLAND_LANES: Dict[str, str] = {
        "-E0": ":J0_w0_0",
        "-E2": ":J0_w2_0",
        "-E1": ":J0_w1_0",
        "-E3": ":J0_w3_0",
    }

    GROUPS = ["NS_through", "NS_right", "NS_left", "EW_through", "EW_right", "EW_left"]

    NS_IN_LANES = ("-E0_1", "-E0_2", "-E0_3", "-E2_1", "-E2_2", "-E2_3")
    EW_IN_LANES = ("-E1_1", "-E1_2", "-E1_3", "-E3_1", "-E3_2", "-E3_3")

    LEFT_LANES = {"-E0_3", "-E2_3", "-E1_3", "-E3_3"}

    RIGHT_TARGETS: Dict[str, str] = {
        "-E0": "E1",  # north -> west
        "-E2": "E3",  # south -> east
        "-E1": "E2",  # west -> south
        "-E3": "E0",  # east -> north
    }

    def __init__(
        self,
        switcher: Any,
        normalize: bool = True,
        vehicle_manager=None,
        pedestrian_manager=None,
    ):
        """ObservationBuilder.

        Parameters:
            switcher: TrafficLightPhaseSwitcher instance (provides timing/state).
            normalize: Whether feature scaling is enabled.
            vehicle_manager: Optional pre-initialised VehicleManager instance.
            pedestrian_manager: Optional pre-initialised PedestrianManager instance.
        """
        self.switcher = switcher
        self.normalize = normalize
        self._feature_names: List[str] = []
        self._build_feature_names()
        self.vehicle_manager = vehicle_manager
        self.pedestrian_manager = pedestrian_manager

    # Convenience helper for late injection of managers
    def set_managers(
        self, vehicle_manager, pedestrian_manager
    ) -> None:  # pragma: no cover
        self.vehicle_manager = vehicle_manager
        self.pedestrian_manager = pedestrian_manager

    def _build_feature_names(self) -> None:
        if self._feature_names:
            return
        names: List[str] = []
        # Queue length per group
        for g in [
            "NS_through",
            "NS_right",
            "NS_left",
            "EW_through",
            "EW_right",
            "EW_left",
        ]:
            names.append(f"grp_queue_{g}")
        # Mean wait per group
        for g in [
            "NS_through",
            "NS_right",
            "NS_left",
            "EW_through",
            "EW_right",
            "EW_left",
        ]:
            names.append(f"grp_mean_wait_{g}")
        # Pedestrian queues & max waits
        for e in self.PED_LANES:
            names.append(f"ped_queue:{e}")
        for e in self.PED_LANES:
            names.append(f"ped_max_wait:{e}")
        # Phase / timing / pressure features
        names += [
            "green_onehot_0",
            "green_onehot_1",
            "green_onehot_2",
            "green_onehot_3",
            "transition_remaining",
            "min_green_reached",
            "elapsed_green_time",
            "max_vehicle_wait_time",
            "pressure_NS",
            "pressure_EW",
        ]
        self._feature_names = names  # directional features appended dynamically later

    def feature_names(self) -> List[str]:  # pragma: no cover
        return list(self._feature_names)

    def reset(self) -> None:
        pass

    # Direction ordering for pedestrian directional features (approach -> outbound)
    # We derive outbound edges from PedestrianManager.outbound_edges; pairs created at runtime.
    def _build_direction_pairs(self, ped_manager) -> List[tuple]:
        # Physical crossings: each approach has two practical outbound targets
        # Mapping: inbound edge -> permitted outbound edges (exclude UNKNOWN/self)
        allowed: Dict[str, List[str]] = {
            "-E0": ["E1", "E2"],  # north -> west (right) or south (straight)
            "-E2": ["E3", "E0"],  # south -> east or north
            "-E1": ["E2", "E3"],  # west -> south or east
            "-E3": ["E0", "E1"],  # east -> north or west
        }
        pairs: List[tuple] = []
        for approach, targets in allowed.items():
            for tgt in targets:
                pairs.append((approach, tgt))
        return pairs

    def _ensure_directional_feature_names(
        self, direction_pairs: List[Tuple[str, str]]
    ) -> None:
        # Expect exactly three features per pair (count, max_wait, vulnerable_frac)
        existing = [n for n in self._feature_names if n.startswith("ped_dir_")]
        expected_count = len(direction_pairs) * 3
        if existing and len(existing) == expected_count:
            return  # already ok
        # Remove old directional entries
        self._feature_names = [
            n for n in self._feature_names if not n.startswith("ped_dir_")
        ]
        extra: List[str] = []
        for ap, tgt in direction_pairs:
            extra.append(f"ped_dir_count:{ap}->{tgt}")
        for ap, tgt in direction_pairs:
            extra.append(f"ped_dir_max_wait:{ap}->{tgt}")
        for ap, tgt in direction_pairs:
            extra.append(f"ped_dir_vulnerable_frac:{ap}->{tgt}")
        self._feature_names.extend(extra)

    def compute(self, traci_mod, tl_id: Optional[str]) -> np.ndarray:  # noqa: C901
        base_size = 30  # without directional features
        if traci_mod is None or tl_id is None or not traci_mod.isLoaded():
            return np.zeros((base_size,), dtype=np.float32)
        # Create managers if they have not been provided
        if self.vehicle_manager is None:
            from vehicle_manager import VehicleManager  # late import

            self.vehicle_manager = VehicleManager(
                traci_mod,
                inbound_lanes=(),
                left_lanes=self.LEFT_LANES,
                right_targets=self.RIGHT_TARGETS,
            )
        if self.pedestrian_manager is None:
            from pedestrian_manager import PedestrianManager  # late import

            self.pedestrian_manager = PedestrianManager(
                traci_mod,
                approaches=list(self.PED_WAIT_EDGES),
                island_lanes=self.PED_ISLAND_LANES,
            )
        # Refresh
        self.vehicle_manager.refresh()
        self.pedestrian_manager.refresh()

        group_queue = self.vehicle_manager.group_queue_counts()
        group_mean_waits = self.vehicle_manager.group_mean_waits()
        _veh_sum_wait, veh_max_wait, _veh_mean_wait = self.vehicle_manager.wait_stats()
        ped_queue_counts = self.pedestrian_manager.waiting_counts()
        ped_max_waits = getattr(
            self.pedestrian_manager, "approach_max_waits", lambda: {}
        )()

        # Directional
        dir_pairs = self._build_direction_pairs(self.pedestrian_manager)
        dir_counts = self.pedestrian_manager.waiting_counts_detailed()
        dir_max_waits = self.pedestrian_manager.directional_max_wait_times()
        dir_vuln_frac = self.pedestrian_manager.directional_vulnerable_fraction()
        self._ensure_directional_feature_names(dir_pairs)

        # Allocate vector
        total_size = len(self._feature_names)
        vec = np.zeros((total_size,), dtype=np.float32)

        # Indices mapping
        grp_order = [
            "NS_through",
            "NS_right",
            "NS_left",
            "EW_through",
            "EW_right",
            "EW_left",
        ]
        # 0..5 queues
        for i, g in enumerate(grp_order):
            qc = group_queue.get(g, 0)
            denom = (
                max(self.vehicle_manager.group_lane_counts.get(g, 1), 1)
                * self.CAP_VEH_PER_LANE
            )
            vec[i] = min(qc / denom, 1.0) if self.normalize else float(qc)
        # 6..11 mean waits
        for i, g in enumerate(grp_order):
            mw = group_mean_waits.get(g, 0.0)
            vec[6 + i] = (
                min(mw / self.CAP_WAIT_TIME, 1.0) if self.normalize else float(mw)
            )
        # 12..15 ped queues
        ped_order = ["-E0", "-E2", "-E1", "-E3"]
        for i, ap in enumerate(ped_order):
            c = ped_queue_counts.get(ap, 0)
            vec[12 + i] = (
                min(c / self.CAP_PED_PER_EDGE, 1.0) if self.normalize else float(c)
            )
        # 16..19 ped max waits
        for i, ap in enumerate(ped_order):
            mw = ped_max_waits.get(ap, 0.0)
            vec[16 + i] = (
                min(mw / self.CAP_WAIT_TIME, 1.0) if self.normalize else float(mw)
            )

        # 20..23 green onehot
        try:
            cur_phase_idx = int(traci_mod.trafficlight.getPhase(tl_id))
        except Exception:
            cur_phase_idx = -1
        if hasattr(self.switcher, "idx") and hasattr(self.switcher, "GROUPS"):
            # Guarantee fixed order: NS_MAIN, EW_MAIN, NS_LEFT, EW_LEFT
            green_indices = [
                self.switcher.idx[g]["green"]
                for g in getattr(self.switcher, "GROUPS", [])
                if g in self.switcher.idx
            ]
            phase_for_onehot = cur_phase_idx
            # During a transition, pre-activate the target green phase in the one-hot vector
            if getattr(self.switcher, "in_transition", lambda: False)():
                tgt = getattr(self.switcher, "target_group", lambda: None)()
                if tgt and tgt in self.switcher.idx:
                    phase_for_onehot = self.switcher.idx[tgt]["green"]
            if phase_for_onehot in green_indices:
                pos = green_indices.index(phase_for_onehot)
                if 0 <= pos < 4:
                    vec[20 + pos] = 1.0

        # 24 transition_remaining
        # 24 transition_remaining (0 when no transition is active, otherwise remaining time / max transition)
        transition_remaining = 0.0
        try:
            if hasattr(self.switcher, "remaining_transition_time"):
                rem = float(self.switcher.remaining_transition_time())
            else:
                rem = 0.0
            max_trans = float(
                getattr(self.switcher, "YELLOW_TIME", 3)
                + getattr(self.switcher, "REDYELLOW_TIME", 1)
            )
            if rem > 0 and max_trans > 0:
                transition_remaining = rem / max_trans if self.normalize else rem
            else:
                transition_remaining = 0.0
        except Exception:
            transition_remaining = 0.0
        vec[24] = min(transition_remaining, 1.0)
        # 25 min_green flag
        vec[25] = (
            1.0
            if getattr(self.switcher, "min_green_satisfied", lambda: False)()
            else 0.0
        )
        # 26 elapsed green
        elapsed_green = 0.0
        if hasattr(self.switcher, "elapsed_green_time"):
            try:
                elapsed_green = float(self.switcher.elapsed_green_time())  # type: ignore[call-arg]
            except Exception:
                pass
        vec[26] = min(elapsed_green / 120.0, 1.0) if self.normalize else elapsed_green
        # 27 max vehicle wait (global)
        vec[27] = (
            min(veh_max_wait / self.CAP_WAIT_TIME, 1.0)
            if self.normalize
            else float(veh_max_wait)
        )
        # 28/29 pressure (Queue Summen)
        pressure_ns = sum(
            group_queue.get(g, 0) for g in ["NS_through", "NS_right", "NS_left"]
        )
        pressure_ew = sum(
            group_queue.get(g, 0) for g in ["EW_through", "EW_right", "EW_left"]
        )
        if self.normalize:
            vec[28] = min(pressure_ns / 120.0, 1.0)
            vec[29] = min(pressure_ew / 120.0, 1.0)
        else:
            vec[28] = float(pressure_ns)
            vec[29] = float(pressure_ew)

        # Directional blocks start at index 30
        cursor = 30
        for ap, tgt in dir_pairs:  # counts
            cnt = dir_counts.get(ap, {}).get(tgt, 0)
            vec[cursor] = (
                min(cnt / self.CAP_PED_PER_EDGE, 1.0) if self.normalize else float(cnt)
            )
            cursor += 1
        for ap, tgt in dir_pairs:  # max waits
            mw = dir_max_waits.get(ap, {}).get(tgt, 0.0)
            vec[cursor] = (
                min(mw / self.CAP_WAIT_TIME, 1.0) if self.normalize else float(mw)
            )
            cursor += 1
        for ap, tgt in dir_pairs:  # vulnerable fraction
            vf = dir_vuln_frac.get(ap, {}).get(tgt, 0.0)
            vf_clamped = 0.0 if vf < 0 else (1.0 if vf > 1 else vf)
            vec[cursor] = float(vf_clamped)
            cursor += 1

        return vec
