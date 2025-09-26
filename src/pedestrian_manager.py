"""PedestrianManager wraps TraCI pedestrian queries.

Goals:
- Count waiting pedestrians per approach (inbound plus island lane)
- Track wait-time statistics (sum, max, mean, count)
- Capture class distribution (agile / regular / vulnerable)
"""

from typing import Dict, List, Optional, Sequence, Tuple


class PedestrianManager:
    """Compute pedestrian wait statistics without trusting ``person.getWaitingTime``.

    SUMO may leave the built-in waiting time unset for pedestrians depending on the
    network and signal configuration. This manager therefore flags pedestrians as
    waiting when they sit on an inbound edge or island walking area while moving
    slower than ``speed_threshold`` and integrates the elapsed wait time over each
    continuous standstill period. Directional aggregations use the first outbound
    edge (E0..E3) discovered along the pedestrian's route.
    """

    def __init__(
        self,
        traci_mod,
        approaches: Sequence[str],
        island_lanes: Dict[str, str],  # lane ids (with _0)
        pedestrian_types: Optional[Sequence[str]] = None,
        outbound_edges: Optional[Sequence[str]] = None,
        long_wait_threshold: float = 10.0,
        speed_threshold: float = 0.05,
    ) -> None:
        self.traci = traci_mod
        self.approaches = list(approaches)
        self.island_lanes = dict(island_lanes)  # approach -> lane id (e.g. :J0_w0_0)
        # Derive walking-area edge IDs (edge id equals lane id without the trailing _0)
        self.walking_area_edges = {
            ap: lane.split("_0")[0] if lane.endswith("_0") else lane
            for ap, lane in self.island_lanes.items()
        }
        self.pedestrian_types = (
            list(pedestrian_types)
            if pedestrian_types
            else [
                "agile_pedestrian",
                "regular_pedestrian",
                "vulnerable_pedestrian",
            ]
        )
        self.outbound_edges = (
            list(outbound_edges) if outbound_edges else ["E0", "E1", "E2", "E3"]
        )
        self.long_wait_threshold = float(long_wait_threshold)
        self.speed_threshold = float(speed_threshold)

        # Persistent state
        self._ped_wait_start: Dict[str, float] = {}  # pid -> sim_time start
        self._ped_last_edge: Dict[str, str] = {}
        self._ped_type: Dict[str, str] = {}

        # Aggregated containers per refresh
        self._waiting_counts: Dict[str, int] = {}
        self._wait_times: List[float] = []
        self._type_counts: Dict[str, int] = {}
        self._type_wait_times: Dict[str, List[float]] = {}
        self._direction_counts: Dict[str, Dict[str, int]] = {}
        self._direction_wait_times: Dict[str, Dict[str, List[float]]] = {}
        self._direction_vulnerable_counts: Dict[str, Dict[str, int]] = {}
        self._direction_long_wait_counts: Dict[str, Dict[str, int]] = {}
        self._last_sim_time: Optional[float] = None

    # --------------- Core Update ---------------
    def refresh(self) -> None:
        if self.traci is None:
            return
        try:
            sim_time = float(self.traci.simulation.getTime())
        except Exception:
            sim_time = 0.0
        # Reset persistent state when the simulation time jumps backwards (episode reset)
        if self._last_sim_time is not None and sim_time < self._last_sim_time:
            self._ped_wait_start.clear()
            self._ped_last_edge.clear()
            self._ped_type.clear()
        self._last_sim_time = sim_time

        # Reinitialise aggregate containers
        self._waiting_counts = {a: 0 for a in self.approaches}
        self._wait_times = []
        self._type_counts = {t: 0 for t in self.pedestrian_types}
        self._type_wait_times = {t: [] for t in self.pedestrian_types}
        self._direction_counts = {
            a: {edge: 0 for edge in self.outbound_edges} | {"UNKNOWN": 0}
            for a in self.approaches
        }
        self._direction_wait_times = {
            a: {edge: [] for edge in self.outbound_edges} | {"UNKNOWN": []}
            for a in self.approaches
        }
        self._direction_vulnerable_counts = {
            a: {edge: 0 for edge in self.outbound_edges} | {"UNKNOWN": 0}
            for a in self.approaches
        }
        self._direction_long_wait_counts = {
            a: {edge: 0 for edge in self.outbound_edges} | {"UNKNOWN": 0}
            for a in self.approaches
        }

        # Fetch all pedestrian IDs once
        try:
            pids = list(self.traci.person.getIDList())  # type: ignore
        except Exception:
            pids = []

        for pid in pids:
            # Base data
            try:
                edge = self.traci.person.getRoadID(pid)
            except Exception:
                edge = ""
            try:
                speed = self.traci.person.getSpeed(pid)
            except Exception:
                speed = 0.0
            try:
                ptype = self.traci.person.getTypeID(pid)
            except Exception:
                ptype = ""
            if ptype:
                self._ped_type[pid] = ptype

            approach = self._classify_approach(edge)
            if approach is None:
                # If the pedestrian just left the inbound area, clear any waiting state
                if pid in self._ped_wait_start:
                    del self._ped_wait_start[pid]
                continue

            # Type counts (overall within inbound/walking-area context)
            if ptype in self._type_counts:
                self._type_counts[ptype] += 1

            waiting = speed < self.speed_threshold and not edge.startswith("E")
            if waiting:
                # Set the start time when the pedestrian begins waiting
                if pid not in self._ped_wait_start:
                    self._ped_wait_start[pid] = sim_time
                wait_time = sim_time - self._ped_wait_start[pid]
                # Aggregate metrics
                self._waiting_counts[approach] += 1
                self._wait_times.append(wait_time)
                if ptype in self._type_wait_times:
                    self._type_wait_times[ptype].append(wait_time)
                target = self._infer_target_outbound_edge(pid)
                tgt_key = target if target in self.outbound_edges else "UNKNOWN"
                self._direction_counts[approach][tgt_key] += 1
                self._direction_wait_times[approach][tgt_key].append(wait_time)
                if wait_time > self.long_wait_threshold:
                    self._direction_long_wait_counts[approach][tgt_key] += 1
                if "vulnerable" in ptype:
                    self._direction_vulnerable_counts[approach][tgt_key] += 1
            else:
                # When moving again, clear any waiting start marker
                if pid in self._ped_wait_start:
                    del self._ped_wait_start[pid]
            self._ped_last_edge[pid] = edge

        # Clean up IDs that disappeared from the network
        stale = set(self._ped_wait_start.keys()) - set(pids)
        for pid in stale:
            del self._ped_wait_start[pid]

    # --------------- Queries ---------------
    def _classify_approach(self, edge: str) -> Optional[str]:
        if not edge:
            return None
        # Directly on an inbound edge (-E0, -E0_0, etc.)
        for ap in self.approaches:
            if edge.startswith(ap):  # matches -E0 or -E0_0
                return ap
        # Walking-area edge
        for ap, walk_edge in self.walking_area_edges.items():
            if edge == walk_edge:
                return ap
        return None

    # --- Query helpers ---
    def waiting_counts(self) -> Dict[str, int]:
        return dict(self._waiting_counts)

    def wait_stats(self) -> Tuple[float, float, float]:
        if not self._wait_times:
            return 0.0, 0.0, 0.0
        s = sum(self._wait_times)
        mx = max(self._wait_times)
        mean = s / len(self._wait_times)
        return s, mx, mean

    def wait_stats_by_type(self) -> Dict[str, Tuple[float, float, float, int]]:
        """Return (sum, max, mean, count_waiting) per pedestrian type."""
        out: Dict[str, Tuple[float, float, float, int]] = {}
        for t, arr in self._type_wait_times.items():
            if not arr:
                out[t] = (0.0, 0.0, 0.0, 0)
            else:
                s = sum(arr)
                mx = max(arr)
                mean = s / len(arr)
                out[t] = (s, mx, mean, len(arr))
        return out

    def waiting_counts_detailed(self) -> Dict[str, Dict[str, int]]:
        """Nested counts: approach -> outbound_edge/"UNKNOWN" -> waiting count."""
        return {a: dict(d) for a, d in self._direction_counts.items()}

    def direction_distribution(self) -> Dict[str, Dict[str, int]]:  # Alias
        return self.waiting_counts_detailed()

    # --- Extended directional metrics ---
    def directional_max_wait_times(self) -> Dict[str, Dict[str, float]]:
        return {
            a: {tgt: (max(wts) if wts else 0.0) for tgt, wts in targets.items()}
            for a, targets in self._direction_wait_times.items()
        }

    def directional_vulnerable_counts(self) -> Dict[str, Dict[str, int]]:
        return {a: dict(d) for a, d in self._direction_vulnerable_counts.items()}

    def directional_vulnerable_fraction(self) -> Dict[str, Dict[str, float]]:
        frac: Dict[str, Dict[str, float]] = {}
        for a, tgt_counts in self._direction_vulnerable_counts.items():
            frac[a] = {}
            for tgt, v_cnt in tgt_counts.items():
                total = self._direction_counts[a][tgt]
                frac[a][tgt] = (v_cnt / total) if total > 0 else 0.0
        return frac

    def directional_long_wait_fraction(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for a, tgt_counts in self._direction_long_wait_counts.items():
            out[a] = {}
            for tgt, l_cnt in tgt_counts.items():
                total = self._direction_counts[a][tgt]
                out[a][tgt] = (l_cnt / total) if total > 0 else 0.0
        return out

    # --- New: max wait per approach ---
    def approach_max_waits(self) -> Dict[str, float]:
        """Return the longest observed wait per approach (considering currently waiting pedestrians).

        The implementation reuses ``_direction_wait_times`` by inspecting all target
        lists (including ``UNKNOWN``) for each approach and taking their maximum to
        avoid storing duplicate structures.
        """
        out: Dict[str, float] = {}
        for ap in self.approaches:
            mx = 0.0
            d = self._direction_wait_times.get(ap, {})
            for lst in d.values():
                if lst:
                    lm = max(lst)
                    if lm > mx:
                        mx = lm
            out[ap] = mx
        return out

    def long_wait_threshold_seconds(self) -> float:
        return self.long_wait_threshold

    def class_distribution(self) -> Dict[str, int]:
        return dict(self._type_counts)

    def wait_count(self) -> int:
        """Total number of waiting pedestrians (sum of ``waiting_counts``)."""
        return sum(self._waiting_counts.values())

    # --- Internal helper functions ---
    def _infer_target_outbound_edge(self, pid: str) -> Optional[str]:
        """Heuristically determine the next outbound edge the pedestrian intends to reach.

        Strategy:
          1. Query the pedestrian's route/edge list (``person.getEdges`` or ``getRoute``).
          2. Identify the current edge via ``person.getRoadID``.
          3. Locate the current edge within the route list.
          4. Iterate forward until an outbound edge (E0..E3) appears.
          5. Return that edge as the target; otherwise return ``None``.
        """
        if self.traci is None:
            return None
        try:
            cur_edge = self.traci.person.getRoadID(pid)
        except Exception:
            cur_edge = None
        # Retrieve the route edges (APIs differ by SUMO version)
        route_edges: List[str] = []
        if cur_edge is None:
            return None
        try:
            # SUMO >=1.10: getEdges available
            route_edges = list(self.traci.person.getEdges(pid))  # type: ignore
        except Exception:
            # Fallback: try getRoute
            try:
                route_edges = list(self.traci.person.getRoute(pid))  # type: ignore
            except Exception:
                route_edges = []
        if not route_edges:
            return None
        try:
            idx = route_edges.index(cur_edge)
        except ValueError:
            # If the current edge is not part of the route, fall back to the entire route
            idx = -1
        search_slice = route_edges[idx + 1 :] if idx >= 0 else route_edges
        for e in search_slice:
            if not e or e.startswith(":"):
                continue  # skip internal edges
            if e in self.outbound_edges:
                return e
        return None
