"""Traffic light phase switcher with fixed groups and safe transitions.

This module defines a minimal, fixed-phase controller that exposes four main
signal groups and enforces a deterministic transition sequence with YELLOW and
REDYELLOW phases, plus a minimum green time to prevent chattering.
"""

from typing import Dict, Iterable, Optional


class TrafficLightPhaseSwitcher:
    """Minimal fixed switcher: 4 main groups with a clear sequence.

    Green phases (Agent actions):
        0: NS_Through+Right
        1: EW_Through+Right
        2: NS_Left
        3: EW_Left

    Transition when a change is needed:
        current green -> its Yellow -> target RedYellow -> target Green
    Direct jump: if already at target RedYellow -> target Green.
    Minimum green protects against chattering.
    """

    GROUPS = ["NS_MAIN", "EW_MAIN", "NS_LEFT", "EW_LEFT"]
    NAME_MAP = {
        "NS_MAIN": {
            "green": "NS_Through+Right",
            "yellow": "NS_Yellow",
            "redyellow": "NS_REDYellow",
        },
        "EW_MAIN": {
            "green": "EW_Through+Right",
            "yellow": "EW_Yellow",
            "redyellow": "EW_REDYellow",
        },
        "NS_LEFT": {
            "green": "NS_Left",
            "yellow": "NS_Left_Yellow",
            "redyellow": "NS_Left_REDYellow",
        },
        "EW_LEFT": {
            "green": "EW_Left",
            "yellow": "EW_Left_Yellow",
            "redyellow": "EW_Left_REDYellow",
        },
    }

    YELLOW_TIME = 3
    REDYELLOW_TIME = 1
    MIN_GREEN = 5
    GREEN_HOLD = 120

    def __init__(self, step_length: float = 1.0):
        self.step_length = float(step_length)
        self.idx = {}
        self._state = "IDLE"  # IDLE|YELLOW|REDYELLOW
        self._pending_group = None
        self._phase_timer = 0.0
        self._remaining = 0.0
        # Ensure tick() effects apply at most once per simulation time
        self._last_tick_sim_time = None

    # -------- Build --------
    def build(self, traci, tl_id: str) -> None:
        self.idx.clear()
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if not logics:
            raise ValueError("No TLS program logics found")
        phases = logics[0].phases  # type: ignore[attr-defined]
        name_to_index: Dict[str, int] = {}
        for i, ph in enumerate(phases):
            pname = getattr(ph, "name", "") or ""
            if pname:
                name_to_index[pname] = i
        missing = []
        for g in self.GROUPS:
            spec = self.NAME_MAP[g]
            entry: Dict[str, int] = {}
            for k, n in spec.items():
                if n not in name_to_index:
                    missing.append(n)
                else:
                    entry[k] = name_to_index[n]
            if len(entry) == 3:
                self.idx[g] = entry
        if missing:
            raise ValueError(f"Missing phase names: {missing}")
        self._state = "IDLE"
        self._pending_group = None
        self._phase_timer = 0.0
        self._remaining = 0.0
        self._last_tick_sim_time = None

    # -------- Agent API --------
    def request_action(self, traci, tl_id: str, action: int) -> None:
        if action < 0 or action >= len(self.GROUPS):
            return
        target_group = self.GROUPS[action]
        if target_group not in self.idx:
            return
        self._request_group(traci, tl_id, target_group)

    def tick(self, traci, tl_id: Optional[str]) -> None:
        if tl_id is None:
            return
        # Guard: apply tick effects at most once per simulation time
        try:
            sim_t_raw = traci.simulation.getTime()
            try:
                sim_t = float(sim_t_raw)
            except Exception:
                sim_t = float(sim_t_raw[0])  # type: ignore[index]
        except Exception:
            sim_t = None
        if (
            sim_t is not None
            and self._last_tick_sim_time is not None
            and sim_t == self._last_tick_sim_time
        ):
            return
        if sim_t is not None:
            self._last_tick_sim_time = sim_t
        if self._state == "IDLE":
            # Increase green timer if currently on a controlled green phase
            cur = int(traci.trafficlight.getPhase(tl_id))
            if self._group_of_green(cur) is not None:
                self._phase_timer += self.step_length
            return
        self._remaining -= self.step_length
        if self._remaining > 1e-9:
            return
        if self._state == "YELLOW":
            self._enter_redyellow(traci, tl_id)
        elif self._state == "REDYELLOW":
            self._enter_green(traci, tl_id)

    # -------- Internal --------
    def _request_group(self, traci, tl_id: str, target_group: str) -> None:
        cur_idx = int(traci.trafficlight.getPhase(tl_id))
        if self._state != "IDLE":
            return
        cur_group = self._group_of_green(cur_idx)
        # Same green -> do nothing
        if cur_group == target_group:
            return
        # Enforce minimum green
        if cur_group is not None and self._phase_timer < self.MIN_GREEN:
            return
        # Already at target RedYellow? -> go directly to target Green
        if self._is_redyellow_of(target_group, cur_idx):
            self._set_phase(
                traci, tl_id, self.idx[target_group]["green"], self.GREEN_HOLD
            )
            self._phase_timer = 0.0
            return
        # Normal start: Yellow of the current group; otherwise go directly to target RedYellow
        self._pending_group = target_group
        if cur_group is not None:
            self._set_phase(
                traci, tl_id, self.idx[cur_group]["yellow"], self.YELLOW_TIME
            )
            self._state = "YELLOW"
            # Off-by-one fix: first tick() occurs immediately after setting -> +step_length
            self._remaining = self.YELLOW_TIME + self.step_length
        else:
            self._enter_redyellow(traci, tl_id)

    def _enter_redyellow(self, traci, tl_id: str) -> None:
        if self._pending_group is None:
            self._state = "IDLE"
            return
        ry = self.idx[self._pending_group]["redyellow"]
        self._set_phase(traci, tl_id, ry, self.REDYELLOW_TIME)
        self._state = "REDYELLOW"
        self._remaining = self.REDYELLOW_TIME

    def _enter_green(self, traci, tl_id: str) -> None:
        if self._pending_group is None:
            self._state = "IDLE"
            return
        g = self.idx[self._pending_group]["green"]
        self._set_phase(traci, tl_id, g, self.GREEN_HOLD)
        self._phase_timer = 0.0
        self._state = "IDLE"
        self._pending_group = None
        self._remaining = 0.0

    # -------- Helpers --------
    def _group_of_green(self, phase_idx: int) -> Optional[str]:
        for g, data in self.idx.items():
            if phase_idx == data["green"]:
                return g
        return None

    def _is_redyellow_of(self, group: str, phase_idx: int) -> bool:
        if group not in self.idx:
            return False
        return phase_idx == self.idx[group]["redyellow"]

    def _set_phase(self, traci, tl_id: str, idx: int, duration: int) -> None:
        try:
            traci.trafficlight.setPhase(tl_id, idx)
            try:
                traci.trafficlight.setPhaseDuration(tl_id, duration)
            except Exception:
                pass
        except Exception:
            pass

    # -------- Public Info --------
    def green_phase_indices(self) -> Iterable[int]:
        return [self.idx[g]["green"] for g in self.GROUPS if g in self.idx]

    def current_state(self) -> str:
        return self._state

    def in_transition(self) -> bool:
        return self._state != "IDLE"

    def phase_time(self) -> int:
        return int(self._phase_timer)

    def validate(self) -> Dict[str, bool]:
        return {g: g in self.idx for g in self.GROUPS}

    # --- Accessors for ObservationBuilder ---
    def remaining_transition_time(self) -> int:
        if self._state == "IDLE":
            return 0
        return int(max(0, round(self._remaining)))

    def min_green_satisfied(self) -> bool:
        return self._phase_timer >= self.MIN_GREEN

    # Additional for observations
    def elapsed_green_time(self) -> float:
        return float(self._phase_timer)

    def target_group(self) -> Optional[str]:
        return self._pending_group
