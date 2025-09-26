"""Action space wrapper for the traffic light controller (4 fixed actions).

Provides a stable mapping between discrete agent actions and the four main
green-phase groups managed by TrafficLightPhaseSwitcher.
"""

from typing import Any, Dict, Optional

from TrafficLightPhaseSwitcher import TrafficLightPhaseSwitcher


class TrafficLightActionSpace:
    """Fixed action space (4 actions) for the four main green phases.

    Mapping:
        Action 0 -> NS_MAIN (NS_Through+Right)
        Action 1 -> EW_MAIN (EW_Through+Right)
        Action 2 -> NS_LEFT (NS_Left)
        Action 3 -> EW_LEFT (EW_Left)

    After build():
        ACTION_TO_GREEN_INDEX: action -> real phase ID
        ACTION_TO_GROUP: action -> group name
        INDEX_TO_NAME: phase_index -> group name
    """

    ACTIONS = 4

    def __init__(self, step_length: float = 1.0):
        self.switcher = TrafficLightPhaseSwitcher(step_length=step_length)
        self.ACTION_TO_GREEN_INDEX: Dict[int, int] = {}
        self.ACTION_TO_GROUP: Dict[int, str] = {}
        self.INDEX_TO_NAME: Dict[int, str] = {}

    def build(self, traci, tl_id: str) -> None:
        self.switcher.build(traci, tl_id)
        self.ACTION_TO_GREEN_INDEX.clear()
        self.ACTION_TO_GROUP.clear()
        self.INDEX_TO_NAME.clear()
        for a, group in enumerate(self.switcher.GROUPS):
            if group in self.switcher.idx:
                g_idx = self.switcher.idx[group]["green"]
                self.ACTION_TO_GREEN_INDEX[a] = g_idx
                self.ACTION_TO_GROUP[a] = group
                self.INDEX_TO_NAME[g_idx] = group

    def action_count(self) -> int:
        return self.ACTIONS

    def apply_action(self, traci, tl_id: Optional[str], action: Any) -> None:
        if tl_id is None:
            return
        try:
            a_int = int(action)
        except Exception:
            return
        if a_int < 0 or a_int >= self.ACTIONS:
            return
        self.switcher.request_action(traci, tl_id, a_int)

    def tick(self, traci, tl_id: Optional[str]) -> None:
        self.switcher.tick(traci, tl_id)
