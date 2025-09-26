"""Unit tests for TrafficLightPhaseSwitcher (FSM)."""

import os
import sys
import unittest

# Ensure project root (one level up) is on sys.path when executed directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from TrafficLightPhaseSwitcher import TrafficLightPhaseSwitcher


class _MockPhase:
    def __init__(self, name: str):
        self.name = name
        self.duration = 0


class _MockLogic:
    def __init__(self, phases):
        self.phases = phases


class _MockTrafficLightAPI:
    def __init__(self, phases):
        self._phases = phases
        self._current_phase = self._find_index("NS_Through+Right")
        self._phase_duration_set = None

    def _find_index(self, name: str) -> int:
        for i, ph in enumerate(self._phases):
            if ph.name == name:
                return i
        raise KeyError(name)

    def getAllProgramLogics(self, tl_id: str):
        return [_MockLogic(self._phases)]

    def getPhase(self, tl_id: str):
        return self._current_phase

    def setPhase(self, tl_id: str, index: int):
        self._current_phase = index

    def setPhaseDuration(self, tl_id: str, dur: int):
        self._phase_duration_set = dur


class _MockTraci:
    def __init__(self, phases):
        self.trafficlight = _MockTrafficLightAPI(phases)


def build_mock_traci():
    names = [
        "NS_REDYellow",
        "NS_Through+Right",
        "NS_Yellow",
        "EW_REDYellow",
        "EW_Through+Right",
        "EW_Yellow",
        "NS_Left_REDYellow",
        "NS_Left",
        "NS_Left_Yellow",
        "EW_Left_REDYellow",
        "EW_Left",
        "EW_Left_Yellow",
    ]
    phases = [_MockPhase(n) for n in names]
    return _MockTraci(phases), phases


class TestSwitcherFSM(unittest.TestCase):
    def setUp(self):
        self.traci, self.phases = build_mock_traci()
        self.switcher = TrafficLightPhaseSwitcher(step_length=1.0)
        self.switcher.build(self.traci, "tls0")

    def _tick_n(self, n: int):
        for _ in range(n):
            self.switcher.tick(self.traci, "tls0")

    def test_min_green_enforced(self):
        for _ in range(self.switcher.MIN_GREEN - 1):
            self.switcher.request_action(self.traci, "tls0", 2)
            self._tick_n(1)
            self.assertEqual(self.switcher.current_state(), "IDLE")
            self.assertEqual(
                self.traci.trafficlight.getPhase("tls0"),
                self.traci.trafficlight._find_index("NS_Through+Right"),
            )

        # Exactly (MIN_GREEN - 1) seconds passed -> still not switchable
        self.switcher.request_action(self.traci, "tls0", 2)
        self.assertEqual(self.switcher.current_state(), "IDLE")
        # One more second passes -> minimum green reached
        self._tick_n(1)
        # Now request again -> should start Yellow
        self.switcher.request_action(self.traci, "tls0", 2)
        self.assertEqual(self.switcher.current_state(), "YELLOW")

    def test_full_transition_durations(self):
        self._tick_n(self.switcher.MIN_GREEN)
        self.switcher.request_action(self.traci, "tls0", 2)
        states = []
        for _ in range(10):
            # Tick first, then record state – avoids off-by-one at start (pre-tick YELLOW)
            self.switcher.tick(self.traci, "tls0")
            states.append(self.switcher.current_state())
            if (
                self.switcher.current_state() == "IDLE"
                and self.traci.trafficlight.getPhase("tls0")
                == self.traci.trafficlight._find_index("NS_Left")
                and self.switcher.phase_time() == 1
            ):
                break
        yellow_count = sum(1 for s in states if s == "YELLOW")
        redyellow_count = sum(1 for s in states if s == "REDYELLOW")
        self.assertEqual(yellow_count, self.switcher.YELLOW_TIME)
        self.assertEqual(redyellow_count, self.switcher.REDYELLOW_TIME)
        self.assertEqual(self.switcher.current_state(), "IDLE")

    def test_direct_redyellow_shortcut(self):
        redyellow_idx = self.traci.trafficlight._find_index("NS_Left_REDYellow")
        self.traci.trafficlight.setPhase("tls0", redyellow_idx)
        self.switcher.request_action(self.traci, "tls0", 2)
        self.assertEqual(
            self.traci.trafficlight.getPhase("tls0"),
            self.traci.trafficlight._find_index("NS_Left"),
        )
        self.assertEqual(self.switcher.current_state(), "IDLE")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
