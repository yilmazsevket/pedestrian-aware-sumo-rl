"""Unit tests for TrafficLightActionSpace."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from TrafficLightActionSpace import TrafficLightActionSpace
from TrafficLightPhaseSwitcher import TrafficLightPhaseSwitcher


class _Phase:
    def __init__(self, name: str):
        self.name = name
        self.duration = 0


class _Logic:
    def __init__(self, phases):
        self.phases = phases


class _TLAPI:
    def __init__(self, phases):
        self._phases = phases
        self._phase = 0

    def getAllProgramLogics(self, _):
        return [_Logic(self._phases)]

    def getPhase(self, _):
        return self._phase

    def setPhase(self, _, idx: int):
        self._phase = idx

    def setPhaseDuration(self, *_):
        pass


class _Traci:
    def __init__(self, phases):
        self.trafficlight = _TLAPI(phases)


def mock_phases():
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
    return [_Phase(n) for n in names]


class TestActionSpace(unittest.TestCase):
    def setUp(self):
        self.traci = _Traci(mock_phases())
        self.action_space = TrafficLightActionSpace(step_length=1.0)
        self.action_space.build(self.traci, "tls0")

    def test_mapping_size(self):
        self.assertEqual(len(self.action_space.ACTION_TO_GREEN_INDEX), 4)
        self.assertEqual(len(self.action_space.ACTION_TO_GROUP), 4)

    def test_request_delegation(self):
        green_idx = self.action_space.ACTION_TO_GREEN_INDEX[0]
        switcher: TrafficLightPhaseSwitcher = self.action_space.switcher
        ns_left_idx = self.action_space.ACTION_TO_GREEN_INDEX[2]
        self.traci.trafficlight.setPhase("tls0", ns_left_idx)
        switcher._phase_timer = switcher.MIN_GREEN
        self.action_space.apply_action(self.traci, "tls0", 0)
        for _ in range(10):
            switcher.tick(self.traci, "tls0")
        self.assertEqual(self.traci.trafficlight.getPhase("tls0"), green_idx)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
    unittest.main(verbosity=2)
