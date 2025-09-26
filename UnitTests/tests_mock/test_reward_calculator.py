"""Mock-based unit test for RewardCalculator.

Runs without SUMO: verifies the base formula and clipping behavior.
Quick run:
    python -m pytest -q tests_mock/test_reward_calculator.py
Direct run without pytest (optional):
    python tests_mock/test_reward_calculator.py
"""

import os
import sys

modul_pfad = "C:\\Users\\yilma\\Desktop\\RealSumo\\reward_calculator.py"
sys.path.append(os.path.dirname(modul_pfad))
from reward_calculator import RewardCalculator


class VM:
    def __init__(self, veh_mean: float, queues: dict):
        self._veh_mean = veh_mean
        self._queues = queues

    def wait_stats(self):  # sum, max, mean
        return (0.0, 0.0, self._veh_mean)

    def group_queue_counts(self):
        return self._queues


class PM:
    def __init__(self, ped_mean: float):
        self._ped_mean = ped_mean

    def wait_stats(self):  # sum, max, mean
        return (0.0, 0.0, self._ped_mean)


def simple_case():
    rc = RewardCalculator(
        alpha=2.0, beta=0.5, clip_min=-100, clip_max=0, pressure_scale=10
    )
    vm = VM(
        veh_mean=10.0,
        queues={
            "NS_through": 5,
            "NS_right": 0,
            "NS_left": 0,
            "EW_through": 5,
            "EW_right": 0,
            "EW_left": 0,
        },
    )
    pm = PM(ped_mean=4.0)
    reward, meta = rc.compute(vm, pm)
    # Expectation: raw = -(10 + 2*4 + 0.5 * ((5+0+0 + 5+0+0)/10)) = -(10 + 8 + 0.5 * (10/10)) = -(18 + 0.5) = -18.5
    assert meta["veh_wait_mean"] == 10.0
    assert meta["ped_wait_mean"] == 4.0
    assert round(meta["pressure"], 3) == 1.0
    assert round(meta["reward_raw"], 3) == -18.5
    assert reward == meta["reward_clipped"]


# Optional manual run
if __name__ == "__main__":
    simple_case()
    print("OK")
    print("OK")
