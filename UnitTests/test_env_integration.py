"""Integration test for SumoEnv (optional, uses real SUMO if available)."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import traci  # noqa: F401

    from SumoEnv import SumoEnv
except Exception:  # pragma: no cover
    traci = None  # type: ignore
    SumoEnv = None  # type: ignore


class TestEnvIntegration(unittest.TestCase):
    def setUp(self):
        if traci is None or SumoEnv is None:
            self.skipTest("TraCI/SUMO not available")

    def test_basic_run(self):
        assert SumoEnv is not None  # for type-checkers
        env = SumoEnv(start_time=0, episode_duration=20, gui=False, fast_forward=False)
        try:
            obs, info = env.reset()
        except Exception as e:  # pragma: no cover
            self.skipTest(f"Could not start SUMO: {e}")
        self.assertEqual(obs.shape[0], 4)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            self.assertEqual(obs.shape[0], 4)
            self.assertIn("sim_time", info)
            if terminated or truncated:
                break
        env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
    unittest.main(verbosity=2)
    unittest.main(verbosity=2)
