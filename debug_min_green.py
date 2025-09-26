from __future__ import annotations

"""Diagnose 5-second min-green rule violations without modifying existing code.

This script:
- Builds a SumoEnv and attaches a non-invasive debugger to the TLS switcher.
- Logs phase transitions, switcher timers, and TraCI calls to detect where/how
  the 5s rule may be bypassed (agent vs SUMO auto progression).
- Produces a CSV timeline and a concise console summary of any violations.

Usage (Windows cmd):
  # quick 5-minute runs per window to validate
  set QUICK=1
  conda run --live-stream --name RealSumoCondaEnv python c:/Users/yilma/Desktop/RealSumo/debug_min_green.py --model "C:\\path\\to\\model.zip"

  # or run without model (random actions)
  conda run --live-stream --name RealSumoCondaEnv python c:/Users/yilma/Desktop/RealSumo/debug_min_green.py
"""

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import traci  # type: ignore

try:
    from stable_baselines3 import PPO  # optional
except Exception:  # pragma: no cover
    PPO = None  # type: ignore

from SumoEnv import SumoEnv
from TrafficLightPhaseSwitcher import TrafficLightPhaseSwitcher


@dataclass
class RunConfig:
    label: str
    start_time: int
    duration: int


class MinGreenDebugger:
    """Attaches to an existing TrafficLightPhaseSwitcher to monitor invariants.

    - Tracks agent requests, _set_phase calls, phase indices over time, and the
      switcher's internal min-green timer (_phase_timer).
    - Detects green->green transitions earlier than MIN_GREEN and classifies
      likely initiator (agent vs SUMO auto) based on observed calls.
    """

    def __init__(
        self, switcher: TrafficLightPhaseSwitcher, tl_id: str, min_green: float
    ) -> None:
        self.switcher = switcher
        self.tl_id = tl_id
        self.min_green = float(min_green)

        # originals
        self._orig_request_action = switcher.request_action
        self._orig__request_group = switcher._request_group  # type: ignore[attr-defined]
        self._orig__set_phase = switcher._set_phase  # type: ignore[attr-defined]
        self._orig_tick = switcher.tick

        # recent activity
        self.last_agent_request: Optional[Dict[str, Any]] = None
        self.last_set_phase: Optional[Dict[str, Any]] = None

        # phase tracking
        self.green_indices = set(int(i) for i in switcher.green_phase_indices())
        self.prev_phase: Optional[int] = None
        self.prev_green_start_time: Optional[float] = None
        self.prev_green_idx: Optional[int] = None

        # log rows
        self.rows: List[Dict[str, Any]] = []
        self.violations: List[Dict[str, Any]] = []

        # cache phase names from active program (best-effort)
        try:
            logics = traci.trafficlight.getAllProgramLogics(self.tl_id)
            phases = logics[0].phases if logics else []  # type: ignore[attr-defined]
            self.index_to_name: Dict[int, str] = {
                i: (getattr(ph, "name", "") or "") for i, ph in enumerate(phases)
            }
        except Exception:
            self.index_to_name = {}

        # patch methods
        self._patch_methods()

    # ---- casting helpers for TraCI's varying return types ----
    @staticmethod
    def _to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            try:
                return float(x[0])  # type: ignore[index]
            except Exception:
                return 0.0

    @staticmethod
    def _to_int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            try:
                return int(x[0])  # type: ignore[index]
            except Exception:
                return -1

    def _patch_methods(self) -> None:
        def wrapped_request_action(traci_mod, tl_id: str, action: int) -> None:
            self.last_agent_request = {
                "time": self._now(),
                "sim_time": self._sim_time(),
                "action": int(action),
                "target": (
                    self.switcher.GROUPS[action]
                    if 0 <= action < len(self.switcher.GROUPS)
                    else None
                ),
                "phase_timer": self.switcher.elapsed_green_time(),
                "min_green": getattr(self.switcher, "MIN_GREEN", self.min_green),
                "blocked_min_green": bool(
                    self.switcher.elapsed_green_time()
                    < getattr(self.switcher, "MIN_GREEN", self.min_green)
                ),
            }
            return self._orig_request_action(traci_mod, tl_id, action)

        def wrapped__set_phase(traci_mod, tl_id: str, idx: int, duration: int) -> None:  # type: ignore[no-redef]
            self.last_set_phase = {
                "time": self._now(),
                "sim_time": self._sim_time(),
                "idx": int(idx),
                "duration": int(duration),
            }
            return self._orig__set_phase(traci_mod, tl_id, idx, duration)

        def wrapped_tick(traci_mod, tl_id: Optional[str]) -> None:
            return self._orig_tick(traci_mod, tl_id)

        # apply patches
        self.switcher.request_action = wrapped_request_action  # type: ignore[assignment]
        self.switcher._set_phase = wrapped__set_phase  # type: ignore[attr-defined,assignment]
        self.switcher.tick = wrapped_tick  # type: ignore[assignment]

    def _now(self) -> float:
        return time.time()

    def _sim_time(self) -> float:
        try:
            return self._to_float(traci.simulation.getTime())
        except Exception:
            return -1.0

    def step_capture(self) -> None:
        sim_t = self._sim_time()
        try:
            phase_idx = self._to_int(traci.trafficlight.getPhase(self.tl_id))
        except Exception:
            phase_idx = -1
        phase_name = self.index_to_name.get(phase_idx, "")
        # whether switcher recognizes this phase as one of its green phases
        try:
            recognized_group = None
            for g, data in self.switcher.idx.items():  # type: ignore[attr-defined]
                if phase_idx == data.get("green"):
                    recognized_group = g
                    break
        except Exception:
            recognized_group = None

        # detect green start/transition
        if phase_idx != self.prev_phase:
            # Phase changed
            if phase_idx in self.green_indices:
                # entering a green
                elapsed = None
                if (
                    self.prev_green_start_time is not None
                    and self.prev_green_idx is not None
                ):
                    # green->green transition
                    elapsed = sim_t - self.prev_green_start_time
                    if elapsed < self.min_green - 1e-6:
                        self._record_violation(sim_t, phase_idx, elapsed)
                # start new green window
                self.prev_green_start_time = sim_t
                self.prev_green_idx = phase_idx
            self.prev_phase = phase_idx

        # collect row
        row: Dict[str, Any] = {
            "sim_time": sim_t,
            "phase_idx": phase_idx,
            "phase_name": phase_name,
            "recognized_green_group": recognized_group,
            "switcher_state": self.switcher.current_state(),
            "phase_timer": self.switcher.elapsed_green_time(),
            "min_green_satisfied": self.switcher.min_green_satisfied(),
            "last_agent_action": (self.last_agent_request or {}).get("action"),
            "last_agent_time": (self.last_agent_request or {}).get("sim_time"),
            "last_set_idx": (self.last_set_phase or {}).get("idx"),
            "last_set_dur": (self.last_set_phase or {}).get("duration"),
        }
        # Next switch if available
        try:
            row["next_switch"] = self._to_float(
                traci.trafficlight.getNextSwitch(self.tl_id)
            )
        except Exception:
            row["next_switch"] = None
        self.rows.append(row)

    def _record_violation(
        self, sim_t: float, new_green_idx: int, elapsed: float
    ) -> None:
        who = "unknown"
        # If _set_phase was called very recently, classify as agent-initiated
        if (
            self.last_set_phase
            and abs(sim_t - float(self.last_set_phase.get("sim_time", -1.0))) <= 1.5
        ):
            who = "agent_setPhase"
        elif (
            self.last_agent_request
            and abs(sim_t - float(self.last_agent_request.get("sim_time", -1.0))) <= 1.5
        ):
            who = "agent_request"
        else:
            who = "sumo_auto"
        self.violations.append(
            {
                "sim_time": sim_t,
                "elapsed_since_prev_green": float(elapsed),
                "min_green": self.min_green,
                "new_green_idx": int(new_green_idx),
                "who": who,
                "phase_timer": self.switcher.elapsed_green_time(),
                "state": self.switcher.current_state(),
            }
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = (
            list(self.rows[0].keys())
            if self.rows
            else [
                "sim_time",
                "phase_idx",
                "switcher_state",
                "phase_timer",
                "min_green_satisfied",
                "last_agent_action",
                "last_agent_time",
                "last_set_idx",
                "last_set_dur",
                "next_switch",
            ]
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


def build_env(
    start_time: int, duration: int, gui: bool, agent_control: bool
) -> SumoEnv:
    return SumoEnv(
        start_time=start_time,
        episode_duration=duration,
        step_length=1.0,
        seed=123,
        gui=gui,
        sumo_config="validatednet.sumocfg",
        fast_forward=True,
        reward_mode="delta",
        agent_control=agent_control,
    )


def main() -> None:
    # config
    quick = os.environ.get("QUICK", "").strip() == "1"
    dur = 300 if quick else 1800
    windows = [
        RunConfig(label="Morning", start_time=7 * 3600 + 30 * 60, duration=dur),
    ]

    # default model path (can be overridden by CLI arg)
    model_path = (
        r"C:\Users\yilma\Desktop\RealSumo\runs\ppo_20250830_122448\ppo_sumo_final.zip"
    )
    if len(sys.argv) > 1 and sys.argv[1]:
        model_path = sys.argv[1]
    model = None
    if model_path and PPO is not None:
        try:
            model = PPO.load(model_path)
        except Exception:
            model = None

    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    for w in windows:
        env = build_env(
            start_time=w.start_time, duration=w.duration, gui=True, agent_control=True
        )
        obs, info = env.reset()
        # access switcher and tl_id
        switcher = env.tl_action_space.switcher  # type: ignore[attr-defined]
        # discover tl_id
        try:
            tls_ids = traci.trafficlight.getIDList()
            tl_id = tls_ids[0] if tls_ids else None
        except Exception:
            tl_id = None
        if not tl_id or switcher is None:
            print("No TLS or switcher found; aborting this window")
            env.close()
            continue

        dbg = MinGreenDebugger(switcher, tl_id, getattr(switcher, "MIN_GREEN", 5))

        # rollout
        steps = 0
        violations_before = len(dbg.violations)
        while True:
            # decide action
            if model is None:
                action = steps % 4  # cycle 0..3 to stress transitions
            else:
                action, _ = model.predict(obs, deterministic=True)
            # apply step
            obs, reward, terminated, truncated, info = env.step(action)
            dbg.step_capture()
            steps += 1
            if terminated or truncated:
                break

        # save CSV log
        safe_label = "morning"
        csv_path = os.path.join(outdir, f"min_green_debug_{safe_label}.csv")
        dbg.save(csv_path)

        # print summary
        print(f"Window {w.label}: steps={steps}")
        if dbg.violations:
            print(f"Min-green violations: {len(dbg.violations)}")
            # show first few
            for v in dbg.violations[:5]:
                print(
                    "Violation at t=%.1f: elapsed=%.2f<%.2f, newGreen=%d, who=%s, state=%s, timer=%.2f"
                    % (
                        v["sim_time"],
                        v["elapsed_since_prev_green"],
                        v["min_green"],
                        v["new_green_idx"],
                        v["who"],
                        v["state"],
                        v["phase_timer"],
                    )
                )
        else:
            print("No min-green violations detected.")

        env.close()


if __name__ == "__main__":
    main()
