"""Realtime latency profiling for SumoEnv + PPO policy.

Measures per-step timings (milliseconds):
- obs_build_ms: time from previous env.step end to policy inference start
    (approximates ObservationBuilder + internal manager cost)
- policy_infer_ms: PPO forward pass (model.predict)
- env_step_ms: duration of env.step (TraCI simulation + reward)
- cycle_total_ms: time from cycle start (after obs) to next obs
    (policy + env.step + minimal overhead)
- end_to_end_ms: time between consecutive policy decisions

Outputs:
- Aggregate stats (count, mean, p50, p90, p95, max) per metric
- Optional JSON file for automated analysis

Usage:
    python profile_realtime_latency.py --model runs/my_model.zip --steps 2000 --warmup 50 \
            --veh-wait-mode accumulated --json-out latency.json

Realtime guideline:
    With step_length=1.0 s, cycle_total_ms should be well below 1000 ms (e.g., <100 ms).
    For step_length=0.5 s it should be <50 ms, etc. A safety factor of ~5–10x is recommended.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO

from SumoEnv import SumoEnv

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class StepTiming:
    obs_build_ms: float
    policy_infer_ms: float
    env_step_ms: float
    cycle_total_ms: float
    end_to_end_ms: float


def _percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    k = (len(arr) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(arr[f])
    return float(arr[f] * (c - k) + arr[c] * (k - f))


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    vals = list(values)
    vals.sort()
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "max": float(vals[-1]),
    }


def profile(
    model_path: str,
    sumocfg: str,
    steps: int,
    warmup: int,
    gui: bool,
    veh_wait_mode: str,
    seed: int = 42,
    step_length: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    model = PPO.load(model_path)

    env = SumoEnv(
        start_time=0,
        episode_duration=steps + warmup + 10,
        step_length=step_length,
        gui=gui,
        sumo_config=sumocfg,
        reward_mode="delta",  # matches training config
        fast_forward=False,
        agent_control=True,
    )
    obs, info = env.reset()

    timings: List[StepTiming] = []

    # Pre-allocate variables for timing
    last_decision_time = time.perf_counter()

    for step_idx in range(steps + warmup):
        cycle_start = time.perf_counter()
        # Observation is already present in 'obs'; we approximate obs build cost by time since last env.step end
        obs_build_ms = (cycle_start - last_decision_time) * 1000.0

        # Policy inference
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        t1 = time.perf_counter()
        policy_infer_ms = (t1 - t0) * 1000.0

        # Environment step
        t2 = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action)
        t3 = time.perf_counter()
        env_step_ms = (t3 - t2) * 1000.0

        cycle_total_ms = (t3 - cycle_start) * 1000.0
        end_to_end_ms = (t3 - last_decision_time) * 1000.0
        last_decision_time = t3

        obs = next_obs

        if step_idx >= warmup:
            timings.append(
                StepTiming(
                    obs_build_ms=obs_build_ms,
                    policy_infer_ms=policy_infer_ms,
                    env_step_ms=env_step_ms,
                    cycle_total_ms=cycle_total_ms,
                    end_to_end_ms=end_to_end_ms,
                )
            )
        if terminated or truncated:
            break

    env.close()

    # Aggregate
    timing_dict = {
        "obs_build_ms": _summary([t.obs_build_ms for t in timings]),
        "policy_infer_ms": _summary([t.policy_infer_ms for t in timings]),
        "env_step_ms": _summary([t.env_step_ms for t in timings]),
        "cycle_total_ms": _summary([t.cycle_total_ms for t in timings]),
        "end_to_end_ms": _summary([t.end_to_end_ms for t in timings]),
    }

    # Add real-time headroom metric: fraction of step_length consumed
    for key in list(timing_dict.keys()):
        if "ms" in key and timing_dict[key].get("mean") is not None:
            mean_ms = timing_dict[key]["mean"]
            timing_dict[key]["mean_fraction_of_step"] = mean_ms / (step_length * 1000.0)

    return timing_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile realtime latency of PPO + SumoEnv"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained PPO model (zip/dir)"
    )
    parser.add_argument(
        "--sumocfg", type=str, default="validatednet.sumocfg", help="SUMO config file"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Profiled steps (post-warmup)"
    )
    parser.add_argument(
        "--warmup", type=int, default=50, help="Warmup steps (excluded from stats)"
    )
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI (slower)")
    parser.add_argument(
        "--veh-wait-mode",
        type=str,
        choices=("accumulated", "snapshot"),
        default="accumulated",
        help="Reserved flag (for symmetry with eval); currently informational only",
    )
    parser.add_argument(
        "--step-length", type=float, default=1.0, help="Simulation step length seconds"
    )
    parser.add_argument(
        "--json-out", type=str, default=None, help="Optional path to write JSON results"
    )
    parser.add_argument(
        "--print-raw", action="store_true", help="Print each step timing (verbose)"
    )
    args = parser.parse_args()

    results = profile(
        model_path=args.model,
        sumocfg=args.sumocfg,
        steps=args.steps,
        warmup=args.warmup,
        gui=args.gui,
        veh_wait_mode=args.veh_wait_mode,
        step_length=args.step_length,
    )

    print("\n=== Realtime Latency Profiling Results ===")
    print(f"Model: {args.model}")
    print(f"SUMO Config: {args.sumocfg}")
    print(f"Steps (profiled): {args.steps} (warmup {args.warmup})")
    print(f"Step length (s): {args.step_length}")
    for name, stats in results.items():
        if not stats:
            continue
        print(f"\n{name}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved JSON results to {os.path.abspath(args.json_out)}")
        except Exception as e:
            print(f"Failed to write JSON: {e}")

    # Interpretation hint
    cycle_mean = results.get("cycle_total_ms", {}).get("mean", 0.0)
    if cycle_mean > 0:
        frac = cycle_mean / (args.step_length * 1000.0)
        print(
            f"\nMean cycle uses {frac*100:.1f}% of allotted step time. "
            f"Headroom factor ≈ {1/max(frac,1e-6):.1f}x."
        )
        if frac < 0.2:
            print("Realtime capable with large margin.")
        elif frac < 0.5:
            print("Realtime likely OK; monitor worst-case spikes.")
        elif frac < 0.8:
            print(
                "Borderline: consider optimization (reduce GUI, lower observation cost, vectorize)."
            )
        else:
            print("Not realtime-safe: optimize or increase step_length.")


if __name__ == "__main__":  # pragma: no cover
    main()
