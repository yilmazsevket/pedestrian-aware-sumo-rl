"""Quick performance benchmark for SumoEnv policy latency.

Measures per-step latency components (ms):
 - action_apply_ms (time applying TL action / FSM updates)
 - simulation_step_ms (traci.simulationStep)
 - obs_build_ms (observation construction)
 - reward_compute_ms (reward calculation + normalization)
 - total_step_ms (end-to-end time inside env.step)

Usage (example):
  python performance_benchmark.py --model runs/ppo_latest.zip --steps 1000 --warmup 50

If --model is omitted a random policy is used to approximate overhead.

The script prints a summary table and saves JSON stats to performance_timing.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, cast

try:
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover
    PPO = None  # type: ignore

from SumoEnv import SumoEnv


def run(
    env: SumoEnv,
    model: Optional[Any],
    steps: int,
    collect_ms: Optional[List[float]] = None,
) -> None:
    obs, _info = env.reset()
    for _ in range(steps):
        if model is None:
            action = env.action_space.sample()
        else:
            try:
                action, _ = model.predict(obs, deterministic=True)
            except Exception:
                action = env.action_space.sample()
        t0 = time.perf_counter()
        obs, _r, term, trunc, _info = env.step(action)
        t1 = time.perf_counter()
        if collect_ms is not None:
            collect_ms.append((t1 - t0) * 1000.0)
        if term or trunc:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark per-step latency of SumoEnv")
    ap.add_argument("--model", type=str, default=None, help="Optional PPO model path")
    ap.add_argument("--steps", type=int, default=500, help="Benchmark steps to record")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup steps (discarded)")
    ap.add_argument(
        "--start-time",
        type=int,
        default=7 * 3600 + 30 * 60,
        help="Simulation start time (s)",
    )
    ap.add_argument(
        "--episode-duration", type=int, default=4000, help="Episode duration (s)"
    )
    ap.add_argument("--gui", action="store_true", help="Run SUMO with GUI (slower)")
    ap.add_argument(
        "--out-json",
        type=str,
        default="performance_timing.json",
        help="Output JSON file",
    )
    args = ap.parse_args()

    model = None
    if args.model and PPO is not None:
        try:
            model = PPO.load(args.model)
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Failed to load model: {e}; using random actions")

    env = SumoEnv(
        start_time=args.start_time,
        episode_duration=args.episode_duration,
        step_length=1.0,
        gui=args.gui,
        reward_mode="delta",
        fast_forward=True,
        agent_control=True,
    )

    # Warmup phase (simulation caches, JIT, etc.)
    if args.warmup > 0:
        run(env, model, args.warmup)
        reset_profile = getattr(env, "reset_timing_stats", None)
        if callable(reset_profile):
            try:
                reset_profile()
            except Exception:
                pass

    # Benchmark phase
    total_step_ms: List[float] = []
    run(env, model, args.steps, collect_ms=total_step_ms)
    stats: Dict[str, Any] = {}
    get_stats = getattr(env, "get_timing_stats", None)
    if callable(get_stats):
        try:
            s = get_stats()
            stats = s if isinstance(s, dict) else {}
        except Exception:
            stats = {}
    if not stats:
        # Local fallback: only total_step_ms is available
        def p95(vals: list[float]) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            idx = int(0.95 * (len(s) - 1))
            return s[idx]

        if total_step_ms:
            stats = {
                "total_step_ms": {
                    "mean": float(sum(total_step_ms) / len(total_step_ms)),
                    "p95": float(p95(total_step_ms)),
                    "max": float(max(total_step_ms)),
                    "count": int(len(total_step_ms)),
                    "util_mean_pct": 0.0,
                    "util_p95_pct": 0.0,
                }
            }
        else:
            stats = {}
    env.close()

    # Pretty print summary
    def fmt(v: Any) -> str:
        return f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v)

    print("\nPer-step latency (ms):")
    print("Metric                Mean     P95     Max    Count  UtilMean%  UtilP95%")
    for k in [
        "action_apply_ms",
        "simulation_step_ms",
        "obs_build_ms",
        "reward_compute_ms",
        "total_step_ms",
    ]:
        if not isinstance(stats, dict) or k not in stats:
            continue
        s = cast(Dict[str, Any], stats[k])
        util_mean = s.get("util_mean_pct", 0.0)
        util_p95 = s.get("util_p95_pct", 0.0)
        print(
            f"{k:20s} {fmt(s['mean']):>7} {fmt(s['p95']):>7} {fmt(s['max']):>7} {int(s['count']):>7} {fmt(util_mean):>9} {fmt(util_p95):>9}"
        )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved timing stats to {os.path.abspath(args.out_json)}")

    # Quick guidance
    if isinstance(stats, dict) and "total_step_ms" in stats:
        mean_ms = cast(Dict[str, Any], stats["total_step_ms"]).get("mean", 0.0)
        if mean_ms < 20:
            print(
                "[INFO] Real-time capable for 1 Hz steps (1s per sim second) with large headroom."
            )
        elif mean_ms < 200:
            print(
                "[INFO] Still below 200 ms; OK for near real-time 1s sim steps, might handle faster than real-time."
            )
        else:
            print(
                "[WARN] High per-step latency; consider profiling simulation_step_ms and obs_build_ms components."
            )


if __name__ == "__main__":
    main()
    main()
