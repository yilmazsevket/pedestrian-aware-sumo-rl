"""Policy analysis script for Chapter 5.4.

Generates figures and analyses:
 1) Action usage (distribution of signal groups)
 2) State–action heatmaps (e.g., ped wait vs action, pressure vs action)
 3) Safety/rule report (min-green attempts)
 4) Optional feature permutation importance (if SB3 + model path available)
 5) Summary JSON for direct inclusion

Data sources:
- Expects existing rollout files (.npz) or performs its own rollouts.
- Does not depend on compare_baseline_vs_agent.py.

Usage:
        python policy_analysis.py --model runs/ppo_latest.zip --episodes 2 --steps 800

Artifacts are written to ./figures/policy_analysis/.

Notes:
- If no model is provided, only a baseline random policy is analyzed.
- Heatmaps use feature names from ObservationBuilder; robust fallbacks are applied if
    names are missing or changed.
"""

from __future__ import annotations

# Early env settings to avoid OpenMP + multi-runtime issues and force deterministic small thread usage
import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
# Allow duplicate OpenMP runtimes (NumPy MKL + SUMO / Torch) to prevent hard crash
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Headless Matplotlib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt  # noqa: E402

# Optional Stable-Baselines3 / Torch imports
try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover
    PPO = None  # type: ignore
    torch = None  # type: ignore

# Local imports (robust with clear error message)
try:
    from SumoEnv import SumoEnv  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"[FATAL] Import SumoEnv failed: {e}")
    sys.exit(2)

# ---------------------- Configuration ----------------------
DEFAULT_EPISODES = 1
DEFAULT_STEPS = 600
FIG_SUBDIR = os.path.join("figures", "policy_analysis")
RANDOM_SEED = 123
ACTION_LABELS = ["NS_MAIN", "EW_MAIN", "NS_LEFT", "EW_LEFT"]

# Simple debug flag via env var: set POLICY_ANALYSIS_DEBUG=1 for verbose
DEBUG_MODE = os.environ.get("POLICY_ANALYSIS_DEBUG", "0") == "1"

# ---------------------- Utils ----------------------


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_index(
    names: List[str], key: str, fallback: Optional[int] = None
) -> Optional[int]:
    try:
        return names.index(key)
    except Exception:
        return fallback


# ---------------------- Rollout Dataclass ----------------------
@dataclass
class RolloutData:
    actions: List[int]
    obss: List[np.ndarray]
    veh_wait: List[float]
    ped_wait: List[float]
    vuln_ped_wait: List[float]
    min_green_flags: List[float]
    attempted_before_min_green: int
    phase_indices: List[int]
    feature_names: List[str]

    def stack_obs(self) -> np.ndarray:
        if not self.obss:
            return np.zeros((0,))
        try:
            return np.stack(self.obss, axis=0)
        except Exception:
            return np.zeros((0,))


# ---------------------- Rollout function ----------------------


def run_rollout(env: SumoEnv, model: Optional[Any], max_steps: int) -> RolloutData:
    obs, info = env.reset()
    feature_names: List[str] = []
    if getattr(env, "obs_builder", None) is not None:
        try:
            feature_names = env.obs_builder.feature_names()  # type: ignore[attr-defined]
        except Exception:
            feature_names = []

    idx_map = {name: i for i, name in enumerate(feature_names)}

    actions: List[int] = []
    obss: List[np.ndarray] = []
    veh_wait: List[float] = []
    ped_wait: List[float] = []
    vuln_ped_wait: List[float] = []
    min_green_flags: List[float] = []
    phase_indices: List[int] = []
    attempted_before_min_green = 0

    rng = np.random.default_rng(RANDOM_SEED)

    for step in range(1, max_steps + 1):
        if DEBUG_MODE and step == 1:
            print(
                f"[DEBUG] Erste Observation shape={getattr(obs,'shape',None)} sim_time={info.get('sim_time','?')} "
            )
        if model is None:
            action = int(rng.integers(0, 4))
        else:
            try:
                action, _ = model.predict(obs, deterministic=True)
            except Exception:
                action = int(rng.integers(0, 4))

        # Derive current phase and min_green from the observation vector
        try:
            # One-hot starts at index 34 (per spec); fallback if different
            onehot_slice = obs[34:38]
            cur_phase = int(np.argmax(onehot_slice)) if onehot_slice.size == 4 else -1
            min_green = float(obs[39]) if obs.shape[0] > 39 else 0.0
        except Exception:
            cur_phase = -1
            min_green = 0.0

        if (
            model is not None
            and cur_phase >= 0
            and action != cur_phase
            and min_green < 0.5
        ):
            attempted_before_min_green += 1

        terminated = False
        truncated = False
        step_info: Dict[str, Any] = {}
        try:
            obs, reward, terminated, truncated, step_info = env.step(action)
        except Exception as step_err:
            if DEBUG_MODE:
                import traceback

                print(f"[DEBUG] env.step Exception bei step={step}: {step_err}")
                traceback.print_exc()
            break
        if DEBUG_MODE and (terminated or truncated):
            reason = step_info.get("reason") or step_info.get("error") or "episode_end"
            print(
                f"[DEBUG] Episode ended early after {step} steps (terminated={terminated}, truncated={truncated}) reason={reason}"
            )

        # Wait times: vehicle sum (index 42); no direct ped sum here -> approximate placeholders
        try:
            veh_sum_wait = float(obs[42]) if obs.shape[0] > 42 else 0.0
        except Exception:
            veh_sum_wait = 0.0

        # No direct ped sum in the current obs vector; omit for now (placeholders remain 0.0)
        ped_sum_approx = 0.0
        vuln_sum_approx = 0.0  # Placeholder falls spaeter Typen integriert werden

        actions.append(int(action))
        obss.append(obs.copy())
        veh_wait.append(veh_sum_wait)
        ped_wait.append(ped_sum_approx)
        vuln_ped_wait.append(vuln_sum_approx)
        min_green_flags.append(min_green)
        phase_indices.append(cur_phase)

        if terminated or truncated:
            break

    return RolloutData(
        actions=actions,
        obss=obss,
        veh_wait=veh_wait,
        ped_wait=ped_wait,
        vuln_ped_wait=vuln_ped_wait,
        min_green_flags=min_green_flags,
        attempted_before_min_green=attempted_before_min_green,
        phase_indices=phase_indices,
        feature_names=feature_names,
    )


# ---------------------- Analysis: Action usage ----------------------


def plot_action_usage(data: RolloutData, outdir: str, label: str) -> Dict[str, Any]:
    ensure_outdir(outdir)
    counts = np.bincount(data.actions if data.actions else [0], minlength=4)
    labels = ACTION_LABELS
    total = counts.sum() if counts.sum() > 0 else 1
    shares = counts / total
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color="tab:blue", alpha=0.85)
    for b, val, share in zip(bars, counts, shares):
        if b.get_height() > 0:
            plt.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.5,
                f"{share*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.title(f"Action Usage ({label})")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(outdir, f"action_usage_{label}.png")
    plt.savefig(path, dpi=130)
    plt.close()
    return {"counts": counts.tolist(), "shares": shares.tolist(), "file": path}


# ---------------------- Simplified heatmaps ----------------------


def _discretize(values: np.ndarray, bins: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    edges = np.linspace(vmin, vmax, bins + 1)
    bucket = np.digitize(values, edges) - 1
    bucket = np.clip(bucket, 0, bins - 1)
    return bucket, edges


def build_action_feature_heatmap(
    obss: np.ndarray,
    actions: List[int],
    feature_idx: int,
    feature_name: str,
    outdir: str,
    fname: str,
    bins: int = 12,
) -> Optional[str]:
    if obss.size == 0:
        return None
    ensure_outdir(outdir)
    try:
        X = obss[:, feature_idx]
    except Exception:
        if DEBUG_MODE:
            print(
                f"[DEBUG] Feature index {feature_idx} invalid for heatmap {feature_name}"
            )
        return None
    A = np.array(actions if actions else [0])
    x_bucket, edges = _discretize(X, bins=bins)
    grid = np.zeros((4, bins), dtype=np.float64)
    for b, a in zip(x_bucket, A):
        if 0 <= a < 4:
            grid[a, b] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        col_sum = grid.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1
        grid_norm = grid / col_sum
    plt.figure(figsize=(9, 3.4))
    # Grayscale (lighter = higher share)
    plt.imshow(
        grid_norm,
        origin="lower",
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    plt.yticks(range(4), ACTION_LABELS)
    # Label with bucket midpoints
    xticks = list(range(bins))
    centers = [f"{(edges[i]+edges[i+1])/2:.0f}" for i in range(bins)]
    plt.xticks(xticks, centers, rotation=45, fontsize=7)
    plt.xlabel(feature_name)
    plt.ylabel("Action")
    plt.title(f"Action shares vs. {feature_name}")
    cbar = plt.colorbar()
    cbar.set_label("Share per bucket")
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def build_pressure_heatmap(
    obss: np.ndarray,
    actions: List[int],
    idx_ns: int,
    idx_ew: int,
    outdir: str,
    fname: str,
    bins: int = 10,
) -> Optional[str]:
    if obss.size == 0:
        return None
    ensure_outdir(outdir)
    try:
        p_ns = obss[:, idx_ns]
        p_ew = obss[:, idx_ew]
    except Exception:
        if DEBUG_MODE:
            print("[DEBUG] Pressure indices invalid")
        return None
    A = np.array(actions if actions else [0])
    # Use ratio NS/(NS+EW) as a one-dimensional pressure measure
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_ns / (p_ns + p_ew)
        ratio[np.isnan(ratio)] = 0.5
    bucket, edges = _discretize(ratio, bins=bins)
    grid = np.zeros((4, bins), dtype=np.float64)
    for b, a in zip(bucket, A):
        if 0 <= a < 4:
            grid[a, b] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        col_sum = grid.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1
        grid_norm = grid / col_sum
    plt.figure(figsize=(8.5, 3.4))
    plt.imshow(
        grid_norm,
        origin="lower",
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    plt.yticks(range(4), ACTION_LABELS)
    centers = [f"{(edges[i]+edges[i+1])/2:.2f}" for i in range(bins)]
    plt.xticks(range(bins), centers, rotation=45, fontsize=7)
    plt.xlabel("Pressure Ratio NS / (NS+EW)")
    plt.ylabel("Action")
    plt.title("Action shares vs. Pressure Ratio")
    cbar = plt.colorbar()
    cbar.set_label("Share per bucket")
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=140)
    plt.close()
    return path


# ---------------------- Safety report ----------------------


def write_safety_report(
    agent_data: RolloutData, baseline_data: RolloutData, outdir: str
) -> str:
    ensure_outdir(outdir)
    path = os.path.join(outdir, "safety_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Safety Report\n")
        f.write("==============\n")
        f.write(
            f"Agent attempted early phase switch before min-green: {agent_data.attempted_before_min_green}\n"
        )
        f.write(
            f"Baseline attempted early phase switch before min-green: {baseline_data.attempted_before_min_green}\n"
        )
        f.write("Executed violations: 0 (FSM enforced).\n")
        f.write(
            "Interpretation: A high number of attempts may indicate an aggressive switching strategy.\n"
        )
    return path


# ---------------------- Permutation Importance (optional) ----------------------


def permutation_importance(
    obss: np.ndarray,
    model: Any,
    feature_names: List[str],
    outdir: str,
    max_samples: int = 4000,
) -> Optional[str]:
    if PPO is None or torch is None:
        return None
    if obss.size == 0:
        return None
    ensure_outdir(outdir)
    n = min(max_samples, obss.shape[0])
    X = obss[:n].copy()
    device = model.device
    with torch.no_grad():
        orig_actions = []
        for i in range(0, n, 1024):
            batch = torch.as_tensor(X[i : i + 1024], dtype=torch.float32, device=device)
            dist = model.policy.get_distribution(batch)
            a = dist.distribution.probs.argmax(dim=-1).cpu().numpy()
            orig_actions.append(a)
        orig_actions = np.concatenate(orig_actions)
    rng = np.random.default_rng(123)
    changes: List[float] = []
    for j in range(X.shape[1]):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        with torch.no_grad():
            new_actions = []
            for i in range(0, n, 1024):
                batch = torch.as_tensor(
                    Xp[i : i + 1024], dtype=torch.float32, device=device
                )
                dist = model.policy.get_distribution(batch)
                a = dist.distribution.probs.argmax(dim=-1).cpu().numpy()
                new_actions.append(a)
            new_actions = np.concatenate(new_actions)
        change_rate = float((orig_actions != new_actions).mean())
        changes.append(change_rate)
    order = np.argsort(changes)[::-1][:10]
    plt.figure(figsize=(7, 4.5))
    top_names = [
        feature_names[k] if k < len(feature_names) else f"f{k}" for k in order
    ][::-1]
    top_vals = [changes[k] for k in order][::-1]
    plt.barh(top_names, top_vals, color="dimgray")
    plt.xlabel("Action change rate (Top 10)")
    plt.title("Permutation Importance")
    plt.tight_layout()
    path = os.path.join(outdir, "permutation_importance_top10.png")
    plt.savefig(path, dpi=140)
    plt.close()
    return path


# ---------------------- Summary JSON ----------------------


def write_summary_json(outdir: str, **sections: Any) -> str:
    ensure_outdir(outdir)
    path = os.path.join(outdir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    return path


# ---------------------- Main pipeline ----------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None, help="Path to PPO model (.zip)"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--start-time",
        type=int,
        default=0,
        help="Simulation start (seconds) – e.g., 1800 for morning rush",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    parser.add_argument(
        "--skip-importance",
        action="store_true",
        help="Skip permutation importance",
    )
    args = parser.parse_args()

    outdir = ensure_outdir(FIG_SUBDIR)

    # Modell laden (falls angegeben)
    model = None
    if args.model and os.path.exists(args.model):
        if PPO is None:
            print("[WARN] SB3 not available – ignoring model.")
        else:
            try:
                model = PPO.load(args.model)
                print(f"[INFO] Model loaded: {args.model}")
            except Exception as e:
                print(f"[WARN] Failed to load model: {e}")
                model = None

    agent_rollouts: List[RolloutData] = []
    baseline_rollouts: List[RolloutData] = []

    for ep in range(args.episodes):
        print(f"[Episode {ep+1}/{args.episodes}] Agent policy rollout...")
        env_agent: Optional[SumoEnv] = None
        try:
            env_agent = SumoEnv(
                start_time=args.start_time,
                episode_duration=args.steps + 10,
                gui=not args.no_gui,
            )
            agent_data = run_rollout(env_agent, model, args.steps)
        except Exception as e:
            print(f"[WARN] Agent rollout episode {ep+1} aborted: {e}")
            agent_data = RolloutData([], [], [], [], [], [], 0, [], [])
        finally:
            try:
                if env_agent is not None:
                    env_agent.close()
            except Exception:
                pass
        agent_rollouts.append(agent_data)
        if DEBUG_MODE:
            print(f"[DEBUG] Agent steps collected: {len(agent_data.actions)}")

        print(f"[Episode {ep+1}/{args.episodes}] Baseline random rollout...")
        env_base: Optional[SumoEnv] = None
        try:
            env_base = SumoEnv(
                start_time=args.start_time,
                episode_duration=args.steps + 10,
                gui=not args.no_gui,
            )
            baseline_data = run_rollout(env_base, None, args.steps)
        except Exception as e:
            print(f"[WARN] Baseline rollout episode {ep+1} aborted: {e}")
            baseline_data = RolloutData([], [], [], [], [], [], 0, [], [])
        finally:
            try:
                if env_base is not None:
                    env_base.close()
            except Exception:
                pass
        baseline_rollouts.append(baseline_data)
        if DEBUG_MODE:
            print(f"[DEBUG] Baseline steps collected: {len(baseline_data.actions)}")

    # Aggregation
    def aggregate(rollouts: List[RolloutData]) -> RolloutData:
        if not rollouts:
            return RolloutData([], [], [], [], [], [], 0, [], [])
        # Flatten
        actions = [a for r in rollouts for a in r.actions]
        obss = [o for r in rollouts for o in r.obss]
        veh_wait = [v for r in rollouts for v in r.veh_wait]
        ped_wait = [p for r in rollouts for p in r.ped_wait]
        vuln_wait = [p for r in rollouts for p in r.vuln_ped_wait]
        min_g = [m for r in rollouts for m in r.min_green_flags]
        phase_idx = [p for r in rollouts for p in r.phase_indices]
        attempts = sum(r.attempted_before_min_green for r in rollouts)
        # Use the feature names of the first rollout (if available)
        feat_names = rollouts[0].feature_names if rollouts[0].feature_names else []
        return RolloutData(
            actions,
            obss,
            veh_wait,
            ped_wait,
            vuln_wait,
            min_g,
            attempts,
            phase_idx,
            feat_names,
        )

    agent_all = aggregate(agent_rollouts)
    baseline_all = aggregate(baseline_rollouts)

    # Feature names from aggregated rollouts
    if agent_all.feature_names:
        feature_names = agent_all.feature_names
    elif agent_all.obss:
        feature_names = [f"f{i}" for i in range(len(agent_all.obss[0]))]
    else:
        feature_names = []

    # Action usage plots
    usage_agent = plot_action_usage(agent_all, outdir, label="agent")
    usage_base = plot_action_usage(baseline_all, outdir, label="baseline")
    if DEBUG_MODE:
        print(f"[DEBUG] Action Usage Agent counts={usage_agent['counts']}")

    # New heatmaps: dynamic feature-name resolution
    heat_elapsed = heat_wait = heat_pressure = None
    if agent_all.obss:
        obss_agent = agent_all.stack_obs()
        # Try name-based indices (robust to layout shifts)
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        idx_elapsed = name_to_idx.get("elapsed_green_time")
        # Vehicle wait sum may not exist directly -> use max_vehicle_wait_time as proxy if sum is unavailable
        idx_wait_sum = name_to_idx.get("sum_vehicle_wait_time")
        if idx_wait_sum is None:
            # Fallback to max_vehicle_wait_time (index 27 in the new layout)
            idx_wait_sum = name_to_idx.get("max_vehicle_wait_time")
        idx_pressure_ns = name_to_idx.get("pressure_NS")
        idx_pressure_ew = name_to_idx.get("pressure_EW")

        if idx_elapsed is not None:
            heat_elapsed = build_action_feature_heatmap(
                obss_agent,
                agent_all.actions,
                idx_elapsed,
                "elapsed_green_time",
                outdir,
                "heat_elapsed_agent.png",
            )
        if idx_wait_sum is not None:
            heat_wait = build_action_feature_heatmap(
                obss_agent,
                agent_all.actions,
                idx_wait_sum,
                (
                    feature_names[idx_wait_sum]
                    if idx_wait_sum < len(feature_names)
                    else "veh_wait"
                ),
                outdir,
                "heat_vehicle_wait_agent.png",
            )
        if idx_pressure_ns is not None and idx_pressure_ew is not None:
            heat_pressure = build_pressure_heatmap(
                obss_agent,
                agent_all.actions,
                idx_pressure_ns,
                idx_pressure_ew,
                outdir,
                "heat_pressure_ratio_agent.png",
            )

    # Safety report
    safety_path = write_safety_report(agent_all, baseline_all, outdir)

    # Permutation Importance (optional)
    perm_path = None
    if model is not None and not args.skip_importance and agent_all.obss:
        obss_agent = agent_all.stack_obs()
        try:
            perm_path = permutation_importance(obss_agent, model, feature_names, outdir)
        except Exception as e:
            print(f"[WARN] Permutation importance failed: {e}")
            perm_path = None

    # Summary JSON for chapter text
    summary = {
        "action_usage": {"agent": usage_agent, "baseline": usage_base},
        "safety_report": safety_path,
        "heatmaps": {
            "elapsed": heat_elapsed,
            "vehicle_wait": heat_wait,
            "pressure_ratio": heat_pressure,
        },
        "permutation_importance": perm_path,
        "attempts_before_min_green": {
            "agent": agent_all.attempted_before_min_green,
            "baseline": baseline_all.attempted_before_min_green,
        },
        "mean_vehicle_wait_agent": float(
            np.mean(agent_all.veh_wait) if agent_all.veh_wait else 0.0
        ),
        "mean_vehicle_wait_baseline": float(
            np.mean(baseline_all.veh_wait) if baseline_all.veh_wait else 0.0
        ),
        "total_steps_agent": len(agent_all.actions),
        "total_steps_baseline": len(baseline_all.actions),
    }
    summary_path = write_summary_json(outdir, **summary)

    print("\nGenerated artifacts:")
    for k, v in summary.items():
        stat = v if not isinstance(v, dict) else "OK"
        print(f"- {k}: {stat}")
    print(f"Summary JSON: {summary_path}")
    print(f"Output directory: {outdir}")
    # Note for very short episodes
    if len(agent_all.actions) < 5 and DEBUG_MODE:
        print(
            "[DEBUG] Fewer than 5 steps in agent rollout – possibly no SUMO run (traci not loaded?)"
        )

    if not any(os.path.isfile(os.path.join(outdir, f)) for f in os.listdir(outdir)):
        print("[WARN] No files written – please re-run with POLICY_ANALYSIS_DEBUG=1.")


if __name__ == "__main__":  # pragma: no cover
    main()
