from __future__ import annotations

# Set before importing libraries that may load OpenMP runtimes (e.g., PyTorch)
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # allow mixed OpenMP runtimes
os.environ.setdefault("OMP_NUM_THREADS", "1")  # reduce thread contention

import re

import matplotlib

matplotlib.use("Agg")  # headless backend for saving figures without a display

"""Baseline vs. Agent comparison at three rush-hour windows with line charts.

It runs each window twice (baseline and agent), collects time series per step of
vehicle/pedestrian waiting times and plots Baseline vs Agent. You can choose between:
    - snapshot: per-step snapshot sums (current total wait seconds; default)
    - accumulated: cumulative waiting times using manager accumulators per step

Comparison modes:
    - macro: line plots for vehicles/pedestrians/vulnerable pedestrians (x=episode step, y=waiting time)
    - ped_hist: per-class pedestrian waiting-time time series (x=episode step, y=waiting time)
    - ped_bar: aggregated per-class wait totals as grouped bar charts

Then it plots three figures (morning, midday, evening) with baseline vs agent lines
for each metric and saves them as PNGs next to this script.
"""

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from SumoEnv import SumoEnv

COMPARE_MODE_ALIASES = {
    "macro": "macro",
    "1": "macro",
    "ped_classes": "ped_hist",
    "classes": "ped_hist",
    "ped_hist": "ped_hist",
    "hist": "ped_hist",
    "2": "ped_hist",
    "ped_classes_hist": "ped_hist",
    "ped_bar": "ped_bar",
    "bar": "ped_bar",
    "ped_classes_bar": "ped_bar",
    "3": "ped_bar",
}


def _normalize_compare_mode(raw: str) -> str:
    value = (raw or "").strip().lower()
    return COMPARE_MODE_ALIASES.get(value, "macro")


def _add_global_legend(fig, show_base: bool = True, show_agent: bool = True) -> None:
    """Add a single, figure-level legend for Baseline/Agent with proxy handles.

    Avoids duplicating entries collected from multiple axes.
    """
    from matplotlib.patches import Patch

    handles = []
    labels = []
    if show_base:
        handles.append(Patch(facecolor="tab:blue", edgecolor="tab:blue"))
        labels.append("Baseline")
    if show_agent:
        handles.append(Patch(facecolor="tab:orange", edgecolor="tab:orange"))
        labels.append("Agent")
    if handles:
        fig.legend(handles, labels, loc="upper right")


@dataclass
class RunConfig:
    label: str
    start_time: int
    duration: int


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


def _get_waits(
    env: SumoEnv, info: Dict[str, float], mode: str
) -> Tuple[float, float, float]:
    """Return (veh_sum, ped_sum, vuln_ped_sum) for selected mode.

    mode: 'snapshot' or 'accumulated'
    """
    mode = (mode or "snapshot").lower()
    # Vehicles may use accumulated mode; pedestrians always use snapshot semantics.
    veh: float
    if mode == "accumulated":
        veh = 0.0
        try:
            vm = getattr(env, "vehicle_manager", None)
            if vm is not None:
                res = vm.wait_stats()  # accumulated vehicles
                if isinstance(res, tuple) and len(res) >= 1:
                    veh = float(res[0])
        except Exception:
            veh = float(info.get("veh_wait_sum", 0.0))
    else:
        veh = float(info.get("veh_wait_sum", 0.0))

    # Pedestrians: snapshot only
    ped = float(info.get("ped_wait_sum", 0.0))
    vuln = float(info.get("vuln_ped_wait_sum", 0.0))
    return float(veh), float(ped), float(vuln)


MacroSeries = Dict[str, List[float]]
PedSeries = Dict[str, Dict[str, List[float]]]


def collect_series(
    env: SumoEnv, model: PPO | None, wait_mode: str
) -> Tuple[MacroSeries, PedSeries]:
    obs, _ = env.reset()
    macro: MacroSeries = {
        "veh_wait_sum": [],
        "ped_wait_sum": [],
        "vuln_ped_wait_sum": [],
    }
    ped_manager = getattr(env, "pedestrian_manager", None)
    ped_types = (
        list(getattr(ped_manager, "pedestrian_types", [])) if ped_manager else []
    )
    ped_series: PedSeries = {p: {"sum": [], "mean": [], "count": []} for p in ped_types}

    while True:
        if model is None:
            action = 0  # unused; no-op when agent_control=False
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        veh, ped, vuln = _get_waits(env, info, wait_mode)
        macro["veh_wait_sum"].append(veh)
        macro["ped_wait_sum"].append(ped)
        macro["vuln_ped_wait_sum"].append(vuln)

        if ped_manager is not None:
            try:
                stats_by_type = ped_manager.wait_stats_by_type()
            except Exception:
                stats_by_type = {}
            if stats_by_type:
                for ptype, data in stats_by_type.items():
                    if not isinstance(data, tuple) or len(data) < 4:
                        continue
                    total, _max_wait, avg_wait, count = data
                    series = ped_series.setdefault(
                        ptype, {"sum": [], "mean": [], "count": []}
                    )
                    series["sum"].append(float(total))
                    series["mean"].append(float(avg_wait))
                    series["count"].append(int(count))
        if terminated or truncated:
            break

    env.close()
    return macro, ped_series


def _aggregate_for_bar(series: Dict[str, List[float]], wait_mode: str) -> float:
    values = series.get("sum", [])
    non_zero = [v for v in values if v > 0]
    if not non_zero:
        return 0.0
    if wait_mode == "snapshot":
        return float(sum(non_zero))
    return float(mean(non_zero))


def plot_ped_hist(
    window: RunConfig,
    base_ped: PedSeries,
    agent_ped: PedSeries,
    outdir: str,
    wait_mode: str,
) -> bool:
    classes = sorted(set(base_ped.keys()) | set(agent_ped.keys()))
    if not classes:
        return False

    num_classes = len(classes)
    fig_height = max(3.0, 2.5 * num_classes)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, fig_height), sharex=False)
    if num_classes == 1:
        axes = [axes]

    slug = re.sub(r"[^a-z0-9]+", "-", window.label.lower()).strip("-")
    out_path = os.path.join(outdir, f"compare_{slug}_classes_timeseries.png")

    for ax, cls in zip(axes, classes):
        base_vals = base_ped.get(cls, {}).get("mean", [])
        agent_vals = agent_ped.get(cls, {}).get("mean", [])
        drew_any = False
        if base_vals:
            ax.plot(range(len(base_vals)), base_vals, color="tab:blue", alpha=0.85)
            drew_any = True
        if agent_vals:
            ax.plot(range(len(agent_vals)), agent_vals, color="tab:orange", alpha=0.85)
            drew_any = True

        if not drew_any:
            ax.text(
                0.5,
                0.5,
                "No waiting pedestrians",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(cls.replace("_", " ").title())
        ax.set_ylabel("Waiting time (s)")
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("Episode step")
    _add_global_legend(fig)
    fig.suptitle(f"{window.label}: Pedestrian class waiting time over episode")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    try:
        fig.savefig(out_path, dpi=120)
    except Exception as e:
        alt_path = os.path.join(outdir, "compare_window_classes_hist.png")
        try:
            fig.savefig(alt_path, dpi=120)
            print(f"Saved class histogram (fallback) due to error '{e}': {alt_path}")
        finally:
            pass
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return True


def plot_ped_bar(
    window: RunConfig,
    base_ped: PedSeries,
    agent_ped: PedSeries,
    outdir: str,
    wait_mode: str,
) -> bool:
    classes = sorted(set(base_ped.keys()) | set(agent_ped.keys()))
    if not classes:
        return False

    width = 0.35
    idx = list(range(len(classes)))
    base_vals = [_aggregate_for_bar(base_ped.get(c, {}), wait_mode) for c in classes]
    agent_vals = [_aggregate_for_bar(agent_ped.get(c, {}), wait_mode) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in idx], base_vals, width, label="Baseline")
    ax.bar([i + width / 2 for i in idx], agent_vals, width, label="Agent")
    ax.set_xticks(idx)
    ax.set_xticklabels([c.replace("_", " ").title() for c in classes], rotation=20)
    ylabel = (
        "Cumulative wait (s)" if wait_mode == "snapshot" else "Avg accumulated wait (s)"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{window.label}: Pedestrian classes – Baseline vs Agent")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    slug = re.sub(r"[^a-z0-9]+", "-", window.label.lower()).strip("-")
    out_path = os.path.join(outdir, f"compare_{slug}_classes_bar.png")
    fig.tight_layout()
    _add_global_legend(fig)
    try:
        fig.savefig(out_path, dpi=120)
    except Exception as e:
        alt_path = os.path.join(outdir, "compare_window_classes_bar.png")
        try:
            fig.savefig(alt_path, dpi=120)
            print(
                f"Saved class bar figure (fallback name) due to error '{e}': {alt_path}"
            )
        finally:
            pass
    plt.close(fig)
    print(f"Saved figure: {out_path}")
    return True


def plot_macro(
    window: RunConfig,
    base_macro: MacroSeries,
    agent_macro: MacroSeries,
    outdir: str,
    wait_mode: str,
) -> None:
    t_base = list(range(len(base_macro["veh_wait_sum"])))
    t_agent = list(range(len(agent_macro["veh_wait_sum"])))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    fig.suptitle(f"{window.label}: Baseline vs Agent")

    axes[0].plot(
        t_base,
        base_macro["veh_wait_sum"],
        label="Baseline",
        color="tab:blue",
        alpha=0.8,
    )
    axes[0].plot(
        t_agent,
        agent_macro["veh_wait_sum"],
        label="Agent",
        color="tab:orange",
        alpha=0.8,
    )
    axes[0].set_title("Vehicle waiting time")
    axes[0].grid(True, linestyle=":", alpha=0.4)

    axes[1].plot(
        t_base,
        base_macro["ped_wait_sum"],
        label="Baseline",
        color="tab:blue",
        alpha=0.8,
    )
    axes[1].plot(
        t_agent,
        agent_macro["ped_wait_sum"],
        label="Agent",
        color="tab:orange",
        alpha=0.8,
    )
    axes[1].set_title("Pedestrian waiting time")
    axes[1].grid(True, linestyle=":", alpha=0.4)

    axes[2].plot(
        t_base,
        base_macro["vuln_ped_wait_sum"],
        label="Baseline",
        color="tab:blue",
        alpha=0.8,
    )
    axes[2].plot(
        t_agent,
        agent_macro["vuln_ped_wait_sum"],
        label="Agent",
        color="tab:orange",
        alpha=0.8,
    )
    axes[2].set_title("Vulnerable pedestrian waiting time")
    axes[2].grid(True, linestyle=":", alpha=0.4)

    fig.supxlabel("Episode step")
    fig.supylabel("Waiting time (s)")
    _add_global_legend(fig)

    slug = re.sub(r"[^a-z0-9]+", "-", window.label.lower()).strip("-")
    suffix = "_acc" if wait_mode == "accumulated" else ""
    out_path = os.path.join(outdir, f"compare_{slug}{suffix}.png")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    try:
        fig.savefig(out_path, dpi=120)
    except Exception as e:
        alt_path = os.path.join(outdir, "compare_window.png")
        try:
            fig.savefig(alt_path, dpi=120)
            print(f"Saved figure (fallback name) due to error '{e}': {alt_path}")
        finally:
            pass
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_window(
    window: RunConfig,
    base_data: Tuple[MacroSeries, PedSeries],
    agent_data: Tuple[MacroSeries, PedSeries],
    outdir: str,
    compare_mode: str,
    wait_mode: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    compare_mode = compare_mode or "macro"
    base_macro, base_ped = base_data
    agent_macro, agent_ped = agent_data

    if compare_mode == "ped_hist":
        if plot_ped_hist(window, base_ped, agent_ped, outdir, wait_mode):
            return
        print("No pedestrian data for histogram – falling back to macro view.")
    elif compare_mode == "ped_bar":
        if plot_ped_bar(window, base_ped, agent_ped, outdir, wait_mode):
            return
        print("No pedestrian data for bar chart – falling back to macro view.")

    plot_macro(window, base_macro, agent_macro, outdir, wait_mode)


def main() -> None:
    # Optional quick mode for smoke tests (set QUICK=1 in env)
    quick = os.environ.get("QUICK", "").strip() == "1"
    dur = 300 if quick else 1800
    compare_mode_raw = os.environ.get("COMPARE_MODE", "")
    compare_mode = _normalize_compare_mode(compare_mode_raw)
    if (
        not compare_mode_raw
        or compare_mode_raw.strip().lower() not in COMPARE_MODE_ALIASES
    ):
        compare_sel = input(
            "Which comparison should be plotted? [macro/ped_hist/ped_bar] (macro): "
        )
        compare_mode = _normalize_compare_mode(compare_sel)

    wait_mode_env = os.environ.get("WAIT_MODE", "").strip().lower()
    if compare_mode == "macro":
        if wait_mode_env in ("snapshot", "accumulated"):
            wait_mode = wait_mode_env
        else:
            sel = (
                input("Vehicle wait aggregation? [snapshot/accumulated] (snapshot): ")
                .strip()
                .lower()
            )
            wait_mode = sel if sel in ("snapshot", "accumulated") else "snapshot"
    else:
        wait_mode = "snapshot"
        if wait_mode_env == "accumulated":
            print(
                "Accumulated mode is ignored for pedestrian-only comparisons; using snapshot waits."
            )
    # Three rush-hour windows
    windows = [
        RunConfig(
            label="Morning Rush Hour (07:30)",
            start_time=7 * 3600 + 30 * 60,
            duration=dur,
        ),
        RunConfig(
            label="Midday Rush Hour (12:00)",
            start_time=12 * 3600,
            duration=dur,
        ),
        RunConfig(
            label="Evening Rush Hour (17:00)",
            start_time=17 * 3600,
            duration=dur,
        ),
    ]

    # Load trained agent
    model_path = r"C:\Users\yilma\Desktop\RealSumo\runs\RUN_4_ppo_6_000_000_20k_accreward\ppo_sumo_4300000_steps.zip"
    model = PPO.load(model_path)

    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    for w in windows:
        # Baseline (no agent control)
        env_base = build_env(
            start_time=w.start_time, duration=w.duration, gui=False, agent_control=False
        )
        base_result = collect_series(env_base, model=None, wait_mode=wait_mode)

        # Agent (deterministic)
        env_agent = build_env(
            start_time=w.start_time, duration=w.duration, gui=False, agent_control=True
        )
        agent_result = collect_series(env_agent, model=model, wait_mode=wait_mode)

        plot_window(
            w,
            base_result,
            agent_result,
            outdir,
            compare_mode=compare_mode,
            wait_mode=wait_mode,
        )

    print("Done. Figures saved in 'figures' folder.")


if __name__ == "__main__":
    main()
