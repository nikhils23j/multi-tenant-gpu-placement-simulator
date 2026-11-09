"""Render the README figures from the saved results.json (no simulation is re-run).

    python3 scripts/make_figures.py            # writes docs/img/*.png
    python3 scripts/make_figures.py --input results.json --outdir docs/img

Every plotted value is read from results.json; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Categorical slots in fixed order (validated: adjacent CVD/normal-vision separation passes on white).
# Colour follows the policy everywhere, so a policy keeps its hue across every figure.
COLOR = {
    "fifo": "#2a78d6",
    "fifo_greedy_backfill": "#eb6834",
    "easy_backfill": "#1baf7a",
    "easy_topo": "#eda100",
    "drf_topo": "#e87ba4",
    "drf_topo_preempt": "#008300",
    "wfq_topo": "#6250d6",
}
LABEL = {
    "fifo": "fifo",
    "fifo_greedy_backfill": "greedy backfill",
    "easy_backfill": "easy_backfill",
    "easy_topo": "easy_topo",
    "wfq_topo": "wfq_topo",
    "drf_topo": "drf_topo",
    "drf_topo_preempt": "drf_topo_preempt",
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DPI = 140

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)


def stat(block: dict, policy: str, metric: str) -> dict:
    return block["policies"][policy]["summary"][metric]


# ---------------------------------------------------------------------------------------------------
def load_sweep(res: dict, out: Path) -> Path:
    """Small multiples over offered load: utilization, batch p95, worst big-job wait, inference p99."""
    sweep = sorted(res["load_sweep"], key=lambda r: r["load"])
    loads = [r["load"] for r in sweep]
    n_seeds, n_jobs = len(sweep[0]["seeds"]), sweep[0]["jobs"]
    shown = ["fifo", "fifo_greedy_backfill", "easy_backfill", "easy_topo", "drf_topo", "drf_topo_preempt"]
    panels = [
        ("gpu_utilization", "GPU utilization (%)", 100.0, True),
        ("batch_p95_wait_s", "Batch p95 wait (hours)", 1 / 3600, False),
        ("big_job_max_wait_s", "Worst 16/32-GPU job wait (hours)", 1 / 3600, True),
        ("interactive_p99_wait_s", "Inference p99 wait (seconds)", 1.0, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for ax, (metric, title, scale, with_fifo) in zip(axes.flat, panels):
        pols = shown if with_fifo else shown[1:]
        for p in pols:
            m = [stat(r, p, metric)["mean"] * scale for r in sweep]
            h = [stat(r, p, metric)["ci95"] * scale for r in sweep]
            ax.fill_between(loads, [a - b for a, b in zip(m, h)], [a + b for a, b in zip(m, h)],
                            color=COLOR[p], alpha=0.10, linewidth=0)
            ax.plot(loads, m, "-", color=COLOR[p], linewidth=2, solid_capstyle="round", label=LABEL[p])
            ax.plot(loads[-1], m[-1], "o", color=COLOR[p], markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=2)
        style(ax)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(loads, [f"{x:g}" for x in loads])
        ax.set_xlabel("offered load (GPU-time demand / capacity)", fontsize=9)
        ax.set_ylim(bottom=0)
        if not with_fifo:
            lo = min(stat(r, "fifo", metric)["mean"] for r in sweep) * scale
            hi = max(stat(r, "fifo", metric)["mean"] * scale for r in sweep)
            unit = "h" if scale != 1.0 else "s"
            ax.text(0.02, 0.97, f"fifo omitted (off scale: {lo:,.0f}–{hi:,.0f} {unit})", transform=ax.transAxes,
                    va="top", fontsize=8.5, color=MUTED)
    axes[0, 0].set_ylim(0, 100)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.965), handlelength=1.6)
    fig.suptitle(f"Load sweep: {n_jobs:,} jobs per run, mean of {n_seeds} seeds, shaded band = 95% CI",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    path = out / "load_sweep.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------------
def policy_comparison(res: dict, out: Path, key: str = "high_load_1.0") -> Path:
    """Dot + 95% CI per policy at one load, three metrics side by side."""
    block = res[key]
    order = ["fifo", "fifo_greedy_backfill", "easy_backfill", "easy_topo", "wfq_topo", "drf_topo",
             "drf_topo_preempt"]
    metrics = [
        ("interactive_p99_wait_s", "Inference p99 wait (s)", 1.0, "s"),
        ("batch_p95_wait_s", "Batch p95 wait (h)", 1 / 3600, "h"),
        ("big_job_max_wait_s", "Worst 16/32-GPU job wait (h)", 1 / 3600, "h"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    ys = list(range(len(order)))[::-1]
    for ax, (metric, title, scale, unit) in zip(axes, metrics):
        others = [stat(block, p, metric) for p in order if p != "fifo"]
        xmax = max((s["mean"] + s["ci95"]) * scale for s in others) * 1.18
        for y, p in zip(ys, order):
            s = stat(block, p, metric)
            m, h = s["mean"] * scale, s["ci95"] * scale
            if m - h > xmax:  # off the scale: say so with the real number instead of squashing the rest
                ax.annotate(f"{LABEL[p]}: {m:,.0f} ± {h:,.0f} {unit}  →", xy=(xmax, y), ha="right", va="center",
                            fontsize=8.5, color=INK2)
                continue
            ax.plot([max(m - h, 0), m + h], [y, y], color=COLOR[p], linewidth=2, solid_capstyle="round")
            ax.plot(m, y, "o", color=COLOR[p], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(m + h + xmax * 0.02, y, f"{m:,.0f}" if unit == "s" else f"{m:.1f}", va="center",
                    fontsize=8.5, color=INK)
        style(ax, grid_axis="x")
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].set_yticks(ys, [LABEL[p] for p in order], fontsize=9.5, color=INK)
    n_seeds = len(block["seeds"])
    fig.suptitle(f"All 7 policies at load {block['load']:.1f}, {block['jobs']:,} jobs per seed: "
                 f"dot = mean of {n_seeds} seeds, bar = 95% CI (t-dist.)", x=0.01, ha="left", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / "policy_comparison_load1.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------------
def paired_seeds(res: dict, out: Path, key: str = "high_load_1.0") -> Path:
    """Per-seed slope charts for the two trade-offs: greedy vs EASY, and drf_topo vs +preemption."""
    block = res[key]
    pairs = [
        ("fifo_greedy_backfill", "easy_backfill", "big_job_max_wait_s", 1 / 3600,
         "Worst 16/32-GPU job wait (hours)"),
        ("drf_topo", "drf_topo_preempt", "interactive_p99_wait_s", 1.0, "Inference p99 wait (seconds)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (a, b, metric, scale, title) in zip(axes, pairs):
        sa = {r["seed"]: r[metric] * scale for r in block["policies"][a]["per_seed"]}
        sb = {r["seed"]: r[metric] * scale for r in block["policies"][b]["per_seed"]}
        seeds = sorted(sa)
        lower = sum(sb[s] < sa[s] for s in seeds)
        for s in seeds:
            ax.plot([0, 1], [sa[s], sb[s]], color=AXIS, linewidth=1.2, zorder=1)
        ax.scatter([0] * len(seeds), [sa[s] for s in seeds], s=64, color=COLOR[a], edgecolors=SURFACE,
                   linewidths=2, zorder=2)
        ax.scatter([1] * len(seeds), [sb[s] for s in seeds], s=64, color=COLOR[b], edgecolors=SURFACE,
                   linewidths=2, zorder=2)
        for x, p in ((0, a), (1, b)):
            m = stat(block, p, metric)["mean"] * scale
            ax.plot([x - 0.12, x + 0.12], [m, m], color=INK, linewidth=2, zorder=3)
        style(ax)
        ax.set_xticks([0, 1], [LABEL[a], LABEL[b]], fontsize=10, color=INK)
        ax.set_xlim(-0.4, 1.4)
        top = max(max(sa.values()), max(sb.values()))
        ax.set_ylim(0, top * 1.15)
        ax.set_title(title, loc="left", fontsize=11)
        ax.text(0.5, 0.985, f"lower under {LABEL[b]} in {lower}/{len(seeds)} seeds", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=INK2)
    fig.suptitle(f"Same workload, same failures: each grey line is one seed (load {block['load']:.1f}, "
                 f"{block['jobs']:,} jobs); black tick = mean", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "paired_seeds_load1.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=ROOT / "results.json")
    ap.add_argument("--outdir", type=Path, default=ROOT / "docs" / "img")
    args = ap.parse_args()
    res = json.loads(args.input.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    for fn in (load_sweep, policy_comparison, paired_seeds):
        p = fn(res, args.outdir)
        print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
