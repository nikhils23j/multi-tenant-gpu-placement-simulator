"""Full evaluation: every policy x many seeds x several loads. Writes results.json.

    python3 run_eval.py                 # full run (~1-3 min on an 8-core M2)
    python3 run_eval.py --quick         # 3 seeds, smaller sweep, for poking around
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import simulator as S
from simulator import POLICIES, ClusterSpec, simulate
from workload import generate_failures, generate_workload, offered_load

# two-sided 95% t critical values, index = degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 29: 2.045}

HEADLINE = ["interactive_p50_wait_s", "interactive_p95_wait_s", "interactive_p99_wait_s", "interactive_max_wait_s",
            "interactive_frac_wait_over_60s", "batch_p50_wait_s", "batch_p95_wait_s", "batch_p99_wait_s",
            "batch_max_wait_s", "big_job_p95_wait_s", "big_job_max_wait_s", "gpu_utilization", "stranded_gpus_avg",
            "jain_fairness_slowdown", "preemptions", "failure_kills", "wasted_gpu_hours",
            "cross_island_frac_small_gangs", "overcommit_events", "jobs_completed"]


def ci(values):
    vals = [v for v in values if not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    mean = statistics.fmean(vals)
    half = T95.get(n - 1, 1.96) * statistics.stdev(vals) / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": mean, "ci95": half, "min": min(vals), "max": max(vals), "n": n}


def run_one(args):
    n_jobs, seed, load, policy, nodes, mtbf = args
    cluster = ClusterSpec(nodes=nodes)
    jobs = generate_workload(n_jobs, seed, load, cluster)
    fails = generate_failures(seed, cluster, jobs[-1].arrival, mtbf_days=mtbf)
    t0 = time.perf_counter()
    m = simulate(jobs, policy, cluster, fails).metrics
    m["sim_wall_s"] = time.perf_counter() - t0
    m["offered_load_realized"] = offered_load(jobs, cluster)
    m["node_failures"] = len(fails)
    m["arrival_span_days"] = jobs[-1].arrival / 86400
    return (n_jobs, seed, load, policy), m


def tasks_for(n_jobs, seeds, load, nodes, mtbf):
    return [(n_jobs, s, load, p, nodes, mtbf) for s in seeds for p in POLICIES]


def experiment(raw, n_jobs, seeds, load):
    out = {"jobs": n_jobs, "load": load, "seeds": list(seeds), "policies": {}}
    for p in POLICIES:
        per_seed = [raw[(n_jobs, s, load, p)] for s in seeds]
        for m in per_seed:
            assert m["jobs_completed"] == n_jobs and m["overcommit_events"] == 0, (p, m)
        out["policies"][p] = {
            "summary": {k: ci([m[k] for m in per_seed]) for k in HEADLINE},
            "per_seed": [{k: m[k] for k in HEADLINE + ["offered_load_realized", "node_failures",
                                                        "arrival_span_days", "tenant_mean_slowdown"]} | {"seed": s}
                         for s, m in zip(seeds, per_seed)],
        }
    return out


def paired(exp, a, b, metric):
    """Per-seed difference b - a (same workload + failure trace), with CI and win count."""
    xa = [r[metric] for r in exp["policies"][a]["per_seed"]]
    xb = [r[metric] for r in exp["policies"][b]["per_seed"]]
    diffs = [y - x for x, y in zip(xa, xb)]
    return {"baseline": a, "policy": b, "metric": metric, "diff": ci(diffs),
            "seeds_where_policy_lower": sum(d < 0 for d in diffs), "seeds_equal": sum(d == 0 for d in diffs),
            "n": len(diffs)}


def fmt(s, scale=1.0, nd=0):
    return f"{s['mean'] / scale:.{nd}f} ± {s['ci95'] / scale:.{nd}f}"


def print_table(exp, title):
    print(f"\n## {title}: load {exp['load']}, {exp['jobs']} jobs, seeds {exp['seeds'][0]}-{exp['seeds'][-1]}"
          f" (mean ± 95% CI)")
    print("| policy | int p95 (s) | int p99 (s) | int >60s (%) | batch p50 (h) | batch p95 (h) | big-job max (h) "
          "| util (%) | stranded GPUs | Jain | preempt | wasted GPU-h |")
    print("|---|" + "---|" * 11)
    for p, d in exp["policies"].items():
        s = d["summary"]
        print(f"| {p} | {fmt(s['interactive_p95_wait_s'], 1, 1)} | {fmt(s['interactive_p99_wait_s'], 1, 0)} "
              f"| {fmt(s['interactive_frac_wait_over_60s'], 0.01, 2)} | {fmt(s['batch_p50_wait_s'], 3600, 2)} "
              f"| {fmt(s['batch_p95_wait_s'], 3600, 1)} | {fmt(s['big_job_max_wait_s'], 3600, 0)} "
              f"| {fmt(s['gpu_utilization'], 0.01, 1)} | {fmt(s['stranded_gpus_avg'], 1, 1)} "
              f"| {fmt(s['jain_fairness_slowdown'], 1, 3)} | {fmt(s['preemptions'], 1, 0)} "
              f"| {fmt(s['wasted_gpu_hours'], 1, 0)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=10_000)
    ap.add_argument("--sweep-jobs", type=int, default=5_000)
    ap.add_argument("--nodes", type=int, default=32)
    ap.add_argument("--mtbf-days", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--output", default="results.json")
    a = ap.parse_args()
    seeds = range(1, (3 if a.quick else a.seeds) + 1)
    sweep_loads = [0.6, 0.8, 1.0] if a.quick else [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    t0 = time.perf_counter()
    specs = [(a.jobs, 0.8), (a.jobs, 1.0)] + [(a.sweep_jobs, ld) for ld in sweep_loads]
    tasks = [t for n, ld in specs for t in tasks_for(n, seeds, ld, a.nodes, a.mtbf_days)]
    tasks.sort(key=lambda t: -(t[0] * t[2]))  # biggest/heaviest first for better packing
    with ProcessPoolExecutor(a.workers) as pool:
        raw = dict(pool.map(run_one, tasks, chunksize=1))
    main_exp = experiment(raw, a.jobs, seeds, 0.8)
    high_exp = experiment(raw, a.jobs, seeds, 1.0)
    sweep = [experiment(raw, a.sweep_jobs, seeds, ld) for ld in sweep_loads]
    wall = time.perf_counter() - t0

    comparisons = []
    for exp in (main_exp, high_exp):
        for base, pol, metric in [
            ("fifo", "fifo_greedy_backfill", "batch_p95_wait_s"),
            ("fifo_greedy_backfill", "easy_backfill", "big_job_max_wait_s"),
            ("easy_backfill", "easy_topo", "batch_p95_wait_s"),
            ("easy_backfill", "easy_topo", "cross_island_frac_small_gangs"),
            ("easy_topo", "drf_topo", "interactive_p99_wait_s"),
            ("drf_topo", "drf_topo_preempt", "interactive_p99_wait_s"),
            ("drf_topo", "drf_topo_preempt", "interactive_max_wait_s"),
            ("drf_topo", "drf_topo_preempt", "batch_p95_wait_s"),
            ("drf_topo", "drf_topo_preempt", "big_job_max_wait_s"),
            ("drf_topo", "drf_topo_preempt", "wasted_gpu_hours"),
            ("wfq_topo", "drf_topo", "batch_p95_wait_s"),
        ]:
            comparisons.append({"load": exp["load"], **paired(exp, base, pol, metric)})

    results = {
        "scope": "Discrete-event simulation with a generated workload. Simulated seconds; no real GPUs, "
                 "no trace replay, no production system.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "cpu_count": os.cpu_count(), "workers": a.workers},
        "commands": ["python3 -m unittest -v test_simulator.py", "python3 " + " ".join(sys.argv)],
        "eval_wall_clock_s": wall,
        "model_constants": {k: getattr(S, k) for k in ("GPUS_PER_NODE", "ISLAND_SIZE", "SLICES_PER_GPU",
                            "MEM_UNITS_PER_GPU", "MIG_PROFILES", "TOPOLOGY_PENALTY", "CHECKPOINT_INTERVAL",
                            "RESTART_OVERHEAD", "MAX_PREEMPTIONS", "AGE_LIMIT", "SCAN_LIMIT")},
        "cluster": {"nodes": a.nodes, "gpus": a.nodes * 8, "frac_80gb_nodes": 0.5, "node_mtbf_days": a.mtbf_days},
        "policies": {k: vars(v) for k, v in POLICIES.items()},
        "main_load_0.8": main_exp,
        "high_load_1.0": high_exp,
        "load_sweep": sweep,
        "paired_comparisons": comparisons,
    }
    with open(a.output, "w") as f:
        json.dump(results, f, indent=1, default=float)
        f.write("\n")

    print_table(main_exp, "Main")
    print_table(high_exp, "High load")
    print("\n## Load sweep (mean over seeds): interactive p99 s / batch p95 h / big-job max h / util %")
    print("| load | " + " | ".join(POLICIES) + " |")
    print("|---|" + "---|" * len(POLICIES))
    for exp in sweep:
        cells = []
        for p in POLICIES:
            s = exp["policies"][p]["summary"]
            cells.append(f"{s['interactive_p99_wait_s']['mean']:.0f} / {s['batch_p95_wait_s']['mean'] / 3600:.1f} / "
                         f"{s['big_job_max_wait_s']['mean'] / 3600:.0f} / {100 * s['gpu_utilization']['mean']:.0f}")
        print(f"| {exp['load']} | " + " | ".join(cells) + " |")
    print("\n## Paired per-seed differences (policy - baseline)")
    for c in comparisons:
        d = c["diff"]
        print(f"load {c['load']}: {c['policy']} vs {c['baseline']} {c['metric']}: {d['mean']:.3f} ± {d['ci95']:.3f}"
              f" (lower in {c['seeds_where_policy_lower']}/{c['n']} seeds, equal in {c['seeds_equal']})")
    print(f"\nwall clock {wall:.1f} s -> {a.output}")


if __name__ == "__main__":
    main()
