"""Quick single-seed demo: one workload, every policy, one table. Good for poking at knobs.

    python3 run_demo.py --load 1.0 --seed 3 --jobs 4000
    python3 run_demo.py --policies easy_topo drf_topo_preempt --mtbf-days 0   # no node failures
"""
from __future__ import annotations

import argparse
import time

from simulator import POLICIES, ClusterSpec, simulate
from workload import generate_failures, generate_workload, offered_load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--load", type=float, default=0.9, help="offered GPU-time / cluster capacity")
    ap.add_argument("--nodes", type=int, default=32)
    ap.add_argument("--frac-80gb", type=float, default=0.5)
    ap.add_argument("--mtbf-days", type=float, default=10.0, help="per-node mean time between failures; 0 = off")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    a = ap.parse_args()

    cluster = ClusterSpec(a.nodes, a.frac_80gb)
    jobs = generate_workload(a.jobs, a.seed, a.load, cluster)
    fails = generate_failures(a.seed, cluster, jobs[-1].arrival, mtbf_days=a.mtbf_days)
    print(f"{len(jobs)} jobs over {jobs[-1].arrival / 86400:.1f} simulated days, offered load "
          f"{offered_load(jobs, cluster):.2f}, {cluster.gpus} GPUs, {len(fails)} node failures\n")
    print(f"{'policy':22s} {'int p99 s':>10s} {'int >60s%':>9s} {'batch p95 h':>11s} {'big max h':>9s} "
          f"{'util%':>6s} {'strand':>6s} {'jain':>5s} {'preempt':>7s} {'xisland%':>8s} {'sim s':>6s}")
    for p in a.policies:
        t0 = time.perf_counter()
        m = simulate(jobs, p, cluster, fails).metrics
        assert m["jobs_completed"] == len(jobs) and m["overcommit_events"] == 0
        print(f"{p:22s} {m['interactive_p99_wait_s']:10.0f} {100 * m['interactive_frac_wait_over_60s']:9.2f} "
              f"{m['batch_p95_wait_s'] / 3600:11.1f} {m['big_job_max_wait_s'] / 3600:9.0f} "
              f"{100 * m['gpu_utilization']:6.1f} {m['stranded_gpus_avg']:6.1f} {m['jain_fairness_slowdown']:5.2f} "
              f"{m['preemptions']:7d} {100 * m['cross_island_frac_small_gangs']:8.1f} {time.perf_counter() - t0:6.1f}")


if __name__ == "__main__":
    main()
