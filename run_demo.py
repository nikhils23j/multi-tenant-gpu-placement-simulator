"""Run equal-workload FIFO and weighted-fair comparisons and write results."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from simulator import generate_jobs, simulate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=10_000)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--arrival-rate", type=float, default=3.2)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    jobs = generate_jobs(args.jobs, args.seed, args.arrival_rate)
    fifo = simulate(jobs, "fifo", args.nodes, args.gpus_per_node)
    backfill = simulate(jobs, "fifo_backfill", args.nodes, args.gpus_per_node)
    fair = simulate(jobs, "weighted_fair", args.nodes, args.gpus_per_node)
    fifo_metrics = fifo.metrics(jobs, args.nodes, args.gpus_per_node)
    backfill_metrics = backfill.metrics(jobs, args.nodes, args.gpus_per_node)
    fair_metrics = fair.metrics(jobs, args.nodes, args.gpus_per_node)
    baseline = fifo_metrics["interactive_p95_wait_s"]
    improvement = 100 * (baseline - fair_metrics["interactive_p95_wait_s"]) / baseline if baseline else 0.0
    backfill_baseline = backfill_metrics["interactive_p95_wait_s"]
    backfill_improvement = (100 * (backfill_baseline - fair_metrics["interactive_p95_wait_s"]) / backfill_baseline
                            if backfill_baseline else 0.0)
    result = {
        "scope": "discrete-event simulation; no physical GPU or Kubernetes cluster",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "workload": {
            "jobs": args.jobs, "nodes": args.nodes, "gpus_per_node": args.gpus_per_node,
            "arrival_rate_per_s": args.arrival_rate, "seed": args.seed,
            "interactive_fraction_target": 0.70,
            "interactive_duration_s": [1, 5], "batch_duration_s": [25, 80],
        },
        "fifo": fifo_metrics, "fifo_backfill": backfill_metrics, "weighted_fair": fair_metrics,
        "interactive_p95_wait_improvement_pct": round(improvement, 2),
        "interactive_p95_wait_improvement_vs_backfill_pct": round(backfill_improvement, 2),
        "target_met": improvement >= 25 and fair.overcommit_events == 0,
    }
    assert fifo.completed == backfill.completed == fair.completed == args.jobs
    assert fifo.overcommit_events == backfill.overcommit_events == fair.overcommit_events == 0
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
