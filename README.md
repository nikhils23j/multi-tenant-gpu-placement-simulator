# Multi-tenant GPU Placement Simulator

**Status:** Personal portfolio simulation built for the Inferact Cloud Orchestration application on 30 October 2026. It models GPU capacity; it does not operate physical GPUs, Kubernetes, or vLLM.

The code and evaluation were prepared with Codex for Nikhil's review. Nikhil should study and modify it before presenting it as his own implementation work.

## Problem and design

When large batch jobs and short inference jobs share GPU nodes, strict FIFO can block a short job behind a large queued job even when some capacity could run the short one. The simulator compares strict FIFO/first-fit, FIFO with backfilling, and a weighted-fair policy that assigns larger service weights to two interactive tenants and uses best-fit placement to reduce GPU fragmentation. All policies process the same deterministic, generated arrivals and durations on 32 logical 8-GPU nodes. Events advance to the next arrival or completion, so idle time and queue waits are measured in simulated seconds.

The weighted-fair rank is each tenant's virtual finish time: `max(previous_finish, arrival) + GPU_seconds / weight`. Jobs with lower rank are considered first, while capacity-aware scanning can bypass a job that cannot fit on any current node. Four tenants are modeled: two interactive with weight 4 each and two batch with weight 1 each. The main outcome is p95 **interactive** queue wait, with batch p95 reported to expose tradeoffs. Every dispatch checks node capacity and all jobs must eventually complete.

## Reproduce

Python 3.10+ standard library only:

```sh
python3 -m unittest -v test_simulator.py
python3 run_demo.py --jobs 10000 --nodes 32 --gpus-per-node 8 --arrival-rate 3.2 --seed 27 --output results.json
```

The JSON includes environment, seed, workload, strict FIFO, FIFO with backfilling, and weighted-fair waits, utilization, capacity checks, and whether the original 25% p95 reduction target was met. The backfilling baseline is important: it separates the gain from fitting around a large queued job from the further gain from tenant weighting. Change the arrival rate or seed to examine sensitivity rather than treating one generated trace as a universal performance claim.

### Measured local result

On Python 3.10.8/macOS arm64 with seed 27, arrival rate 3.2 jobs per simulated second, 10,000 generated jobs, and 32 logical 8-GPU nodes, p95 interactive queue wait was **168.225 s** for strict FIFO, **1.584 s** for FIFO with backfilling, and **0.167 s** for weighted fair scheduling. The weighted policy reduced p95 by **89.46% versus the stronger backfilling baseline** and completed all 10,000 jobs with zero capacity overcommit. Batch p95 increased from **69.840 s** under FIFO with backfilling to **93.285 s** under weighted fair scheduling; this is the principal tradeoff. Seeds 28 and 29 gave backfilling/weighted interactive p95 pairs of 1.722/0.095 s and 2.478/0.113 s on the same workload settings. These are simulated seconds and generated jobs, not measured production inference latency.

## Limits

Node GPU units are fungible; the model omits memory limits, topology, preemption, runtime interference, network transfer, and hardware failures. The scheduler is a single-process study and does not claim real cluster-management experience or production inference latency. Weighted fairness may improve interactive waits at the cost of batch waits; inspect both metrics.
