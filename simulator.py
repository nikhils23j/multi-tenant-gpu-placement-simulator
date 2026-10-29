"""Deterministic discrete-event simulation of shared GPU-node scheduling."""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import random
from statistics import mean


WEIGHTS = {"interactive-a": 4, "interactive-b": 4, "batch-a": 1, "batch-b": 1}


@dataclass(frozen=True)
class Job:
    id: int
    arrival: float
    duration: float
    gpus: int
    tenant: str

    @property
    def kind(self) -> str:
        return "interactive" if self.tenant.startswith("interactive") else "batch"


@dataclass
class Node:
    id: int
    capacity: int = 8
    free: int = 8


@dataclass
class Result:
    policy: str
    completed: int
    waits: dict[int, float]
    starts: dict[int, float]
    peak_used_gpus: int
    overcommit_events: int
    makespan: float
    busy_gpu_seconds: float
    per_tenant_p95: dict[str, float] = field(default_factory=dict)

    def metrics(self, jobs: list[Job], nodes: int, capacity: int) -> dict:
        by_id = {job.id: job for job in jobs}
        interactive = [wait for jid, wait in self.waits.items() if by_id[jid].kind == "interactive"]
        batch = [wait for jid, wait in self.waits.items() if by_id[jid].kind == "batch"]
        per_tenant = {tenant: percentile([self.waits[jid] for jid in self.waits if by_id[jid].tenant == tenant], 95)
                      for tenant in WEIGHTS}
        return {
            "policy": self.policy, "jobs_completed": self.completed,
            "interactive_p95_wait_s": round(percentile(interactive, 95), 3),
            "interactive_mean_wait_s": round(mean(interactive), 3),
            "batch_p95_wait_s": round(percentile(batch, 95), 3),
            "tenant_p95_wait_s": {k: round(v, 3) for k, v in per_tenant.items()},
            "peak_used_gpus": self.peak_used_gpus,
            "overcommit_events": self.overcommit_events,
            "makespan_s": round(self.makespan, 3),
            "gpu_utilization_fraction": round(self.busy_gpu_seconds / (nodes * capacity * self.makespan), 4),
        }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("percentile of empty sample")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def generate_jobs(count: int = 10_000, seed: int = 27, arrival_rate: float = 3.2) -> list[Job]:
    """Generated public-data-free trace; seconds and GPU units are simulated."""
    if count <= 0 or arrival_rate <= 0:
        raise ValueError("count and arrival_rate must be positive")
    rng = random.Random(seed)
    arrival = 0.0
    jobs = []
    for jid in range(count):
        arrival += rng.expovariate(arrival_rate)
        if rng.random() < 0.70:
            tenant = rng.choice(("interactive-a", "interactive-b"))
            gpus = rng.choice((1, 1, 1, 2))
            duration = rng.uniform(1.0, 5.0)
        else:
            tenant = rng.choice(("batch-a", "batch-b"))
            gpus = rng.choice((2, 4, 4, 6, 8))
            duration = rng.uniform(25.0, 80.0)
        jobs.append(Job(jid, arrival, duration, gpus, tenant))
    return jobs


def simulate(jobs: list[Job], policy: str, node_count: int = 32, node_capacity: int = 8) -> Result:
    if policy not in ("fifo", "fifo_backfill", "weighted_fair"):
        raise ValueError("policy must be fifo, fifo_backfill, or weighted_fair")
    if node_count <= 0 or node_capacity <= 0:
        raise ValueError("node count and capacity must be positive")
    jobs = sorted(jobs, key=lambda j: (j.arrival, j.id))
    if any(j.gpus <= 0 or j.gpus > node_capacity or j.duration <= 0 for j in jobs):
        raise ValueError("every job must fit a node and have positive duration")
    nodes = [Node(i, node_capacity, node_capacity) for i in range(node_count)]
    running: list[tuple[float, int, int, int]] = []  # finish, job id, node id, GPU units
    pending: list[tuple[float, int, Job]] = []  # virtual finish or arrival, id, job
    last_virtual_finish = {tenant: 0.0 for tenant in WEIGHTS}
    waits: dict[int, float] = {}
    starts: dict[int, float] = {}
    peak_used = overcommit = 0
    busy_gpu_seconds = 0.0
    idx = 0
    now = 0.0
    while idx < len(jobs) or running or pending:
        next_arrival = jobs[idx].arrival if idx < len(jobs) else math.inf
        next_finish = running[0][0] if running else math.inf
        if next_arrival == math.inf and next_finish == math.inf and pending:
            raise RuntimeError("queued jobs cannot fit any node")
        now = min(next_arrival, next_finish)
        while running and running[0][0] <= now + 1e-10:
            _, _, node_id, gpu_units = heapq.heappop(running)
            nodes[node_id].free += gpu_units
            assert nodes[node_id].free <= node_capacity
        while idx < len(jobs) and jobs[idx].arrival <= now + 1e-10:
            job = jobs[idx]
            if policy != "weighted_fair":
                rank = job.arrival
            else:
                rank = max(last_virtual_finish[job.tenant], job.arrival) + job.gpus * job.duration / WEIGHTS[job.tenant]
                last_virtual_finish[job.tenant] = rank
            pending.append((rank, job.id, job))
            idx += 1
        if policy == "weighted_fair":
            pending.sort(key=lambda entry: (entry[0], entry[1]))
        while pending:
            selected = None
            selected_node = None
            for position, (_, _, job) in enumerate(pending):
                fitting = [node for node in nodes if node.free >= job.gpus]
                if fitting:
                    selected = position
                    if policy != "weighted_fair":
                        selected_node = fitting[0]
                    else:
                        # Best-fit leaves the smallest residual fragment on a node.
                        selected_node = min(fitting, key=lambda node: (node.free - job.gpus, node.id))
                    break
                if policy == "fifo":
                    break  # strict head-of-line blocking
            if selected is None:
                break
            _, _, job = pending.pop(selected)
            node = selected_node
            node.free -= job.gpus
            if node.free < 0:
                overcommit += 1
            peak_used = max(peak_used, node_count * node_capacity - sum(n.free for n in nodes))
            waits[job.id] = max(0.0, now - job.arrival)
            starts[job.id] = now
            busy_gpu_seconds += job.gpus * job.duration
            heapq.heappush(running, (now + job.duration, job.id, node.id, job.gpus))
    if len(waits) != len(jobs):
        raise AssertionError("not all jobs completed")
    return Result(policy, len(waits), waits, starts, peak_used, overcommit, now, busy_gpu_seconds)
