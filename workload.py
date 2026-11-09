"""Synthetic workload + failure traces.

The shapes are *inspired by* published cluster-trace studies (Microsoft Philly,
Jeon et al., ATC'19; Alibaba PAI, Weng et al., NSDI'22): most jobs are small,
a few multi-node gang jobs eat a large share of GPU-time, durations are heavy
tailed, and many inference jobs need only a fraction of a GPU. Nothing here is
fitted to or replayed from those traces; every parameter is a hand-picked guess.
"""
from __future__ import annotations

import math
import random

from simulator import GPUS_PER_NODE, SLICES_PER_GPU, ClusterSpec, Job, mig_profile

INTERACTIVE_FRACTION = 0.6
DIURNAL_AMPLITUDE = 0.5            # arrival rate swings between 0.5x and 1.5x of the mean
DAY = 86_400.0

# tenant -> (gang-size distribution, median duration s, lognormal sigma, P(needs >40 GB per GPU))
TRAINING = {
    "train-a": ({1: 0.20, 2: 0.10, 4: 0.20, 8: 0.25, 16: 0.15, 32: 0.10}, 7200.0, 1.4, 0.5),
    "train-b": ({1: 0.60, 2: 0.20, 4: 0.15, 8: 0.05}, 1800.0, 1.5, 0.25),
}
TRAIN_CLIP = (120.0, 72 * 3600.0)
INFER_MEDIAN_S, INFER_SIGMA, INFER_CLIP = 120.0, 1.0, (5.0, 3600.0)
INFER_MEM_MEDIAN_GB, INFER_MEM_SIGMA, INFER_MEM_CLIP = 10.0, 0.9, (2.0, 80.0)


def _lognormal(rng: random.Random, median: float, sigma: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, rng.lognormvariate(math.log(median), sigma)))


def _choice(rng: random.Random, dist: dict) -> int:
    u, acc = rng.random(), 0.0
    for k, p in dist.items():
        acc += p
        if u < acc:
            return k
    return k


def gpu_equivalent(gpus: int, mem_gb: float) -> float:
    if gpus:
        return float(gpus)
    c, _ = mig_profile(mem_gb, 80.0)
    return c / SLICES_PER_GPU


def generate_workload(n_jobs: int, seed: int, load: float, cluster: ClusterSpec = ClusterSpec(),
                      diurnal_amplitude: float = DIURNAL_AMPLITUDE) -> list[Job]:
    """Generate n_jobs whose GPU-seconds demand averages `load` x cluster capacity over the arrival span."""
    if n_jobs <= 0 or load <= 0 or not 0 <= diurnal_amplitude < 1:
        raise ValueError("n_jobs and load must be positive, amplitude in [0, 1)")
    rng = random.Random(seed)
    attrs = []
    for _ in range(n_jobs):
        if rng.random() < INTERACTIVE_FRACTION:
            tenant = rng.choice(("infer-a", "infer-b"))
            dur = _lognormal(rng, INFER_MEDIAN_S, INFER_SIGMA, *INFER_CLIP)
            mem = _lognormal(rng, INFER_MEM_MEDIAN_GB, INFER_MEM_SIGMA, *INFER_MEM_CLIP)
            gpus, est = 0, dur * rng.uniform(1.0, 2.0)
        else:
            tenant = rng.choice(tuple(TRAINING))
            dist, med, sigma, p_big = TRAINING[tenant]
            gpus = _choice(rng, dist)
            dur = _lognormal(rng, med, sigma, *TRAIN_CLIP)
            mem = rng.uniform(41.0, 78.0) if rng.random() < p_big else rng.uniform(8.0, 40.0)
            est = dur * rng.uniform(1.0, 3.0)
        attrs.append((tenant, dur, est, gpus, round(mem, 2)))
    demand = sum(gpu_equivalent(g, m) * d for _, d, _, g, m in attrs)
    mean_rate = n_jobs / (demand / (load * cluster.gpus))
    # Non-homogeneous Poisson arrivals by thinning: rate(t) = mean * (1 - A cos(2 pi t / day)).
    t, arrivals, lam_max = 0.0, [], mean_rate * (1 + diurnal_amplitude)
    while len(arrivals) < n_jobs:
        t += rng.expovariate(lam_max)
        if rng.random() * (1 + diurnal_amplitude) <= 1 - diurnal_amplitude * math.cos(2 * math.pi * t / DAY):
            arrivals.append(t)
    return [Job(i, tenant, arrivals[i], dur, est, gpus, mem) for i, (tenant, dur, est, gpus, mem) in enumerate(attrs)]


def offered_load(jobs: list[Job], cluster: ClusterSpec) -> float:
    span = max(j.arrival for j in jobs)
    return sum(gpu_equivalent(j.gpus, j.mem_gb) * j.duration for j in jobs) / (cluster.gpus * span)


def generate_failures(seed: int, cluster: ClusterSpec, horizon: float, mtbf_days: float = 10.0,
                      mean_repair_h: float = 2.0) -> list[tuple[float, int, float]]:
    """Per-node exponential failures; identical for every policy given the same seed."""
    if mtbf_days <= 0:
        return []
    rng = random.Random(seed * 1_000_003 + 17)
    events = []
    for nid in range(cluster.nodes):
        t = 0.0
        while True:
            t += rng.expovariate(1.0 / (mtbf_days * DAY))
            if t >= horizon:
                break
            repair = 900.0 + rng.expovariate(1.0 / (mean_repair_h * 3600.0))
            events.append((t, nid, repair))
            t += repair
    return sorted(events)


__all__ = ["generate_workload", "generate_failures", "offered_load", "gpu_equivalent", "GPUS_PER_NODE"]
