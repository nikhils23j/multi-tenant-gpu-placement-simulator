"""Discrete-event simulator for a shared, multi-tenant GPU cluster.

Everything here is simulated: seconds are simulated seconds, GPUs are integer
counters, and the workload comes from workload.py. No real hardware is touched.

Model summary
-------------
* Nodes have 8 GPUs split into two NVLink "islands" of 4. A node is either an
  80 GB-GPU node or a 40 GB-GPU node.
* Each GPU has 7 compute slices and 8 memory units (A100-style MIG). Inference
  jobs get a MIG profile (1g/2g/3g/7g) sized by their memory need. Training
  jobs take whole GPUs (all 7 slices + 8 memory units).
* Training jobs are gangs: all requested GPUs are allocated at the same instant
  or none are. <=8 GPUs must sit on one node; 16/32 GPUs take whole nodes.
* Splitting a <=4-GPU training job across islands when one island would have
  done makes it run TOPOLOGY_PENALTY times slower (an assumption, not measured).
* Batch jobs checkpoint every CHECKPOINT_INTERVAL seconds of work. Preemption or
  a node failure throws away work since the last checkpoint and adds
  RESTART_OVERHEAD seconds. Inference jobs restart from scratch.
"""
from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass, field

GPUS_PER_NODE = 8
ISLAND_SIZE = 4
SLICES_PER_GPU = 7
MEM_UNITS_PER_GPU = 8
# (compute slices, memory eighths) for 1g, 2g, 3g, 7g profiles, A100 style.
MIG_PROFILES = ((1, 1), (2, 2), (3, 4), (7, 8))

TOPOLOGY_PENALTY = 1.3
CHECKPOINT_INTERVAL = 1800.0
RESTART_OVERHEAD = 120.0
MAX_PREEMPTIONS = 2
AGE_LIMIT = 6 * 3600.0
SCAN_LIMIT = 500
SLOWDOWN_FLOOR = 60.0  # bounded slowdown denominator, seconds
EPS = 1e-9

# tenant -> (kind, weight)
TENANTS = {
    "infer-a": ("interactive", 4.0),
    "infer-b": ("interactive", 4.0),
    "train-a": ("batch", 1.0),
    "train-b": ("batch", 1.0),
}


@dataclass(frozen=True)
class Job:
    id: int
    tenant: str
    arrival: float
    duration: float      # work in seconds at full speed
    est: float           # user-supplied runtime estimate (>= duration)
    gpus: int            # whole GPUs; 0 means a fractional (MIG) inference job
    mem_gb: float        # per-GPU memory for training, total for inference

    @property
    def kind(self) -> str:
        return TENANTS[self.tenant][0]


@dataclass(frozen=True)
class Policy:
    name: str
    order: str           # fifo | wfq | drf
    backfill: str        # none | greedy | easy
    placement: str       # first_fit | topo_best_fit
    preempt: bool = False
    aging: bool = False


POLICIES = {
    "fifo": Policy("fifo", "fifo", "none", "first_fit"),
    "fifo_greedy_backfill": Policy("fifo_greedy_backfill", "fifo", "greedy", "first_fit"),
    "easy_backfill": Policy("easy_backfill", "fifo", "easy", "first_fit"),
    "easy_topo": Policy("easy_topo", "fifo", "easy", "topo_best_fit"),
    "wfq_topo": Policy("wfq_topo", "wfq", "easy", "topo_best_fit", aging=True),
    "drf_topo": Policy("drf_topo", "drf", "easy", "topo_best_fit", aging=True),
    "drf_topo_preempt": Policy("drf_topo_preempt", "drf", "easy", "topo_best_fit", preempt=True, aging=True),
}


@dataclass(frozen=True)
class ClusterSpec:
    nodes: int = 32
    frac_80gb: float = 0.5

    def node_mem(self) -> list[float]:
        n80 = round(self.nodes * self.frac_80gb)
        return [80.0 if i < n80 else 40.0 for i in range(self.nodes)]

    @property
    def gpus(self) -> int:
        return self.nodes * GPUS_PER_NODE


def mig_profile(mem_gb: float, gpu_mem_gb: float):
    """Smallest (slices, mem_units) profile holding mem_gb on this GPU type, or None."""
    need = math.ceil(mem_gb / (gpu_mem_gb / MEM_UNITS_PER_GPU) - EPS)
    for c, m in MIG_PROFILES:
        if m >= need:
            return c, m
    return None


def percentile(values, pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    rank = (len(s) - 1) * pct / 100.0
    lo, hi = math.floor(rank), math.ceil(rank)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def jain(xs) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) ** 2 / (len(xs) * sum(x * x for x in xs))


class Node:
    __slots__ = ("id", "mem_gb", "up", "free_c", "free_m", "free_whole", "jobs")

    def __init__(self, nid: int, mem_gb: float):
        self.id = nid
        self.mem_gb = mem_gb
        self.up = True
        self.free_c = [SLICES_PER_GPU] * GPUS_PER_NODE
        self.free_m = [MEM_UNITS_PER_GPU] * GPUS_PER_NODE
        self.free_whole = GPUS_PER_NODE
        self.jobs: set[int] = set()


class _Run:
    __slots__ = ("remaining", "est_remaining", "first_start", "finish", "alloc", "slowdown",
                 "run_start", "epoch", "preemptions", "kills", "rank", "shape", "queued")

    def __init__(self, job: Job):
        self.remaining = job.duration
        self.est_remaining = job.est
        self.first_start = None
        self.finish = None
        self.alloc = ()
        self.slowdown = 1.0
        self.run_start = 0.0
        self.epoch = 0
        self.preemptions = 0
        self.kills = 0
        self.rank = job.arrival
        self.queued = False
        if job.gpus == 0:
            self.shape = ("i", mig_profile(job.mem_gb, 80.0), mig_profile(job.mem_gb, 40.0))
        else:
            self.shape = ("t", job.gpus, job.mem_gb > 40.0)


@dataclass
class SimResult:
    policy: str
    metrics: dict
    audit: list = field(default_factory=list)
    starts: dict = field(default_factory=dict)


class Simulator:
    def __init__(self, jobs, policy: Policy | str, cluster: ClusterSpec = ClusterSpec(),
                 failures=(), audit: bool = False):
        self.policy = POLICIES[policy] if isinstance(policy, str) else policy
        if self.policy.order not in ("fifo", "wfq", "drf") or self.policy.backfill not in ("none", "greedy", "easy") \
                or self.policy.placement not in ("first_fit", "topo_best_fit"):
            raise ValueError(f"bad policy {self.policy}")
        self.cluster = cluster
        self.nodes = [Node(i, m) for i, m in enumerate(cluster.node_mem())]
        self.jobs = sorted(jobs, key=lambda j: (j.arrival, j.id))
        if len({j.id for j in self.jobs}) != len(self.jobs):
            raise ValueError("duplicate job ids")
        self.by_id = {j.id: j for j in self.jobs}
        self._validate()
        self.failures = sorted(failures)
        self.audit_on = audit
        self.audit: list = []
        self.run = {j.id: _Run(j) for j in self.jobs}
        self.now = 0.0
        # queues
        self.by_arrival: list[tuple[float, int]] = []
        self.tq: dict[str, list[tuple[float, int]]] = {t: [] for t in TENANTS}
        self.vfinish = {t: 0.0 for t in TENANTS}
        # running state
        self.running: dict[int, float] = {}       # jid -> estimated end
        self.completions: list = []
        self.repairs: list = []
        self.t_c = {t: 0 for t in TENANTS}
        self.t_m = {t: 0 for t in TENANTS}
        self.total_c = len(self.nodes) * GPUS_PER_NODE * SLICES_PER_GPU
        self.total_m = len(self.nodes) * GPUS_PER_NODE * MEM_UNITS_PER_GPU
        self.used_slices = 0
        self.nodes_up = len(self.nodes)
        self.version = 0          # bumps on any allocation or node state change; keys the caches below
        self._res_cache = None
        self._share_cache = (-1, None)
        self.completed = 0
        self.seq = 0
        # stats
        self.window_end = self.jobs[-1].arrival if self.jobs else 0.0
        self.t_prev = 0.0
        self.busy_int = 0.0
        self.stranded_int = 0.0
        self.down_int = 0.0
        self.stranded_now = 0
        self.overcommit = 0
        self.preemptions = 0
        self.failure_kills = 0
        self.wasted_gpu_s = 0.0
        self.small_gang_starts = 0
        self.cross_island_starts = 0
        self.max_queue = 0

    # ------------------------------------------------------------------ setup
    def _validate(self):
        mems = [n.mem_gb for n in self.nodes]
        for j in self.jobs:
            if j.duration <= 0 or j.est <= 0 or j.mem_gb <= 0 or j.tenant not in TENANTS:
                raise ValueError(f"bad job {j}")
            if j.gpus == 0:
                if not any(mig_profile(j.mem_gb, m) for m in mems):
                    raise ValueError(f"job {j.id} memory fits no GPU")
            elif j.gpus <= GPUS_PER_NODE:
                if not any(m >= j.mem_gb for m in mems):
                    raise ValueError(f"job {j.id} fits no node")
            else:
                if j.gpus % GPUS_PER_NODE or sum(m >= j.mem_gb for m in mems) < j.gpus // GPUS_PER_NODE:
                    raise ValueError(f"job {j.id} gang cannot fit the cluster")

    # -------------------------------------------------------------- resources
    def _apply(self, jid: int, alloc, sign: int):
        job = self.by_id[jid]
        self.version += 1
        for nid, g, c, m in alloc:
            node = self.nodes[nid]
            was_whole = node.free_c[g] == SLICES_PER_GPU
            node.free_c[g] -= sign * c
            node.free_m[g] -= sign * m
            if node.free_c[g] < 0 or node.free_m[g] < 0 or node.free_c[g] > SLICES_PER_GPU \
                    or node.free_m[g] > MEM_UNITS_PER_GPU:
                self.overcommit += 1
            is_whole = node.free_c[g] == SLICES_PER_GPU
            node.free_whole += is_whole - was_whole
            self.t_c[job.tenant] += sign * c
            self.t_m[job.tenant] += sign * m
            self.used_slices += sign * c
            if sign > 0:
                node.jobs.add(jid)
            else:
                node.jobs.discard(jid)
            if self.audit_on:
                self.audit.append((self.now, sign, jid, nid, g, c, m))

    # -------------------------------------------------------------- placement
    def _place_inference(self, job: Job, avoid):
        best = None
        best_key = None
        topo = self.policy.placement == "topo_best_fit"
        _, p80, p40 = self.run[job.id].shape
        for node in self.nodes:
            if not node.up or (avoid and node.id in avoid):
                continue
            prof = p80 if node.mem_gb == 80.0 else p40 if node.mem_gb == 40.0 else mig_profile(job.mem_gb, node.mem_gb)
            if prof is None:
                continue
            c, m = prof
            for g in range(GPUS_PER_NODE):
                fc, fm = node.free_c[g], node.free_m[g]
                if fc >= c and fm >= m:
                    if not topo:
                        return [(node.id, g, c, m)], 1.0
                    # pack onto already-shared GPUs, then onto busy nodes, smallest leftover
                    key = (fc == SLICES_PER_GPU, node.free_whole, (fc - c) + (fm - m), node.mem_gb, node.id, g)
                    if best_key is None or key < best_key:
                        best_key, best = key, [(node.id, g, c, m)]
        return (best, 1.0) if best else None

    @staticmethod
    def _pick_gpus(node: Node, g: int, topo: bool):
        free = [i for i in range(GPUS_PER_NODE) if node.free_c[i] == SLICES_PER_GPU]
        if len(free) < g:
            return None
        chosen = free[:g]
        if topo and g <= ISLAND_SIZE:
            islands = {}
            for i in free:
                islands.setdefault(i // ISLAND_SIZE, []).append(i)
            fits = [lst for lst in islands.values() if len(lst) >= g]
            if fits:
                chosen = min(fits, key=lambda lst: (len(lst), lst[0]))[:g]
        used = len({i // ISLAND_SIZE for i in chosen})
        slow = TOPOLOGY_PENALTY if g > 1 and used > math.ceil(g / ISLAND_SIZE) else 1.0
        return chosen, slow

    def _place_gang(self, job: Job, avoid):
        topo = self.policy.placement == "topo_best_fit"
        g = job.gpus
        if g <= GPUS_PER_NODE:
            best = None
            best_key = None
            for node in self.nodes:
                if not node.up or node.mem_gb < job.mem_gb or node.free_whole < g or (avoid and node.id in avoid):
                    continue
                picked = self._pick_gpus(node, g, topo)
                if picked is None:
                    continue
                gpus, slow = picked
                if not topo:
                    best = (node, gpus, slow)
                    break
                key = (slow, node.mem_gb, node.free_whole - g, node.id)
                if best_key is None or key < best_key:
                    best_key, best = key, (node, gpus, slow)
            if best is None:
                return None
            node, gpus, slow = best
            return [(node.id, i, SLICES_PER_GPU, MEM_UNITS_PER_GPU) for i in gpus], slow
        k = g // GPUS_PER_NODE
        cands = [n for n in self.nodes if n.up and n.mem_gb >= job.mem_gb and n.free_whole == GPUS_PER_NODE
                 and not (avoid and n.id in avoid)]
        if len(cands) < k:
            return None
        if topo:
            cands.sort(key=lambda n: (n.mem_gb, n.id))
        return [(n.id, i, SLICES_PER_GPU, MEM_UNITS_PER_GPU) for n in cands[:k] for i in range(GPUS_PER_NODE)], 1.0

    def _find_place(self, job: Job, avoid=None):
        return self._place_inference(job, avoid) if job.gpus == 0 else self._place_gang(job, avoid)

    # ------------------------------------------------------------------ queue
    def _enqueue(self, jid: int):
        job, r = self.by_id[jid], self.run[jid]
        r.queued = True
        bisect.insort(self.by_arrival, (job.arrival, jid))
        bisect.insort(self.tq[job.tenant], (r.rank, jid))
        self.max_queue = max(self.max_queue, len(self.by_arrival))

    def _dequeue(self, jid: int):
        job, r = self.by_id[jid], self.run[jid]
        r.queued = False
        for lst, key in ((self.by_arrival, (job.arrival, jid)), (self.tq[job.tenant], (r.rank, jid))):
            i = bisect.bisect_left(lst, key)
            del lst[i]

    def _share(self, tenant: str) -> float:
        return max(self.t_c[tenant] / self.total_c, self.t_m[tenant] / self.total_m) / TENANTS[tenant][1]

    def _ordered(self):
        seen = set()
        if self.policy.aging:
            cutoff = self.now - AGE_LIMIT
            for a, jid in self.by_arrival:
                if a > cutoff:
                    break
                seen.add(jid)
                yield self.by_id[jid]
        order = self.policy.order
        if order == "fifo":
            for _, jid in self.by_arrival:
                if jid not in seen:
                    yield self.by_id[jid]
        elif order == "wfq":
            for _, jid in heapq.merge(*self.tq.values()):
                if jid not in seen:
                    yield self.by_id[jid]
        else:  # progressive-filling DRF: always serve the tenant with the lowest weighted dominant share
            ptr = {t: 0 for t in TENANTS}
            queues = [(t, self.tq[t]) for t in sorted(TENANTS)]
            while True:
                if self._share_cache[0] != self.version:
                    self._share_cache = (self.version, {x: self._share(x) for x in TENANTS})
                shares = self._share_cache[1]
                t, best = None, None
                for x, q in queues:
                    if ptr[x] < len(q) and (best is None or shares[x] < best):
                        t, best = x, shares[x]
                if t is None:
                    return
                jid = self.tq[t][ptr[t]][1]
                ptr[t] += 1
                if jid not in seen:
                    yield self.by_id[jid]

    # --------------------------------------------------------------- running
    def _start(self, job: Job, alloc, slow: float):
        r = self.run[job.id]
        self._apply(job.id, alloc, +1)
        r.alloc, r.slowdown, r.run_start = alloc, slow, self.now
        if r.first_start is None:
            r.first_start = self.now
        r.epoch += 1
        self.seq += 1
        heapq.heappush(self.completions, (self.now + r.remaining * slow, self.seq, job.id, r.epoch))
        self.running[job.id] = self.now + r.est_remaining
        if 1 < job.gpus <= ISLAND_SIZE:
            self.small_gang_starts += 1
            self.cross_island_starts += slow > 1.0

    def _gpu_equiv(self, job: Job, alloc) -> float:
        return sum(c for _, _, c, _ in alloc) / SLICES_PER_GPU

    def _interrupt(self, jid: int, preempted: bool):
        job, r = self.by_id[jid], self.run[jid]
        alloc = r.alloc
        self._apply(jid, alloc, -1)
        del self.running[jid]
        r.epoch += 1
        elapsed = self.now - r.run_start
        done = elapsed / r.slowdown
        saved = math.floor(done / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL if job.kind == "batch" else 0.0
        saved = min(saved, r.remaining)
        self.wasted_gpu_s += self._gpu_equiv(job, alloc) * (elapsed - saved * r.slowdown)
        r.remaining = r.remaining - saved + (RESTART_OVERHEAD if job.kind == "batch" else 0.0)
        r.est_remaining = max(job.est * r.remaining / job.duration, r.remaining)
        r.alloc = ()
        if preempted:
            r.preemptions += 1
            self.preemptions += 1
        else:
            r.kills += 1
            self.failure_kills += 1
        self._enqueue(jid)

    def _complete(self, jid: int):
        r = self.run[jid]
        self._apply(jid, r.alloc, -1)
        del self.running[jid]
        r.alloc = ()
        r.finish = self.now
        self.completed += 1

    # ------------------------------------------------------------- scheduling
    def _reservation(self, job: Job):
        """EASY shadow: earliest time (by estimates) the head gang job could start, and the nodes it would use."""
        cached = self._res_cache
        if cached is not None and cached[0] == job.id and cached[1] == self.version:
            raw = cached[2]
        else:
            raw = self._reservation_raw(job)
            self._res_cache = (job.id, self.version, raw)
        return None if raw is None else (max(raw[0], self.now), raw[1])

    def _reservation_raw(self, job: Job):
        compat = [n for n in self.nodes if n.up and n.mem_gb >= job.mem_gb]
        free = {n.id: n.free_c[:] for n in compat}
        whole = {n.id: n.free_whole for n in compat}
        g = job.gpus
        k = g // GPUS_PER_NODE if g > GPUS_PER_NODE else 0
        full = {nid for nid, w in whole.items() if w == GPUS_PER_NODE}
        for est_end, jid in sorted((e, j) for j, e in self.running.items()):
            touched = set()
            for nid, gi, c, _ in self.run[jid].alloc:
                if nid in free:
                    free[nid][gi] += c
                    if free[nid][gi] == SLICES_PER_GPU:
                        whole[nid] += 1
                    touched.add(nid)
            t = est_end
            if k == 0:
                for nid in sorted(touched):
                    if whole[nid] >= g:
                        return t, {nid}
            else:
                full |= {nid for nid in touched if whole[nid] == GPUS_PER_NODE}
                if len(full) >= k:
                    pref = sorted(full, key=lambda i: (self.nodes[i].mem_gb, i))
                    return t, set(pref[:k])
        return None

    def _schedule(self):
        started = []
        blocked = set()
        reservation = None
        examined = 0
        pol = self.policy
        for job in self._ordered():
            if examined >= SCAN_LIMIT or self.nodes_up * GPUS_PER_NODE * SLICES_PER_GPU == self.used_slices:
                break  # scan budget used up, or not a single free slice left
            examined += 1
            r = self.run[job.id]
            if reservation is None or self.now + r.est_remaining <= reservation[0] + EPS:
                avoid, mode = None, 0
            else:
                avoid, mode = reservation[1], 1
            key = (r.shape, mode)
            placed = None
            if key not in blocked:
                placed = self._find_place(job, avoid)
                if placed is None:
                    blocked.add(key)
            if placed is not None:
                self._start(job, *placed)
                started.append(job.id)
                continue
            if pol.backfill == "none":
                break
            if pol.backfill == "easy" and reservation is None and job.gpus > 0:
                reservation = self._reservation(job)
        for jid in started:
            self._dequeue(jid)

    def _preempt(self):
        """Let blocked interactive jobs evict the cheapest single-node batch job that makes room."""
        cands = [self.by_id[jid] for _, jid in self.by_arrival if self.by_id[jid].kind == "interactive"][:32]
        for job in cands:
            if self._find_place(job) is not None:
                continue  # it fits; the normal pass (with its reservation rules) will start it
            best = None
            for node in self.nodes:
                if not node.up:
                    continue
                prof = mig_profile(job.mem_gb, node.mem_gb)
                if prof is None:
                    continue
                c, m = prof
                for vid in node.jobs:
                    v, vr = self.by_id[vid], self.run[vid]
                    if v.kind != "batch" or v.gpus > GPUS_PER_NODE or vr.preemptions >= MAX_PREEMPTIONS:
                        continue
                    freed = {gi for nid, gi, _, _ in vr.alloc}
                    if not any(gi in freed or (node.free_c[gi] >= c and node.free_m[gi] >= m)
                               for gi in range(GPUS_PER_NODE)):
                        continue
                    done = (self.now - vr.run_start) / vr.slowdown
                    lost = done - math.floor(done / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
                    cost = (v.gpus * (lost + RESTART_OVERHEAD), vid)
                    if best is None or cost < best[0]:
                        best = (cost, vid, node.id)
            if best is None:
                continue
            _, vid, nid = best
            self._interrupt(vid, preempted=True)
            others = {n.id for n in self.nodes if n.id != nid}
            placed = self._find_place(job, avoid=others)
            if placed is None:  # cannot happen: the victim freed a whole GPU on nid
                raise AssertionError("preemption freed no usable capacity")
            self._start(job, *placed)
            self._dequeue(job.id)

    # ------------------------------------------------------------ accounting
    def _sample_stranded(self):
        """Free whole GPUs count as 'stranded' while the oldest waiting training job would fit by count but is not running."""
        oldest = None
        for _, jid in self.by_arrival[:64]:
            if self.by_id[jid].gpus > 0:
                oldest = self.by_id[jid]
                break
        self.stranded_now = 0
        if oldest is not None:
            free = sum(n.free_whole for n in self.nodes if n.up and n.mem_gb >= oldest.mem_gb)
            if free >= oldest.gpus:
                self.stranded_now = sum(n.free_whole for n in self.nodes if n.up)

    def _advance(self, t: float):
        if self.t_prev < self.window_end:
            seg = min(t, self.window_end) - self.t_prev
            self.busy_int += self.used_slices / SLICES_PER_GPU * seg
            self.stranded_int += self.stranded_now * seg
            self.down_int += sum(GPUS_PER_NODE for n in self.nodes if not n.up) * seg
        self.t_prev = t
        self.now = t

    # ------------------------------------------------------------------- run
    def run_sim(self) -> SimResult:
        n = len(self.jobs)
        idx = fi = 0
        inf = math.inf
        while self.completed < n:
            while self.completions and self.run[self.completions[0][2]].epoch != self.completions[0][3]:
                heapq.heappop(self.completions)
            t = min(self.jobs[idx].arrival if idx < n else inf,
                    self.completions[0][0] if self.completions else inf,
                    self.failures[fi][0] if fi < len(self.failures) else inf,
                    self.repairs[0][0] if self.repairs else inf)
            if t == inf:
                raise RuntimeError("deadlock: queued jobs can never be placed")
            self._advance(t)
            while self.completions and self.completions[0][0] <= t + EPS:
                _, _, jid, ep = heapq.heappop(self.completions)
                if self.run[jid].epoch == ep:
                    self._complete(jid)
            while self.repairs and self.repairs[0][0] <= t + EPS:
                _, nid = heapq.heappop(self.repairs)
                self.nodes[nid].up = True
                self.nodes_up += 1
                self.version += 1
            while fi < len(self.failures) and self.failures[fi][0] <= t + EPS:
                _, nid, repair = self.failures[fi]
                fi += 1
                if nid >= len(self.nodes) or not self.nodes[nid].up:
                    continue
                node = self.nodes[nid]
                for jid in sorted(node.jobs):
                    self._interrupt(jid, preempted=False)
                node.up = False
                self.nodes_up -= 1
                self.version += 1
                heapq.heappush(self.repairs, (t + repair, nid))
            while idx < n and self.jobs[idx].arrival <= t + EPS:
                job = self.jobs[idx]
                idx += 1
                r = self.run[job.id]
                if self.policy.order == "wfq":
                    w = TENANTS[job.tenant][1]
                    work = (job.gpus or (r.shape[1] or (7, 8))[0] / SLICES_PER_GPU) * job.est
                    r.rank = max(self.vfinish[job.tenant], job.arrival) + work / w
                    self.vfinish[job.tenant] = r.rank
                self._enqueue(job.id)
            self._schedule()
            if self.policy.preempt:
                self._preempt()
            self._sample_stranded()
        return SimResult(self.policy.name, self._metrics(),
                         self.audit, {j: r.first_start for j, r in self.run.items()})

    def _metrics(self) -> dict:
        waits = {"interactive": [], "batch": []}
        big, slow_by_tenant = [], {t: [] for t in TENANTS}
        for j in self.jobs:
            r = self.run[j.id]
            w = r.first_start - j.arrival
            waits[j.kind].append(w)
            if j.gpus >= 2 * GPUS_PER_NODE:
                big.append(w)
            slow_by_tenant[j.tenant].append((r.finish - j.arrival) / max(j.duration, SLOWDOWN_FLOOR))
        tenant_slow = {t: sum(v) / len(v) if v else float("nan") for t, v in slow_by_tenant.items()}
        cap = self.cluster.gpus
        win = max(self.window_end, EPS)
        out = {"policy": self.policy.name, "jobs": len(self.jobs), "jobs_completed": self.completed}
        for kind, ws in waits.items():
            for p in (50, 95, 99):
                out[f"{kind}_p{p}_wait_s"] = percentile(ws, p)
            out[f"{kind}_max_wait_s"] = max(ws) if ws else float("nan")
            out[f"{kind}_frac_wait_over_60s"] = sum(w > 60.0 for w in ws) / len(ws) if ws else float("nan")
        out.update({
            "big_job_count": len(big),
            "big_job_p95_wait_s": percentile(big, 95),
            "big_job_max_wait_s": max(big) if big else float("nan"),
            "gpu_utilization": self.busy_int / (cap * win),
            "stranded_gpus_avg": self.stranded_int / win,
            "down_gpus_avg": self.down_int / win,
            "jain_fairness_slowdown": jain(tenant_slow.values()),
            "tenant_mean_slowdown": tenant_slow,
            "preemptions": self.preemptions,
            "failure_kills": self.failure_kills,
            "wasted_gpu_hours": self.wasted_gpu_s / 3600.0,
            "cross_island_frac_small_gangs": self.cross_island_starts / max(self.small_gang_starts, 1),
            "makespan_s": self.now,
            "max_queue_len": self.max_queue,
            "overcommit_events": self.overcommit,
        })
        return out


def simulate(jobs, policy, cluster: ClusterSpec = ClusterSpec(), failures=(), audit=False) -> SimResult:
    return Simulator(jobs, policy, cluster, failures, audit).run_sim()
