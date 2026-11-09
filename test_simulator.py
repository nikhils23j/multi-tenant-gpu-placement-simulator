import math
import unittest
from collections import defaultdict

import simulator as S
from simulator import (GPUS_PER_NODE, MEM_UNITS_PER_GPU, POLICIES, SLICES_PER_GPU, ClusterSpec, Job, Policy,
                       Simulator, jain, mig_profile, percentile, simulate)
from workload import generate_failures, generate_workload, offered_load

SMALL = ClusterSpec(nodes=8, frac_80gb=0.5)


def small_workload(seed=3, n=600, load=0.95):
    jobs = generate_workload(n, seed, load, SMALL)
    fails = generate_failures(seed, SMALL, jobs[-1].arrival, mtbf_days=2.0)
    return jobs, fails


def replay_audit(audit, cluster):
    """Independently replay every allocation/release and return the worst per-GPU usage seen."""
    used_c = defaultdict(int)
    used_m = defaultdict(int)
    worst = 0.0
    for _, sign, _, nid, g, c, m in audit:
        used_c[nid, g] += sign * c
        used_m[nid, g] += sign * m
        assert used_c[nid, g] >= 0 and used_m[nid, g] >= 0
        worst = max(worst, used_c[nid, g] / SLICES_PER_GPU, used_m[nid, g] / MEM_UNITS_PER_GPU)
    return worst, used_c, used_m


class InvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs, cls.fails = small_workload()
        cls.results = {p: simulate(cls.jobs, p, SMALL, cls.fails, audit=True) for p in POLICIES}

    def test_capacity_never_overcommitted(self):
        for p, res in self.results.items():
            worst, used_c, used_m = replay_audit(res.audit, SMALL)
            self.assertLessEqual(worst, 1.0 + 1e-12, p)
            self.assertEqual(res.metrics["overcommit_events"], 0, p)
            self.assertTrue(all(v == 0 for v in used_c.values()), f"{p}: leaked allocation")
            self.assertTrue(all(v == 0 for v in used_m.values()), f"{p}: leaked memory")

    def test_every_job_completes(self):
        for p, res in self.results.items():
            self.assertEqual(res.metrics["jobs_completed"], len(self.jobs), p)
            self.assertTrue(all(res.starts[j.id] is not None and res.starts[j.id] >= j.arrival for j in self.jobs))

    def test_gang_atomicity(self):
        by_id = {j.id: j for j in self.jobs}
        for p, res in self.results.items():
            grants = defaultdict(list)  # (job, time) -> GPU grants made at that instant
            for t, sign, jid, nid, g, c, m in res.audit:
                if sign > 0:
                    grants[jid, t].append((nid, g, c, m))
            for (jid, _), alloc in grants.items():
                job = by_id[jid]
                if job.gpus == 0:
                    self.assertEqual(len(alloc), 1)
                    continue
                self.assertEqual(len({(n, g) for n, g, _, _ in alloc}), job.gpus, f"{p}: partial gang {jid}")
                self.assertTrue(all(c == SLICES_PER_GPU and m == MEM_UNITS_PER_GPU for _, _, c, m in alloc))
                nodes = {n for n, _, _, _ in alloc}
                if job.gpus <= GPUS_PER_NODE:
                    self.assertEqual(len(nodes), 1)
                else:
                    self.assertEqual(len(nodes), job.gpus // GPUS_PER_NODE)

    def test_memory_constraint_respected(self):
        mems = SMALL.node_mem()
        by_id = {j.id: j for j in self.jobs}
        for res in self.results.values():
            for _, sign, jid, nid, _, _, _ in res.audit:
                if sign > 0 and by_id[jid].gpus > 0:
                    self.assertGreaterEqual(mems[nid], by_id[jid].mem_gb)

    def test_failures_and_preemption_exercised(self):
        self.assertGreater(len(self.fails), 0)
        self.assertGreater(self.results["fifo_greedy_backfill"].metrics["failure_kills"], 0)
        self.assertGreater(self.results["drf_topo_preempt"].metrics["preemptions"], 0)
        for p, res in self.results.items():
            if p != "drf_topo_preempt":
                self.assertEqual(res.metrics["preemptions"], 0)

    def test_deterministic_per_seed(self):
        jobs2, fails2 = small_workload()
        self.assertEqual(jobs2, self.jobs)
        self.assertEqual(fails2, self.fails)
        for p in ("easy_topo", "drf_topo_preempt"):
            again = simulate(jobs2, p, SMALL, fails2).metrics
            self.assertEqual(again, self.results[p].metrics)
        self.assertNotEqual(generate_workload(50, 4, 0.8, SMALL), generate_workload(50, 5, 0.8, SMALL))


def gang(jid, t, dur, gpus, tenant="train-b", mem=20.0, est=None):
    return Job(jid, tenant, t, dur, est or dur, gpus, mem)


class BehaviourTests(unittest.TestCase):
    def test_mig_profile_sizing(self):
        self.assertEqual(mig_profile(10, 80), (1, 1))
        self.assertEqual(mig_profile(10, 40), (2, 2))
        self.assertEqual(mig_profile(35, 80), (3, 4))
        self.assertEqual(mig_profile(60, 80), (7, 8))
        self.assertIsNone(mig_profile(60, 40))

    def test_strict_fifo_blocks_but_backfill_does_not(self):
        one = ClusterSpec(nodes=1, frac_80gb=1.0)
        jobs = [gang(0, 0, 100, 6), gang(1, 1, 100, 4), gang(2, 2, 10, 2)]
        fifo = simulate(jobs, "fifo", one).starts
        bf = simulate(jobs, "easy_backfill", one).starts
        self.assertEqual(fifo[2], 100)   # stuck behind job 1
        self.assertEqual(bf[2], 2)       # ends (t=12) before job 1's reservation (t=100)

    def test_easy_reservation_protects_head_job(self):
        one = ClusterSpec(nodes=1, frac_80gb=1.0)
        # job 1 needs the whole node at t=100; a long 2-GPU job would delay it under greedy backfill.
        jobs = [gang(0, 0, 100, 6), gang(1, 1, 100, 8), gang(2, 2, 500, 2)]
        greedy = simulate(jobs, "fifo_greedy_backfill", one).starts
        easy = simulate(jobs, "easy_backfill", one).starts
        self.assertEqual(easy[1], 100)
        self.assertEqual(greedy[1], 502)
        self.assertEqual(easy[2], 200)

    def test_topology_aware_keeps_gang_on_one_island(self):
        one = ClusterSpec(nodes=1, frac_80gb=1.0)
        # Job 0 takes GPUs 0-2, leaving GPU 3 (island 0) and 4-7 (island 1) free.
        jobs = [gang(0, 0, 50, 3), gang(2, 1, 10, 2)]
        ff = Simulator(jobs, "easy_backfill", one, audit=True)
        ff.run_sim()
        tp = Simulator(jobs, "easy_topo", one, audit=True)
        tp.run_sim()
        isl = lambda sim: {g // 4 for _, s, j, _, g, _, _ in sim.audit if s > 0 and j == 2}
        self.assertEqual(len(isl(ff)), 2)          # first-fit grabs GPU 3 + GPU 4
        self.assertEqual(len(isl(tp)), 1)
        self.assertAlmostEqual(ff.run[2].finish - ff.run[2].first_start, 10 * S.TOPOLOGY_PENALTY)
        self.assertAlmostEqual(tp.run[2].finish - tp.run[2].first_start, 10)

    def test_preemption_rolls_back_to_checkpoint(self):
        one = ClusterSpec(nodes=1, frac_80gb=1.0)
        jobs = [gang(0, 0, 4000, 8), Job(1, "infer-a", 2000, 60, 60, 0, 10)]
        sim = Simulator(jobs, "drf_topo_preempt", one)
        res = sim.run_sim()
        self.assertEqual(res.starts[1], 2000)      # interactive job ran immediately
        self.assertEqual(res.metrics["preemptions"], 1)
        # 2000 s done, 1800 s checkpointed, 200 s lost, +120 s restart. The 8-GPU gang can only
        # restart once the 60 s inference job frees its GPU (t=2060).
        self.assertAlmostEqual(sim.run[0].finish, 2060 + (4000 - 1800 + S.RESTART_OVERHEAD))
        self.assertAlmostEqual(res.metrics["wasted_gpu_hours"], 8 * 200 / 3600)

    def test_failure_kills_whole_gang(self):
        two = ClusterSpec(nodes=2, frac_80gb=1.0)
        jobs = [gang(0, 0, 1000, 16)]
        res = simulate(jobs, "easy_backfill", two, failures=[(500.0, 1, 100.0)], audit=True)
        self.assertEqual(res.metrics["failure_kills"], 1)
        releases = [(t, n) for t, s, _, n, _, _, _ in res.audit if s < 0 and t == 500.0]
        self.assertEqual(len(releases), 16)         # both nodes released together
        self.assertEqual(res.metrics["jobs_completed"], 1)

    def test_validation(self):
        with self.assertRaises(ValueError):
            simulate([gang(0, 0, 10, 8, mem=60)], "fifo", ClusterSpec(nodes=1, frac_80gb=0.0))
        with self.assertRaises(ValueError):
            simulate([gang(0, 0, 10, 24)], "fifo", ClusterSpec(nodes=2))
        with self.assertRaises(ValueError):
            Simulator([], Policy("x", "lifo", "none", "first_fit"))

    def test_helpers_and_load(self):
        self.assertEqual(percentile([0, 10], 50), 5)
        self.assertTrue(math.isnan(percentile([], 50)))
        self.assertAlmostEqual(jain([1, 1, 1]), 1.0)
        self.assertAlmostEqual(jain([1, 0]), 0.5)
        jobs = generate_workload(3000, 1, 0.8, ClusterSpec())
        self.assertAlmostEqual(offered_load(jobs, ClusterSpec()), 0.8, delta=0.08)


if __name__ == "__main__":
    unittest.main()
