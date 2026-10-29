import unittest

from simulator import Job, generate_jobs, percentile, simulate


class SimulatorTests(unittest.TestCase):
    def test_reproducible_trace(self):
        self.assertEqual(generate_jobs(100, seed=7), generate_jobs(100, seed=7))

    def test_no_overcommit_and_all_complete(self):
        jobs = generate_jobs(300, seed=4, arrival_rate=3.2)
        for policy in ("fifo", "fifo_backfill", "weighted_fair"):
            result = simulate(jobs, policy, node_count=4, node_capacity=8)
            self.assertEqual(result.completed, 300)
            self.assertEqual(result.overcommit_events, 0)
            self.assertLessEqual(result.peak_used_gpus, 32)
            self.assertEqual(set(result.waits), {job.id for job in jobs})

    def test_weighted_fair_can_bypass_head_of_line(self):
        jobs = [
            Job(0, 0.0, 10.0, 3, "batch-a"),
            Job(1, 0.1, 10.0, 3, "batch-a"),
            Job(2, 0.2, 10.0, 3, "batch-a"),
            Job(3, 0.3, 1.0, 1, "interactive-a"),
        ]
        fifo = simulate(jobs, "fifo", node_count=2, node_capacity=4)
        fair = simulate(jobs, "weighted_fair", node_count=2, node_capacity=4)
        self.assertLess(fair.waits[3], fifo.waits[3])

    def test_percentile_and_invalid_job(self):
        self.assertEqual(percentile([0, 10], 50), 5)
        with self.assertRaises(ValueError):
            simulate([Job(0, 0, 1, 9, "batch-a")], "fifo", node_count=1, node_capacity=8)


if __name__ == "__main__":
    unittest.main()
