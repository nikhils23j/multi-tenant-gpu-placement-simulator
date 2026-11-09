# Interview notes: Multi-tenant GPU Placement Simulator

## 1. What it is

It's a discrete-event simulator, in plain Python, of a 256-GPU cluster (32 nodes × 8 GPUs) shared by two
inference tenants and two training tenants. It compares seven scheduling policies on identical generated
workloads and identical node-failure schedules, from strict FIFO up to DRF with topology-aware placement and
preemption. It's a personal project built with AI pair-programming help. Everything runs in simulated time;
there's no real hardware.

## 2. How it works (and why I made each choice)

- **The event loop.** There are four event sources: arrivals (a pre-sorted list), completions (a heap),
  node failures (a pre-generated list) and repairs (a heap). At each event time I process completions, then
  repairs, failures and arrivals, then run one scheduling pass. I picked discrete-event over time-stepping
  because jobs range from 5 seconds to 72 hours. A fixed tick would either be too coarse for inference or
  far too slow for multi-day jobs.
- **Stale completions.** Every job has an `epoch` counter. Preemption or a failure bumps it, and completion
  events with an old epoch are ignored when popped (lazy deletion). That's simpler than deleting from the
  middle of a heap.
- **The resource model.** Each GPU has 7 compute slices and 8 memory units, like A100 MIG: 1g=(1,1),
  2g=(2,2), 3g=(3,4), 7g=(7,8). I tracked memory separately from compute because that's where fragmentation
  really comes from. Two 3g slices use all 8 memory units but only 6 compute slices, so one slice is
  stranded. Half the nodes have 40 GB GPUs, so memory also limits *where* a job can run, not just how much
  room it takes.
- **Gangs.** A training job asks for N whole GPUs. With N ≤ 8 it has to fit on one node, and 16 or 32 means
  whole free nodes. The placement function either returns the full allocation or nothing, and the resources
  are applied in one call, so a partial gang can't exist. There's a test that replays the audit log to
  check this.
- **Topology.** Each node has two NVLink islands of 4. Best-fit picks the island with the *fewest* free GPUs
  that still fits the job, which keeps big holes free for bigger jobs. First-fit just takes the
  lowest-numbered free GPUs. A cross-island split costs an assumed 1.3× slowdown.
- **EASY backfill.**
  - **Reservation:** the first blocked gang job gets one. To compute it, I walk the running jobs in order
    of *estimated* end time, give back their resources on a copy of the cluster state, and stop at the first
    point where the job fits. That gives a shadow time and a set of reserved nodes.
  - **Who can jump ahead:** a later job can start now if its estimate ends before the shadow time, or if it
    avoids the reserved nodes.
  - **Estimates:** user estimates are 1–3× the true runtime.
  - **Caching:** the reservation is cached by a state-version counter so it isn't recomputed when nothing
    changed.
- **Ordering.**
  - **WFQ** gives each job a virtual finish time: `max(tenant_last_finish, arrival) + work/weight`.
  - **DRF** keeps serving the tenant whose weighted dominant share is lowest, where the share is the larger
    of its compute-slice fraction and memory fraction.
  - **Aging:** both put jobs that have waited more than 6 hours first. Otherwise WFQ can push a heavy tenant
    days into the future.
- **Preemption.** When an inference job is blocked, I look for the single-node batch job whose eviction
  makes room and costs the least: GPUs × (un-checkpointed work + restart cost). Checkpoints happen every 30
  minutes of work, a restart costs 120 s, and a job can be preempted at most 2 times so batch jobs always
  finish.
- **Failures.** Each node fails on average every 10 days and takes about 2 hours to repair. The schedule
  comes from its own seeded RNG, so every policy sees exactly the same failures. Killing one node of a
  multi-node gang kills the whole gang.
- **Making the comparison fair.** Every policy gets the same jobs and failures for a given seed. I report
  mean ± 95% t-interval over 10 seeds, plus *paired* per-seed differences, which remove the
  workload-to-workload noise.
- **Speed.** A scheduling pass scans at most 500 jobs, similar to Slurm's `bf_max_job_test`. A shape that
  failed to place is cached within the pass. DRF shares are cached by state version. Runs go in parallel
  with `ProcessPoolExecutor`. The full eval is 560 runs and took 390 s.

## 3. Key results (all measured from my run of `python3 run_eval.py`)

These come from 10 seeds × 10,000 jobs, mean ± 95% CI, unless noted.

- **Strict FIFO is unusable with gang jobs.**
  - Utilization stayed at 57–65% across loads 0.6–1.1.
  - On average 52–54 whole GPUs sat idle while the oldest training job would have fit by count.
  - Inference p99 wait was already 69,332 s at load 0.6 (sweep, 5,000 jobs).
- **Greedy backfill vs EASY at load 1.0.**
  - The worst wait for a 16/32-GPU job dropped from 297 ± 45 h to 82 ± 17 h, better in 10/10 seeds.
  - The cost: inference p99 rose from 1,365 s to 4,316 s and batch p95 from 19.4 h to 31.6 h.
- **Topology-aware best-fit at load 1.0.**
  - Cross-island splits of 2–4 GPU jobs fell from 33% to 7% (2.5% at load 0.8).
  - Batch p95 fell by 19,728 ± 9,566 s, paired, lower in 10/10 seeds. That rests on the assumed 1.3× penalty.
- **Preemption on top of DRF at load 1.0.**
  - Inference p99 went from 869 ± 347 s to 0 s, lower in 10/10 seeds.
  - Inference jobs waiting more than 60 s went from 3.6% to 0.04%.
  - About 203 preemptions per 10,000 jobs.
  - Batch p95 changed by +1,899 ± 6,373 s and wasted GPU-hours by +11 ± 21, both within noise.
  - Jain's index fell from 0.59 to 0.52.
- **WFQ ≈ DRF here.** Batch p95 was identical in 8 of 10 seeds at load 0.8.
- **Correctness:** all 560 runs completed every job with zero overcommit events, and 14 unit tests pass.

**What these results do NOT show**

- Nothing about real latency, real GPUs, or any production system. These are simulated seconds on a
  synthetic workload.
- That the workload matches Philly or PAI. It's only shaped like them by hand.
- That the topology gain is real-world sized. It comes straight from the 1.3× number I assumed.
- That preemption is free in general. Its low cost here depends on 30-minute checkpoints, a 120 s restart,
  and only evicting single-node jobs.
- At load 0.8 (the setting I chose before looking), inference p95 is 0 for every policy except FIFO.
  Differences only show up in the tail and at higher load. I added load 1.0 afterwards and say so in the
  README.

## 4. Two-minute answer: "Tell me about a project you're proud of" (STAR)

### Situation
"Shared GPU machines are a classic campus annoyance. You've got a two-minute inference job, and it's stuck
behind somebody's multi-day training run. I wanted to know if that was just bad luck or a scheduling
problem, and if you fix it, who ends up paying. I didn't have a real cluster to play with, so I decided to
simulate one."

### Task
"I gave myself the job of building a simulator realistic enough that the answer would actually mean
something. That meant more than counting GPUs. It needed GPU memory, MIG-style fractional slices,
NVLink topology, all-or-nothing gang scheduling for training jobs, preemption with checkpoint costs, and
node failures. Then I wanted to compare a ladder of policies honestly, from FIFO up to DRF with preemption,
across many random seeds instead of one lucky run."

### Action
"It's a discrete-event simulator in plain Python: 32 nodes of 8 GPUs, each GPU split into 7 compute slices
and 8 memory units. I built it with an AI pair-programmer. I used it to move faster, but I read and tested
every part, and some of the bugs I caught were actually in my own test setups. I implemented EASY
backfilling, where the blocked big job gets a reservation computed from user runtime estimates. I also
wrote weighted fair queueing, Dominant Resource Fairness, topology-aware best-fit, and preemption that picks
the batch job that loses the least un-checkpointed work. Every policy sees the exact same jobs and the exact
same node failures, and I report 95% confidence intervals plus paired per-seed differences. For
correctness, I added tests that replay every allocation to prove no GPU is ever overcommitted and no gang is
ever half-placed."

### Result
"The clearest results were the tradeoffs. Greedy backfilling looked best on batch wait, but the worst
32-GPU job waited about 297 simulated hours. EASY reservations cut that to 82, but inference tail latency
got worse. Topology-aware placement cut split jobs from 33% to 7%. Adding preemption on top of DRF took the
inference p99 wait from about 870 seconds to zero at full load, with about 200 preemptions per 10,000 jobs
and no batch change I could tell apart from noise. And strict FIFO couldn't get past 65% utilization even at
light load. The big lesson was that there's no free lunch, only a choice of who waits, and you have to
report spread and tradeoffs, not just one nice number. It's a simulation with a synthetic workload, so I'd
treat the comparisons as directional, not real-world numbers."

## 5. Likely follow-up questions

1. **Why is the inference p99 exactly 0 with preemption?** A blocked inference job can almost always evict
   a single-node batch job right away, and in this model eviction is instant, so it starts at the same
   simulated second. Real preemption takes seconds to minutes to drain and free memory, which I don't model.
   The max wait isn't always 0 either: 3,297 ± 7,193 s at load 1.0. That's because some inference jobs need
   a full 80 GB GPU and no evictable victim is available.
2. **How do you know there's no overcommit?** Every allocation goes through one `_apply` function that
   counts violations. Separately, the tests record an audit log and replay it independently, checking that
   per-GPU slices and memory never exceed capacity and that everything is released at the end. All 560 eval
   runs reported 0 overcommit events.
3. **Why does EASY make inference latency worse?** The reserved nodes are protected. A job whose *estimate*
   runs past the shadow time can't use them, and estimates are inflated 1–3×. So capacity sits idle while
   inference waits. That's the cost of keeping big jobs from starving.
4. **Is the workload realistic?** It's *inspired by* the published Philly and PAI studies: mostly small
   jobs, a few big gangs dominating GPU-time, heavy tails, fractional-GPU inference. But I picked the
   parameters by hand and didn't replay a trace. The absolute numbers don't transfer. The comparisons
   between policies on identical inputs are what I trust, and even those depend on my assumptions.
5. **Why were WFQ and DRF so similar?** With EASY backfill, ordering mostly decides who gets the single
   reservation. Both rules put the light inference tenants first and the lighter training tenant ahead of
   the heavy one, so they usually pick the same job. With more tenants, or tenants using different
   resources (memory-heavy vs compute-heavy), I'd expect them to diverge.
6. **What would you do next?** Replay a real public trace, such as the Philly trace or Alibaba's GPU
   traces. Add conservative backfilling (a reservation for every job) for comparison. Model MIG placement
   rules properly. Add a preemption grace period with an inference SLO. Sweep the topology penalty and
   checkpoint interval to see how sensitive the conclusions are.
7. **How did you use AI help, and how do you know the code is right?** I used it as a pair-programmer for
   drafting and speeding things up. I checked it by writing hand-built scenarios with known answers, like
   checkpoint rollback arithmetic, EASY protecting the reserved job, and topology splits. When two of those
   tests failed, I traced both to mistakes in my own test setups, not the simulator, and fixed the tests
   with reasoning I can explain.
8. **Why mean ± CI over seeds instead of one run?** Heavy-tailed workloads make single runs misleading.
   One 32-GPU, 72-hour job can swing a whole seed. Paired differences on identical seeds give much tighter
   comparisons than looking at the means separately.

## 6. Things I should be able to explain cold

- [ ] The event loop order (completions → repairs → failures → arrivals → schedule → preempt) and why
      epochs implement lazy deletion
- [ ] The MIG profile math: why a 10 GB model needs 1g on an 80 GB GPU but 2g on a 40 GB one, and why
      two 3g slices strand a slice
- [ ] How gang atomicity is guaranteed and how the test checks it
- [ ] EASY backfill: how the shadow time and reserved nodes are computed, the two rules for jumping ahead,
      and why estimates matter
- [ ] Greedy backfill vs EASY: why greedy starves big jobs and what EASY costs
- [ ] WFQ virtual finish time formula; DRF dominant share; why aging is needed
- [ ] Topology-aware best-fit (smallest island that fits), and that the 1.3× penalty is assumed
- [ ] Preemption victim choice, checkpoint rollback, `MAX_PREEMPTIONS`, and why every job still completes
- [ ] Definitions of utilization, stranded GPUs, and Jain's index over mean slowdown
- [ ] How offered load is set (demand ÷ capacity → arrival rate), the daily cycle, and why FIFO is unstable
      even at 0.6
- [ ] 95% t-interval vs paired differences; why CIs are wide with heavy tails
- [ ] The headline numbers and their limits: 297 → 82 h, 33% → 7%, 869 → 0 s p99, about 203 preemptions,
      and what's simulated or assumed
