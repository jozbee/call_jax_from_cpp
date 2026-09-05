# 05 — Results

MPC fixture, case 2, on the **aarch64 devcontainer**, machine otherwise idle.
28 runs × 4000 calls per API = 112,000 calls each, interleaved. Container
numbers are relative signals; absolute ones belong on the target hardware.

| | legacy (per-call buffers) | `Runtime`/`Function` |
|---|---|---|
| median p50 | 4841.8 µs | **3979.4 µs** (−18%) |
| median p99.9 | 5416.9 µs | **4416.5 µs** |
| worst single call | 20168.7 µs | **6832.0 µs** |
| worst max/p50 | 4.198 | **1.719** |
| runs with a >2x outlier | 2 of 28 | **0 of 28** |
| allocations/call (ours) | ~520 | **0** |
| allocations/call (inside XLA) | ~15,400 | ~15,400 |

Baseline for context: before any of this work, 4000 MPC steps gave mean 3016 µs,
p50 2996, p99 3516, p99.9 4665, **max 17279 µs — 5.7x the median**.

## What moved the numbers

**Persistent zero-copy input buffers.** ~18% off the median on MPC, ~16% on the
synthetic fixed-cost twin — the agreement is what confirms the win is in the
call path rather than anything MPC-specific. Removing the per-call
host-to-device copies also removes the per-call allocation and event churn
that produced the large spikes.

**Inline synchronous execution** (`asynchronous=false` via the fork patch)
removes the hand-off to the dispatch thread pool.

## What did not move them

**Plugin build mode.** `-c opt` vs bazel's default `fastbuild` (`-O0`): 4718.7
vs 4728.4 µs p50. See [04-xla-fork.md](04-xla-fork.md).

## How much of this to believe

- The **magnitude** difference is solid: the largest call the new path produced
  in 112,000 was 1.7x its median; the old path reached 4.2x.
- The **frequency** claim is the soft part. Two outlier runs against zero, at
  n=28, is not on its own a significant difference. Do not quote it as one.
- Everything here is aarch64 in a container, with no `SCHED_FIFO` and no core
  pinning (the container lacked `CAP_SYS_NICE` at the time of measurement).
  Treat it as a relative signal.

## Targets for sign-off on the production box

From the plan, on the native Intel amd64 Linux machine, MPC, >=4000 steps, RT
environment:

- p99.9/p50 <= 1.3
- max/p50 <= 2.0 (the intrinsic bound from in-process Python PJRT is 1.14–1.25)
- p50 <= baseline p50
- absolute max < 6 ms
- wrapper allocations = 0

## Published write-up

A report artifact exists with the full data and charts:
<https://claude.ai/code/artifact/1cea846f-2ac6-4312-a6a9-31779b194dd1>
("Jitter in the JAX Call Path"). Raw CSVs from the campaign are in
`artifacts/` (`rare.csv`, `mpc_rt_long.csv`, `mpc_legacy_long.csv`,
`plugin_ab.csv`, `matrix_*.csv`) — note `artifacts/` is gitignored, so these
are local to whichever machine produced them.
