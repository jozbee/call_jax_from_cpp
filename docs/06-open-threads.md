# 06 — Open threads and things that will bite you

Status as of 2026-09-05. Verify against the tree before acting; this file ages.

## Not done, deliberately

**Deadline / watchdog mode.** The owner chose "inline-sync first, watchdog
later". Note the hard constraint if you build it: **PJRT cannot cancel a
running CPU computation.** A watchdog can only return a stale result and
discard the late one. Design it as double-buffered result slots behind a
pinned executor thread, and be explicit in the docs that overrun means stale
data, not cancellation.

**Donation.** Outputs are small (~10 KB) and constant-size, and donation forces
per-call buffer recreation plus bookkeeping. Deferred pending evidence. The
hardcoded `non_donatable_input_indices = {0}` was removed regardless.

**Pooling allocator fork patch.** See [04-xla-fork.md](04-xla-fork.md).
Evidence-gated on real hardware.

## Not done, still owed

**Artifact format v2** (plan Phase 5, not started):
- `donate_argnums` passthrough in `src/jax2exec/jax2exec.py`
- emit `{name}.mlirbc` via `jax.export` alongside the `.binpb`
- richer JSON sidecar: `schema: 2`, dtypes, shapes (rank-general), donation
  map, jax/jaxlib versions, export platform/arch — keeping v1 keys intact
- C++ load fallback: on `.binpb` load failure, `PJRT_Client_Compile` the
  `.mlirbc` (~2 s, host-tuned codegen). This is the answer to the
  architecture-lock problem, and it also enables an A/B quantifying the penalty
  of running container-exported artifacts on the production box.
- pin `jaxlib` in `pyproject.toml` to match the pinned `jax==0.9.0.1`

**`eng/comp` migration.** `/Users/jozbee/work/eng/comp/cpp/src/mpc_example.cpp`
still uses the legacy API — one file, ~6 call sites. It migrates at the next
submodule bump. The legacy `Client`/`Buffer`/`AOTComputation` classes and
`src/lc0/` are slated for deletion only after that.

**Native amd64 validation.** Everything is wired and waiting: `tools/rt_check.sh`
to audit the host, `make fixtures` **on-box** (serialized executables embed
target machine code), then `tools/run_matrix.sh` plus the 28×4000 campaign,
with `SCHED_FIFO` and core pinning enabled. The container could not exercise
those: `set_realtime_priority: pthread_setschedparam(SCHED_FIFO) -- needs
CAP_SYS_NICE: Operation not permitted`. `.devcontainer/compose.yml` has since
gained `cap_add: SYS_NICE` and `rtprio`/`memlock` ulimits, but the running
container predates that.

## Environment caveats

**The running `jax_from_cpp_arm64` container has packages installed at runtime
that are not baked into the image**: `liblapack-dev`, `libblas-dev`, `xxd`, the
`control` Python package, and jax pinned to 0.9.0.1. Recreating the container —
which is required to pick up the new `cap_add`/`ulimits` — means either
`docker compose build` or redoing that setup by hand. Symptoms of missing them:
link failures for `dgesvd_`/`sgeev_` (LAPACK), and `xxd: command not found`
during the bazel plugin build.

**`docker compose up` reported recreating an orphan `siso_arm64` container**
from another project sharing the compose namespace. It did not appear in
`docker ps -a` afterwards, but be aware the namespace is shared.

## Small things worth knowing

- `artifacts/` is gitignored. All measurement CSVs and built plugins live
  there and do not travel between machines.
- The exporter asserts **rank <= 1 and float64** for every input and output.
  Anything else needs exporter work first.
- `tools/export_fixture.py` builds the synthetic twin with
  `HORIZON=200, WIDTH=96, DEPTH=7`, and its seed deliberately depends on every
  input so XLA cannot prune parameters (see
  [02-measurement.md](02-measurement.md), trap 5).
- `make test_alloc` preloads `tests/support/malloc_guard.c`; the markers are
  `pjrt_guard_arm` / `pjrt_guard_disarm`, resolved via `dlsym(RTLD_DEFAULT,…)`,
  so the benchmark no-ops cleanly when the guard is absent.
- On macOS the preload variable is `DYLD_INSERT_LIBRARIES`, not `LD_PRELOAD`;
  the Makefile handles this. All RT helpers are no-ops on macOS.
