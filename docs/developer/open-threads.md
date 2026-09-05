# Open threads

**Status as of 2026-09-05.** This page ages faster than the rest of the
directory; verify against the tree before acting on any of it.

## Deferred deliberately

### A watchdog or deadline mode

Not built. The order of work was inline-synchronous execution first, watchdog
later, and the first part carried enough of the win to leave the second unbuilt.

The hard constraint, if it is ever built: **PJRT cannot cancel a running CPU
computation.** There is no cancellation in the C API and none underneath it. A
watchdog can only return the previous result and discard the late one when it
eventually arrives. An overrun therefore means *stale data*, not a cancelled
call, and a caller that treats a deadline as a guarantee of freshness will be
wrong exactly when it matters.

The shape that follows from that: double-buffered result slots behind a pinned
executor thread, with the control thread reading whichever slot is complete and
being told how old it is. Anything that promises to abort a call is promising
something the runtime cannot do, and the documentation for it has to say so in
those words.

### Donation

Deferred pending evidence. Donating an input lets XLA write a result into the
buffer the caller supplied, which sounds like exactly the right thing for a
loop. It also consumes the donated buffer, which is incompatible with the
persistent zero-copy input buffers that produced most of the measured win: a
donated input has to be rebuilt every call, which is the cost this design exists
to remove.

The exporter passes `donate_argnums` through and the sidecar records which
arguments were donated, so the information is there when somebody wants to
measure it. The historical `non_donatable_input_indices = {0}` hardcoding was
removed regardless — it was a guess about one caller's arguments.

### The pooling-allocator fork patch

Roughly thousands of allocations happen per call inside XLA's thunk runtime, about one
per StableHLO op. That is the last known structural jitter surface, and it is
not reachable through the PJRT C API: it would take a third fork patch
installing a pooling allocator behind `CpuClientOptions::allocator`. See
[the XLA fork](xla-fork.md).

Gated on evidence, deliberately: build it when a measurement on real hardware,
on a tuned idle host, shows that plugin-internal allocation drives a residual
tail. Not before.

## Owed, not deferred

### Native x86-64 validation

Every number this project has published was taken on **aarch64 in a container**,
with no `SCHED_FIFO` and no core pinning, because the container lacked
`CAP_SYS_NICE` at the time. It is a relative signal: the comparison between two
call paths on the same host is meaningful, the absolute microseconds are not.

The sign-off targets on the native x86-64 box, on an idle tuned host, are still
open:

| | target |
|---|---|
| p99.9 / p50 | ≤ 1.3 |
| max / p50 | ≤ 2.0 |
| p50 | ≤ the previous path's p50 |
| absolute max | < 6 ms |
| wrapper allocations per call | 0 |

The max/p50 target comes from a measured bound rather than a wish: in-process
blocking PJRT from Python, on the same module, shows max/avg of 1.14–1.25x, so
about 1.2 is the floor and 2.0 leaves room for the operating system.

Everything needed is wired: `tools/rt_check.sh` to audit the host, `make export`
**on that box** because serialized executables embed target machine code, then
`tools/run_matrix.sh` and a campaign of the shape described in
[measurement](measurement.md), with `SCHED_FIFO` and core pinning enabled.

### The headline campaign is not reproducible from this tree

The 112,000-call comparison on the benchmarks page predates the current
examples, and the fixtures it used do not ship. It stands as a record of what
was measured, not as something a reader can re-run. `examples/02_trajopt` is the
workload to quote reproducible numbers from, and any new claim should be made
with it.

## Caveats that will bite someone

**The sparse and tridiagonal FFI handlers are dropped.** The fork vendors
jaxlib's LAPACK kernels but not the sparse or tridiagonal ones, so their
handler registrations are removed rather than left dangling. An executable that
lowers to `cpu_csr_sparse_dense_ffi` or `tridiagonal_solve_perturbed_ffi` fails
at load with `No FFI handler registered ... on a platform Host`. That message is
the intended outcome — it names the missing handler — and the fix is to vendor
that kernel into `//jaxlib_cpu_kernels` the same way as the rest.

**There is no real-time support on macOS.** Every helper in `pjrt::rt` is a
no-op there, and the preload variable differs (`DYLD_INSERT_LIBRARIES` rather
than `LD_PRELOAD`, which the Makefile handles). macOS is fine for correctness
work and is not a machine to take latency numbers on.

**Half-precision and complex dtypes are unsupported.** The supported set is
`bool`, the four signed and four unsigned integer widths, `float32` and
`float64`. `float16`, `bfloat16` and the complex types are not supported by the
exporter and have no `DType` here. Adding one means the sidecar spelling, the
`DType` entry, the PJRT element type, and a reference case that exercises it —
in that order, and the exporter refuses clearly in the meantime rather than
writing an artifact nobody can load.

**`artifacts/` and `build/` are gitignored, and their contents are
machine-specific.** Exported executables are locked to the exporting
architecture and ISA, measurement CSVs belong to the machine that produced them,
and the plugin is a downloaded binary. None of it travels; re-export and
re-download rather than copying.

**Container timings are relative signals.** `docker/compose.yml` grants
`CAP_SYS_NICE` and the `rtprio`/`memlock` ulimits so the hardening path can be
exercised, and the real-time steps report `[skip]` rather than failing when the
privilege is absent. Being able to run the code is not the same as being able to
measure it: a shared or virtualised host is not a machine to quote numbers from.
