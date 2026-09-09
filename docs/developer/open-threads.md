# Open threads

**Status as of 2026-09-08.** This page ages faster than the rest of the
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

### The JAX pin is held one release behind on purpose

`versions.env` pins JAX {{ jax_version }}, and the newer 0.11.1 exists. That is
not staleness. [jax-ml/jax#40101](https://github.com/jax-ml/jax/issues/40101)
is an XLA:CPU code-generation regression, present from 0.11.1 onward, in which
a `dynamic-update-slice` writing a small slice into a large buffer inside a
loop body costs time proportional to the whole destination buffer rather than
to the slice written. The HLO is identical across the two releases, so nothing
upstream of code generation shows the problem.

**The threshold is in bytes, and where it falls depends on the machine.** The
body of the issue reports a cliff at 256 bytes of update size, measured on
native aarch64 and on emulated amd64. The reporter's own re-measurement on a
native x86-64 runner with AVX-512
([comment](https://github.com/jax-ml/jax/issues/40101#issuecomment-5352919579))
puts it between 448 and 480 bytes instead, and retracts the "batch writes to at
least 256 bytes" workaround as unsafe on x86 — 512 is the safe number there.
Below the threshold, slow; the cliff is sharp on both. This project targets
x86-64, so 448 is the figure to reason with here and 256 would wrongly clear a
workload writing 300 bytes an iteration.

It matters here because of *where* the defect lives. jaxlib compiles the
machine code that a `.binpb` embeds, at export time; the plugin relinks that
code and never recompiles it. So the cost is baked into the artifact, and no
patch carried in the XLA fork can remove it. Pinning jaxlib is the fix.

**This host reproduces the regression, and it does not touch this workload.**
Both halves were measured, and the second is only trustworthy because of the
first.

The reporter's own case, in-process, second-call timing, on an Intel i9-9880H,
x86-64, bare metal, kernel 7.1.8-arch1, `powersave` governor:

| case | jaxlib 0.11.0 | jaxlib 0.11.1 |
|---|---|---|
| `fori_loop`, n = 100,000 | 0.0006 s | 2.1821 s |
| `lax.scan` stacked, n = 200,000 | 0.0009 s | 14.4420 s |

Three to four orders of magnitude, so the probe is sensitive to what it
perturbs and a negative from it means something.

`examples/02_trajopt` at the default preset shows no difference at all. Six
interleaved rounds of 2000 calls per arm, 12,000 calls in total, the *same*
plugin throughout and the only variable the jaxlib that compiled the artifact,
on the host above with the load average at 0.48 before and 0.95 after
(`tools/rt_check.sh`: 0 ok, 9 worth fixing, so untuned):

| median across rounds | 0.11.1 artifact | 0.11.0 artifact |
|---|---|---|
| p50 | 3929 µs | 3931 µs |
| p99.9 / p50 | 1.5 | 1.7 |
| max / p50 | 1.8 | 1.9 |

The p50 difference is 2 µs against a round-to-round spread of 90 µs, so it is
nil. The tail columns move the wrong way and are dominated by one outlier round
in each arm; nothing there is a signal either.

Why the workload escapes is worth writing down, because it is the thing to
re-check rather than a conclusion to reuse. The cost is proportional to the
destination buffer *times the trip count*, and these loops are short: a horizon
of 50 against the reporter's 200,000. The per-iteration writes do sit under the
byte threshold, so the slow path is presumably taken; there is just almost no
work behind it. A fixture with a long `lax.scan` would not be so lucky.

**So the pin is held for insurance, not for a measured win here.** Before moving
it forward, do not simply take the next release. A fix was reported in flight on
2026-08-20, so the wait may be short; the regression was still present on
`0.11.2.dev20260819`. Check whether the issue is closed, and whether the XLA
revision that release pins still contains
`e9204c9ce05359855a42dae3ad5e69ad9e532d82` — the commit the reporter's bisect
points at, a tentative attribution rather than a confirmed root cause. Then
re-run the two-line reproducer above, which costs seconds and answers the
question directly.

The workload comparison is cheap to repeat because the two plugins are
interchangeable: a plugin built from either XLA revision loads and runs an
artifact exported against the other, verified in both directions here with
`load_kind=deserialized` and all four reference cases agreeing to 1e-15. Export
`examples/02_trajopt` under both candidate releases, run both artifacts against
the *same* plugin, and the difference is attributable to export-time code
generation alone. Interleave them, per [measurement](measurement.md); do not run
them back to back.

### The fork cannot find LAPACK on a non-Debian host

Patch 1 adds the Debian and Ubuntu multiarch LAPACK directories to the plugin's
link path, and nothing else, because a general library directory on that list
relinks the whole C runtime against the build host's glibc — measured, and
described on [the fork page](xla-fork.md). So a build on Arch, Fedora or
anything else that keeps LAPACK in `/usr/lib` fails at the final link unless the
builder passes `--arch-flags "--linkopt=-L/usr/lib"`, and that build is then not
publishable.

There is no safe generic path, because every candidate directory also holds
`libc`. A real fix would link the two libraries by absolute file path rather
than by search: resolve them once at configure time and pass the resolved paths
to the rule. That is a bazel change to a patch this project has to rebase every
bump, so it is worth doing only if someone actually needs to build releases off
Debian. Until then the cost is one flag and a note.

### A run report does not record which JAX produced the artifact

`bench --json` writes the host audit, the plugin path and the PJRT API version,
and `report.hpp` adds the plugin's advertised attributes. None of it says which
jaxlib compiled the executable being timed. The sidecar beside every artifact
does record `jax_version` and `jaxlib_version`, so the information exists two
files away and simply is not copied into the report.

That is why the matrix figures below lost their pin: nothing in the recorded
output would have contradicted a reader who assumed they were current. Copying
the sidecar's `jax_version`, `jaxlib_version` and `generator` into the report,
next to the host block, would make the omission impossible rather than
merely discouraged. Small, and not done.

### The exporter under-claims the ISA level when `/proc/cpuinfo` is unreadable

On x86, `jax2exec._isa.host_isa_level` reads the flag list from
`/proc/cpuinfo` and tests the levels strongest first, so when the file cannot
be read — a sandbox that hides `/proc`, for one — every test fails and it
reports the weakest level of the family, `x86-64-v1`, rather than `unknown`.
The sidecar then under-claims, and the loader's ISA guard cannot catch it: a
`.binpb` carrying AVX-512 code is passed onto a host without it, where it
fails as the illegal instruction the guard exists to convert into a
`LoadError`. The C++ side reads CPUID and has no such case. The fix is to
return `unknown` when the flags cannot be read, which the guard refuses rather
than trusts; it is not done, and until it is, an `isa_level` of `x86-64-v1` in
a sidecar written on a machine that is not one is the thing to be suspicious
of.

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
