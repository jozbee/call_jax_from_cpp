# The XLA fork

`third_party/xla` is a submodule of a fork of OpenXLA — {{ xla_fork_repo }},
branch `{{ xla_fork_branch }}`, currently at `{{ xla_fork_commit }}`. It sits on
top of `{{ xla_commit }}`, the XLA revision that JAX {{ jax_version }} pins
(`third_party/xla/revision.bzl` at the `jax-v{{ jax_version }}` tag).

The fork exists for one reason: there is no official prebuilt CPU PJRT C-API
plugin. jaxlib links its CPU client statically and never exports `GetPjrtApi`,
so a C++ caller has to build the plugin, and once you are building it there are
two things worth changing.

Nothing in this repository depends on the fork at build time. The plugin is
`dlopen`-ed at run time, `make` never touches bazel, and a stock unpatched
plugin will load and run most artifacts. The fork buys LAPACK support and
inline execution, not the ability to function.

## The two patches

Two changes, carried as three commits, because re-vendoring the kernels for a
new JAX release is its own step and deserves its own diff. They are also kept
as `third_party/patches/*.patch` so the fork can be reconstructed from this
repository alone.

### Patch 1 — jaxlib's LAPACK FFI kernels

Vendors `jaxlib/cpu/{lapack_kernels,cpu_kernels,ffi_helpers}.*` from the
`jax-ml/jax` repository at the pinned release tag into a new
`//jaxlib_cpu_kernels` package, and links `pjrt_c_api_cpu_plugin.so` against it
plus the system `liblapack` and `libblas`. These are the files the JAX team
ships specifically for calling JAX-generated HLO from outside JAX: their own
guidance is that a C++ user should link against LAPACK directly.

**What breaks without it.** An executable compiled from a JAX function that
lowers to a LAPACK-backed custom call — `jnp.linalg.inv` needs
`lapack_dgetrf_ffi` — fails at load with:

```
No FFI handler registered for lapack_dgetrf_ffi on a platform Host
```

A bare plugin never registers those handlers, because what registers them
normally is jaxlib's Python extension, on import. Nothing imports it here.

The second commit re-vendors these kernels for the current JAX release. They
changed substantially between the last pin and this one — `lapack_kernels.cc`
grew from 1656 to 2644 lines — and the FFI handler signatures have to match
what the pinned release lowers to, so the copies are refreshed rather than
carried forward. It also drops the sparse and tridiagonal handler
registrations, because their kernels are not vendored and registering them
would not link: an executable that lowers to `cpu_csr_sparse_dense_ffi` or
`tridiagonal_solve_perturbed_ffi` fails to load with the same clear message
above.

### Patch 2 — CPU plugin create options and their advertisement

Sixty lines in one file, `xla/pjrt/c/pjrt_c_api_cpu_internal.cc`. It accepts
`max_inflight_computations` (mapped to
`CpuClientOptions::max_inflight_computations_per_device`) and adds a
`PJRT_Plugin_Attributes_Cpu` that returns `GetXlaPluginCAttributes()` plus three
`int64` markers: `supports_synchronous_execution`,
`supports_max_inflight_computations` and `cjfc_plugin_patch_level`.

**Why the markers matter more than the option.** Upstream now parses
`asynchronous` itself, so that half of the older patch is gone. Upstream also
now *validates* option names and rejects anything it does not know, which means
a caller can no longer discover the surface by trying an option — an unknown
name fails `PJRT_Client_Create` outright. `PJRT_Plugin_Attributes` is the only
place left where a plugin can say what it accepts, and it is readable before any
client exists. That is what the markers are for, and it is why
`Runtime` queries attributes before it creates a client.

**Why `asynchronous=false` is worth having at all.** It makes XLA run
computations inline on the calling thread instead of handing them to its
dispatch pool, which is the largest structural cut to tail latency available to
a real-time caller. The PJRT C API offers no other route:
`PJRT_ExecuteOptions` has no execution-mode field.

**Why it is safe to lose.** Dropping this patch costs performance, never
correctness. Without it the plugin advertises nothing, so `Runtime` withholds
`max_inflight_computations` rather than having creation rejected, and reports
`SyncMode::Accepted` at best — inline execution may still be in effect through
upstream's own parsing, but nothing can confirm it. A stock plugin still loads,
still runs, and still produces the same numbers, more slowly at the tail.

## Building the plugin

```
make plugin-source                 # tools/build_plugin.sh --out build/plugin
docker compose -f docker/compose.yml run --rm plugin-builder \
    tools/build_plugin.sh          # same build, in the image that has bazel
```

Budget 30–60 minutes on a real machine with a cold cache — bazel compiles LLVM
from source — and around 25 GB of free disk. The build needs `bazel` (or
bazelisk), `xxd`, and `liblapack-dev`/`libblas-dev` for patch 1's link step;
`tools/build_plugin.sh` checks for each and says what is missing rather than
failing halfway through a link.

Two details that cost time to rediscover. XLA at this pin sets
`common --noenable_bzlmod --enable_workspace` in its own `.bazelrc`, so the tree
still builds in WORKSPACE mode and no `--config=bzlmod` is wanted. And
`.bazelversion` is 8.7.0, which bazelisk will fetch for you; a system bazel of
another version is the usual cause of a confusing early failure.

`--package` produces the release tarball and its `.sha256`, plus a
`PLUGIN_INFO.txt` recording the JAX version, the XLA commit, the fork commit
and branch, the PJRT API minor, the build mode, the host architecture and the
maximum glibc symbol version the binary requires. That file is what makes a
published asset auditable later. Release assets are built **without**
`ARCH_FLAGS`, baseline for the architecture, so they run on any machine of that
kind; the ISA-locked part of this system is the serialized executable, not the
plugin.

**Do not run this build concurrently with a benchmark.** It saturates every
core, and a number measured next to it is invalid rather than noisy — see
[measurement](measurement.md).

## Build mode does not matter — do not repeat this experiment

The plugin originally shipped from a bazel `fastbuild` (`-O0`) build, and
rebuilding it `-c opt` was expected to be a large win. It moved p50 by **0.2%**:
4718.7 µs versus 4728.4 µs.

The reason is worth remembering, because it generalises. The compute kernels are
LLVM-compiled at *export* time and embedded in the `.binpb`. The plugin only
orchestrates: it creates buffers, dispatches, and waits. Optimising the
orchestration does not touch the arithmetic.

`opt` is still the default and is what gets published, because that is what a
shipped binary should be. But the twelve-minute rebuild is off the critical
path, and a "let me just rebuild the plugin with better flags" idea has already
been tried.

## A candidate third patch, not yet justified

A pooling allocator installed behind `CpuClientOptions::allocator` — also
unreachable through the C API — would address the thousands of allocations per call
inside XLA's thunk runtime, which is the last known structural jitter surface.

It is gated on evidence, deliberately. Build it when a measurement on real
hardware, on an idle tuned host, shows that plugin-internal allocation drives a
residual tail. Until then it is a large patch to carry against a cost nobody has
demonstrated.

## The lost-commit lesson

The previous version of patch 2 was committed locally, and the submodule pointer
in this repository named that commit. It was never pushed. The commit therefore
could not be fetched by anyone — including its author, on the same machine,
after the working tree had moved on — and the patch had to be written again from
the prose description that survived in these notes.

Nothing about that failure was visible from this repository. The submodule
pointer looked correct, `git submodule status` was happy, and the local build
worked. It broke for the first time when somebody else, or CI, tried to check
out the tree.

So: **a patch must exist in three places before anything is allowed to depend on
it.**

1. **A pushed branch**, referenced by commit hash. Pushed, not committed.
2. **A regenerated patch file** under `third_party/patches/`, from
   `git -C third_party/xla format-patch "$XLA_COMMIT..HEAD" -o third_party/patches --no-signature`.
3. **`XLA_FORK_COMMIT` in `versions.env`**, naming that same pushed hash.

A rebase is not finished until all three agree. The check is one command:

```
git -C third_party/xla rev-parse HEAD
git -C third_party/xla ls-remote origin <branch>    # must contain that hash
grep XLA_FORK_COMMIT versions.env                   # must equal it too
```

[Bumping JAX](bumping-jax.md) has this as an explicit step with a verify line,
which is the only reason to trust that it happens.
