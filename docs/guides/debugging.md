# Debugging

Almost everything that can go wrong goes wrong at load. That is deliberate: the
sidecar is read and cross-checked, the artifact is chosen, the arenas are
allocated and the warm-up calls are made before the caller's loop starts, so a
control process either fails at startup or runs.

Two exception types carry the failures. `pjrt::LoadError` is an artifact or
plugin problem — something a caller can act on. `pjrt::Error` wraps an error the
PJRT plugin returned and carries its status code. The rest are standard library
exceptions raised by the optional debug checks, and those are marked **(debug)**
below.

## Every message, and what to do about it

Angle brackets stand for whatever the message interpolates.

| Message | From | Cause | Fix |
|---|---|---|---|
| `no PJRT CPU plugin: set RuntimeOptions::plugin_path or $PJRT_CPU_PLUGIN, or run make plugin` | `LoadError`, plugin search | Nothing named a plugin: no option, no environment variable, and the build compiled in no default. | `make plugin`, or set `PJRT_CPU_PLUGIN`. |
| `dlopen(<path>): <dlerror>` | `LoadError`, plugin load | The file is missing, is for another architecture, or has an unresolved symbol — it is opened `RTLD_NOW`, so a missing `liblapack` shows up here rather than on the first `jnp.linalg` call. | `ldd` the plugin; install `liblapack3`/`libblas3`; check the path. |
| `<path>: no GetPjrtApi symbol (<why>); a PJRT plugin exports exactly this one function` | `LoadError`, plugin load | The shared object is not a PJRT plugin. A jaxlib `.so` is the usual mistake. | `tools/get_plugin.sh --check`. |
| `<path>: GetPjrtApi returned null` | `LoadError`, plugin load | The plugin loaded but declined to hand back an API table. | Rebuild or re-download the plugin. |
| `<path>: PJRT C API major version <n> but this build needs <m>` | `LoadError`, plugin load | ABI mismatch: struct fields have moved, so every call would be filled in at the wrong offsets. | Match the plugin to `PJRT_API_MAJOR` in `versions.env`. |
| `<path>: PJRT C API 0.<n> is older than the oldest version this code has been tested against (0.90)` | `LoadError`, plugin load | A plugin from an older XLA. | Rebuild the plugin from the pinned fork. |
| `<path>: PJRT plugin does not implement <PJRT_Function>, which pjrt_exec calls (plugin reports PJRT C API 0.<n>)` | `LoadError`, plugin load | A required entry point is null or past the plugin's `struct_size`. Checked up front so the failure names the function instead of being a segfault later. | Use a plugin built from the pinned XLA commit. |
| `Unexpected option name passed to PJRT_Client_Create` | `pjrt::Error` (`INVALID_ARGUMENT`), from the plugin | This XLA version validates create option names. Normally the runtime drops the named option and retries; you see this only with `allow_async_fallback = false`, or when the offending name cannot be matched to an option that was sent. | Leave the fallback on, or use the patched plugin that accepts the option. |
| `PJRT client reported no devices` | `LoadError`, client creation | `cpu_device_count` left the client with nothing addressable. | Set `cpu_device_count >= 1`. |
| `cannot open sidecar <path>: <strerror>` | `LoadError`, sidecar | Wrong base path, or the export never ran. Remember the path is given **without** an extension. | `make export`; check the path. |
| `<path> is not valid JSON: <detail>` | `LoadError`, sidecar | Truncated or hand-edited sidecar. | Re-export. |
| `sidecar schema <n> is newer than this loader (supports 1-2)` | `LoadError`, sidecar | The artifact was written by a newer `jax2exec` than this C++ build understands. | Upgrade the C++ side, or re-export with the matching exporter. |
| `<path>: a schema 2 sidecar must have both "inputs" and "outputs"` | `LoadError`, sidecar | Required keys missing. | Re-export. |
| `<path>: "inputs" and "outputs" must be arrays, in the order the executable takes and returns them` | `LoadError`, sidecar | Hand-edited into the wrong shape. | Re-export. |
| `<path>: a schema 1 sidecar must have both "args_info" and "out_info"` | `LoadError`, sidecar | A v1 sidecar missing its two blocks. | Re-export with the current exporter, which writes schema 2. |
| `<path>: input <i> ('<name>') has dtype '<dtype>', which pjrt_exec does not support (supported: bool, int8..int64, uint8..uint64, float32, float64)` | `LoadError`, sidecar | An element type with no C++ storage type. | Cast inside the traced function; see {doc}`exporting`. |
| `<path>: input <i> ('<name>') declares index <j>, but entries must be listed in executable order` | `LoadError`, sidecar | The array order was rearranged. Order *is* the calling convention. | Re-export. |
| `<path>: output <i> ('<name>') has dimension <n>; pjrt_exec loads static shapes only` | `LoadError`, sidecar | A negative (dynamic) dimension. | Export with static shapes. |
| `<path>: input <i> ('<name>') declares numel <n> but its shape <dtype[dims]> holds <m>` | `LoadError`, sidecar | The sidecar's arithmetic disagrees with its own shape. `numel` is what the loader allocates against, so it is rejected rather than reinterpreted. | Re-export; do not hand-edit. |
| `<path>: input <i> ('<name>') declares nbytes <n> but <m> <dtype> elements are <k> bytes` | `LoadError`, sidecar | As above, for the byte count. | Re-export. |
| `<path> is not a sidecar this loader can read: <detail>` | `LoadError`, sidecar | A type error inside the JSON — a string where a number belongs. | Re-export. |
| `cannot read <path>.binpb (missing or empty); export it, or load with LoadPolicy::CompileOnly` | `LoadError`, artifact | The executable is absent or zero bytes. | `make export`, or compile the `.mlirbc`. |
| `cannot read <path>.mlirbc (missing or empty); export it, or load with LoadPolicy::BinaryOnly` | `LoadError`, artifact | The fallback bytecode is absent — `write_mlir=False` at export, most likely. | Re-export with `write_mlir=True`. |
| `could not load <path>.binpb (<why>) and could not compile <path>.mlirbc (<why>)` | `LoadError`, artifact | Both routes failed. The two reasons are both in the message. | Read both halves; usually a stale `.binpb` plus a missing `.mlirbc`. |
| `could not allocate a <n>-byte aligned arena` | `LoadError`, arenas | `posix_memalign` failed. | Out of memory, or an `RLIMIT_AS` that `mlockall` is fighting. |
| `<name>.json declares <n> outputs but <name>.binpb produces <m>` | `LoadError`, metadata | The sidecar has gone stale relative to its executable. Without this check it is a buffer overrun found later as corrupted output. | Re-export both together; they are one unit. |
| `<name>.binpb reports <n> output element types for <m> outputs` / `reports dimensions for <n> outputs but <m> were expected` / `reported no output dimensions` | `LoadError`, metadata | The executable's self-description is internally inconsistent. | Re-export; report it if it persists. |
| `output <i> has element type <TYPE>, which pjrt_exec does not support (supported: ...)` | `LoadError`, metadata | The executable returns something like `BF16` that the sidecar did not mention. | Cast the output inside the traced function. |
| `<name>.json declares output <i> as <dtype[dims]> but <name>.binpb produces <dtype[dims]>` | `LoadError`, metadata | Sidecar and executable disagree about a result. | Re-export both. |
| `warm-up call <n> of <m> failed: <plugin error> (an input count/shape/dtype mismatch between the sidecar and the executable shows up here; the PJRT C API has no parameter-shape query)` | `LoadError` wrapping a `pjrt::Error` | The inputs are the half that cannot be cross-checked, so an input mismatch surfaces as a failed warm-up. | Read the wrapped message; the next two rows are the common cases. |
| `Execution supplied <n> buffers but compiled program expected <m>` | `pjrt::Error`, from the plugin, inside warm-up | **XLA pruned parameters the computation never reads**, so the executable takes fewer arguments than the sidecar declares. | Make every output depend on every input, or stop passing the argument. The exporter refuses this at export time now — an artifact that produces it predates that check. |
| `No FFI handler registered for lapack_dgetrf_ffi on a platform Host` | `pjrt::Error`, at deserialize or compile | The function lowers to a LAPACK custom call and the plugin registers no FFI handlers — i.e. it is a stock plugin, not one built from this project's fork. | Use the patched plugin ({doc}`../getting-started/installation`), or avoid `jnp.linalg` in the exported function. |
| `input index <i> is out of range: function '<name>' has <n> inputs` | `std::out_of_range` — always in `input_spec`/`output_spec`, **(debug)** in the accessors | An index past the end. | Resolve names with `find_input` instead of hard-coding indices. |
| `input <i> ('<name>') has dtype <a> but was accessed as <b>` | `std::invalid_argument` **(debug)** | `input<double>(i)` on a float32 arena — usually an artifact re-exported without `jax_enable_x64` under calling code that still says `double`. | Fix the export, or the accessor type. This is the check that catches the x64 trap. |
| `Function '<name>'::call() re-entered` | `std::logic_error` **(debug)** | `call()` entered from inside a call — a signal handler, or a callback. | Do not call from a signal handler. |
| `input <i> ('<name>') element <k> is nan` (or `inf`, `-inf`) | `std::domain_error`, `check_values` only | A non-finite value going in or coming out, named down to the element. | Look at what wrote that element; the check runs before *and* after the call so the caller and the computation can be told apart. |
| `output <i> ('<name>') element <k> is <v>, bool arenas must hold 0 or 1` | `std::domain_error`, `check_values` only | A `PRED` byte that is neither 0 nor 1. | See below — this is not pedantry. |
| *(no message; `load_kind()` reports `compiled` for an artifact exported on this machine)* | not an error | The `.binpb` did not deserialize and the `.mlirbc` was compiled instead. Usually the IFRT envelope jaxlib wraps the executable in changed shape and the exporter's unwrap no longer recognises it, so the written bytes are not what PJRT expects. Correct answers, a slower load, and no ahead-of-time compilation. | Check `artifacts.executable_source` in the sidecar and read `load_detail()`, which names the deserialize failure. See [Bumping JAX](../developer/bumping-jax.md). |

## The debug-only rows

Everything marked **(debug)** exists only under `FunctionOptions::debug`
(`check_values` for the last two). They cost a predictable branch on a member
that is always in cache, which is why they are cheap enough to leave on outside
a control loop.

:::{warning}
**In production, with debug off, the same mistakes are silent memory
corruption.** A wrong index reads or writes past an arena; a wrong accessor
type reads a float32 arena as doubles and walks off the end of it. Neither
throws, and neither is visible until something far away produces a wrong
number.

Develop with `debug = true` and `check_values = true`. Turn them off for the
measured run, not for the whole project.
:::

A stray byte in a bool arena deserves its own note. XLA does **not** normalize a
`PRED` byte: a 2 in there can make one predicate read as true and another read
as false within a single computation. The result is a wrong answer with no
failure anywhere to attach a bug report to, which is why `check_values` audits
bool arenas instead of trusting the caller to have written a clean 0 or 1.

## Failures that are not exceptions

**`SIGILL` inside the executable.** A `.binpb` carries machine code for the
machine that exported it, and deserializing relinks it without checking whether
this CPU has those instructions. With `isa_guard` on (the default under
`LoadPolicy::Auto`) the loader compares the sidecar's `isa_level` against this
host's and compiles the `.mlirbc` instead. With the guard off, or under
`BinaryOnly`, the failure is an illegal instruction with nothing in the
backtrace pointing at an artifact. `python -m jax2exec check <base>` answers the
question directly:

```text
instruction set
  host is x86-64-v2, exported on x86-64-v3: TOO WEAK
```

**A latency number that is wrong rather than noisy.** Not a failure the program
can report. See {doc}`measuring`.

## Getting a verbose load

Three things describe a running process, and all three belong in a bug report.

**`Runtime::describe()`** — log it once at startup:

```text
pjrt_exec 0.2.0 on cpu (...) through build/plugin/libpjrt_c_api_cpu_plugin.so,
PJRT C API <major>.<minor>; execution is inline on the calling thread (the
plugin advertises supports_synchronous_execution); 1 CPU device, PJRT_NPROC=1.
```

Read it in four parts: which plugin was actually opened (after the
option/environment/default search); the platform and API version, with a
warning appended when the plugin's minor version differs from the vendored
header's, because then the two may disagree about the trailing fields of the
`Args` structs; the execution mode, in the four flavours
{doc}`calling` describes; and the device and thread configuration. A
`max_inflight_computations` that was withheld because the plugin does not
advertise it is named too.

**`Function::load_detail()`** — which artifact, and why:

```text
deserialized artifacts/trajopt.binpb
compiled artifacts/trajopt.mlirbc in 2140.3 ms because artifacts/trajopt.binpb
  was exported for x86-64-v4 and this host is x86-64-v3
compiled artifacts/trajopt.mlirbc in 1980.1 ms because artifacts/trajopt.binpb
  could not be deserialized: <plugin error>
```

The second and third are the ISA guard and the post-upgrade fallback. Both are
successful loads that cost seconds instead of milliseconds, which is exactly
the kind of thing that should never be a surprise in a deployment — hence
`LoadPolicy::BinaryOnly`.

**`Function::fingerprint()`** — `PJRT_Executable_Fingerprint`, empty when the
plugin does not implement it. Two processes reporting the same fingerprint are
running the same compiled program. It is the cheap way to confirm that a
benchmark and a deployment are measuring the same thing.

And from the shell, before any of that:

```console
$ build/bin/plugin_probe            # what the plugin is and what it accepts
$ python -m jax2exec check <base>   # what the artifacts declare, and whether they run here
$ tools/rt_check.sh                 # what the host is willing to give a real-time loop
```
