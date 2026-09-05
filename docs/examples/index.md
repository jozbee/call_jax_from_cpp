# Examples

Three examples, each buildable from the tree with `make examples` and runnable
with `make run-examples`. They are ordered by how much of the real-time
machinery they use, and they share one artifact set: example 03 runs the
function example 02 exports.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 01 · Basic
:link: 01-basic
:link-type: doc

Export a small function and call it once. The shortest complete path from JAX
to a C++ result.

**Needs:** the fork plugin's LAPACK patch — this one calls `jnp.linalg.inv`.
:::

:::{grid-item-card} 02 · Trajectory optimization
:link: 02-trajopt
:link-type: doc

A workload heavy enough to time, with reference cases, a latency summary and a
JSON report. Numbers you can reproduce on your own machine.

**Needs:** any CPU plugin. Inline execution if you want the tail it reports.
:::

:::{grid-item-card} 03 · Real-time loop
:link: 03-realtime
:link-type: doc

The `pjrt::rt` hardening helpers, a fixed period, and a report of jitter,
deadline misses and allocations.

**Needs:** example 02's artifacts. `CAP_SYS_NICE` and an unlimited memlock for
the hardening to take effect.
:::

::::

## Before any of them

```console
$ uv sync                 # the Python environment; uv comes from mise
$ make plugin             # the prebuilt PJRT CPU plugin, sha256-verified
$ make                    # the library and the three example binaries
$ make export             # run the export scripts with JAX
```

`make export` is the step that has to happen on the machine that will run the
examples. A serialized executable embeds machine code for the exporting host,
so `artifacts/` is gitignored and every run target depends on the export rather
than on a committed file.

The plugin is a separate errand from the build. Nothing links against it — it
is `dlopen`-ed at run time — so a plain `make` never touches bazel or the
network, and a plugin supplied through `$PJRT_CPU_PLUGIN` is just as valid as
one in `build/plugin`.

## Which plugin features each one needs

Example 01 lowers `jnp.linalg.inv` to a LAPACK custom call, and **a bare PJRT
CPU plugin does not register LAPACK FFI handlers** — jaxlib registers those
from Python on import, which is not happening here. Without the fork's first
patch the load fails with:

```text
No FFI handler registered for lapack_dgetrf_ffi on a platform Host
```

`make plugin` downloads a plugin built from the fork, so this is only a problem
for someone substituting their own. Examples 02 and 03 do not lower to LAPACK
and run against any CPU plugin.

Inline synchronous execution is a different matter: it is advertised by the
fork's second patch, and a plugin without it still runs everything correctly,
just with the dispatch hand-off left in the call path.
`Runtime::synchronous_mode()` says which you have, and every example prints it.
