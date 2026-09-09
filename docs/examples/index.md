# Examples

Five programs. Four build from the tree with `make examples` and run with
`make run-examples`; the fifth is a `ros2_control` package and builds with
colcon, inside a ROS 2 workspace. They share one artifact set: 01 and 03 run
the function example 01 exports, 02 and 04 the one example 02 exports, and 05
its own. Read them in any order; each page names what it borrows from the
others.

::::{grid} 1 2 2 2
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

:::{grid-item-card} 03 · Minimal real-time loop
:link: 03-minimal
:link-type: doc

The `pjrt::rt` calls in the one safe order, an absolute-time sleep and two
recorders, in one file you can copy. Skip 01 and 02 if the loop is your use
case.

**Needs:** example 01's artifact. `CAP_SYS_NICE` and an unlimited memlock for
the hardening to take effect.
:::

:::{grid-item-card} 04 · Real-time loop, instrumented
:link: 04-realtime
:link-type: doc

Example 03 with the measurement attached: the host audit, deadline
accounting, page faults and context switches, the allocation census, a JSON
report.

**Needs:** example 02's artifact; the same privileges as 03.
:::

:::{grid-item-card} 05 · ros2_control
:link: 05-ros2-control
:link-type: doc

The same hardening from inside a `controller_manager` plugin: one `Runtime`
for the process, one `Function` per controller, and an `update()` that only
copies, calls and writes back.

**Needs:** the fork plugin's LAPACK patch, and a ROS 2 Jazzy workspace — the
`ros2` compose service is one.
:::

::::

## Before any of them

```console
$ uv sync                 # the Python environment; uv comes from mise
$ make plugin             # the prebuilt PJRT CPU plugin, sha256-verified
$ make                    # the library and the four example binaries
$ make export             # run the export scripts with JAX
```

Example 05 is not built by `make`: it is an `ament_cmake` package, and
`examples/05_ros2_control/run.sh` builds and launches it with colcon inside a
ROS 2 workspace.

`make export` is the step that has to happen on the machine that will run the
examples. A serialized executable embeds machine code for the exporting host,
so `artifacts/` is gitignored and every run target depends on the export rather
than on a committed file.

The plugin is a separate errand from the build. Nothing links against it — it
is `dlopen`-ed at run time — so a plain `make` never touches bazel or the
network, and a plugin supplied through `$PJRT_CPU_PLUGIN` is just as valid as
one in `build/plugin`.

## Which plugin features each one needs

Example 01 lowers `jnp.linalg.inv` to a LAPACK custom call, which only the
fork's plugin can load; a stock plugin fails with `No FFI handler registered
for lapack_dgetrf_ffi`. Example 03 loads the same artifact, and example 05's
`jnp.linalg.solve` needs the same patch. Examples 02 and 04 run against any
CPU plugin. {term}`Inline execution <inline execution>`
needs the fork's second patch; without it everything is still correct, only
slower in the tail, and {cpp:func}`~pjrt::Runtime::synchronous_mode` says
which you have.

## The shared layer

Examples 01 through 04 share `examples/common/`, namespace `cjfc` —
call_jax_from_cpp — a set of headers written to be copied: a flag parser, the
absolute-time sleep and stop flag, the host audit and the hardening order,
the JSON report, and the trajopt workload's I/O contract
(`cjfc::workload`). It is documented on {doc}`/api/cpp/examples` and is not
part of the library. Example 05 uses none of it: a controller is a shared
object in someone else's process, and it includes the library and nothing
else.
