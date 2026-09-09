# 05 · ros2_control

*Assumes the {doc}`Quickstart </getting-started/quickstart>` and a reading
knowledge of `ros2_control`. Nothing from examples 01–04, though 03 is the
same hardening outside a plugin.*

`examples/05_ros2_control` is the executable form of
{ref}`Recipe 1 <rec-ros2>`: a `ros2_control` position controller for a
two-link planar arm, hosted by `controller_manager` and built with colcon,
whose numerics are an exported JAX function. What it demonstrates is where the
pieces go when the loop is not yours — one {cpp:class}`~pjrt::Runtime` for the
process, one {cpp:class}`~pjrt::Function` per controller created in
`on_configure`, the memory hardening in the same place, and an `update()` that
only copies, calls and writes back.

:::{note}
**Before the code.** Four namespaces appear. `pjrt::` is the library
({doc}`/api/index`) and `pjrt::rt::` its hardening layer ({doc}`/api/cpp/rt`).
`jax_arm_controller::` is this example's own and holds one class.
`controller_interface::` and `hardware_interface::` are
[ros2_control](https://control.ros.org/jazzy/index.html)'s: the controller base
class, and the loaned interfaces the manager hands `update()`. Nothing from
`examples/common/` appears — a controller is a shared object in someone else's
process, and the fewer headers it drags in the better.
:::

## The function

```{literalinclude} ../../examples/05_ros2_control/export.py
:language: python
:start-after: docs: begin arm-export
:end-before: docs: end arm-export
```

Resolved-rate control: `fk` is written once and `jax.jacfwd` differentiates
it, so changing the arm means changing `fk` and re-exporting rather than
deriving a Jacobian by hand. That is the reason the numerics are in JAX.
`jnp.linalg.solve` lowers to a LAPACK custom call, as example 01's
`jnp.linalg.inv` does, so this {term}`artifact` needs the fork's plugin and
fails at load against a stock one. {py:func}`~jax2exec.export` writes
`arm.binpb`, `arm.mlirbc` and `arm.json`; the {term}`sidecar` names the inputs
`q`, `t` and `dt`, which is the contract `on_configure` checks.

## Configure

```{literalinclude} ../../examples/05_ros2_control/src/jax_arm_controller.cpp
:language: cpp
:start-after: docs: begin ros2-configure
:end-before: docs: end ros2-configure
```

| Call | What it removes | Needs |
|---|---|---|
| {cpp:func}`~pjrt::rt::harden_malloc` | the allocator returning memory to the kernel, to be faulted back in by a later call | nothing; first, before the `Runtime` allocates in bulk |
| {cpp:func}`~pjrt::rt::lock_memory` | a {term}`page fault` inside a call | `RLIMIT_MEMLOCK` unlimited; after the load, so what it prefaults is what a call touches |
| {cpp:func}`~pjrt::rt::corral_xla_threads` | XLA's {term}`thread pool` waking on the update thread's core | the `Runtime` to exist, and the `xla_cpus` parameter to name somewhere to put them |

Two helpers are deliberately absent.
{cpp:func}`~pjrt::rt::pin_current_thread` and
{cpp:func}`~pjrt::rt::set_realtime_priority` act on the calling thread, and the
update thread is not this controller's: calling either from a plugin would
promote whichever thread happened to run `on_configure`. `controller_manager`
owns that thread and configures it from its own parameters —
`thread_priority` for {term}`SCHED_FIFO`, `lock_memory` for the process — both
set in `config/controllers.yaml`.

`runtime()`, the file's only free function, is a function-local static: one
client per process, shared by every controller the manager hosts, because a
second one starts a second set of XLA pools mid-run.
{cpp:enumerator}`~pjrt::LoadPolicy::BinaryOnly` makes a missing or foreign
`.binpb` a {cpp:class}`~pjrt::LoadError` at configure time rather than seconds
of silent compilation at startup; both exceptions the load can raise are
logged, and the controller reports `CallbackReturn::ERROR`.

## Update

```{literalinclude} ../../examples/05_ros2_control/src/jax_arm_controller.cpp
:language: cpp
:start-after: docs: begin ros2-update
:end-before: docs: end ros2-update
```

| Name | What it is |
|---|---|
| `state_interfaces_`, `command_interfaces_` | the loaned interfaces `controller_interface::ControllerInterface` fills in, in the order the two `*_interface_configuration()` methods asked for them — which is joint order |
| `get_optional()` | `hardware_interface::LoanedStateInterface`'s read, `std::nullopt` when another thread holds the handle; `value_or(q_[i])` then keeps last cycle's value rather than a zero |
| `set_value()` | `[[nodiscard]] bool` since ros2_control 4; a refused write is late data reaching the actuator, so it is reported as `return_type::ERROR` rather than dropped |
| `q_`, `t_`, `dt_`, `q_cmd_` | the four {term}`arenas <arena>`, resolved once in `on_configure` |
| {cpp:func}`~pjrt::Function::call` | execute, one await, one `memcpy` per output; allocation-free, lock-free and silent |

Nothing in the body allocates, locks or logs. `period` comes from the manager
rather than from a clock read here, so a cycle that ran late integrates the
time it actually took. An overrun is stale data, not a cancelled call: PJRT
cannot stop a running CPU computation.

## Build and run

The package is not part of `make`: colcon builds it, inside a ROS 2 workspace,
and the `ros2` compose service is one.

```console
$ make plugin && make export                            # the plugin, and artifacts/arm
$ docker compose -f docker/compose.yml build ros2       # ROS 2 Jazzy, once
$ docker compose -f docker/compose.yml run --rm ros2 \
    examples/05_ros2_control/run.sh --seconds 8         # colcon build, launch, stop, report
```

`run.sh --help` is its own header comment. Without `--seconds` the launch runs
until Ctrl-C; with it, the script samples `/joint_states` while the launch is
up and prints the first and last vectors it saw. Everything colcon writes goes
under `build/ros2/`.

## Reading the output

From a run of the command above, in the `ros2` service on this project's
development host, with the epoch timestamps elided:

```text
[ros2_control_node-2] [INFO] [controller_manager]: Successful set up FIFO RT scheduling policy with priority 50.
[ros2_control_node-2] [INFO] [controller_manager]: Configuring controller: 'jax_arm_controller'
[ros2_control_node-2] [INFO] [jax_arm_controller]: [ok  ] harden_malloc: M_TRIM_THRESHOLD=-1 M_MMAP_MAX=0 M_ARENA_MAX=1
[ros2_control_node-2] [INFO] [jax_arm_controller]: [ok  ] lock_memory: mlockall(MCL_CURRENT|MCL_FUTURE)
[ros2_control_node-2] [INFO] [jax_arm_controller]: deserialized /workspace/artifacts/arm.binpb (LoadPolicy::BinaryOnly)
[ros2_control_node-2] [INFO] [controller_manager]: Activating controllers: [ jax_arm_controller ]
[spawner-3] [INFO] [spawner_joint_state_broadcaster]: Configured and activated jax_arm_controller

=== /joint_states ===
first 0.480172438184073, 1.022513746356958
last  -0.35332078464935923, 2.223814587740879
```

Both steps report `[ok  ]` because the compose service grants `CAP_SYS_NICE`
and an unlimited memlock; without them `lock_memory` reports `[skip]` and the
run is still correct. `corral_xla_threads` prints nothing here: `xla_cpus` is
empty, so it is never attempted. Nothing touches `/dev/cpu_dma_latency` — the
{term}`C-state` hold is the host's business, not a plugin's. The first line is
the manager's, and it is the half of the hardening this controller cannot do
for itself.

The two joint vectors are the evidence that the loop ran: the arm starts at
the pose `urdf/arm.urdf` gives the mock hardware and ends somewhere else, so
`update()` was called, `call()` returned, and the commands reached the
interfaces. `tests/test_examples_ros2.py` asserts exactly those four things.

## Making it yours

Replace `fk` and the joint list; keep the order in `on_configure`. The three
parameters are the whole configuration surface — `joints`, `artifact` (a base
path with no extension) and `xla_cpus` — and the artifact path is one line of
`config/controllers.yaml`, naming a `.binpb` exported on the machine that will
run it. A package of your own carries the library as a submodule and installs
its own copy of the plugin and the artifacts, which is what
{doc}`/guides/integration` spells out; this one reaches two directories up
into the checkout instead, so that what the docs show is what builds.
{doc}`/guides/realtime` says what each hardening call buys, and
{doc}`03-minimal` is the same order in a program that owns its loop.
