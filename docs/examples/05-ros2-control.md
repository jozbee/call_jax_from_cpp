# 05 · ros2_control

*Assumes the {doc}`Quickstart </getting-started/quickstart>` and a reading
knowledge of `ros2_control`. Nothing from examples 01–04, though 03 is the
same hardening outside a plugin.*

`examples/05_ros2_control` is the executable form of
{ref}`Recipe 1 <rec-ros2>`: a `ros2_control` position controller for a
two-link planar arm whose numerics are an exported JAX function. It shows
where the pieces go when the loop is not yours — one
{cpp:class}`~pjrt::Runtime` for the process, one {cpp:class}`~pjrt::Function`
per controller created in `on_configure`, the memory hardening in the same
place, and an `update()` that only copies, calls and writes back.

:::{note}
**Before the code.** Three namespaces appear. `pjrt::` is the library
({doc}`/api/index`) and `pjrt::rt::` its hardening layer ({doc}`/api/cpp/rt`).
`controller_interface::` is
[ros2_control](https://control.ros.org/jazzy/index.html)'s controller base
class; `RCLCPP_ERROR` and `RCLCPP_INFO` are rclcpp's logging macros. Nothing
from `examples/common/` appears.
:::

## The function

```{literalinclude} ../../examples/05_ros2_control/export.py
:language: python
:start-after: docs: begin arm-export
:end-before: docs: end arm-export
```

Resolved-rate control: `fk` is written once and `jax.jacfwd` differentiates
it, so changing the arm means changing `fk` and re-exporting, not deriving a
Jacobian by hand. `DAMPING` keeps the solve finite where the arm is straight
and the Jacobian singular, which is also why `urdf/arm.urdf` starts it bent.
`jnp.linalg.solve` lowers to a LAPACK custom call, so this {term}`artifact`
needs the fork's plugin. {py:func}`~jax2exec.export` writes the `.binpb`,
`.mlirbc` and `.json`; the {term}`sidecar` names the inputs `q`, `t` and
`dt`. `on_configure` resolves them by index and checks only that `q` has one
element per joint; `tests/test_examples_ros2.py` pins the names and their
order, so a re-export that reorders them fails there, not as an arm that
moves strangely.

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

Each returns a {cpp:struct}`~pjrt::rt::Status`; `report`, the file's own
helper, logs it as the `[ok  ]`/`[skip]` lines under
[Reading the output](#reading-the-output).

{cpp:func}`~pjrt::rt::pin_current_thread` and
{cpp:func}`~pjrt::rt::set_realtime_priority` are absent: both act on the
calling thread, and the update thread is `controller_manager`'s, whose
`thread_priority` parameter in `config/controllers.yaml` does that half.
`runtime()` is the process's one `Runtime`.
{cpp:enumerator}`~pjrt::LoadPolicy::BinaryOnly` makes a missing or foreign
`.binpb` a {cpp:class}`~pjrt::LoadError` at configure time rather than
seconds of silent compilation; the `catch` turns it, and anything else the
load throws, into a logged line and `CallbackReturn::ERROR`, so the manager
never activates a controller with no function.

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
| `set_value()` | `[[nodiscard]] bool` since ros2_control 4; a refused write leaves last cycle's command standing — stale data, the same policy as an overrun — and is counted, reported once at deactivation. Returning `return_type::ERROR` instead would have the manager deactivate the controller |
| `q_`, `t_`, `dt_`, `q_cmd_` | the four {term}`arenas <arena>`, resolved once in `on_configure` |
| {cpp:func}`~pjrt::Function::call` | execute, one await, one `memcpy` per output; allocation-free, lock-free and silent |

Nothing in the body allocates, blocks or logs: the loaned handles try-lock
and yield rather than wait, and give up after ten tries. `period` comes from
the manager, so a cycle that ran late integrates the time it actually took.
An overrun is stale data, not a cancelled call: PJRT cannot stop a running
CPU computation.

## Build and run

```console
$ make plugin && make export                            # the plugin, and artifacts/arm
$ docker compose -f docker/compose.yml build ros2       # ROS 2 Jazzy, once
$ docker compose -f docker/compose.yml run --rm ros2 \
    examples/05_ros2_control/run.sh --seconds 8         # colcon build, launch, stop, report
```

The package is not part of `make`: colcon builds it inside a ROS 2 workspace,
and the `ros2` compose service is one. Without `--seconds` the launch runs
until Ctrl-C; with it, `run.sh` samples `/joint_states` while the launch is
up, prints the first and last vectors it saw, and exits non-zero unless the
run showed the four facts below. CI runs that command and relies on the exit
status; `run.sh --help` is its own header comment.

## Reading the output

From a run of the command above in the `ros2` service on this project's
development host: the lines about this controller, timestamps elided. Cut:
the colcon build, the launch's and the manager's other lines, the
broadcaster's, and the plugin's absl start-up chatter.

```text
[ros2_control_node-2] [INFO] [controller_manager]: Successful set up FIFO RT scheduling policy with priority 50.
[ros2_control_node-2] [INFO] [controller_manager]: Configuring controller: 'jax_arm_controller'
[ros2_control_node-2] [INFO] [jax_arm_controller]: [ok  ] harden_malloc: M_TRIM_THRESHOLD=-1 M_MMAP_MAX=0 M_ARENA_MAX=1
[ros2_control_node-2] [INFO] [jax_arm_controller]: [ok  ] lock_memory: mlockall(MCL_CURRENT|MCL_FUTURE)
[ros2_control_node-2] [INFO] [jax_arm_controller]: deserialized /workspace/artifacts/arm.binpb (LoadPolicy::BinaryOnly)
[ros2_control_node-2] [INFO] [controller_manager]: Activating controllers: [ jax_arm_controller ]
[spawner-3] [INFO] [spawner_joint_state_broadcaster]: Configured and activated jax_arm_controller

=== /joint_states ===
first 0.4607781236559495, 1.0439951470612612
last  -0.26244714029634797, 2.1463853760531397
```

`harden_malloc` needs no privilege. `lock_memory` reports `[ok  ]` because
the compose service grants an unlimited memlock; without it the line reads
`[skip]` and the run is still correct. The manager's first line needs
`CAP_SYS_NICE`, which the service also grants; it is the half of the
hardening this controller cannot do for itself. `corral_xla_threads` prints
nothing: `xla_cpus` is empty, so it is never attempted. Nothing touches
`/dev/cpu_dma_latency` — the {term}`C-state` hold is the host's business.

The two joint vectors are the evidence that the loop ran: the arm starts at
the pose `urdf/arm.urdf` gives the mock hardware and ends somewhere else, so
`update()` was called, `call()` returned, and the commands reached the
interfaces. `run.sh` and `tests/test_examples_ros2.py` both assert the two
hardening lines, the activation line, and that the two vectors were
published and differ.

## Making it yours

Replace `fk` and the joint list; keep the order in `on_configure`. The three
parameters — `joints`, `artifact` (a base path, no extension) and `xla_cpus`
— are the whole configuration surface, and the artifact path is one line of
`config/controllers.yaml`, naming a `.binpb` exported on the machine that will
run it. The plugin is the other path: `PJRT_EXEC_PLUGIN_PATH` in
`CMakeLists.txt` is the `.so` compiled in as the default, so point it at your
own copy — or set {cpp:member}`~pjrt::RuntimeOptions::plugin_path` in
`runtime()` and drop the line. A package of your own installs the plugin and
the artifacts itself, as {doc}`/guides/integration` spells out.
{doc}`/guides/realtime` says what each hardening call buys, and
{doc}`03-minimal` is the same order in a program that owns its loop.
