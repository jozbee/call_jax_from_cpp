# Integrating into your build

*Assumes the {doc}`Quickstart </getting-started/quickstart>` built the tree
once. Pick the recipe for your build and stop there.*

For dropping this library into an existing C++ project. Pick the row that
matches your build, follow that recipe, and stop — each one is complete.

Three facts shape all of them:

- **Nothing links against the {term}`PJRT plugin`.** It is `dlopen`-ed at run time, so
  integrating adds two include roots and `-ldl -lpthread` to a link line — no
  bazel, no network access, no XLA anywhere in your build.
- **The library is vendored, not installed.** Three translation units and no
  `install()` rules. A system-wide library shared between two projects pinned
  to different JAX versions would couple them for no gain.
- **Do not clone recursively.** `third_party/xla` is a full XLA checkout, used
  only to build the plugin from source.

## Which recipe

| Your build | Sources | Recipe |
|---|---|---|
| ROS 2 (`ament_cmake` / colcon) | Submodule inside the package | [1](#rec-ros2) |
| CMake, and you can carry a submodule | Git submodule + `add_subdirectory` | [2](#rec-submodule) |
| CMake, and you would rather not | `FetchContent` on a pinned tag | [3](#rec-fetchcontent) |
| Make, Bazel, Meson, anything else | Copied or vendored sources | [4](#rec-make) |

All four end in the same place: a binary that links `pjrt_exec::pjrt_exec` (or
its three objects), finds a plugin at run time, and loads artifacts written by
`jax2exec`.

(rec-ros2)=
## Recipe 1 — ROS 2

The motivating application is a controller under `ros2_control`, so this
recipe comes first. **No ROS code ships in this repository.** What follows is
the build integration and the three design constraints a controller has to
respect, stated once.

**Package layout.**

```text
my_controller/
  package.xml
  CMakeLists.txt
  src/my_controller.cpp
  plugin/libpjrt_c_api_cpu_plugin.so   # from `make plugin`, for the target machine
  artifacts/                           # exported on the target machine
  third_party/call_jax_from_cpp/       # git submodule; never --recursive
```

```console
$ git submodule add https://github.com/jozbee/call_jax_from_cpp.git \
    third_party/call_jax_from_cpp
```

**`package.xml`.**

```xml
<buildtool_depend>ament_cmake</buildtool_depend>
<depend>ament_index_cpp</depend>   <!-- get_package_share_directory -->
```

**`CMakeLists.txt`.**

```cmake
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Vendored, not found: three translation units pinned to one JAX version.
set(PJRT_EXEC_FETCH_PLUGIN OFF CACHE BOOL "" FORCE)   # colcon builds offline
add_subdirectory(third_party/call_jax_from_cpp)

add_library(my_controller SHARED src/my_controller.cpp)
target_link_libraries(my_controller PRIVATE pjrt_exec::pjrt_exec
                                            ament_index_cpp::ament_index_cpp)

# The plugin and the artifacts are runtime data, not build outputs.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/plugin/libpjrt_c_api_cpu_plugin.so
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/
        DESTINATION share/${PROJECT_NAME}/artifacts)
ament_package()
```

Resolve both paths at run time from
`ament_index_cpp::get_package_share_directory`, and pass the plugin through
{cpp:member}`~pjrt::RuntimeOptions::plugin_path` rather than a compiled-in
default.

**Where the objects live.** `controller_manager` hosts every controller in one
process, and a {cpp:class}`~pjrt::Runtime` is one per process — creating a client starts XLA's
thread pools, and creating a second one mid-run is a latency spike. So:

- **One `Runtime` for the process**, behind an accessor — a function-local
  `static std::shared_ptr<pjrt::Runtime>` is enough — shared by every
  controller.
- **One {cpp:class}`~pjrt::Function` per controller**, created in
  `on_configure` with {cpp:enumerator}`~pjrt::LoadPolicy::BinaryOnly`, never
  in `update()`. Loading takes milliseconds
  for a `.binpb` and seconds if it falls back to compiling the `.mlirbc`.

**One side effect to decide once.** The first `Runtime` constructed calls
`setenv("PJRT_NPROC", worker_threads)` — that is where XLA sizes its pools —
and every client created later in the process inherits it. Choose
`worker_threads` once, process-wide, and say so in the controller's own
documentation.

**Hardening inside a plugin.** The update thread belongs to
`controller_manager`. Call the memory helpers —
{cpp:func}`~pjrt::rt::harden_malloc`, {cpp:func}`~pjrt::rt::lock_memory` —
from `on_configure`, and leave affinity and {term}`SCHED_FIFO` to the
manager's configuration rather than calling
{cpp:func}`~pjrt::rt::pin_current_thread` or
{cpp:func}`~pjrt::rt::set_realtime_priority` on a thread you do not own.
{doc}`realtime` says what each one buys.

**A deadline is not a cancellation.** PJRT cannot cancel a running CPU
computation. An overrun means the controller reads a stale result, not that
the call was aborted; design the fallback around stale data.

**Verify.** `colcon build --packages-select my_controller` → builds with no
plugin on the link line, and
`ls install/my_controller/lib/my_controller/libpjrt_c_api_cpu_plugin.so` finds
the installed plugin.

(rec-submodule)=
## Recipe 2 — CMake + git submodule

The consumer controls the version by moving a submodule pointer, and nothing
is fetched during a build.

**Prerequisites.** CMake ≥ 3.21, a C++17 compiler, and a PJRT CPU plugin for
the JAX version the artifacts were exported with (see
[the plugin at run time](#the-plugin-at-run-time)).

**Files to add.**

```console
$ git submodule add https://github.com/jozbee/call_jax_from_cpp.git \
    third_party/call_jax_from_cpp
```

`git submodule add` does not descend into our submodules, which is what you
want; `--recursive` would pull down `third_party/xla`.

**Snippet.** `examples/01_basic/CMakeLists.txt` is exactly this, configured on
its own so that what the docs show is what builds:

```{literalinclude} ../../examples/01_basic/CMakeLists.txt
:language: cmake
:start-after: docs: begin cmake-submodule
:end-before: docs: end cmake-submodule
```

The library brings its own include directories, `cxx_std_17`, `dl` and
`Threads::Threads`, and is position-independent. Cache options:
`PJRT_EXEC_FETCH_PLUGIN` (download the published plugin at configure time,
default `ON`), `PJRT_EXEC_PLUGIN_PATH` (use one you already have), and
`PJRT_EXEC_BUILD_EXAMPLES` / `_TESTS` / `_BENCH`, which default to `OFF` when
this project is not the top level.

**Verify.**

```console
$ cmake -S . -B build && cmake --build build --target controller
$ ldd build/controller | grep pjrt || echo "plugin not linked, as intended"
```

→ expected: `plugin not linked, as intended`. If the plugin appears in `ldd`
output, some part of your build is linking it, which fails on any machine
whose plugin lives somewhere else.

(rec-fetchcontent)=
## Recipe 3 — CMake + FetchContent

For a project that does not want a submodule. The cost is a network fetch at
configure time and the requirement to pin.

```cmake
include(FetchContent)
FetchContent_Declare(pjrt_exec
  GIT_REPOSITORY https://github.com/jozbee/call_jax_from_cpp.git
  GIT_TAG        v0.2.0          # pin a tag; never a branch
  GIT_SHALLOW    TRUE
  GIT_SUBMODULES ""              # do NOT fetch third_party/xla
)
FetchContent_MakeAvailable(pjrt_exec)

target_link_libraries(controller PRIVATE pjrt_exec::pjrt_exec)
```

`GIT_SUBMODULES ""` is the line that matters: without it FetchContent clones
the XLA submodule.

**Verify.** `cmake -S . -B build` → expected: a configure-time status line
`pjrt_exec 0.2.0: JAX <version>, plugin release <tag>, default plugin <path>`.

(rec-make)=
## Recipe 4 — Copied sources plus `pjrt_exec.mk`

For a build that is not CMake. Copy or submodule the tree — at minimum
`include/`, `src/pjrt_exec/`, `third_party/pjrt/`, `third_party/nlohmann/`
and `pjrt_exec.mk` — and include one fragment. It defines
`PJRT_EXEC_CPPFLAGS`, `PJRT_EXEC_SRCS`, `PJRT_EXEC_OBJS`, `PJRT_EXEC_LIB` and
`PJRT_EXEC_LDLIBS`, and touches nothing else:

```{literalinclude} ../../pjrt_exec.mk
:language: make
:start-after: docs: begin make-fragment
:end-before: docs: end make-fragment
```

**Verify.** `make controller && nm -C build/controller | grep -c 'pjrt::Function'`
→ expected: a non-zero count on an unstripped binary.

## The plugin at run time

Every recipe needs a plugin on the target machine, matching the JAX version
the artifacts were exported with. `Runtime` looks in three places, in order:
`RuntimeOptions::plugin_path`, `$PJRT_CPU_PLUGIN`, then the path compiled in
at build time — wherever `make plugin` writes.

Getting one: `make plugin` downloads the published, sha256-verified asset,
`make plugin-source` builds it from the XLA fork, and a CMake consumer can let
`PJRT_EXEC_FETCH_PLUGIN` do it at configure time. A stock plugin also works,
with two losses: a LAPACK custom call will not load, and inline execution
cannot be confirmed ({doc}`../developer/xla-fork`).

## The artifacts at run time

A `.binpb` embeds machine code for the machine that exported it; it is
relinked at load, never recompiled. Export on the deployment machine, or on
one with the same architecture and instruction-set level, and ship the
`.mlirbc` alongside so the loader has somewhere to fall back to.
`python -m jax2exec check <base>` answers "will this artifact run here"
without running it.

## Toggles

Each is independent of the recipe.

**Real-time hardening.** Optional, Linux-only, applied once after loading and
before the loop, in an order that matters. {doc}`realtime` lists every helper
and what each one buys.

**Debug mode.**

```cpp
pjrt::FunctionOptions opts;
opts.debug = true;         // bounds, typed-accessor dtype, call() re-entrancy
opts.check_values = true;  // no nan/inf in float arenas, no stray bool bytes
pjrt::Function f(runtime, "artifacts/trajopt", opts);
```

Run with `debug` on in development: off, the same mistake is silent memory
corruption rather than an exception, and the checks cost nothing when off.
`check_values` walks every element of every arena, so it belongs in
development and acceptance tests, not in a loop.

**The compile fallback.**

```cpp
pjrt::FunctionOptions opts;
opts.load_policy = pjrt::LoadPolicy::BinaryOnly;  // for a deployment
```

`LoadPolicy::Auto` prefers the `.binpb` and compiles the `.mlirbc` when the
binary is missing or built for a wider instruction set than this host. Right
for a workstation, wrong for a control process: compiling takes seconds,
silently, at startup. Ask for `BinaryOnly` and get a `LoadError` naming the
mismatch instead — or, if a heterogeneous fleet makes per-machine export
impractical, allow the fallback and log `Function::load_kind()` at startup.
