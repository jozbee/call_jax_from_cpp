# Integration recipes

For dropping this library into an existing C++ project. Pick the row of the
table that matches the build, follow that recipe, and stop — the recipes are
complete, not sketches.

Three facts shape all of them:

- **Nothing links against the PJRT plugin.** It is `dlopen`-ed at run time, so
  integrating adds two include roots and `-ldl -lpthread` to a link line, and
  no bazel, no network access, and no XLA anywhere in the build.
- **The library is vendored, not installed.** Three translation units, no
  `install()` rules, and that is deliberate: a system-wide `libpjrt_exec.so`
  shared between two projects pinned to different JAX versions is a support
  burden with no upside.
- **Do not clone this repository recursively.** `third_party/xla` is a full XLA
  checkout and is needed *only* to build the plugin from source. `git submodule
  update --init` on it costs gigabytes for something a consumer never uses.

## Which recipe

| Your build | How you want the sources | Recipe |
|---|---|---|
| CMake, and you can carry a submodule | Git submodule + `add_subdirectory` | [1](#rec-submodule) — recommended |
| CMake, and you would rather not | `FetchContent` on a pinned tag | [2](#rec-fetchcontent) |
| Make, Bazel, Meson, anything else | Copied or vendored sources | [3](#rec-make) |
| CMake under ament/colcon (ROS 2) | Submodule inside the package | [4](#rec-ros2) |

All four end in the same place: a binary that links `pjrt_exec::pjrt_exec` (or
its three objects), finds a plugin at run time, and loads artifacts exported by
`jax2exec`.

(rec-submodule)=
## Recipe 1 — CMake + git submodule

The recommended one. The consumer controls the version by moving a submodule
pointer, and nothing is fetched during a build.

**Prerequisites.** CMake ≥ 3.21, a C++17 compiler, and a PJRT CPU plugin for the
JAX version the artifacts were exported with (see [the plugin at run
time](#the-plugin-at-run-time)).

**Files to add.**

```
git submodule add https://github.com/jozbee/call_jax_from_cpp.git \
    third_party/call_jax_from_cpp
```

`git submodule add` does not descend into our submodules, which is what you
want. Keep it that way: `git submodule update --init --recursive` in your
project would pull down `third_party/xla`, a full XLA checkout that only the
source build of the plugin uses.

**Snippet.** In your `CMakeLists.txt`:

```cmake
# The examples, tests and benchmark default to OFF when this project is not the
# top-level one, so nothing of ours enters your build but the library.
add_subdirectory(third_party/call_jax_from_cpp)

add_executable(controller controller.cpp)
target_link_libraries(controller PRIVATE pjrt_exec::pjrt_exec)
```

The library brings its own include directories (`include/` and `third_party/`,
the second because the public headers include `"pjrt/pjrt_c_api.h"` and
`"nlohmann/json.hpp"`), `cxx_std_17`, `dl` and `Threads::Threads`. It is built
`POSITION_INDEPENDENT_CODE`, so it can go into a shared object of yours;
`BUILD_SHARED_LIBS` decides whether it is static or shared.

Options worth knowing, all `OFF`/`ON` cache variables:
`PJRT_EXEC_FETCH_PLUGIN` (download the published plugin at configure time,
default `ON`), `PJRT_EXEC_PLUGIN_PATH` (use a plugin you already have),
`PJRT_EXEC_PLUGIN_SOURCE_BUILD` (build it from the fork; bazel, 30–60 minutes),
and `PJRT_EXEC_BUILD_EXAMPLES` / `_TESTS` / `_BENCH`.

`examples/01_basic/CMakeLists.txt` is a standalone version of exactly this,
configured on its own so it can be built and run as written:

```{literalinclude} ../../examples/01_basic/CMakeLists.txt
:language: cmake
```

**Verify.**

```
cmake -S . -B build && cmake --build build --target controller
ldd build/controller | grep pjrt || echo "plugin not linked, as intended"
```

→ expected: the build succeeds and the `ldd` line prints
`plugin not linked, as intended`. If the plugin appears in `ldd` output, some
part of your build is linking it, which will fail on any machine whose plugin
lives somewhere else.

(rec-fetchcontent)=
## Recipe 2 — CMake + FetchContent

For a project that does not want a submodule. The cost is a network fetch at
configure time, and the requirement to pin.

**Prerequisites.** As recipe 1, plus network access when configuring a fresh
build tree.

**Snippet.**

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

`GIT_SUBMODULES ""` is the line that matters. Without it, FetchContent clones
the XLA submodule — gigabytes, for a tree that is only ever used to build the
plugin from source.

**Verify.** `cmake -S . -B build` → expected: a configure-time status line
reading `pjrt_exec 0.2.0: JAX <version>, plugin release <tag>, default plugin
<path>`. That line is the project reporting which plugin it compiled in as the
default.

(rec-make)=
## Recipe 3 — Copied sources plus `pjrt_exec.mk`

For a build that is not CMake. Copy or submodule the tree, include one
fragment, and build the three sources into your own objects.

**Prerequisites.** GNU Make, a C++17 compiler. Nothing else.

**Files to add.** The repository (or at minimum `include/`, `src/pjrt_exec/`,
`third_party/pjrt/`, `third_party/nlohmann/` and `pjrt_exec.mk`) under, say,
`third_party/call_jax_from_cpp/`.

**Snippet.** The fragment defines `PJRT_EXEC_CPPFLAGS`, `PJRT_EXEC_SRCS`,
`PJRT_EXEC_OBJS`, `PJRT_EXEC_LIB` and `PJRT_EXEC_LDLIBS`, and touches nothing
else — your `CPPFLAGS` and `CXXFLAGS` stay yours:

```{literalinclude} ../../pjrt_exec.mk
:language: make
:start-after: docs: begin make-fragment
:end-before: docs: end make-fragment
```

**Verify.** `make controller && nm -C build/controller | grep -c 'pjrt::Function'`
→ expected: a non-zero count on an unstripped binary, meaning the library's
symbols are in yours.
If `make` picked the library as its default goal instead of yours, the fragment
was included before your first target *and* something went wrong with the
`.DEFAULT_GOAL` restore — check that you are on GNU Make.

(rec-ros2)=
## Recipe 4 — CMake under ament / colcon (ROS 2)

The motivating application is a control loop alongside `ros2_control`, so this
row exists. **No control code ships in this repository** — this is the build
integration only.

**Prerequisites.** An `ament_cmake` package, and a decision about where the
plugin and the artifacts live on the target machine.

**Snippet.** In the package's `CMakeLists.txt`:

```cmake
find_package(ament_cmake REQUIRED)

# The library is vendored inside the package rather than found: it is three
# translation units with no install rules, pinned to one JAX version.
set(PJRT_EXEC_FETCH_PLUGIN OFF CACHE BOOL "" FORCE)   # colcon builds offline
add_subdirectory(third_party/call_jax_from_cpp)

add_library(my_controller SHARED src/my_controller.cpp)
target_link_libraries(my_controller PRIVATE pjrt_exec::pjrt_exec)

# The plugin and the artifacts are runtime data, not build outputs. Install
# them alongside the package and resolve them at run time.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/plugin/libpjrt_c_api_cpu_plugin.so
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/
        DESTINATION share/${PROJECT_NAME}/artifacts)
ament_package()
```

Then in the controller, resolve both paths from the package share directory
(`ament_index_cpp::get_package_share_directory`) and pass the plugin through
`RuntimeOptions::plugin_path` rather than relying on a compiled-in default.

Three things to get right in a `ros2_control` context, none of them build
issues:

- Load in `on_configure`, never in `update()`. Loading takes milliseconds for a
  `.binpb` and seconds if it falls back to compiling the `.mlirbc`.
- One `Function` per thread. It owns its arenas; `Runtime` is the object to
  share.
- **A deadline is not a cancellation.** PJRT cannot cancel a running CPU
  computation, so an overrun means the controller reads a stale result, not that
  the call was aborted. Design the fallback around stale data.

**Verify.** `colcon build --packages-select my_controller` → expected: builds
with no reference to the plugin on the link line, and
`ls install/my_controller/lib/my_controller/libpjrt_c_api_cpu_plugin.so` finds
the installed plugin.

## The plugin at run time

Every recipe needs a plugin on the target machine, matching the JAX version the
artifacts were exported with. `Runtime` looks in three places, in order:

1. `RuntimeOptions::plugin_path`, if non-empty;
2. `$PJRT_CPU_PLUGIN`;
3. the path compiled in at build time, which this project's own build sets to
   wherever `make plugin` writes.

Getting one: `make plugin` downloads the published, sha256-verified asset;
`make plugin-source` builds it from the XLA fork; a CMake consumer can let
`PJRT_EXEC_FETCH_PLUGIN` do it at configure time or build the `pjrt_plugin`
target later. A stock unpatched plugin also works, with two caveats — an
executable that lowers to a LAPACK custom call will not load, and inline
execution cannot be confirmed. See [the XLA fork](xla-fork.md).

## The artifacts at run time

A `.binpb` embeds machine code for the machine that exported it: it is relinked
at load, never recompiled. Export on the deployment machine, or on one with the
same architecture and instruction-set level, and ship the `.mlirbc` alongside so
the loader has somewhere to fall back to. `python -m jax2exec check <base>`
answers "will this artifact run here" without running it.

## Toggles

Each of these is independent of the recipes above.

### Real-time hardening

Optional, Linux-only, and each helper reports whether it took effect instead of
failing the program. Apply them **once, after loading and before the loop**, in
this order — the order is not arbitrary:

```cpp
pjrt::rt::harden_malloc();        // 1. before anything allocates in bulk
pjrt::rt::lock_memory();          // 2. prefault and lock what exists
pjrt::rt::pin_current_thread(4);  // 3. the loop owns one core
// ... construct Runtime and Function here ...
pjrt::rt::corral_xla_threads({5, 6, 7});  // 4. XLA's pools exist only now
pjrt::rt::set_realtime_priority(80);      // 5. LAST, after warm-up
```

`corral_xla_threads` has to come after the `Runtime` exists, because XLA's
pools are created with the client. `set_realtime_priority` goes last so that
loading, warm-up and the allocations that come with them do not run at
real-time priority, where a long operation would starve the rest of the
machine.

`examples/common/rt_env.hpp` does exactly this, reports each step, and is worth
copying rather than re-deriving:

:::{dropdown} The full sequence, as the examples apply it
```{literalinclude} ../../examples/common/rt_env.hpp
:language: cpp
:start-after: docs: begin rt-harden-impl
:end-before: docs: end rt-harden-impl
```
:::

An unprivileged run skips most of this and is still a correct run — it is just
not a run to quote tail numbers from. `tools/rt_check.sh` audits what the host
provides.

### Debug mode

```cpp
pjrt::FunctionOptions opts;
opts.debug = true;         // bounds, typed-accessor dtype, call() re-entrancy
opts.check_values = true;  // no nan/inf in float arenas, no stray bool bytes
pjrt::Function f(runtime, "artifacts/trajopt", opts);
```

Run with `debug` on in development. With it off, the same mistake is silent
memory corruption rather than an exception, and the checks cost nothing when
they are off — one predictable branch on a member. `check_values` walks every
element of every arena, so it belongs in development and acceptance tests, not
in a loop.

### The compile fallback

```cpp
pjrt::FunctionOptions opts;
opts.load_policy = pjrt::LoadPolicy::BinaryOnly;  // for a deployment
```

`LoadPolicy::Auto` (the default) prefers the `.binpb` and compiles the
`.mlirbc` when the binary is missing or was built for a wider instruction set
than this host implements. That is the right default for a workstation and a
poor one for a control process: compiling takes seconds, silently, at startup.

Allow the fallback in production only when a several-second load is acceptable
and a heterogeneous fleet makes per-machine export impractical — and then log
`Function::load_kind()` at startup, so "it was slow to start today" is
answerable. Otherwise ask for `BinaryOnly` and get a `LoadError` naming the
mismatch instead.
