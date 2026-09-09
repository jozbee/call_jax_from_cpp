# call_jax_from_cpp -- call an ahead-of-time exported JAX function from C++
# through the PJRT C API, with the tail of the latency distribution as the
# figure of merit.
#
# The Makefile is the primary build; this file exists so the project can be
# vendored into a CMake tree (add_subdirectory or FetchContent) and so that
# `cmake --build` produces the same layout the Makefile does: build/bin,
# build/lib, build/plugin.
#
# 3.21 for PROJECT_IS_TOP_LEVEL, which is how every "only when we are the main
# project" decision below is made.
cmake_minimum_required(VERSION 3.21)

project(pjrt_exec
        VERSION 0.2.0
        DESCRIPTION "Call an AOT-exported JAX function from C++ over the PJRT C API"
        HOMEPAGE_URL "https://github.com/jozbee/call_jax_from_cpp"
        LANGUAGES C CXX)   # C for the preloadable allocation counter

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GetPjrtPlugin)

# ------------------------------------------------------------------- options

# Nobody vendoring this library wants our examples, tests and benchmark in
# their build, so those default to the answer that is right for the situation.
option(PJRT_EXEC_BUILD_EXAMPLES "Build the example programs" ${PROJECT_IS_TOP_LEVEL})
option(PJRT_EXEC_BUILD_TESTS    "Build the C++ test binaries" ${PROJECT_IS_TOP_LEVEL})
option(PJRT_EXEC_BUILD_BENCH    "Build the benchmark" ${PROJECT_IS_TOP_LEVEL})

# Plugin resolution is available to consumers too: a vendored copy still has to
# find a plugin to dlopen at run time.
option(PJRT_EXEC_FETCH_PLUGIN
       "Download the prebuilt PJRT CPU plugin at configure time" ON)
option(PJRT_EXEC_PLUGIN_SOURCE_BUILD
       "Build the PJRT CPU plugin from the XLA fork at configure time (bazel, 30-60 min)" OFF)

pjrt_exec_versions_env(PLUGIN_RELEASE PJRT_EXEC_DEFAULT_PLUGIN_RELEASE)
pjrt_exec_versions_env(JAX_VERSION PJRT_EXEC_JAX_VERSION)
set(PJRT_EXEC_PLUGIN_VERSION "${PJRT_EXEC_DEFAULT_PLUGIN_RELEASE}" CACHE STRING
    "Release tag of the prebuilt PJRT CPU plugin (default: PLUGIN_RELEASE from versions.env)")
set(PJRT_EXEC_PLUGIN_PATH "" CACHE FILEPATH
    "An existing libpjrt_c_api_cpu_plugin.so to use instead of fetching one")

# ------------------------------------------------------ top-level-only setup

if(PROJECT_IS_TOP_LEVEL)
  # Same layout as the Makefile, so a tree built either way is inspected the
  # same way and the docs need only one set of paths.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

  # -O2 with symbols: the same optimisation level the Makefile uses, and `perf`
  # is unusable without the symbols. An empty build type would mean -O0, and an
  # -O0 latency number is not a latency number.
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
  endif()
endif()

find_package(Threads REQUIRED)

# Warnings are ours to fix, not a consumer's: imposing them on a vendored build
# that runs -Werror would break someone else's tree over our code.
if(PROJECT_IS_TOP_LEVEL)
  set(PJRT_EXEC_WARNINGS -Wall -Wextra)
else()
  set(PJRT_EXEC_WARNINGS "")
endif()

# ------------------------------------------------------------------- library

# No explicit STATIC/SHARED: BUILD_SHARED_LIBS decides, which is what a
# consumer expects of a vendored library.
add_library(pjrt_exec
  src/pjrt_exec/runtime.cpp
  src/pjrt_exec/rt.cpp
  src/pjrt_exec/isa.cpp)
add_library(pjrt_exec::pjrt_exec ALIAS pjrt_exec)

target_compile_features(pjrt_exec PUBLIC cxx_std_17)
target_compile_options(pjrt_exec PRIVATE ${PJRT_EXEC_WARNINGS})

# BUILD_INTERFACE only: there is no install interface, see the note below.
# third_party/ is public because the public headers include "pjrt/pjrt_c_api.h"
# and "nlohmann/json.hpp".
target_include_directories(pjrt_exec PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>)

# dl for dlopen of the plugin, pthread because XLA's CPU runtime starts
# threads. The plugin itself is never linked: it is opened at run time.
target_link_libraries(pjrt_exec PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

# A consumer may link this into a shared object of their own.
set_target_properties(pjrt_exec PROPERTIES POSITION_INDEPENDENT_CODE ON)

pjrt_exec_get_plugin(PJRT_EXEC_RESOLVED_PLUGIN)
if(NOT PJRT_EXEC_RESOLVED_PLUGIN)
  # Nothing was resolved, so compile in the location `make plugin` and the
  # pjrt_plugin target write to. Being wrong here costs nothing: the runtime
  # reports a LoadError naming the path, and $PJRT_CPU_PLUGIN overrides it.
  set(PJRT_EXEC_RESOLVED_PLUGIN
      "${PJRT_EXEC_PLUGIN_DIR}/libpjrt_c_api_cpu_plugin.so")
endif()
target_compile_definitions(pjrt_exec PRIVATE
  PJRT_EXEC_DEFAULT_PLUGIN_PATH="${PJRT_EXEC_RESOLVED_PLUGIN}")

# NO install() RULES, DELIBERATELY. This project is vendored per consumer, not
# installed system-wide: the artifacts it loads embed machine code for the
# exporting host, the plugin is pinned to one JAX version, and the library is
# three translation units. A system-wide libpjrt_exec.so shared between two
# projects on different JAX pins is a support burden with no upside. Vendor the
# tree, or use pjrt_exec.mk.

# ------------------------------------------------------- allocation counter

# Preloaded, never linked: the binaries resolve its markers with dlsym and
# no-op when it is absent. SHARED rather than MODULE because dyld will not
# insert a bundle, and DYLD_INSERT_LIBRARIES is how this is used on macOS;
# `cc -shared` on Apple means -dynamiclib, which is what the Makefile produces.
if(PJRT_EXEC_BUILD_TESTS OR PJRT_EXEC_BUILD_BENCH)
  add_library(malloc_guard SHARED tests/support/malloc_guard.c)
  target_link_libraries(malloc_guard PRIVATE ${CMAKE_DL_LIBS})
  target_compile_options(malloc_guard PRIVATE ${PJRT_EXEC_WARNINGS})
  set_target_properties(malloc_guard PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON)
endif()

# ------------------------------------------------------------------ programs

# The example and test sources belong to other parts of the tree; they are
# listed here explicitly rather than pulled in with add_subdirectory, so that
# this file is the one place a binary's sources are declared.
#
# examples/01_basic/CMakeLists.txt is the exception in the other direction: it
# is a standalone consumer example, configured on its own to show what using
# this project from CMake looks like. It is not added from here, because both
# files would define example_01_basic. (Configuring it standalone works because
# PJRT_EXEC_BUILD_EXAMPLES defaults OFF when this project is not top level.)

if(PJRT_EXEC_BUILD_EXAMPLES)
  foreach(_pair "example_01_basic;examples/01_basic/basic.cpp"
                "example_02_trajopt;examples/02_trajopt/trajopt.cpp"
                "example_03_minimal;examples/03_minimal/minimal.cpp"
                "example_04_realtime;examples/04_realtime/realtime.cpp")
    list(GET _pair 0 _name)
    list(GET _pair 1 _src)
    add_executable(${_name} "${_src}")
    target_link_libraries(${_name} PRIVATE pjrt_exec::pjrt_exec)
    target_compile_options(${_name} PRIVATE ${PJRT_EXEC_WARNINGS})
    # The examples share helpers they include as "common/cli.hpp".
    target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/examples")
  endforeach()
endif()

if(PJRT_EXEC_BUILD_TESTS)
  # Named rather than globbed, so adding a test binary is a visible edit here.
  # Each is added only if its source is on disk: naming a file CMake cannot find
  # is a *generate*-time error, which would take the library, the examples and
  # the benchmark down with it over a test that has not been written yet.
  set(PJRT_EXEC_TEST_BINS "")
  foreach(_name fn_info test_debug_checks test_load_errors test_latency
                test_guard_selftest)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/${_name}.cpp")
      message(STATUS "pjrt_exec: skipping test binary ${_name} (no tests/cpp/${_name}.cpp)")
      continue()
    endif()
    add_executable(${_name} "tests/cpp/${_name}.cpp")
    target_link_libraries(${_name} PRIVATE pjrt_exec::pjrt_exec)
    target_compile_options(${_name} PRIVATE ${PJRT_EXEC_WARNINGS})
    # The tests reuse the examples' helpers and the benchmark's fixture loader
    # on the same terms the benchmark does.
    target_include_directories(${_name} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/tests"
      "${CMAKE_CURRENT_SOURCE_DIR}/examples"
      "${CMAKE_CURRENT_SOURCE_DIR}/bench")
    list(APPEND PJRT_EXEC_TEST_BINS ${_name})
  endforeach()

  # A developer tool rather than a deliverable, and absent from a trimmed
  # checkout.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/plugin_probe.cpp")
    add_executable(plugin_probe tools/plugin_probe.cpp)
    target_link_libraries(plugin_probe PRIVATE pjrt_exec::pjrt_exec)
    target_compile_options(plugin_probe PRIVATE ${PJRT_EXEC_WARNINGS})
  endif()
endif()

if(PJRT_EXEC_BUILD_BENCH)
  add_executable(bench bench/bench_main.cpp)
  target_link_libraries(bench PRIVATE pjrt_exec::pjrt_exec)
  target_compile_options(bench PRIVATE ${PJRT_EXEC_WARNINGS})
  # examples/ as well as bench/: the benchmark reuses the examples' CLI parser,
  # JSON report writer and host-environment probe rather than carrying a second
  # copy of all three, so it includes them as "common/cli.hpp" too.
  target_include_directories(bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/bench"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples")
endif()

# ------------------------------------------------------------ chore targets

# Mirrors of the Makefile's run targets, for a CMake-only workflow. Top level
# only: these names (`test` above all) are far too generic to inflict on a
# consumer's build.
#
# Note that CTest is deliberately not enabled. The suite is pytest-driven --
# the C++ binaries above are fixtures it runs -- and include(CTest) would both
# claim the `test` target and add a dashboard this project has no use for.
if(PROJECT_IS_TOP_LEVEL)
  # ;-lists, because these are commands with arguments. uv comes from mise and
  # is not on PATH in a non-interactive shell; override when that bites.
  set(PJRT_EXEC_PYTHON "uv;run;python" CACHE STRING
      "Interpreter used by the export target, as a ;-list")
  set(PJRT_EXEC_PYTEST "uv;run;pytest" CACHE STRING
      "Test runner used by the test target, as a ;-list")

  # Exported artifacts live in the source tree (gitignored) rather than the
  # build tree: they are shared with the Makefile, and every consumer of them
  # -- the examples, the benchmark, the Python tests -- looks there.
  set(PJRT_EXEC_ARTIFACTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artifacts" CACHE PATH
      "Where the exported .binpb/.json/.mlirbc artifacts are written")
  set(_reports "${PJRT_EXEC_ARTIFACTS_DIR}/reports")

  if(APPLE)
    set(_preload_var DYLD_INSERT_LIBRARIES)
  else()
    set(_preload_var LD_PRELOAD)
  endif()

  # Serialized executables embed target machine code, so exporting has to
  # happen on the machine that will run them. Always out of date, on purpose:
  # a stale artifact is the failure this project takes most seriously.
  add_custom_target(export
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PJRT_EXEC_ARTIFACTS_DIR}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
            ${PJRT_EXEC_PYTHON} examples/01_basic/export.py
            --out "${PJRT_EXEC_ARTIFACTS_DIR}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
            ${PJRT_EXEC_PYTHON} examples/02_trajopt/export.py
            --out "${PJRT_EXEC_ARTIFACTS_DIR}" --cases 4
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
            ${PJRT_EXEC_PYTHON} examples/05_ros2_control/export.py
            --out "${PJRT_EXEC_ARTIFACTS_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Exporting the example artifacts with JAX"
    USES_TERMINAL
    VERBATIM)

  if(PJRT_EXEC_BUILD_EXAMPLES)
    add_custom_target(run-examples
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_reports}"
      COMMAND $<TARGET_FILE:example_01_basic>
      COMMAND $<TARGET_FILE:example_01_basic> --debug
      COMMAND $<TARGET_FILE:example_02_trajopt>
              --iterations 500 --json "${_reports}/trajopt.json"
      COMMAND $<TARGET_FILE:example_03_minimal> artifacts/basic 1000 1000
      COMMAND ${CMAKE_COMMAND} -E env
              "${_preload_var}=$<TARGET_FILE:malloc_guard>"
              $<TARGET_FILE:example_04_realtime>
              --iterations 1000 --json "${_reports}/realtime.json"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMENT "Running the examples end to end"
      USES_TERMINAL
      VERBATIM)
    add_dependencies(run-examples export example_01_basic example_02_trajopt
                     example_03_minimal example_04_realtime malloc_guard)
  endif()

  add_custom_target(test
    COMMAND ${PJRT_EXEC_PYTEST}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running the test suite"
    USES_TERMINAL
    VERBATIM)
  if(PJRT_EXEC_BUILD_TESTS)
    add_dependencies(test malloc_guard)
    if(PJRT_EXEC_TEST_BINS)
      add_dependencies(test ${PJRT_EXEC_TEST_BINS})
    endif()
  endif()
  if(PJRT_EXEC_BUILD_EXAMPLES)
    add_dependencies(test example_01_basic example_02_trajopt
                     example_03_minimal example_04_realtime)
  endif()

  # The gate that matters: the wrapper's own allocations in the steady-state
  # call path, which must be zero. The ~15,400 per call inside XLA's thunk
  # runtime are not reachable from here and are reported separately.
  if(PJRT_EXEC_BUILD_BENCH)
    add_custom_target(test-alloc
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_reports}"
      COMMAND ${CMAKE_COMMAND} -E env
              "${_preload_var}=$<TARGET_FILE:malloc_guard>"
              $<TARGET_FILE:bench>
              --iterations 200 --warmup 20 --alloc-gate self --require-guard
              --json "${_reports}/alloc_census.json"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMENT "Checking that the call path adds no allocations"
      USES_TERMINAL
      VERBATIM)
    add_dependencies(test-alloc export bench malloc_guard)
  endif()

  message(STATUS
    "pjrt_exec ${PROJECT_VERSION}: JAX ${PJRT_EXEC_JAX_VERSION}, "
    "plugin release ${PJRT_EXEC_PLUGIN_VERSION}, "
    "default plugin ${PJRT_EXEC_RESOLVED_PLUGIN}")
endif()
