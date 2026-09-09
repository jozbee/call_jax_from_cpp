# call_jax_from_cpp -- call an ahead-of-time exported JAX function from C++
# through the PJRT C API.
#
# The Makefile is the primary build. This file exists so the project can be
# vendored into a CMake tree (add_subdirectory or FetchContent); it produces
# the Makefile's layout: build/bin, build/lib, build/plugin.
#
# 3.21 for PROJECT_IS_TOP_LEVEL, which every top-level-only decision uses.
cmake_minimum_required(VERSION 3.21)

project(pjrt_exec
        VERSION 0.2.0
        DESCRIPTION "Call an AOT-exported JAX function from C++ over the PJRT C API"
        HOMEPAGE_URL "https://github.com/jozbee/call_jax_from_cpp"
        LANGUAGES C CXX)   # C for the preloadable allocation counter

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GetPjrtPlugin)

# ------------------------------------------------------------------- options

# Off when vendored: a consumer does not want our examples, tests and benchmark.
option(PJRT_EXEC_BUILD_EXAMPLES "Build the example programs" ${PROJECT_IS_TOP_LEVEL})
option(PJRT_EXEC_BUILD_TESTS    "Build the C++ test binaries" ${PROJECT_IS_TOP_LEVEL})
option(PJRT_EXEC_BUILD_BENCH    "Build the benchmark" ${PROJECT_IS_TOP_LEVEL})

# For consumers too: a vendored copy still has to find a plugin at run time.
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
  # The Makefile's layout, so the docs need only one set of paths.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

  # An empty build type means -O0, and an -O0 latency number is not a latency
  # number. RelWithDebInfo is the Makefile's -O2 plus the symbols `perf` needs.
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
  endif()
endif()

find_package(Threads REQUIRED)

# Top level only: a vendored build running -Werror must not break over our
# warnings.
if(PROJECT_IS_TOP_LEVEL)
  set(PJRT_EXEC_WARNINGS -Wall -Wextra)
else()
  set(PJRT_EXEC_WARNINGS "")
endif()

# ------------------------------------------------------------------- library

# No STATIC/SHARED: BUILD_SHARED_LIBS decides, as a consumer expects.
add_library(pjrt_exec
  src/pjrt_exec/runtime.cpp
  src/pjrt_exec/rt.cpp
  src/pjrt_exec/isa.cpp)
add_library(pjrt_exec::pjrt_exec ALIAS pjrt_exec)

target_compile_features(pjrt_exec PUBLIC cxx_std_17)
target_compile_options(pjrt_exec PRIVATE ${PJRT_EXEC_WARNINGS})

# third_party/ is public because the public headers include "pjrt/pjrt_c_api.h"
# and "nlohmann/json.hpp". BUILD_INTERFACE only: nothing is installed (below).
target_include_directories(pjrt_exec PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>)

# dl for dlopen of the plugin, pthread because XLA's CPU runtime starts threads.
target_link_libraries(pjrt_exec PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

# A consumer may link this into a shared object of their own.
set_target_properties(pjrt_exec PROPERTIES POSITION_INDEPENDENT_CODE ON)

pjrt_exec_get_plugin(PJRT_EXEC_RESOLVED_PLUGIN)
if(NOT PJRT_EXEC_RESOLVED_PLUGIN)
  # Compile in where `make plugin` writes. A wrong guess costs a LoadError
  # naming the path, and $PJRT_CPU_PLUGIN overrides it.
  set(PJRT_EXEC_RESOLVED_PLUGIN
      "${PJRT_EXEC_PLUGIN_DIR}/libpjrt_c_api_cpu_plugin.so")
endif()
target_compile_definitions(pjrt_exec PRIVATE
  PJRT_EXEC_DEFAULT_PLUGIN_PATH="${PJRT_EXEC_RESOLVED_PLUGIN}")

# No install() rules, deliberately: the plugin is pinned to one JAX version,
# and a system-wide libpjrt_exec shared between two projects on different pins
# is a support burden with no upside. Vendor the tree, or use pjrt_exec.mk.

# ------------------------------------------------------- allocation counter

# Preloaded, never linked. SHARED rather than MODULE: dyld will not insert a
# bundle, and DYLD_INSERT_LIBRARIES is how this is used on macOS.
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

# Every binary's sources are declared here rather than through
# add_subdirectory. examples/01_basic/CMakeLists.txt is a standalone consumer
# example, configured on its own; adding it from here would define
# example_01_basic twice.

# One source plus the library; the extra include roots are relative to the
# source root.
function(pjrt_exec_add_program name source)
  add_executable(${name} "${source}")
  target_link_libraries(${name} PRIVATE pjrt_exec::pjrt_exec)
  target_compile_options(${name} PRIVATE ${PJRT_EXEC_WARNINGS})
  foreach(_dir IN LISTS ARGN)
    target_include_directories(${name} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/${_dir}")
  endforeach()
endfunction()

if(PJRT_EXEC_BUILD_EXAMPLES)
  # -Iexamples: the examples share helpers they include as "common/cli.hpp".
  foreach(_pair "example_01_basic;examples/01_basic/basic.cpp"
                "example_02_trajopt;examples/02_trajopt/trajopt.cpp"
                "example_03_minimal;examples/03_minimal/minimal.cpp"
                "example_04_realtime;examples/04_realtime/realtime.cpp")
    list(GET _pair 0 _name)
    list(GET _pair 1 _src)
    pjrt_exec_add_program(${_name} "${_src}" examples)
    list(APPEND PJRT_EXEC_EXAMPLE_BINS ${_name})
  endforeach()
endif()

if(PJRT_EXEC_BUILD_TESTS)
  # Named rather than globbed, so a new test binary is a visible edit here.
  # Skipped rather than named when absent: a source CMake cannot find is a
  # generate-time error that takes the library and the examples down with it.
  foreach(_name fn_info test_debug_checks test_load_errors test_latency
                test_guard_selftest)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/${_name}.cpp")
      message(STATUS "pjrt_exec: skipping test binary ${_name} (no tests/cpp/${_name}.cpp)")
      continue()
    endif()
    pjrt_exec_add_program(${_name} "tests/cpp/${_name}.cpp"
                          tests examples bench)
    list(APPEND PJRT_EXEC_TEST_BINS ${_name})
  endforeach()

  # A developer tool, absent from a trimmed checkout.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/plugin_probe.cpp")
    pjrt_exec_add_program(plugin_probe tools/plugin_probe.cpp)
  endif()
endif()

if(PJRT_EXEC_BUILD_BENCH)
  # -Iexamples too: the benchmark reuses the examples' helpers.
  pjrt_exec_add_program(bench bench/bench_main.cpp bench examples)
endif()

# ------------------------------------------------------------ chore targets

# The Makefile's run targets, for a CMake-only workflow. Top level only: a
# name like `test` must not land in a consumer's build. CTest is not enabled on
# purpose: the suite is pytest-driven, and include(CTest) would claim `test`.
if(PROJECT_IS_TOP_LEVEL)
  # ;-lists: commands with arguments. uv comes from mise and is not on PATH in
  # a non-interactive shell.
  set(PJRT_EXEC_PYTHON "uv;run;python" CACHE STRING
      "Interpreter used by the export target, as a ;-list")
  set(PJRT_EXEC_PYTEST "uv;run;pytest" CACHE STRING
      "Test runner used by the test target, as a ;-list")

  # In the source tree, not the build tree: the Makefile and the Python tests
  # look there.
  set(PJRT_EXEC_ARTIFACTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/artifacts" CACHE PATH
      "Where the exported .binpb/.json/.mlirbc artifacts are written")
  set(_reports "${PJRT_EXEC_ARTIFACTS_DIR}/reports")

  if(APPLE)
    set(_preload_var DYLD_INSERT_LIBRARIES)
  else()
    set(_preload_var LD_PRELOAD)
  endif()

  # Always out of date, on purpose: artifacts embed machine code for this host,
  # and a stale one is the failure this project takes most seriously.
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
    add_dependencies(run-examples export malloc_guard ${PJRT_EXEC_EXAMPLE_BINS})
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
    add_dependencies(test ${PJRT_EXEC_EXAMPLE_BINS})
  endif()

  # `--alloc-gate self`: the wrapper's own allocations must be zero. XLA's
  # own, inside the plugin, are out of reach and reported separately.
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
