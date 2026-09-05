# examples/01_basic/CMakeLists.txt -- a standalone consumer of this project.
#
# The parent build does NOT add this directory; it compiles basic.cpp from the
# top-level CMakeLists.txt instead, so that a binary's sources are declared in
# one place. This file exists to be read and copied: it is what a tree that
# vendored call_jax_from_cpp as a git submodule would look like, and it is
# configured on its own -- `cmake -S examples/01_basic -B build/example01` --
# so that what the docs show is what actually builds.

# docs: begin cmake-submodule
cmake_minimum_required(VERSION 3.21)

# Guarded so that this file also survives being pulled into a larger build with
# add_subdirectory(), where the enclosing project() and the library target
# already exist. Standalone, none of that is true and every line below runs.
if(NOT TARGET pjrt_exec::pjrt_exec)
  project(basic_example LANGUAGES CXX)

  # In a consumer tree, this is the line that brings the library in.
  # EXCLUDE_FROM_ALL keeps its examples, tests and benchmark out of `all`;
  # they are off by default anyway once this project is not the top level.
  # add_subdirectory(third_party/call_jax_from_cpp EXCLUDE_FROM_ALL)

  # This copy lives inside the project it is an example of, so the submodule is
  # two directories up. Delete these three lines along with the comment above
  # once the line above is the real one.
  if(NOT TARGET pjrt_exec::pjrt_exec)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." call_jax_from_cpp
                     EXCLUDE_FROM_ALL)
  endif()
endif()

# The parent build declares this target too, under the same name.
if(NOT TARGET example_01_basic)
  add_executable(example_01_basic basic.cpp)

  # One link line is the whole integration: pjrt_exec carries its own include
  # directories (include/ and third_party/, the second because the public
  # headers include "pjrt/pjrt_c_api.h" and "nlohmann/json.hpp"), cxx_std_17,
  # dl and Threads. The PJRT plugin is never linked -- it is dlopen'd at run
  # time from RuntimeOptions::plugin_path, $PJRT_CPU_PLUGIN, or the path
  # compiled in as the default.
  target_link_libraries(example_01_basic PRIVATE pjrt_exec::pjrt_exec)

  # Only the examples need this: they include their shared helpers as
  # "common/cli.hpp". A consumer's own program does not.
  target_include_directories(example_01_basic PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
# docs: end cmake-submodule
