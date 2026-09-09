# examples/01_basic/CMakeLists.txt -- a standalone consumer of this project.
#
# Written to be copied: what a tree that vendored call_jax_from_cpp as a git
# submodule looks like. The parent build does not add this directory (it
# compiles basic.cpp itself); this file is configured on its own, so what the
# docs show is what builds:  cmake -S examples/01_basic -B build/example01

# docs: begin cmake-submodule
cmake_minimum_required(VERSION 3.21)

# Guarded so the file also survives add_subdirectory() from a build that
# already has the library target. Standalone, every line below runs.
if(NOT TARGET pjrt_exec::pjrt_exec)
  project(basic_example LANGUAGES CXX)

  # In a consumer tree, this is the line that brings the library in;
  # EXCLUDE_FROM_ALL keeps its examples, tests and benchmark out of `all`.
  # add_subdirectory(third_party/call_jax_from_cpp EXCLUDE_FROM_ALL)

  # This copy lives inside the project it is an example of, so the submodule is
  # two directories up; delete these lines once the line above is the real one.
  if(NOT TARGET pjrt_exec::pjrt_exec)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." call_jax_from_cpp
                     EXCLUDE_FROM_ALL)
  endif()
endif()

# The parent build declares this target too, under the same name.
if(NOT TARGET example_01_basic)
  add_executable(example_01_basic basic.cpp)

  # One link line is the whole integration: pjrt_exec carries its include
  # directories, cxx_std_17, dl and Threads. The plugin is dlopen-ed at run
  # time, never linked.
  target_link_libraries(example_01_basic PRIVATE pjrt_exec::pjrt_exec)

  # Only the examples need this; a consumer's own program does not.
  target_include_directories(example_01_basic PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()
# docs: end cmake-submodule
