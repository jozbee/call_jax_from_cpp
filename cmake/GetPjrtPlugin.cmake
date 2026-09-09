# Resolve a PJRT CPU plugin for this build tree.
#
# There is no official prebuilt CPU PJRT C-API plugin (jaxlib never exports
# GetPjrtApi), so this project publishes its own per JAX version:
# tools/plugin_versions.txt is the manifest, versions.env names the release
# tag. Nothing links against the plugin, so a missing one is never a configure
# error: the binaries then have no compiled-in default and read
# $PJRT_CPU_PLUGIN.
#
#   pjrt_exec_get_plugin(<out-var>)
#
# Sets <out-var> to the .so, or to "" when none could be resolved. Honours, in
# order:
#
#   PJRT_EXEC_PLUGIN_PATH          an existing plugin, used as given
#   PJRT_EXEC_PLUGIN_SOURCE_BUILD  build it from the fork now (bazel; slow)
#   PJRT_EXEC_FETCH_PLUGIN         download the published asset (default)
#
# Also defines the `pjrt_plugin` target, which does the same work at build
# time.

include_guard(GLOBAL)

# Baked into the cache because the functions below run in the caller's scope,
# which may be a subdirectory that never saw this file's include().
get_filename_component(_pjrt_exec_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(PJRT_EXEC_ROOT "${_pjrt_exec_root}" CACHE INTERNAL
    "Source root of the vendored pjrt_exec tree")

set(PJRT_EXEC_PLUGIN_DIR "${CMAKE_BINARY_DIR}/plugin" CACHE PATH
    "Where a downloaded or locally built PJRT CPU plugin is installed")

#[==[
Read one KEY=VALUE line out of versions.env. Sets <out-var> to the value, or
to "" when the key is absent.
]==]
function(pjrt_exec_versions_env KEY OUT_VAR)
  set(${OUT_VAR} "" PARENT_SCOPE)
  set(_file "${PJRT_EXEC_ROOT}/versions.env")
  if(NOT EXISTS "${_file}")
    return()
  endif()
  file(STRINGS "${_file}" _lines REGEX "^${KEY}=")
  if(_lines)
    list(GET _lines 0 _line)
    string(REGEX REPLACE "^${KEY}=[ \t]*" "" _value "${_line}")
    string(STRIP "${_value}" _value)
    set(${OUT_VAR} "${_value}" PARENT_SCOPE)
  endif()
endfunction()

#[==[
This host as the <os>-<arch> spelling the manifest and the release assets use
(linux/darwin, x86_64/aarch64), or "" on anything else.
]==]
function(_pjrt_exec_platform OUT_VAR)
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" _os)
  if(NOT _os MATCHES "^(linux|darwin)$")
    set(${OUT_VAR} "" PARENT_SCOPE)
    return()
  endif()
  set(_arch "${CMAKE_SYSTEM_PROCESSOR}")
  if(_arch MATCHES "^([Aa][Mm][Dd]64|x86_64|x64)$")
    set(_arch "x86_64")
  elseif(_arch MATCHES "^([Aa][Rr][Mm]64|aarch64)$")
    set(_arch "aarch64")
  else()
    set(${OUT_VAR} "" PARENT_SCOPE)
    return()
  endif()
  set(${OUT_VAR} "${_os}-${_arch}" PARENT_SCOPE)
endfunction()

# The same three remedies get_plugin.sh prints; the caller says what went wrong.
function(_pjrt_exec_remedies)
  message(STATUS
    "pjrt_exec: the build continues without a compiled-in plugin path. "
    "Options:\n"
    "   1. build it from the XLA fork:  cmake -DPJRT_EXEC_PLUGIN_SOURCE_BUILD=ON\n"
    "      (or: docker compose -f docker/compose.yml run --rm plugin-builder \\\n"
    "            tools/build_plugin.sh)\n"
    "   2. publish the asset:           tools/release_plugin.sh\n"
    "   3. point at one you already have:\n"
    "        cmake -DPJRT_EXEC_PLUGIN_PATH=/path/to/libpjrt_c_api_cpu_plugin.so\n"
    "        or export PJRT_CPU_PLUGIN=... before running the binaries")
endfunction()

#[==[
Find, download or build the PJRT CPU plugin. See the file header.
]==]
function(pjrt_exec_get_plugin OUT_VAR)
  set(${OUT_VAR} "" PARENT_SCOPE)

  # An explicit path is used as given: a consumer may have their own plugin.
  if(PJRT_EXEC_PLUGIN_PATH)
    if(NOT EXISTS "${PJRT_EXEC_PLUGIN_PATH}")
      message(WARNING
        "pjrt_exec: PJRT_EXEC_PLUGIN_PATH=${PJRT_EXEC_PLUGIN_PATH} does not "
        "exist yet; it is compiled in as the default and must be there before "
        "anything runs")
    endif()
    set(${OUT_VAR} "${PJRT_EXEC_PLUGIN_PATH}" PARENT_SCOPE)
    return()
  endif()

  set(_so "${PJRT_EXEC_PLUGIN_DIR}/libpjrt_c_api_cpu_plugin.so")

  # Already in the build tree, whichever way it was asked for.
  if(EXISTS "${_so}" AND
     (PJRT_EXEC_PLUGIN_SOURCE_BUILD OR PJRT_EXEC_FETCH_PLUGIN))
    message(STATUS "pjrt_exec: reusing the plugin at ${_so}")
    set(${OUT_VAR} "${_so}" PARENT_SCOPE)
    return()
  endif()

  # Opt-in only: this is bazel compiling LLVM, at configure time.
  if(PJRT_EXEC_PLUGIN_SOURCE_BUILD)
    message(STATUS
      "pjrt_exec: building the PJRT CPU plugin from source; this runs bazel "
      "and takes 30-60 minutes on a cold cache")
    execute_process(
      COMMAND "${PJRT_EXEC_ROOT}/tools/build_plugin.sh" --out "${PJRT_EXEC_PLUGIN_DIR}"
      WORKING_DIRECTORY "${PJRT_EXEC_ROOT}"
      RESULT_VARIABLE _rc)
    if(_rc EQUAL 0 AND EXISTS "${_so}")
      set(${OUT_VAR} "${_so}" PARENT_SCOPE)
    else()
      message(STATUS
        "pjrt_exec: tools/build_plugin.sh failed (${_rc}); see its output "
        "above. The build continues without a compiled-in plugin path.")
    endif()
    return()
  endif()

  # The published asset.
  if(PJRT_EXEC_FETCH_PLUGIN)
    set(_release "${PJRT_EXEC_PLUGIN_VERSION}")
    if(NOT _release)
      pjrt_exec_versions_env(PLUGIN_RELEASE _release)
    endif()

    _pjrt_exec_platform(_platform)
    if(NOT _platform)
      message(STATUS
        "pjrt_exec: no published plugin for ${CMAKE_SYSTEM_NAME}/"
        "${CMAKE_SYSTEM_PROCESSOR}; build one with "
        "-DPJRT_EXEC_PLUGIN_SOURCE_BUILD=ON")
      return()
    endif()

    set(_manifest "${PJRT_EXEC_ROOT}/tools/plugin_versions.txt")
    if(NOT EXISTS "${_manifest}")
      message(STATUS "pjrt_exec: no plugin manifest at ${_manifest}")
      return()
    endif()

    # Rows: <release-tag> <os-arch> <url> <sha256>; the sha may be "-".
    file(STRINGS "${_manifest}" _rows REGEX "^[ \t]*[^#\t ]")
    set(_url "")
    set(_sha "")
    foreach(_row IN LISTS _rows)
      if(_row MATCHES "^[ \t]*([^ \t]+)[ \t]+([^ \t]+)[ \t]+([^ \t]+)[ \t]*([^ \t]*)")
        if(CMAKE_MATCH_1 STREQUAL "${_release}" AND CMAKE_MATCH_2 STREQUAL "${_platform}")
          set(_url "${CMAKE_MATCH_3}")
          set(_sha "${CMAKE_MATCH_4}")
          break()
        endif()
      endif()
    endforeach()

    if(NOT _url)
      message(STATUS
        "pjrt_exec: no prebuilt plugin published for ${_platform} at release "
        "'${_release}'")
      _pjrt_exec_remedies()
      return()
    endif()

    set(_archive "${CMAKE_BINARY_DIR}/pjrt_plugin-${_release}-${_platform}.tar.gz")
    message(STATUS "pjrt_exec: downloading ${_url}")
    file(DOWNLOAD "${_url}" "${_archive}" STATUS _status TLS_VERIFY ON)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
      list(GET _status 1 _reason)
      file(REMOVE "${_archive}")
      message(STATUS "pjrt_exec: download failed: ${_reason}")
      _pjrt_exec_remedies()
      return()
    endif()

    # Compared here rather than with DOWNLOAD's EXPECTED_HASH, which is a hard
    # CMake error: a bad download must leave a tree that still configures. The
    # archive is discarded either way.
    if(_sha AND NOT _sha STREQUAL "-")
      file(SHA256 "${_archive}" _actual)
      string(TOLOWER "${_sha}" _sha_lc)
      string(TOLOWER "${_actual}" _actual_lc)
      if(NOT _sha_lc STREQUAL _actual_lc)
        file(REMOVE "${_archive}")
        message(WARNING
          "pjrt_exec: checksum mismatch for ${_url}\n"
          "  expected ${_sha_lc}\n"
          "  actual   ${_actual_lc}\n"
          "The download was discarded and no plugin was installed.")
        _pjrt_exec_remedies()
        return()
      endif()
      message(STATUS "pjrt_exec: sha256 ok")
    else()
      message(STATUS "pjrt_exec: the manifest carries no checksum for this asset")
    endif()

    file(MAKE_DIRECTORY "${PJRT_EXEC_PLUGIN_DIR}")
    file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${PJRT_EXEC_PLUGIN_DIR}")
    file(REMOVE "${_archive}")
    if(EXISTS "${_so}")
      message(STATUS "pjrt_exec: installed ${_so}")
      set(${OUT_VAR} "${_so}" PARENT_SCOPE)
    else()
      message(STATUS
        "pjrt_exec: the archive did not contain libpjrt_c_api_cpu_plugin.so")
    endif()
    return()
  endif()

  message(STATUS
    "pjrt_exec: not resolving a plugin (PJRT_EXEC_FETCH_PLUGIN is OFF); set "
    "PJRT_CPU_PLUGIN at run time")
endfunction()

# The same errand at build time, for a tree configured without a plugin (or one
# whose release tag has moved since): `cmake --build build --target pjrt_plugin`.
if(NOT TARGET pjrt_plugin)
  if(PJRT_EXEC_PLUGIN_SOURCE_BUILD)
    add_custom_target(pjrt_plugin
      COMMAND "${PJRT_EXEC_ROOT}/tools/build_plugin.sh" --out "${PJRT_EXEC_PLUGIN_DIR}"
      WORKING_DIRECTORY "${PJRT_EXEC_ROOT}"
      COMMENT "Building the PJRT CPU plugin from the XLA fork (bazel; slow)"
      USES_TERMINAL
      VERBATIM)
  else()
    add_custom_target(pjrt_plugin
      COMMAND "${PJRT_EXEC_ROOT}/tools/get_plugin.sh" --dest "${PJRT_EXEC_PLUGIN_DIR}"
      WORKING_DIRECTORY "${PJRT_EXEC_ROOT}"
      COMMENT "Downloading the prebuilt PJRT CPU plugin"
      USES_TERMINAL
      VERBATIM)
  endif()
endif()
