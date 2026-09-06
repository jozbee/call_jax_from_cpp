/**
 * @file names.hpp
 * @brief One spelling of `SyncMode` and `LoadKind`, for the reports, the tests
 *        and anything that prints them.
 *
 * Separate from `%report.hpp` so that a program which only wants to name what it
 * loaded does not pull in nlohmann/json to do it.  Two spellings of "inline" in
 * one project is one too many: these strings are a machine interface -- the
 * test suite parses them -- and a second copy is a second thing to reword.
 */
#pragma once

#include "pjrt_exec/runtime.hpp"

// call_jax_from_cpp: helpers the examples share; not the library
namespace cjfc {

/// @brief `SyncMode` as the string the reports use.
inline const char* sync_mode_name(pjrt::SyncMode mode) {
  switch (mode) {
    case pjrt::SyncMode::Inline:
      return "inline";
    case pjrt::SyncMode::Accepted:
      return "accepted";
    case pjrt::SyncMode::Rejected:
      return "rejected";
    case pjrt::SyncMode::Async:
      return "async";
  }
  return "unknown";
}

/// @brief `LoadKind` as the string the reports use.
inline const char* load_kind_name(pjrt::LoadKind kind) {
  return kind == pjrt::LoadKind::Deserialized ? "deserialized" : "compiled";
}

}  // namespace cjfc
