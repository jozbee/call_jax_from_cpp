/**
 * @file fn_info.cpp
 * @brief Print everything a loaded `pjrt::Function` knows about itself, as one
 *        JSON object.
 *
 * The examples print `key=value` lines for a human reading a terminal.  This
 * prints JSON for a test reading a pipe, and it exists so that the assertions
 * about an artifact's signature live in pytest -- where a table of expected
 * dtypes and shapes is a data structure -- rather than in C++, where each one
 * would be another `if` and another exit code.
 *
 * Everything reported here is fixed at load time, so nothing below calls the
 * function.  What a `Function` says about its inputs and outputs is the
 * sidecar's description after the loader has cross-checked it against the
 * executable, which is the thing worth asserting on: a sidecar alone could be
 * read in Python without loading anything.
 *
 *     fn_info <base_path> [--debug]
 *
 * `--debug` loads with the per-call checks on.  They change nothing this
 * program prints -- it makes no calls -- and the flag is here so a test can
 * confirm that a debug load still succeeds and still reports the same
 * signature.  It is echoed back as `debug` for exactly that reason.
 *
 * stdout carries the JSON object and nothing else; diagnostics go to stderr,
 * and a load failure is exit 1 with the message on stderr rather than a JSON
 * object with an error field in it.  A test that meant to read a signature
 * should fail, not parse an apology.
 */
#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>

#include "nlohmann/json.hpp"
#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

using nlohmann::json;

constexpr const char* kUsage =
    "usage: fn_info <base_path> [--debug]\n"
    "       fn_info --artifact <base_path> [--debug]\n";

/// `SyncMode` in the vocabulary the examples and the reports already use.
const char* sync_mode_name(pjrt::SyncMode mode) {
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

/// `LoadKind` as `example_01_basic` spells it, so the two outputs can be
/// compared without a translation table.
const char* load_kind_name(pjrt::LoadKind kind) {
  return kind == pjrt::LoadKind::Deserialized ? "deserialized" : "compiled";
}

/// One entry of the `inputs` or `outputs` array.
json spec_json(std::size_t index, const pjrt::ArraySpec& spec) {
  return json{
      {"index", index},
      {"name", spec.name},
      {"dtype", pjrt::dtype_name(spec.dtype)},
      {"shape", spec.shape},
      {"numel", spec.numel},
      {"nbytes", spec.nbytes},
  };
}

}  // namespace

int main(int argc, char** argv) {
  std::string base;
  bool debug = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--debug") {
      debug = true;
    } else if (arg == "--artifact" && i + 1 < argc) {
      base = argv[++i];
    } else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "fn_info: unknown flag '%s'\n%s", arg.c_str(),
                   kUsage);
      return 1;
    } else if (base.empty()) {
      base = arg;
    } else {
      std::fprintf(stderr, "fn_info: unexpected argument '%s'\n%s", arg.c_str(),
                   kUsage);
      return 1;
    }
  }
  if (base.empty()) {
    // Worded so that a caller written against a different command line reads
    // "this is not the program you think it is" instead of a bare usage block.
    std::fprintf(stderr, "fn_info: unexpected argument list\n%s", kUsage);
    return 1;
  }

  try {
    pjrt::Runtime runtime;

    pjrt::FunctionOptions options;
    options.debug = debug;
    options.check_values = debug;
    pjrt::Function function(runtime, base, options);

    json info;
    info["name"] = function.name();
    info["load_kind"] = load_kind_name(function.load_kind());
    info["load_detail"] = function.load_detail();
    info["fingerprint"] = function.fingerprint();
    info["sync_mode"] = sync_mode_name(runtime.synchronous_mode());
    info["synchronous_supported"] = runtime.synchronous_supported();
    info["debug"] = debug;

    json inputs = json::array();
    for (std::size_t i = 0; i < function.num_inputs(); ++i) {
      inputs.push_back(spec_json(i, function.input_spec(i)));
    }
    info["inputs"] = std::move(inputs);

    json outputs = json::array();
    for (std::size_t i = 0; i < function.num_outputs(); ++i) {
      outputs.push_back(spec_json(i, function.output_spec(i)));
    }
    info["outputs"] = std::move(outputs);

    // The fingerprint is whatever bytes the plugin handed back -- a decimal
    // string on this one, but nothing in the C API promises that. `replace`
    // keeps a plugin with a binary fingerprint from turning this into a
    // serialization exception halfway through a line of output.
    std::printf(
        "%s\n",
        info.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "fn_info: %s\n", error.what());
    return 1;
  }
}
