/**
 * @file fn_info.cpp
 * @brief Print everything a loaded `pjrt::Function` knows about itself, as one
 *        JSON object.
 *
 * The examples print `key=value` lines for a human; this prints JSON for a
 * test reading a pipe, so that the assertions about an artifact's signature
 * live in pytest, where a table of expected dtypes and shapes is a data
 * structure.  Nothing here calls the function: what a `Function` reports is
 * the sidecar's description after the loader has cross-checked it against the
 * executable, which is the thing worth asserting on.
 *
 *     fn_info <base_path> [--debug]
 *
 * `--debug` loads with the per-call checks on and is echoed back as `debug`,
 * so a test can confirm that a debug load reports the same signature.
 *
 * stdout carries the JSON object and nothing else.  A load failure is exit 1
 * with the message on stderr, not a JSON object with an error field: a test
 * that meant to read a signature should fail, not parse an apology.
 */
#include <cstddef>
#include <cstdio>
#include <exception>
#include <string>

#include "common/names.hpp"
#include "nlohmann/json.hpp"
#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

using nlohmann::json;

constexpr const char* kUsage =
    "usage: fn_info <base_path> [--debug]\n"
    "       fn_info --artifact <base_path> [--debug]\n";

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
    // "unexpected argument" rather than a bare usage block, so a caller
    // written against a different command line can tell that from a failure.
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
    info["load_kind"] = cjfc::load_kind_name(function.load_kind());
    info["load_detail"] = function.load_detail();
    info["fingerprint"] = function.fingerprint();
    info["sync_mode"] = cjfc::sync_mode_name(runtime.synchronous_mode());
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

    // The fingerprint is whatever bytes the plugin handed back; nothing in the
    // C API promises UTF-8, and `replace` keeps a binary one from becoming a
    // serialization exception halfway through the line.
    std::printf(
        "%s\n",
        info.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "fn_info: %s\n", error.what());
    return 1;
  }
}
