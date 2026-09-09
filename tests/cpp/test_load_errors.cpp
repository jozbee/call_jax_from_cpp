/**
 * @file test_load_errors.cpp
 * @brief Load one artifact under one deliberately awkward condition and report
 *        what came back.
 *
 * Every failure this project can diagnose happens at load, and a load failure
 * is only useful if it names the file and says what was wrong with it.  That
 * is a property of a string, which pytest is better at asserting on than C++
 * is, so this program loads and prints, and the assertions live there.
 *
 *     test_load_errors <base_path> <scenario>
 *
 * The scenarios that need a damaged artifact are prepared by the caller, which
 * copies the good artifacts into a temporary directory and edits them there.
 * This program only ever loads what it is pointed at:
 *
 *   | scenario     | the directory the caller prepares | what this sets |
 *   |--------------|-----------------------------------|----------------|
 *   | ok           | intact                            | defaults       |
 *   | stale        | a sidecar one output short        | defaults       |
 *   | fallback     | a truncated `.binpb`              | defaults       |
 *   | both         | truncated `.binpb`, no `.mlirbc`  | defaults       |
 *   | compile-only | intact                            | CompileOnly    |
 *   | binary-only  | intact                            | BinaryOnly     |
 *   | missing      | nothing at that path              | defaults       |
 *   | bad-plugin   | intact                            | a bad plugin   |
 *
 * `bad-plugin` points `RuntimeOptions` at the artifact's own sidecar, which
 * is guaranteed to exist and guaranteed not to be a shared object, so the
 * scenario cannot succeed by finding a real plugin at a made-up path.
 *
 * Prints one line -- `LOADED kind=... detail=...` or `THREW: <what()>` -- and
 * exits 0 either way: a failed load is the expected result of five of the
 * eight scenarios.  Exit 1 is reserved for a bad command line.
 */
#include <cstdio>
#include <exception>
#include <optional>
#include <string>

#include "common/names.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

constexpr const char* kUsage =
    "usage: test_load_errors <base_path> <scenario>\n"
    "       test_load_errors --artifact <base_path> <scenario>\n"
    "scenarios: ok stale fallback both compile-only binary-only missing "
    "bad-plugin\n";

/// What the scenario name changes about the load; the rest is the state of
/// the directory, which the caller owns.
struct Setup {
  pjrt::LoadPolicy policy = pjrt::LoadPolicy::Auto;
  bool break_plugin = false;
};

std::optional<Setup> setup_for(const std::string& scenario) {
  Setup setup;
  if (scenario == "ok" || scenario == "stale" || scenario == "fallback" ||
      scenario == "both" || scenario == "missing") {
    return setup;
  }
  if (scenario == "compile-only") {
    setup.policy = pjrt::LoadPolicy::CompileOnly;
    return setup;
  }
  if (scenario == "binary-only") {
    setup.policy = pjrt::LoadPolicy::BinaryOnly;
    return setup;
  }
  if (scenario == "bad-plugin") {
    setup.break_plugin = true;
    return setup;
  }
  return std::nullopt;
}

/// The sidecar path for @p base, accepting both spellings the loader accepts.
std::string sidecar_path(const std::string& base) {
  const std::string suffix = ".json";
  if (base.size() >= suffix.size() &&
      base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return base;
  }
  return base + suffix;
}

}  // namespace

int main(int argc, char** argv) {
  std::string base;
  std::string scenario;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--artifact" && i + 1 < argc) {
      base = argv[++i];
    } else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "test_load_errors: unknown flag '%s'\n%s",
                   arg.c_str(), kUsage);
      return 1;
    } else if (base.empty()) {
      base = arg;
    } else if (scenario.empty()) {
      scenario = arg;
    } else {
      std::fprintf(stderr, "test_load_errors: unexpected argument '%s'\n%s",
                   arg.c_str(), kUsage);
      return 1;
    }
  }
  if (base.empty() || scenario.empty()) {
    std::fprintf(stderr, "test_load_errors: unexpected argument list\n%s",
                 kUsage);
    return 1;
  }

  const std::optional<Setup> setup = setup_for(scenario);
  if (!setup) {
    std::fprintf(stderr, "test_load_errors: unknown scenario '%s'\n%s",
                 scenario.c_str(), kUsage);
    return 1;
  }

  try {
    pjrt::RuntimeOptions runtime_options;
    if (setup->break_plugin) {
      runtime_options.plugin_path = sidecar_path(base);
    }
    // Inside the try: `bad-plugin` fails here rather than at the Function.
    pjrt::Runtime runtime(runtime_options);

    pjrt::FunctionOptions options;
    options.load_policy = setup->policy;
    pjrt::Function function(runtime, base, options);

    std::printf("LOADED kind=%s detail=%s\n",
                cjfc::load_kind_name(function.load_kind()),
                function.load_detail().c_str());
  } catch (const std::exception& error) {
    std::printf("THREW: %s\n", error.what());
  }
  return 0;
}
