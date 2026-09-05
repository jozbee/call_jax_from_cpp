/**
 * @file test_load_errors.cpp
 * @brief Load one artifact under one deliberately awkward condition and report
 *        what came back.
 *
 * Every failure this project can diagnose happens at load: a sidecar that has
 * gone stale relative to its executable, a `.binpb` that will not deserialize,
 * a missing `.mlirbc` to fall back to, a plugin that is not a plugin.  The
 * point of loading them all through one binary is that the caller can compare
 * the *messages*: a load failure is only useful if it names the file and says
 * what was wrong with it, and that is a property of a string, which pytest is
 * better at asserting on than C++ is.
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
 * Two different messages answer to "stale", and which one appears depends on
 * how the caller damaged the sidecar.  Dropping an entry from `outputs` gives
 * the count mismatch, `"<sidecar> declares 1 outputs but <artifact> produces
 * 2"`; changing an entry's dtype or shape gives the per-output one, `"<sidecar>
 * declares output 1 as float64[] but <artifact> produces float32[]"`.  Both
 * are the same class of bug -- a sidecar that no longer describes its
 * executable, which would otherwise be discovered as a buffer overrun on call
 * ten thousand.
 *
 * `bad-plugin` takes no path of its own: it points `RuntimeOptions` at the
 * artifact's own sidecar, which is guaranteed to exist and guaranteed not to
 * be an ELF shared object, so the scenario cannot accidentally succeed by
 * finding a real plugin at a made-up path.
 *
 * Prints one line -- `LOADED kind=... detail=...` or `THREW: <what()>` -- and
 * exits 0 either way.  A failed load is the expected result of five of these
 * eight scenarios, so it is not this program's business to call it an error;
 * exit 1 is reserved for a bad command line.
 */
#include <cstdio>
#include <exception>
#include <string>

#include "pjrt_exec/runtime.hpp"

namespace {

constexpr const char* kUsage =
    "usage: test_load_errors <base_path> <scenario>\n"
    "       test_load_errors --artifact <base_path> <scenario>\n"
    "scenarios: ok stale fallback both compile-only binary-only missing "
    "bad-plugin\n";

/// What the scenario name changes about the load.  Everything else about a
/// scenario is the state of the directory, which the caller owns.
struct Setup {
  pjrt::LoadPolicy policy = pjrt::LoadPolicy::Auto;
  /// Point the runtime at something that is not a plugin.
  bool break_plugin = false;
};

/// @return false when @p scenario is not one of the eight.
bool setup_for(const std::string& scenario, Setup* setup) {
  if (scenario == "ok" || scenario == "stale" || scenario == "fallback" ||
      scenario == "both" || scenario == "missing") {
    return true;
  }
  if (scenario == "compile-only") {
    setup->policy = pjrt::LoadPolicy::CompileOnly;
    return true;
  }
  if (scenario == "binary-only") {
    setup->policy = pjrt::LoadPolicy::BinaryOnly;
    return true;
  }
  if (scenario == "bad-plugin") {
    setup->break_plugin = true;
    return true;
  }
  return false;
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

const char* load_kind_name(pjrt::LoadKind kind) {
  return kind == pjrt::LoadKind::Deserialized ? "deserialized" : "compiled";
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
    // See fn_info: a caller assuming a different command line should be told
    // that is what happened.
    std::fprintf(stderr, "test_load_errors: unexpected argument list\n%s",
                 kUsage);
    return 1;
  }

  Setup setup;
  if (!setup_for(scenario, &setup)) {
    std::fprintf(stderr, "test_load_errors: unknown scenario '%s'\n%s",
                 scenario.c_str(), kUsage);
    return 1;
  }

  try {
    pjrt::RuntimeOptions runtime_options;
    if (setup.break_plugin) {
      runtime_options.plugin_path = sidecar_path(base);
    }
    // Inside the try: `bad-plugin` fails here rather than at the Function.
    pjrt::Runtime runtime(runtime_options);

    pjrt::FunctionOptions options;
    options.load_policy = setup.policy;
    pjrt::Function function(runtime, base, options);

    std::printf("LOADED kind=%s detail=%s\n",
                load_kind_name(function.load_kind()),
                function.load_detail().c_str());
  } catch (const std::exception& error) {
    std::printf("THREW: %s\n", error.what());
  }
  return 0;
}
