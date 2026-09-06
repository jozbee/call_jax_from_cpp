/**
 * @file report.hpp
 * @brief Turning a run into a report: printed for a human, JSON for a script.
 *
 * Every example prints the same summary and, when asked, writes the same JSON.
 * That is the point of putting it here: a latency number is only comparable to
 * another one if the run that produced it is described the same way, and the
 * description has to include the things that invalidate the number -- the load
 * average, the governor, whether the loop was pinned, whether execution was
 * actually inline.  A JSON report that carries the summary without the host is
 * a number without a provenance.
 *
 * **Exit codes.**  The examples share one set, so a script can tell the
 * failures apart:
 *
 * | code | meaning |
 * |---|---|
 * | 0 | ok |
 * | 1 | error (a load failure, a bad flag, an exception) |
 * | 2 | correctness: the computation produced the wrong answer |
 * | 3 | the allocation gate failed: the steady-state path allocated |
 * | 4 | an allocation gate was required and the guard was not preloaded |
 *
 * 3 and 4 are distinct on purpose.  "The path allocated" and "nobody measured
 * whether the path allocated" look identical in a green CI log otherwise, and
 * the second one is how an allocation-free claim quietly stops being true.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "common/names.hpp"
#include "common/rt_env.hpp"
#include "nlohmann/json.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/dtype.hpp"
#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/runtime.hpp"

// call_jax_from_cpp: helpers the examples share; not the library
namespace cjfc {

/// The JSON type the reports are built from.
using json = nlohmann::json;

/// Everything worked.
inline constexpr int kExitOk = 0;
/// An error: a missing artifact, a bad flag, an exception out of `main`.
inline constexpr int kExitError = 1;
/// The computation ran but produced the wrong answer.
inline constexpr int kExitCorrectness = 2;
/// The steady-state path allocated when it was required not to.
inline constexpr int kExitAllocGate = 3;
/// An allocation gate was demanded but the guard was not preloaded, so nothing
/// was actually measured.
inline constexpr int kExitGuardMissing = 4;

/// @brief The summary, field for field, in microseconds.
inline json summary_json(const pjrt::LatencySummary& s) {
  return json{
      {"count", s.count},
      {"dropped", s.dropped},
      {"mean_us", s.mean_us},
      {"stddev_us", s.stddev_us},
      {"min_us", s.min_us},
      {"p50_us", s.p50_us},
      {"p90_us", s.p90_us},
      {"p99_us", s.p99_us},
      {"p999_us", s.p999_us},
      {"p9999_us", s.p9999_us},
      {"max_us", s.max_us},
      {"max_over_p50", s.max_over_p50},
      {"p999_over_p50", s.p999_over_p50},
  };
}

/**
 * @brief The histogram as an array of `{lo_us, hi_us, count}`.
 *
 * The underflow and overflow edges become `null` rather than a number: JSON has
 * no infinity, and a literal -9.2e12 microseconds in a report reads as a
 * measurement rather than as the sentinel it is.
 */
inline json histogram_json(const pjrt::HistogramBin* bins, std::size_t count) {
  json bars = json::array();
  if (bins == nullptr) {
    return bars;
  }
  for (std::size_t i = 0; i < count; ++i) {
    json bar;
    bar["lo_us"] = bins[i].lo_ns == std::numeric_limits<std::int64_t>::min()
                       ? json(nullptr)
                       : json(static_cast<double>(bins[i].lo_ns) * 1e-3);
    bar["hi_us"] = bins[i].hi_ns == std::numeric_limits<std::int64_t>::max()
                       ? json(nullptr)
                       : json(static_cast<double>(bins[i].hi_ns) * 1e-3);
    bar["count"] = bins[i].count;
    bars.push_back(std::move(bar));
  }
  return bars;
}

/**
 * @brief The allocation census, or an explicit statement that there is none.
 *
 * The absent case is a record, not an omission: `guard_present: false` says
 * "not measured", which is a different claim from zero allocations and must not
 * be readable as one.
 */
inline json alloc_json(const pjrt::AllocGuard& guard, std::size_t iterations) {
  if (!guard.present()) {
    return json{
        {"guard_present", false},
        {"note",
         "not measured; run with LD_PRELOAD=build/lib/malloc_guard.so"},
    };
  }
  const double n =
      iterations > 0 ? static_cast<double>(iterations) : 1.0;
  const auto per = [n](unsigned long value) {
    return static_cast<double>(value) / n;
  };
  return json{
      {"guard_present", true},
      {"classified", guard.classified()},
      {"armed_total", guard.allocs()},
      {"self", guard.allocs_self()},
      {"plugin", guard.allocs_plugin()},
      {"runtime", guard.allocs_runtime()},
      {"frees", guard.frees()},
      {"process_total", guard.total()},
      {"per_iteration",
       json{
           {"total", per(guard.allocs())},
           {"self", per(guard.allocs_self())},
           {"plugin", per(guard.allocs_plugin())},
           {"runtime", per(guard.allocs_runtime())},
       }},
  };
}

/// @brief The host audit, field for field.  `busy` is the one to check before
///        believing anything else in the report.
inline json host_json(const HostEnv& env) {
  return json{
      {"kernel", env.kernel},
      {"preempt_rt", env.preempt_rt},
      {"in_container", env.in_container},
      {"cpus_online", env.cpus_online},
      {"isolated", env.isolated},
      {"nohz_full", env.nohz_full},
      {"affinity", env.affinity},
      {"governor", env.governor},
      {"thp", env.thp},
      {"smt", env.smt},
      {"rlimit_rtprio", env.rlimit_rtprio},
      {"rlimit_memlock", env.rlimit_memlock},
      {"rt_runtime_us", env.rt_runtime_us},
      {"cpu_dma_latency_writable", env.cpu_dma_latency_writable},
      {"loadavg1", env.loadavg1},
      {"loadavg5", env.loadavg5},
      {"loadavg15", env.loadavg15},
      {"busy", env.busy},
  };
}

/// @brief The hardening steps, in the order they were applied.
inline json steps_json(const std::vector<Step>& steps) {
  json out = json::array();
  for (const Step& step : steps) {
    out.push_back(json{
        {"name", step.name}, {"ok", step.ok}, {"detail", step.detail}});
  }
  return out;
}

/// @brief One input or output, as the loader resolved it.
inline json spec_json(const pjrt::ArraySpec& spec) {
  return json{
      {"name", spec.name},
      {"dtype", pjrt::dtype_name(spec.dtype)},
      {"shape", spec.shape},
      {"numel", spec.numel},
      {"nbytes", spec.nbytes},
      {"donated", spec.donated},
  };
}

/**
 * @brief Everything about the plugin, the client and the loaded function that a
 *        later reader will wish had been recorded.
 *
 * @param runtime        The client the run was made through; its plugin,
 *                       options and device count are recorded from here.
 * @param function       The loaded executable, read for its signature and the
 *                       kind of load that produced it.
 * @param runtime_ms     Wall time to create the `Runtime` (plugin load, client
 *                       creation, XLA's pools starting).
 * @param load_ms        Wall time to construct the `Function`, warm-up
 *                       included.  Load time only: a figure that also contains
 *                       the timed run says nothing about either.
 * @param first_call_us  The first call after warm-up, which is systematically
 *                       the slowest and is therefore worth its own field rather
 *                       than being buried in the tail.
 */
inline json runtime_json(const pjrt::Runtime& runtime,
                         const pjrt::Function& function, double runtime_ms,
                         double load_ms, double first_call_us) {
  const pjrt::PluginInfo& info = runtime.plugin();
  const pjrt::RuntimeOptions& options = runtime.options();

  json attributes = json::object();
  for (const auto& attribute : info.attributes) {
    attributes[attribute.first] = attribute.second;
  }

  json inputs = json::array();
  for (std::size_t i = 0; i < function.num_inputs(); ++i) {
    inputs.push_back(spec_json(function.input_spec(i)));
  }
  json outputs = json::array();
  for (std::size_t i = 0; i < function.num_outputs(); ++i) {
    outputs.push_back(spec_json(function.output_spec(i)));
  }

  return json{
      {"pjrt_exec_version", pjrt::version()},
      {"plugin",
       json{
           {"path", info.path},
           {"api_major", info.api_major},
           {"api_minor", info.api_minor},
           {"vendored_api_minor", pjrt::vendored_pjrt_api_minor()},
           {"platform_name", info.platform_name},
           {"platform_version", info.platform_version},
           {"advertises_synchronous_execution",
            info.advertises_synchronous_execution},
           {"advertises_max_inflight", info.advertises_max_inflight},
           {"attributes", attributes},
       }},
      {"options",
       json{
           {"synchronous", options.synchronous},
           {"cpu_device_count", options.cpu_device_count},
           {"worker_threads", options.worker_threads},
           {"max_inflight_computations", options.max_inflight_computations},
       }},
      {"sync_mode", sync_mode_name(runtime.synchronous_mode())},
      {"synchronous_supported", runtime.synchronous_supported()},
      {"function",
       json{
           {"name", function.name()},
           {"load_kind", load_kind_name(function.load_kind())},
           {"load_detail", function.load_detail()},
           {"fingerprint", function.fingerprint()},
           {"inputs", inputs},
           {"outputs", outputs},
       }},
      {"timing",
       json{
           {"runtime_ms", runtime_ms},
           {"load_ms", load_ms},
           {"first_call_us", first_call_us},
       }},
  };
}

/**
 * @brief Print a summary in the same layout `pjrt::LatencyRecorder::report`
 *        uses, minus the histogram.
 *
 * Same layout on purpose: an example's output and the library's output end up
 * pasted into the same issue, and two spellings of p99.9 in one thread is one
 * too many.
 */
inline void print_summary(const char* label, const pjrt::LatencySummary& s,
                          bool signed_samples = false) {
  std::printf("\n=== %s (n=%zu, microseconds) ===\n",
              label != nullptr ? label : "latency", s.count);
  if (s.count == 0) {
    std::printf("  no samples\n");
    return;
  }
  if (s.dropped != 0) {
    std::printf("  WARNING: %zu samples dropped (recorder was full)\n",
                s.dropped);
  }
  std::printf("  mean   %10.1f     stddev %10.1f\n", s.mean_us, s.stddev_us);
  std::printf("  min    %10.1f     p50    %10.1f\n", s.min_us, s.p50_us);
  std::printf("  p90    %10.1f     p99    %10.1f\n", s.p90_us, s.p99_us);
  std::printf("  p99.9  %10.1f     p99.99 %10.1f\n", s.p999_us, s.p9999_us);
  std::printf("  max    %10.1f\n", s.max_us);
  // A median near zero, which is what signed period jitter has by
  // construction, makes these ratios arbitrarily large and meaningless.
  if (signed_samples) {
    return;
  }
  std::printf("  --- tail ratios ---\n");
  std::printf("  max/p50   %7.3f      p99.9/p50 %7.3f\n", s.max_over_p50,
              s.p999_over_p50);
}

/**
 * @brief Write @p report to @p path, creating the directories above it.
 *
 * Reports land in `artifacts/reports/`, which is not in the repository, so the
 * first run of a fresh checkout would otherwise fail at the very end -- after
 * the measurement, which is the worst moment to lose it.
 *
 * @throws std::runtime_error when the directory or the file cannot be created.
 */
inline void write_json(const std::string& path, const json& report) {
  const std::filesystem::path file(path);
  if (file.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    if (ec) {
      throw std::runtime_error("cannot create " + file.parent_path().string() +
                               ": " + ec.message());
    }
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  out << report.dump(2) << '\n';
  if (!out) {
    throw std::runtime_error("short write to " + path);
  }
}

/**
 * @brief Decide what an allocation gate says about this run.
 *
 * @param guard         The guard the window was recorded into, already
 *                      disarmed.  Whether it was preloaded at all is part of
 *                      the verdict, not a precondition of asking.
 * @param gate          `"self"` gates on the wrapper's own allocations, which
 *                      is the number that must be zero; `"all"` gates on every
 *                      allocation in the armed window, including the thousands
 *                      per call inside XLA's thunk runtime, so it only makes
 *                      sense for a run that calls nothing.  Anything else
 *                      (`"none"`) gates on nothing.
 * @param require_guard Fail when the interposer was not preloaded, instead of
 *                      passing a test that never ran.
 * @return `kExitOk`, `kExitAllocGate` or `kExitGuardMissing`.
 *
 * When the guard is present but cannot attribute allocations to a module -- the
 * classifier needs module ranges that are not built on every platform -- the
 * `"self"` gate falls back to the total.  That is stricter than asked for and
 * will fail a run that only the plugin allocated in, which is the right way
 * round: a gate that cannot see what it is gating should complain, not pass.
 */
inline int alloc_gate_exit_code(const pjrt::AllocGuard& guard,
                                const std::string& gate, bool require_guard) {
  if (!guard.present()) {
    return require_guard ? kExitGuardMissing : kExitOk;
  }
  if (gate == "self") {
    const unsigned long counted =
        guard.classified() ? guard.allocs_self() : guard.allocs();
    return counted > 0 ? kExitAllocGate : kExitOk;
  }
  if (gate == "all") {
    return guard.allocs() > 0 ? kExitAllocGate : kExitOk;
  }
  return kExitOk;
}

}  // namespace cjfc
