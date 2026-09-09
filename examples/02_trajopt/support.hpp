/**
 * @file support.hpp
 * @brief Everything `trajopt.cpp` needs that is not the timed loop: the flags,
 *        the finite-output audit, the fault injector and the two reports.
 *
 * Only `all_finite` and `inject_fault` are called from inside the armed
 * window; neither allocates.
 */
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "common/report.hpp"
#include "common/rt_env.hpp"
#include "common/workload.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/runtime.hpp"

namespace trajopt {

/// Not `high_resolution_clock`, which is the wall clock on some standard
/// libraries: an NTP step would show up as an outlier that never happened.
using Clock = std::chrono::steady_clock;

inline double millis(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

inline double micros(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

// ---------------------------------------------------------------- the flags

constexpr char kUsage[] =
    "usage: example_02_trajopt [options]\n"
    "\n"
    "  --artifact PATH   artifact base path (default artifacts/trajopt)\n"
    "  --iterations N    timed calls (default 2000)\n"
    "  --warmup N        untimed calls before them (default 50)\n"
    "  --threads N       XLA worker threads; 0 is XLA's default (default 1)\n"
    "  --async           do not ask for inline execution\n"
    "  --json PATH       write the report as JSON as well as printing it\n"
    "  --samples PATH    write every raw sample as index,ns\n"
    "  --no-check        skip the per-call finite-output audit\n"
    "  --inject-fault K  corrupt one cycle so a gate can be seen to fire:\n"
    "                    none (default), step, nonfinite\n"
    "  --alloc-gate G    off, self or all; exit 3 when it fails (default off)\n"
    "  --require-guard   exit 4 when malloc_guard.so was not preloaded\n"
    "  --quiet           print nothing on success\n"
    "\n"
    "Exit codes: 0 ok, 1 error, 2 wrong answer, 3 allocation gate, 4 no guard.";

inline cjfc::Cli make_cli(int argc, char** argv) {
  return cjfc::Cli(argc, argv,
                   {"artifact", "iterations", "warmup", "threads", "async",
                    "json", "samples", "no-check", "alloc-gate",
                    "require-guard", "quiet", "inject-fault"},
                   kUsage);
}

/// What `inject_fault` does to cycle `kFaultCycle`.
enum class Fault { None, Step, NonFinite };

/// Everything the command line can say, resolved once at startup.
struct Options {
  std::string artifact = "artifacts/trajopt";
  std::size_t iterations = 2000;
  std::size_t warmup = 50;
  long threads = 1;
  bool synchronous = true;
  std::string json_path;
  std::string samples_path;
  bool audit_values = true;
  Fault inject_fault = Fault::None;
  std::string alloc_gate = "off";
  bool require_guard = false;
  bool quiet = false;
};

/// Read the command line, refusing a misspelled value rather than ignoring it:
/// a gate nobody spelled correctly is a gate that passes everything.
inline Options parse_options(const cjfc::Cli& cli) {
  Options options;
  options.artifact = cli.get("artifact", options.artifact);
  options.iterations = cli.get_size("iterations", options.iterations);
  options.warmup = cli.get_size("warmup", options.warmup);
  options.threads = cli.get_long("threads", options.threads);
  options.synchronous = !cli.flag("async");
  options.json_path = cli.get("json", "");
  options.samples_path = cli.get("samples", "");
  options.audit_values = !cli.flag("no-check");
  options.alloc_gate = cli.get("alloc-gate", options.alloc_gate);
  options.require_guard = cli.flag("require-guard");
  options.quiet = cli.flag("quiet");

  const std::string fault = cli.get("inject-fault", "none");
  if (fault == "none") {
    options.inject_fault = Fault::None;
  } else if (fault == "step") {
    options.inject_fault = Fault::Step;
  } else if (fault == "nonfinite") {
    options.inject_fault = Fault::NonFinite;
  } else {
    throw std::runtime_error("unknown --inject-fault '" + fault +
                             "' (none, step or nonfinite)");
  }

  if (options.alloc_gate != "off" && options.alloc_gate != "self" &&
      options.alloc_gate != "all") {
    throw std::runtime_error("'--alloc-gate' expects off, self or all, got '" +
                             options.alloc_gate + "'");
  }
  if (options.threads < 0 || options.threads > 4096) {
    throw std::runtime_error("'--threads' expects a count between 0 and 4096");
  }
  return options;
}

/// On stderr even under `--quiet`: a latency number from a busy machine is not
/// noisy, it is wrong.
inline void warn_if_busy(const cjfc::HostEnv& env) {
  if (env.busy) {
    std::fprintf(stderr,
                 "WARNING: load average is %.2f; these numbers describe a "
                 "contended machine and are not comparable with anything. "
                 "Discard the run rather than correcting it.\n",
                 env.loadavg1);
  }
}

// ------------------------------------------------------ the per-call checks

/// One float output arena, resolved before the loop: the arenas live as long
/// as the `Function`, so the audit walks memory rather than spec vectors.
struct FloatArena {
  const void* data;
  std::size_t numel;
  bool is_double;
};

/// Allocation-free: called from inside the timed loop.
inline bool all_finite(const std::vector<FloatArena>& arenas) {
  for (const FloatArena& arena : arenas) {
    if (arena.is_double) {
      const double* values = static_cast<const double*>(arena.data);
      for (std::size_t i = 0; i < arena.numel; ++i) {
        if (!std::isfinite(values[i])) {
          return false;
        }
      }
    } else {
      const float* values = static_cast<const float*>(arena.data);
      for (std::size_t i = 0; i < arena.numel; ++i) {
        if (!std::isfinite(values[i])) {
          return false;
        }
      }
    }
  }
  return true;
}

inline std::vector<FloatArena> float_outputs(const pjrt::Function& function) {
  std::vector<FloatArena> arenas;
  arenas.reserve(function.num_outputs());
  for (std::size_t i = 0; i < function.num_outputs(); ++i) {
    const pjrt::ArraySpec& spec = function.output_spec(i);
    if (spec.dtype == pjrt::DType::Float64 ||
        spec.dtype == pjrt::DType::Float32) {
      arenas.push_back(FloatArena{function.output_raw(i), spec.numel,
                                  spec.dtype == pjrt::DType::Float64});
    }
  }
  return arenas;
}

/// The first cycle after the cold call, so the gate fires however short the
/// run is.
constexpr std::int64_t kFaultCycle = 1;

/// Corrupt cycle `kFaultCycle`, after the call and before the check, where a
/// real fault would appear.  A gate nobody has seen fail is not evidence of
/// anything: a quiet gate and a right answer look identical from outside.
inline void inject_fault(Fault fault, pjrt::Function& function,
                         const std::vector<FloatArena>& audited,
                         std::int64_t k) {
  if (fault == Fault::None || k != kFaultCycle) {
    return;
  }
  if (fault == Fault::Step) {
    // The next cycle's check sees a step that does not follow from this one.
    *function.input<std::int32_t>(cjfc::workload::kInStep) += 1;
  }
  if (fault == Fault::NonFinite && !audited.empty()) {
    // The outputs are const to callers; writing one is the whole point here.
    auto* poisoned =
        static_cast<double*>(const_cast<void*>(audited.front().data));
    poisoned[0] = std::numeric_limits<double>::quiet_NaN();
  }
}

// ------------------------------------------------------------- the reports

/// Everything the run produced, gathered once the timed loop is over.
struct Outcome {
  double runtime_ms = 0.0;     ///< Wall time to create the `Runtime`.
  double load_ms = 0.0;        ///< Wall time to construct the `Function`.
  double first_call_us = 0.0;  ///< The cold call, reported on its own.
  pjrt::LatencySummary compute;
  cjfc::Rusage faults;
  std::size_t step_errors = 0;
  bool finite_outputs = true;

  float cost = 0.0f;
  double grad_norm = 0.0;
  std::int32_t iterations_used = 0;
  std::int32_t backtracks_used = 0;

  int exit_code = cjfc::kExitOk;
};

inline void read_solution(const pjrt::Function& function, Outcome& outcome) {
  outcome.cost = *function.output<float>(cjfc::workload::kOutCost);
  outcome.grad_norm = *function.output<double>(cjfc::workload::kOutGradNorm);
  outcome.iterations_used =
      *function.output<std::int32_t>(cjfc::workload::kOutIterationsUsed);
  outcome.backtracks_used =
      *function.output<std::int32_t>(cjfc::workload::kOutBacktracksUsed);
}

inline void print_report(const Options& options, const pjrt::Runtime& runtime,
                         const pjrt::Function& function, const Outcome& outcome,
                         const pjrt::AllocGuard& guard) {
  if (options.quiet) {
    return;
  }
  std::printf("%s\n", runtime.describe().c_str());
  std::printf("artifact:   %s (%s)\n", options.artifact.c_str(),
              function.load_detail().c_str());
  std::printf("load:       runtime %.1f ms, function %.1f ms\n",
              outcome.runtime_ms, outcome.load_ms);
  std::printf("first call: %.1f us\n", outcome.first_call_us);

  cjfc::print_summary("steady state", outcome.compute);
  guard.report(stdout, options.iterations);

  std::printf("\n=== run ===\n");
  std::printf("  faults (%s): minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld\n",
              cjfc::Rusage::scope(), outcome.faults.minflt,
              outcome.faults.majflt, outcome.faults.nvcsw,
              outcome.faults.nivcsw);
  if (options.audit_values) {
    std::printf("  checks: step_counter_ok=%d finite_outputs=%d\n",
                outcome.step_errors == 0 ? 1 : 0,
                outcome.finite_outputs ? 1 : 0);
  } else {
    std::printf(
        "  checks: step_counter_ok=%d finite_outputs=skipped (--no-check)\n",
        outcome.step_errors == 0 ? 1 : 0);
  }
  std::printf(
      "  result: cost=%.6f grad_norm=%.6e iterations_used=%d "
      "backtracks_used=%d\n",
      static_cast<double>(outcome.cost), outcome.grad_norm,
      static_cast<int>(outcome.iterations_used),
      static_cast<int>(outcome.backtracks_used));
}

/// A samples file that was asked for and not written is worse than a failure.
inline void write_samples(const Options& options,
                          const pjrt::LatencyRecorder& compute) {
  if (!compute.write_samples(options.samples_path.c_str())) {
    throw std::runtime_error("cannot write samples to " + options.samples_path);
  }
}

/// The summary, and everything that decides whether the summary means anything.
inline void write_report(const Options& options, const cjfc::HostEnv& env,
                         const pjrt::Runtime& runtime,
                         const pjrt::Function& function,
                         const pjrt::LatencyRecorder& compute,
                         const Outcome& outcome,
                         const pjrt::AllocGuard& guard) {
  pjrt::HistogramBin bins[40];
  const std::size_t nbins =
      compute.histogram(bins, sizeof bins / sizeof bins[0]);

  cjfc::json report;
  report["schema"] = 1;
  report["example"] = "02_trajopt";
  report["artifact"] = options.artifact;
  report["config"] = cjfc::json{
      {"iterations", options.iterations},
      {"warmup", options.warmup},
      {"threads", options.threads},
      {"synchronous", options.synchronous},
  };
  report["host"] = cjfc::host_json(env);
  report["runtime"] =
      cjfc::runtime_json(runtime, function, outcome.runtime_ms, outcome.load_ms,
                         outcome.first_call_us);
  // The schema also names these at the top of the runtime object, so a
  // reader need not know which example wrote the file.
  report["runtime"]["load_kind"] = cjfc::load_kind_name(function.load_kind());
  report["runtime"]["runtime_ms"] = outcome.runtime_ms;
  report["runtime"]["load_ms"] = outcome.load_ms;
  report["runtime"]["first_call_us"] = outcome.first_call_us;
  report["compute_us"] = cjfc::summary_json(outcome.compute);
  report["compute_us"]["histogram"] = cjfc::histogram_json(bins, nbins);
  report["rusage"] = cjfc::json{
      {"minflt", outcome.faults.minflt},
      {"majflt", outcome.faults.majflt},
      {"nvcsw", outcome.faults.nvcsw},
      {"nivcsw", outcome.faults.nivcsw},
  };
  report["allocations"] = cjfc::alloc_json(guard, options.iterations);
  report["checks"] = cjfc::json{
      {"step_counter_ok", outcome.step_errors == 0},
      {"step_errors", outcome.step_errors},
      {"finite_outputs", outcome.finite_outputs},
  };
  report["exit_code"] = outcome.exit_code;
  cjfc::write_json(options.json_path, report);
}

}  // namespace trajopt
