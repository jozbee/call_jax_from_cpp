/**
 * @file support.hpp
 * @brief Everything `realtime.cpp` needs that is not the periodic loop: the
 *        flags, the host audit, the summaries and the two reports.
 *
 * Split out so that the example itself is the sequence a control process
 * actually performs -- harden, load, warm up, arm, loop, report -- and so that
 * the loop body sits on one page.  Everything here runs before the first cycle
 * or after the last one; nothing in it may be called from inside the window the
 * allocation guard is armed over.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "common/report.hpp"
#include "common/rt_env.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/runtime.hpp"

namespace rt {

/// Sample budget for a run that was told to go until SIGINT.  A million cycles
/// is 2.8 hours at 100 Hz; past that the recorders count drops rather than
/// growing, because growing would allocate inside the window being measured.
constexpr std::size_t kUnboundedCapacity = 1000000;

/// The clock for everything timed outside the loop.  Not
/// `high_resolution_clock`, which is an alias for the wall clock on some
/// standard libraries.
using Clock = std::chrono::steady_clock;

/// @brief Milliseconds since @p t0.
inline double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// @brief Microseconds since @p t0.
inline double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------- the flags

const char* const kUsage =
    "usage: example_03_realtime [options]\n"
    "\n"
    "Run the 02_trajopt artifact on a fixed period and report the jitter,\n"
    "the deadline misses and the allocations.  Needs `make export` first.\n"
    "\n"
    "  --artifact PATH     artifact base path (default: artifacts/trajopt)\n"
    "  --period-us N       control period in microseconds (default: 10000)\n"
    "  --iterations N      timed cycles; 0 runs until SIGINT (default: 3000)\n"
    "  --warmup N          untimed cycles before the window (default: 200)\n"
    "  --cpu auto|none|N   which cpu to pin the loop to (default: auto)\n"
    "  --rt-priority N     SCHED_FIFO priority; 0 disables (default: 80)\n"
    "  --no-malloc-tune    do not call harden_malloc()\n"
    "  --no-mlock          do not call lock_memory()\n"
    "  --no-corral         leave XLA's pool threads where they are\n"
    "  --dma-latency auto|off  hold /dev/cpu_dma_latency at 0 us (needs root)\n"
    "  --threads N         XLA worker threads; 0 is XLA's default (default: "
    "1)\n"
    "  --json PATH         write the report as JSON as well as printing it\n"
    "  --samples PATH      write every raw sample as CSV (one file per "
    "series)\n"
    "  --alloc-gate off|self|all  which allocations fail the run (default: "
    "self)\n"
    "  --require-guard     fail when malloc_guard.so was not preloaded\n"
    "  --no-check          do not fail on a step-counter mismatch\n"
    "  --quiet             print one summary line instead of the full report\n"
    "  -h, --help          print this and exit\n"
    "\n"
    "exit: 0 ok, 1 error, 2 wrong answer, 3 the loop allocated,\n"
    "      4 an allocation gate was required but nothing was measured\n";

/// @brief Parse @p argv against the flags `kUsage` documents.
///
/// The known-flag list lives here rather than at the call site because it and
/// the usage text are one statement: a flag added to one and not the other is
/// either undocumented or refused.
inline cjfc::Cli make_cli(int argc, char** argv) {
  return cjfc::Cli(
      argc, argv,
      {"artifact", "period-us", "iterations", "warmup", "cpu", "rt-priority",
       "no-malloc-tune", "no-mlock", "no-corral", "dma-latency", "threads",
       "json", "samples", "alloc-gate", "require-guard", "no-check", "quiet"},
      kUsage);
}

/// @brief What the loop asks the operating system for, before any flag is read.
///
/// Everything on, at the priority the usage text documents; `--rt-priority 0`
/// is how a caller opts out of `SCHED_FIFO` rather than a separate flag.
inline cjfc::HardeningOptions default_hardening() {
  cjfc::HardeningOptions hardening;
  hardening.rt_priority = 80;
  return hardening;
}

/// Everything the command line can say, resolved once at startup.
struct Options {
  std::string artifact = "artifacts/trajopt";
  std::size_t period_us = 10000;
  std::size_t iterations = 3000;  ///< 0 runs until SIGINT.
  std::size_t warmup = 200;
  int threads = 1;
  cjfc::HardeningOptions hardening = default_hardening();
  std::string json_path;
  std::string samples_path;
  std::string alloc_gate = "self";
  bool require_guard = false;
  bool check = true;
  bool quiet = false;
};

/// One of @p allowed, or a `std::runtime_error` naming the flag and what it
/// takes.  A mistyped `--alloc-gate slef` that silently disabled the gate would
/// be a green run that checked nothing.
inline std::string enum_flag(const cjfc::Cli& cli, const char* name,
                             const char* fallback,
                             std::initializer_list<const char*> allowed) {
  const std::string value = cli.get(name, fallback);
  for (const char* candidate : allowed) {
    if (value == candidate) {
      return value;
    }
  }
  std::string expected;
  for (const char* candidate : allowed) {
    if (!expected.empty()) {
      expected += "|";
    }
    expected += candidate;
  }
  throw std::runtime_error("'--" + std::string(name) + "' expects " + expected +
                           ", got '" + value + "'");
}

/// Read the command line into `Options`, rejecting values the program cannot
/// honour rather than rounding them into something it can.
inline Options parse_options(const cjfc::Cli& cli) {
  Options options;
  options.artifact = cli.get("artifact", options.artifact);
  options.period_us = cli.get_size("period-us", options.period_us);
  options.iterations = cli.get_size("iterations", options.iterations);
  options.warmup = cli.get_size("warmup", options.warmup);
  options.threads = static_cast<int>(cli.get_long("threads", options.threads));
  options.hardening.cpu = cli.get("cpu", options.hardening.cpu);
  options.hardening.rt_priority = static_cast<int>(
      cli.get_long("rt-priority", options.hardening.rt_priority));
  options.hardening.malloc_tune = !cli.flag("no-malloc-tune");
  options.hardening.mlock = !cli.flag("no-mlock");
  options.hardening.corral = !cli.flag("no-corral");
  options.hardening.dma_latency =
      enum_flag(cli, "dma-latency", "off", {"auto", "off"}) == "auto";
  options.json_path = cli.get("json");
  options.samples_path = cli.get("samples");
  options.alloc_gate =
      enum_flag(cli, "alloc-gate", "self", {"off", "self", "all"});
  options.require_guard = cli.flag("require-guard");
  options.check = !cli.flag("no-check");
  options.quiet = cli.flag("quiet");

  if (options.period_us == 0) {
    throw std::runtime_error("'--period-us' must be at least 1");
  }
  // 99 is the ceiling POSIX guarantees, and a priority above the kernel's own
  // migration and watchdog threads is how a machine stops responding.
  if (options.hardening.rt_priority < 0 || options.hardening.rt_priority > 99) {
    throw std::runtime_error("'--rt-priority' must be between 0 and 99, got " +
                             std::to_string(options.hardening.rt_priority));
  }
  if (options.threads < 0) {
    throw std::runtime_error("'--threads' must be 0 or more");
  }
  return options;
}

// ---------------------------------------------------------- the host audit

/// @brief A CPU set the way the kernel spells one: `2-5,8`, or `(none)`.
inline std::string cpulist(const std::vector<int>& cpus) {
  if (cpus.empty()) {
    return "(none)";
  }
  std::string text;
  for (std::size_t i = 0; i < cpus.size();) {
    std::size_t j = i;
    while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) {
      ++j;
    }
    if (!text.empty()) {
      text += ",";
    }
    text += std::to_string(cpus[i]);
    if (j > i) {
      text += "-" + std::to_string(cpus[j]);
    }
    i = j + 1;
  }
  return text;
}

/// @brief An rlimit soft limit, where -1 means the kernel imposes none.
inline std::string rlimit_text(long value) {
  return value < 0 ? std::string("unlimited") : std::to_string(value);
}

/// @brief Print the host audit: the settings that decide whether any number
///        below it is worth reading.  Silent under `--quiet`.
inline void print_host(const Options& options, const cjfc::HostEnv& env) {
  if (options.quiet) {
    return;
  }
  std::printf("=== host ===\n");
  std::printf("  kernel        %s%s%s\n", env.kernel.c_str(),
              env.preempt_rt ? "  PREEMPT_RT" : "  (not PREEMPT_RT)",
              env.in_container ? "  [in a container]" : "");
  std::printf("  cpus          %d online, isolated %s, nohz_full %s\n",
              env.cpus_online, cpulist(env.isolated).c_str(),
              cpulist(env.nohz_full).c_str());
  std::printf("  affinity      %s\n", cpulist(env.affinity).c_str());
  std::printf("  governor      %s     thp %s     smt %s\n",
              env.governor.c_str(), env.thp.c_str(), env.smt.c_str());
  std::printf(
      "  limits        rtprio %s, memlock %s, sched_rt_runtime_us %ld\n",
      rlimit_text(env.rlimit_rtprio).c_str(),
      rlimit_text(env.rlimit_memlock).c_str(), env.rt_runtime_us);
  std::printf("  c-states      /dev/cpu_dma_latency %s\n",
              env.cpu_dma_latency_writable ? "writable" : "not writable");
  std::printf("  loadavg       %.2f %.2f %.2f\n", env.loadavg1, env.loadavg5,
              env.loadavg15);
}

/// @brief Say on stderr that this host cannot produce a usable tail number.
///
/// stderr, so it survives `--json` piping and `--quiet` both: a tail number
/// from a busy machine is not noisy, it is wrong.
inline void warn_if_busy(const cjfc::HostEnv& env) {
  if (env.busy) {
    std::fprintf(stderr,
                 "WARNING: loadavg1=%.2f > 1.0 -- numbers from a busy "
                 "machine are wrong, not noisy\n",
                 env.loadavg1);
  }
}

/// @brief Print what the client is.  Silent under `--quiet`.
inline void print_runtime(const Options& options,
                          const pjrt::Runtime& runtime) {
  if (options.quiet) {
    return;
  }
  std::printf("\n=== runtime ===\n  %s\n", runtime.describe().c_str());
}

/// @brief Print what each hardening step did, and why when it could not.
///        Silent under `--quiet`.
inline void print_steps(const Options& options,
                        const std::vector<cjfc::Step>& steps) {
  if (options.quiet) {
    return;
  }
  std::printf("\n=== hardening ===\n");
  for (const cjfc::Step& step : steps) {
    cjfc::print_step(step);
  }
}

// ------------------------------------------------------- the loop's numbers

/// Wall clock for the three things that happen once, before the loop.
struct Timing {
  double runtime_ms = 0.0;     ///< Plugin, client, XLA pools.
  double load_ms = 0.0;        ///< Deserializing or compiling the executable.
  double first_call_us = 0.0;  ///< The cold call, excluded from every number.
};

/// The four distributions, reserved before the loop and never resized.
struct Recorders {
  pjrt::LatencyRecorder compute;
  pjrt::LatencyRecorder cycle;
  pjrt::LatencyRecorder wake;
  pjrt::LatencyRecorder jitter;

  explicit Recorders(std::size_t capacity)
      : compute(capacity), cycle(capacity), wake(capacity), jitter(capacity) {}
};

/// What the loop counts about its own schedule, incremented in the loop body
/// and read once it has stopped.
struct Deadlines {
  std::size_t missed = 0;
  std::int64_t worst_overrun_ns = 0;
  std::size_t step_errors = 0;

  /// @brief Count one cycle, whose work finished @p overrun_ns after its
  ///        deadline.  A non-positive overrun made it.
  ///
  /// Two integers and a compare: this is called from inside the armed window,
  /// so it may not allocate, and does not.
  void observe(std::int64_t overrun_ns) {
    if (overrun_ns > 0) {
      ++missed;
      worst_overrun_ns = std::max(worst_overrun_ns, overrun_ns);
    }
  }
};

/**
 * @brief The whole run, summarized once the loop is over.
 *
 * The derived numbers are accessors rather than fields so that the printed
 * report and the JSON cannot drift apart: there is one definition of
 * utilization, and both readers use it.
 */
struct Results {
  pjrt::LatencySummary compute;
  pjrt::LatencySummary cycle;
  pjrt::LatencySummary wake;
  pjrt::LatencySummary jitter;
  std::size_t completed = 0;
  Deadlines counters;
  cjfc::Rusage rusage;
  double period_us = 0.0;
  int exit_code = cjfc::kExitOk;

  Results(const Recorders& recorders, std::size_t completed_cycles,
          const Deadlines& deadlines, const cjfc::Rusage& usage, double period)
      : compute(recorders.compute.summary()),
        cycle(recorders.cycle.summary()),
        wake(recorders.wake.summary()),
        jitter(recorders.jitter.summary()),
        completed(completed_cycles),
        counters(deadlines),
        rusage(usage),
        period_us(period) {}

  bool step_counter_ok() const { return counters.step_errors == 0; }
  double worst_overrun_us() const {
    return static_cast<double>(counters.worst_overrun_ns) * 1e-3;
  }
  double miss_rate() const {
    return completed > 0 ? static_cast<double>(counters.missed) /
                               static_cast<double>(completed)
                         : 0.0;
  }
  double utilization_p50() const { return cycle.p50_us / period_us; }
  double utilization_max() const { return cycle.max_us / period_us; }
};

// ------------------------------------------------------------- the reports

/// @brief Print the load costs, once, before the loop starts.  Silent under
///        `--quiet`.
inline void print_cold_start(const Options& options, const Timing& timing,
                             const std::string& detail) {
  if (options.quiet) {
    return;
  }
  std::printf("\n=== cold start ===\n");
  std::printf("  runtime     %10.1f ms   (plugin, client, XLA pools)\n",
              timing.runtime_ms);
  std::printf("  load        %10.1f ms   (%s)\n", timing.load_ms,
              detail.c_str());
  std::printf("  first call  %10.1f us   (excluded from every number below)\n",
              timing.first_call_us);
}

/// @brief Print the four distributions, the schedule and the checks.
inline void print_report(const Options& options, const Results& results,
                         const pjrt::AllocGuard& guard) {
  cjfc::print_summary("call latency (compute)", results.compute);
  cjfc::print_summary("cycle time (wake to done)", results.cycle);
  cjfc::print_summary("wake-up latency (late by)", results.wake);
  cjfc::print_summary("period jitter (signed)", results.jitter,
                      /*signed_samples=*/true);

  std::printf("\n=== periodic behaviour ===\n");
  std::printf(
      "  deadlines: %zu missed of %zu (%.3f%%), worst overrun %.1f us, "
      "utilization p50 %.3f max %.3f\n",
      results.counters.missed, results.completed, results.miss_rate() * 100.0,
      results.worst_overrun_us(), results.utilization_p50(),
      results.utilization_max());
  std::printf(
      "  rusage:    minflt %ld, majflt %ld, nvcsw %ld, nivcsw %ld (%s scope)\n",
      results.rusage.minflt, results.rusage.majflt, results.rusage.nvcsw,
      results.rusage.nivcsw, cjfc::Rusage::scope());
  guard.report(stdout, results.completed);
  std::printf("\n=== checks ===\n");
  std::printf("  step counter %s (%zu mismatches of %zu%s)\n",
              results.step_counter_ok() ? "ok" : "FAILED",
              results.counters.step_errors, results.completed,
              options.check ? "" : ", not gated: --no-check");
  std::printf("  exit code    %d\n", results.exit_code);
}

/// @brief The one line `--quiet` prints instead of the report.
inline void print_quiet_line(const Options& options, const Results& results,
                             const pjrt::AllocGuard& guard) {
  std::printf(
      "03_realtime n=%zu period=%zuus compute p50=%.1f p99.9=%.1f max=%.1f "
      "jitter p99.9=%.1f missed=%zu self_allocs=%lu exit=%d\n",
      results.completed, options.period_us, results.compute.p50_us,
      results.compute.p999_us, results.compute.max_us, results.jitter.p999_us,
      results.counters.missed, guard.allocs_self(), results.exit_code);
}

/**
 * @brief Write the JSON report.
 *
 * `hardening` is an object keyed by step name rather than an array, so a reader
 * can ask whether `lock_memory` worked without scanning a list.
 */
inline void write_report(const Options& options, const cjfc::HostEnv& env,
                         const std::vector<cjfc::Step>& steps,
                         const pjrt::Runtime& runtime,
                         const pjrt::Function& function, const Timing& timing,
                         const Results& results, const pjrt::AllocGuard& guard,
                         int cpu_chosen) {
  cjfc::json hardening_json = cjfc::json::object();
  for (const cjfc::Step& step : steps) {
    hardening_json[step.name] =
        cjfc::json{{"ok", step.ok}, {"detail", step.detail}};
  }

  const cjfc::json report = {
      {"schema", 1},
      {"example", "03_realtime"},
      {"artifact", options.artifact},
      {"config",
       {{"period_us", options.period_us},
        {"iterations", options.iterations},
        {"warmup", options.warmup},
        {"cpu", options.hardening.cpu},
        {"cpu_chosen", cpu_chosen},
        {"rt_priority", options.hardening.rt_priority},
        {"threads", options.threads},
        {"synchronous", runtime.options().synchronous}}},
      {"host", cjfc::host_json(env)},
      {"hardening", hardening_json},
      {"runtime", cjfc::runtime_json(runtime, function, timing.runtime_ms,
                                     timing.load_ms, timing.first_call_us)},
      {"compute_us", cjfc::summary_json(results.compute)},
      {"cycle_us", cjfc::summary_json(results.cycle)},
      {"wake_latency_us", cjfc::summary_json(results.wake)},
      {"period_jitter_us", cjfc::summary_json(results.jitter)},
      {"deadlines",
       {{"period_us", results.period_us},
        {"missed", results.counters.missed},
        {"miss_rate", results.miss_rate()},
        {"worst_overrun_us", results.worst_overrun_us()},
        {"utilization_p50", results.utilization_p50()},
        {"utilization_max", results.utilization_max()}}},
      {"rusage",
       {{"minflt", results.rusage.minflt},
        {"majflt", results.rusage.majflt},
        {"nvcsw", results.rusage.nvcsw},
        {"nivcsw", results.rusage.nivcsw}}},
      {"allocations", cjfc::alloc_json(guard, results.completed)},
      {"checks",
       {{"step_counter_ok", results.step_counter_ok()},
        {"step_errors", results.counters.step_errors}}},
      {"exit_code", results.exit_code},
  };
  cjfc::write_json(options.json_path, report);
}

/// @brief `raw.csv` + `cycle` -> `raw_cycle.csv`; a path with no extension just
///        gets the suffix appended.
inline std::string with_suffix(const std::string& path, const char* suffix) {
  const std::size_t dot = path.find_last_of('.');
  const std::size_t slash = path.find_last_of('/');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return path + "_" + suffix;
  }
  return path.substr(0, dot) + "_" + suffix + path.substr(dot);
}

/// @brief Write every raw sample, four series to four files.
///
/// A summary cannot say *when* an outlier happened, and a spike on cycle 3 and
/// a spike on cycle 30,000 have the same p99.9 and completely different causes.
inline void write_samples(const std::string& path, const Recorders& r) {
  r.compute.write_samples(path.c_str());
  r.cycle.write_samples(with_suffix(path, "cycle").c_str());
  r.wake.write_samples(with_suffix(path, "wake").c_str());
  r.jitter.write_samples(with_suffix(path, "jitter").c_str());
}

}  // namespace rt
