/**
 * @file bench_main.cpp
 * @brief The measurement spine: one configuration, measured properly, and
 *        reported with everything a later reader needs to believe it.
 *
 * Every change to the call path is judged by the tail statistics this binary
 * prints -- p99.9/p50 and max/p50, not the mean.  It measures exactly one
 * configuration per run and records what that configuration was, because a
 * latency number without its provenance cannot be compared with anything.
 *
 * Three traps are designed around here, each of them learned by measuring
 * wrongly first:
 *
 *   - **A busy machine does not add noise to a tail measurement, it
 *     invalidates it.**  The same configuration measured during a concurrent
 *     build reported p50 2.4x high and max/p50 4.4 instead of 1.1.  The load
 *     average is printed with the numbers and `host.busy` is in the JSON
 *     report; check it before quoting anything.
 *   - **XLA dispatch is asynchronous.**  A warm-up that does not wait on
 *     *every* output leaves a backlog that lands on the first timed call and
 *     turns a 1.2x max/p50 into a 27x one.  Every call here, warm-up included,
 *     goes through `pjrt::Function::call()`, which blocks on all outputs.
 *   - **Sequential A/B comparisons drift with CPU temperature.**  Anything
 *     comparative is run as short interleaved bursts that append to one CSV
 *     (`tools/run_matrix.sh`); this binary produces one row of that, and
 *     `--csv` is what makes the interleaving possible.
 *
 * @code
 *   bench --fixture trajopt --iterations 20000 --csv artifacts/reports/rt.csv
 *   bench --all-cases                       # correctness sweep, no timing
 *   LD_PRELOAD=build/lib/malloc_guard.so \
 *     bench --alloc-gate self --require-guard
 * @endcode
 */

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "common/report.hpp"
#include "common/rt_env.hpp"
#include "fixture.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

constexpr char kUsage[] =
    "usage: bench [options]\n"
    "\n"
    "  --fixture NAME       reference fixture and artifact base name"
    " (trajopt)\n"
    "  --assets-dir DIR     where <fixture>_cases.json lives (artifacts)\n"
    "  --artifacts-dir DIR  where <fixture>.json and .binpb live (artifacts)\n"
    "  --iterations N       timed calls (2000)\n"
    "  --warmup N           calls before timing; the first is timed alone"
    " (50)\n"
    "  --case I             reference case fed to the timed loop (0)\n"
    "  --all-cases          sweep every case for correctness, then exit\n"
    "  --no-check           skip the correctness gate (not recommended)\n"
    "  --async              ask for asynchronous execution, not inline\n"
    "  --threads N          PJRT_NPROC for XLA's pools (1)\n"
    "  --devices N          CPU devices to create (1)\n"
    "  --rt                 apply the real-time hardening from rt_env.hpp,\n"
    "                       SCHED_FIFO 80 included\n"
    "  --cpu auto|none|N    which cpu --rt pins the loop to (auto)\n"
    "  --csv PATH           append one summary row\n"
    "  --samples PATH       write every raw sample as index,ns\n"
    "  --json PATH          write the full report, host audit included\n"
    "  --label NAME         label for the summary and the csv row (--fixture)\n"
    "  --alloc-gate MODE    off|self|all: fail if the armed window allocated"
    " (off)\n"
    "  --require-guard      fail when malloc_guard.so was not preloaded\n"
    "\n"
    "exit: 0 ok, 1 error, 2 wrong answer, 3 allocation gate, 4 guard missing";

/// Everything the run was asked for, resolved from the command line once.
struct Args {
  std::string fixture = "trajopt";
  std::string assets_dir = "artifacts";
  std::string artifacts_dir = "artifacts";
  std::string csv;
  std::string samples;
  std::string json;
  std::string label;
  std::string alloc_gate = "off";
  std::string cpu = "auto";
  std::size_t iterations = 2000;
  std::size_t warmup = 50;
  std::size_t case_index = 0;
  bool check = true;
  bool all_cases = false;
  bool synchronous = true;
  bool rt_harden = false;
  bool require_guard = false;
  int worker_threads = 1;
  int cpu_device_count = 1;
};

/// The computation ran and produced the wrong answer.  Distinct from every
/// other failure because a script has to tell "this build is slower" apart
/// from "this build is broken"; it becomes exit code 2.
class CorrectnessFailure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

Args read_args(const cjfc::Cli& cli) {
  Args args;
  args.fixture = cli.get("fixture", args.fixture);
  args.assets_dir = cli.get("assets-dir", args.assets_dir);
  args.artifacts_dir = cli.get("artifacts-dir", args.artifacts_dir);
  args.csv = cli.get("csv");
  args.samples = cli.get("samples");
  args.json = cli.get("json");
  args.alloc_gate = cli.get("alloc-gate", args.alloc_gate);
  args.cpu = cli.get("cpu", args.cpu);
  args.iterations = cli.get_size("iterations", args.iterations);
  args.warmup = cli.get_size("warmup", args.warmup);
  args.case_index = cli.get_size("case", args.case_index);
  args.check = !cli.flag("no-check");
  args.all_cases = cli.flag("all-cases");
  args.synchronous = !cli.flag("async");
  args.rt_harden = cli.flag("rt");
  args.require_guard = cli.flag("require-guard");
  args.worker_threads = static_cast<int>(cli.get_long("threads", 1));
  args.cpu_device_count = static_cast<int>(cli.get_long("devices", 1));
  args.label = cli.get("label", args.fixture);

  // A mistyped gate that quietly gates on nothing is the exact failure
  // report.hpp warns about: "the path allocated" and "nobody measured whether
  // the path allocated" must not look the same in a green log.
  if (args.alloc_gate != "off" && args.alloc_gate != "self" &&
      args.alloc_gate != "all") {
    throw std::runtime_error("--alloc-gate expects off, self or all, got '" +
                             args.alloc_gate + "'");
  }
  return args;
}

/**
 * @brief Record the knobs that actually move the numbers.
 *
 * Call it *after* the `Runtime` is constructed: the runtime is what sets
 * `PJRT_NPROC`, and it is the runtime that knows whether the synchronous
 * option was accepted or quietly refused.  Without this string a CSV row is
 * unattributable a month later.
 */
std::string describe_config(const Args& args, const pjrt::Runtime& runtime) {
  const auto env = [](const char* key) -> std::string {
    const char* value = std::getenv(key);
    return value != nullptr ? value : "unset";
  };
  return std::string("sync=") +
         cjfc::sync_mode_name(runtime.synchronous_mode()) +
         ";devices=" + std::to_string(args.cpu_device_count) +
         ";nproc=" + env("PJRT_NPROC") + ";xla_flags=" + env("XLA_FLAGS");
}

/// @brief One comparison, for the JSON report.
cjfc::json comparison_json(const bench::Fixture::Comparison& comparison) {
  return cjfc::json{
      {"max_rel_err", comparison.max_rel_err},
      {"exact_mismatches", comparison.exact_mismatches},
      {"nan_count", comparison.nan_count},
      {"ok", comparison.ok},
  };
}

/// @brief Print one comparison the same way in both paths.
void print_comparison(const bench::Fixture::Comparison& comparison) {
  std::printf("max rel err %.3e, %zu exact mismatches, %zu nan -- %s\n",
              comparison.max_rel_err, comparison.exact_mismatches,
              comparison.nan_count, comparison.ok ? "ok" : "FAIL");
}

/// Wall-clock duration in microseconds.
double to_us(std::chrono::steady_clock::duration d) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(d).count()) *
         1e-3;
}

/// What the measured run cost outside the steady-state distribution.
struct Timings {
  /// Creating the `Runtime`: dlopen, plugin init, client creation, XLA's pools
  /// starting.
  std::chrono::steady_clock::duration runtime{0};
  /// Constructing the `Function`: sidecar, relink, arenas, buffers.
  std::chrono::steady_clock::duration load{0};
  /// The first call, timed on its own.
  std::chrono::steady_clock::duration first_call{0};
  /// The correctness gate's verdict, when it ran.
  bench::Fixture::Comparison comparison;
  bool checked = false;
};

// docs: begin bench_measured
/**
 * @brief First call, warm-up, correctness gate, then the timed loop.
 *
 * The order is the whole point of this function:
 *
 *   1. The **first call** is timed on its own.  It carries page faults, lazy
 *      binding and whatever the runtime still initializes on demand, and a
 *      steady-state number must not include any of that -- but a control
 *      loop's first cycle really does pay it, so it is reported rather than
 *      discarded.
 *   2. **Warm-up**, every call of it blocking on all outputs.
 *   3. The **correctness gate**, before any timing.  A wrong answer must never
 *      be reported as a fast one, so this is a gate and not a warning.
 *   4. The allocation guard is armed for **exactly** the timed loop.  Arming
 *      it earlier would count warm-up's page faults and lazy initialization as
 *      allocations of the call path.
 *
 * @throws CorrectnessFailure when the outputs disagree with the reference.
 */
template <typename Call>
Timings run_measured(const Args& args, const bench::Fixture& fixture,
                     const std::vector<const void*>& output_ptrs,
                     pjrt::AllocGuard& guard, pjrt::LatencyRecorder& recorder,
                     Call&& one_call) {
  Timings timings;

  const auto first_start = std::chrono::steady_clock::now();
  one_call();
  timings.first_call = std::chrono::steady_clock::now() - first_start;

  for (std::size_t i = 1; i < args.warmup; ++i) {
    one_call();
  }

  if (args.check) {
    timings.comparison = fixture.compare(args.case_index, output_ptrs);
    timings.checked = true;
    std::printf("correctness: ");
    print_comparison(timings.comparison);
    if (!timings.comparison.ok) {
      throw CorrectnessFailure("case " + std::to_string(args.case_index) +
                               " disagrees with the reference");
    }
  }

  {
    pjrt::AllocGuardScope armed(guard);
    for (std::size_t i = 0; i < args.iterations; ++i) {
      recorder.time(one_call);
    }
  }
  return timings;
}
// docs: end bench_measured

/// The result of the `--all-cases` sweep.
struct Sweep {
  bool ok = true;
  cjfc::json cases = cjfc::json::array();
  double first_call_us = 0.0;
};

/**
 * @brief Run every reference case through one `Function`, forwards then
 *        backwards.
 *
 * The steady-state path reuses one set of zero-copy input buffers for the life
 * of the `Function`, so the case that matters for correctness is *changing*
 * inputs between calls -- which a benchmark feeding the same values every time
 * never exercises.  Two passes rather than one because the reverse pass gives
 * every case a different predecessor: a buffer left stale by the previous call
 * produces the right answer for exactly one ordering, and one pass would find
 * that acceptable.
 */
Sweep run_all_cases(const bench::Fixture& fixture, pjrt::Function& function,
                    const std::vector<const void*>& output_ptrs) {
  Sweep sweep;
  const std::size_t n_cases = fixture.num_cases();

  for (std::size_t pass = 0; pass < 2; ++pass) {
    for (std::size_t k = 0; k < n_cases; ++k) {
      const std::size_t c = pass == 0 ? k : n_cases - 1 - k;
      fixture.load_inputs(c, function);

      const auto start = std::chrono::steady_clock::now();
      function.call();
      const auto elapsed = std::chrono::steady_clock::now() - start;
      if (pass == 0 && k == 0) {
        sweep.first_call_us = to_us(elapsed);
      }

      const bench::Fixture::Comparison comparison =
          fixture.compare(c, output_ptrs);
      sweep.ok = sweep.ok && comparison.ok;
      std::printf("  pass %zu case %zu: ", pass, c);
      print_comparison(comparison);

      cjfc::json entry = comparison_json(comparison);
      entry["pass"] = pass;
      entry["case"] = c;
      sweep.cases.push_back(std::move(entry));
    }
  }
  return sweep;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const cjfc::Cli cli(argc, argv,
                        {"fixture", "assets-dir", "artifacts-dir", "csv",
                         "samples", "json", "label", "iterations", "warmup",
                         "case", "all-cases", "no-check", "async", "threads",
                         "devices", "rt", "cpu", "alloc-gate", "require-guard"},
                        kUsage);
    if (cli.help()) {
      return cjfc::kExitOk;
    }
    const Args args = read_args(cli);

    const bench::Fixture fixture(args.assets_dir, args.fixture);
    if (args.case_index >= fixture.num_cases()) {
      throw std::runtime_error("--case " + std::to_string(args.case_index) +
                               ": the fixture has " +
                               std::to_string(fixture.num_cases()) + " cases");
    }

    // Read the host before anything is measured, and say so loudly when it is
    // busy: the numbers below would be wrong rather than merely noisy.
    const cjfc::HostEnv host = cjfc::detect_host_env();
    std::printf("fixture:    %s (case %zu of %zu)\n", args.fixture.c_str(),
                args.case_index, fixture.num_cases());
    std::printf("iterations: %zu, warmup %zu\n", args.iterations, args.warmup);
    std::printf(
        "load avg:   %.2f %.2f %.2f%s\n", host.loadavg1, host.loadavg5,
        host.loadavg15,
        host.busy ? "   <- BUSY: these numbers are not comparable" : "");

    pjrt::RuntimeOptions runtime_options;
    runtime_options.synchronous = args.synchronous;
    runtime_options.worker_threads = args.worker_threads;
    runtime_options.cpu_device_count = args.cpu_device_count;

    Timings timings;
    const auto runtime_start = std::chrono::steady_clock::now();
    pjrt::Runtime runtime(runtime_options);
    const auto runtime_end = std::chrono::steady_clock::now();
    timings.runtime = runtime_end - runtime_start;

    pjrt::FunctionOptions function_options;
    // Warm-up belongs to this harness, so that the first call can be timed on
    // its own; the Function must not quietly do it too.
    function_options.warmup_calls = 0;
    const auto load_start = std::chrono::steady_clock::now();
    pjrt::Function function(runtime, args.artifacts_dir + "/" + args.fixture,
                            function_options);
    timings.load = std::chrono::steady_clock::now() - load_start;

    std::printf("%s\n", runtime.describe().c_str());

    // Hardening comes after the client exists: corral_xla_threads finds XLA's
    // pools by name, and those threads are created with the client.  That
    // costs harden_malloc its ideal position -- before the bulk of startup
    // allocates -- and the trade is deliberate, since the pools are the larger
    // source of jitter.
    cjfc::DmaLatencyHold dma;
    std::vector<cjfc::Step> steps;
    if (args.rt_harden) {
      cjfc::HardeningOptions hardening;
      hardening.cpu = args.cpu;
      // --rt is already an explicit opt-in, so it asks for SCHED_FIFO too;
      // rt_env leaves priority at 0 by default because a real-time thread
      // nobody asked for can take the machine with it.
      hardening.rt_priority = 80;
      int chosen_cpu = -1;
      steps = cjfc::apply_hardening(host, hardening, dma, &chosen_cpu);
      std::printf("\nreal-time hardening:\n");
      for (const cjfc::Step& step : steps) {
        cjfc::print_step(step);
      }
    } else if (cli.flag("cpu")) {
      std::printf("note: --cpu only takes effect with --rt\n");
    }

    std::vector<const void*> output_ptrs(function.num_outputs());
    for (std::size_t i = 0; i < function.num_outputs(); ++i) {
      output_ptrs[i] = function.output_raw(i);
    }

    const std::string config = describe_config(args, runtime);

    if (args.all_cases) {
      std::printf("\nsweeping %zu cases, forwards then backwards:\n",
                  fixture.num_cases());
      const Sweep sweep = run_all_cases(fixture, function, output_ptrs);
      std::printf("%s\n", sweep.ok ? "all cases agree with the reference"
                                   : "FAIL: a case disagreed");
      if (!args.json.empty()) {
        cjfc::json report;
        report["tool"] = "bench --all-cases";
        report["label"] = args.label;
        report["config"] = config;
        report["fixture"] = cjfc::json{{"name", args.fixture},
                                       {"dir", args.assets_dir},
                                       {"num_cases", fixture.num_cases()}};
        report["cases"] = sweep.cases;
        report["ok"] = sweep.ok;
        report["runtime"] =
            cjfc::runtime_json(runtime, function, to_us(timings.runtime) * 1e-3,
                               to_us(timings.load) * 1e-3, sweep.first_call_us);
        report["host"] = cjfc::host_json(host);
        report["hardening"] = cjfc::steps_json(steps);
        cjfc::write_json(args.json, report);
        std::printf("wrote %s\n", args.json.c_str());
      }
      return sweep.ok ? cjfc::kExitOk : cjfc::kExitCorrectness;
    }

    pjrt::AllocGuard guard;
    pjrt::LatencyRecorder recorder(args.iterations);

    // The copy into the input arenas is inside the timed region on purpose: a
    // control loop writes fresh inputs every step, so a benchmark that skips
    // the copy is measuring something no caller does.
    const Timings measured =
        run_measured(args, fixture, output_ptrs, guard, recorder, [&]() {
          fixture.load_inputs(args.case_index, function);
          function.call();
        });
    timings.first_call = measured.first_call;
    timings.comparison = measured.comparison;
    timings.checked = measured.checked;

    std::printf("\n=== cold start (microseconds) ===\n");
    std::printf("  runtime    %10.1f\n", to_us(timings.runtime));
    std::printf("  load       %10.1f\n", to_us(timings.load));
    std::printf("  first call %10.1f\n", to_us(timings.first_call));
    std::printf("config:     %s\n", config.c_str());

    recorder.report(stdout, args.label.c_str());
    guard.report(stdout, args.iterations);

    int status = cjfc::kExitOk;
    if (!args.csv.empty() &&
        !recorder.write_csv_row(args.csv.c_str(), args.label.c_str(),
                                config.c_str())) {
      std::fprintf(stderr, "error: cannot append to %s\n", args.csv.c_str());
      status = cjfc::kExitError;
    }
    if (!args.samples.empty() &&
        !recorder.write_samples(args.samples.c_str())) {
      std::fprintf(stderr, "error: cannot write %s\n", args.samples.c_str());
      status = cjfc::kExitError;
    }

    if (!args.json.empty()) {
      pjrt::HistogramBin bins[40];
      const std::size_t nbins =
          recorder.histogram(bins, sizeof bins / sizeof bins[0]);

      cjfc::json report;
      report["tool"] = "bench";
      report["label"] = args.label;
      report["config"] = config;
      report["fixture"] = cjfc::json{{"name", args.fixture},
                                     {"dir", args.assets_dir},
                                     {"case", args.case_index},
                                     {"num_cases", fixture.num_cases()}};
      report["iterations"] = args.iterations;
      report["warmup"] = args.warmup;
      report["checked"] = timings.checked;
      if (timings.checked) {
        report["correctness"] = comparison_json(timings.comparison);
      }
      report["runtime"] = cjfc::runtime_json(
          runtime, function, to_us(timings.runtime) * 1e-3,
          to_us(timings.load) * 1e-3, to_us(timings.first_call));
      report["latency"] = cjfc::summary_json(recorder.summary());
      report["histogram"] = cjfc::histogram_json(bins, nbins);
      report["allocations"] = cjfc::alloc_json(guard, args.iterations);
      report["alloc_gate"] = cjfc::json{{"mode", args.alloc_gate},
                                        {"require_guard", args.require_guard}};
      report["host"] = cjfc::host_json(host);
      report["hardening"] = cjfc::steps_json(steps);
      cjfc::write_json(args.json, report);
      std::printf("wrote %s\n", args.json.c_str());
    }

    const int gate =
        cjfc::alloc_gate_exit_code(guard, args.alloc_gate, args.require_guard);
    if (gate == cjfc::kExitAllocGate) {
      std::fprintf(stderr,
                   "error: the steady-state path allocated (--alloc-gate %s)\n",
                   args.alloc_gate.c_str());
    } else if (gate == cjfc::kExitGuardMissing) {
      std::fprintf(stderr,
                   "error: --require-guard was given but malloc_guard.so was "
                   "not preloaded; nothing was measured\n");
    }
    return gate != cjfc::kExitOk ? gate : status;
  } catch (const CorrectnessFailure& failure) {
    std::fprintf(stderr, "correctness: %s\n", failure.what());
    return cjfc::kExitCorrectness;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return cjfc::kExitError;
  }
}
