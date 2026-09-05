/**
 * @file trajopt.cpp
 * @brief Time the trajectory-optimisation artifact, and check that the fast
 *        answer is also the right one.
 *
 * This is the example to quote numbers from.  It loads the artifact
 * `examples/02_trajopt/export.py` writes, drives it the way a receding-horizon
 * controller would -- reference in, controls out, outputs fed back into the
 * next call's inputs -- and reports the tail of the call latency rather than
 * its average.
 *
 * The shape of the run is the point:
 *
 *   - **The first call is reported on its own.**  It carries page faults, lazy
 *     symbol binding and whatever the runtime defers to its first execution,
 *     and folding it into a percentile would let one call dominate the tail it
 *     is supposed to describe.  `FunctionOptions::warmup_calls` is therefore 0:
 *     the harness owns warm-up, so the first call really is the first.
 *   - **Warm-up runs the same loop body** the timed section does, including the
 *     feedback, so the pages the loop touches are the pages warm-up touched.
 *   - **The allocation guard is armed around the timed loop only.**  Arming it
 *     over warm-up would fold in the initialization warm-up exists to pay for
 *     and turn a clean path into a few thousand allocations.
 *   - **Nothing in the timed loop allocates, locks, logs or flushes.**  The
 *     reference is written with plain stores into an arena, the outputs are
 *     read out of arenas, and the recorder stores one integer per call.
 *
 * Two checks run alongside the stopwatch, because a call path that runs is not
 * the same as a call path that is right.  `step_next` must be exactly
 * `step + 1` -- an integer identity that a stale or unread input arena breaks
 * and a float tolerance would not catch -- and every floating-point output must
 * be finite.  Either failure exits 2, whatever the latency looked like.
 *
 * @code
 *   ./build/bin/example_02_trajopt --iterations 2000 --json report.json
 *   LD_PRELOAD=build/lib/malloc_guard.so ./build/bin/example_02_trajopt \
 *       --alloc-gate self --require-guard
 * @endcode
 */

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "common/report.hpp"
#include "common/rt_env.hpp"
#include "common/trajopt_signature.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

/// The clock for every measurement here.  Not `high_resolution_clock`, which
/// is an alias for the wall clock on some standard libraries: an NTP step
/// during a run would show up as a spectacular outlier that never happened.
using Clock = std::chrono::steady_clock;

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
    "  --alloc-gate G    off, self or all; exit 3 when it fails (default off)\n"
    "  --require-guard   exit 4 when malloc_guard.so was not preloaded\n"
    "  --quiet           print nothing on success\n"
    "\n"
    "Exit codes: 0 ok, 1 error, 2 wrong answer, 3 allocation gate, 4 no guard.";

/// Milliseconds between two samples of `Clock`.
double millis(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

/// Microseconds between two samples of `Clock`.
double micros(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

/**
 * @brief One floating-point output arena, resolved once for the audit.
 *
 * Resolved before the loop rather than looked up inside it: the arenas live as
 * long as the `Function`, so their addresses and lengths are loop invariants,
 * and the audit then costs a walk over memory that is already hot instead of a
 * walk through the spec vectors.
 */
struct FloatArena {
  const void* data;   ///< The arena itself.
  std::size_t numel;  ///< Elements in it.
  bool is_double;     ///< float64 when true, float32 when false.
};

/// @brief Whether every element of every audited arena is finite.
/// Allocation-free and branch-light: safe to call from the timed loop.
bool all_finite(const std::vector<FloatArena>& arenas) {
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

/// @brief The floating-point outputs of @p function, in index order.
std::vector<FloatArena> float_outputs(const pjrt::Function& function) {
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

int run(int argc, char** argv) {
  cjfc::Cli cli(argc, argv,
                {"artifact", "iterations", "warmup", "threads", "async",
                 "json", "samples", "no-check", "alloc-gate", "require-guard",
                 "quiet"},
                kUsage);
  if (cli.help()) {
    return cjfc::kExitOk;
  }

  const std::string artifact = cli.get("artifact", "artifacts/trajopt");
  const std::size_t iterations = cli.get_size("iterations", 2000);
  const std::size_t warmup = cli.get_size("warmup", 50);
  const long threads = cli.get_long("threads", 1);
  const bool synchronous = !cli.flag("async");
  const std::string json_path = cli.get("json", "");
  const std::string samples_path = cli.get("samples", "");
  const bool audit_values = !cli.flag("no-check");
  const std::string gate = cli.get("alloc-gate", "off");
  const bool require_guard = cli.flag("require-guard");
  const bool quiet = cli.flag("quiet");

  // Refused rather than ignored: a gate nobody spelled correctly is a gate
  // that passes everything, which is the failure this project takes least
  // well.
  if (gate != "off" && gate != "self" && gate != "all") {
    throw std::runtime_error("'--alloc-gate' expects off, self or all, got '" +
                             gate + "'");
  }
  if (threads < 0 || threads > 4096) {
    throw std::runtime_error(
        "'--threads' expects a count between 0 and 4096");
  }

  // Read before anything is measured, and warned about on stderr even under
  // --quiet: a latency number from a busy machine is not noisy, it is wrong.
  const cjfc::HostEnv env = cjfc::detect_host_env();
  if (env.busy) {
    std::fprintf(stderr,
                 "WARNING: load average is %.2f; these numbers describe a "
                 "contended machine and are not comparable with anything. "
                 "Discard the run rather than correcting it.\n",
                 env.loadavg1);
  }

  pjrt::RuntimeOptions runtime_options;
  runtime_options.synchronous = synchronous;
  runtime_options.worker_threads = static_cast<int>(threads);

  const auto t_start = Clock::now();
  pjrt::Runtime runtime(runtime_options);
  const auto t_runtime = Clock::now();

  pjrt::FunctionOptions function_options;
  // The harness warms up, so that the first call below is genuinely the first
  // one and can be reported as such.
  function_options.warmup_calls = 0;
  pjrt::Function function(runtime, artifact, function_options);
  const auto t_loaded = Clock::now();

  const double runtime_ms = millis(t_start, t_runtime);
  const double load_ms = millis(t_runtime, t_loaded);

  const cjfc::Dims dims = cjfc::check_signature(function);
  cjfc::init_inputs(function, dims);

  // Resolved once.  Everything the loop touches is a pointer or an integer by
  // the time the stopwatch starts.
  double* const x_ref = function.input<double>(cjfc::kInXRef);
  const std::vector<FloatArena> audited =
      audit_values ? float_outputs(function) : std::vector<FloatArena>();

  std::int64_t cycle = 0;
  std::size_t step_errors = 0;
  bool finite_outputs = true;

  // Recirculate one cycle's outputs into the next cycle's inputs, and check
  // what came back.  Run for every call, warm-up included: a step counter that
  // goes wrong during warm-up is exactly as broken as one that goes wrong
  // later.  The step check is one integer comparison and always runs;
  // --no-check drops only the value audit, which is the part that walks every
  // element of every float arena.
  const auto finish_cycle = [&](std::int64_t k) {
    if (!cjfc::feedback(function, dims, k)) {
      ++step_errors;
    }
    if (audit_values && !all_finite(audited)) {
      finite_outputs = false;
    }
  };

  cjfc::write_reference(x_ref, dims, cycle);
  const auto t_call = Clock::now();
  function.call();
  const double first_call_us = micros(t_call, Clock::now());
  finish_cycle(cycle);
  ++cycle;

  for (std::size_t i = 0; i < warmup; ++i) {
    cjfc::write_reference(x_ref, dims, cycle);
    function.call();
    finish_cycle(cycle);
    ++cycle;
  }

  pjrt::AllocGuard guard;

// docs: begin latency-recorder
  // Capacity is reserved once, here: the recorder drops rather than grows,
  // because growing would allocate in the middle of the run being measured.
  pjrt::LatencyRecorder compute(iterations);

  const cjfc::Rusage before = cjfc::Rusage::now();
  {
    pjrt::AllocGuardScope armed(guard);
    for (std::size_t i = 0; i < iterations; ++i) {
      // Fresh reference for this cycle, written straight into the input arena
      // XLA will read.  Between calls, never during one.
      cjfc::write_reference(x_ref, dims, cycle);
      {
        pjrt::ScopedLatency sample(compute);
        function.call();
      }
      finish_cycle(cycle);
      ++cycle;
    }
  }
  const cjfc::Rusage after = cjfc::Rusage::now();

  const pjrt::LatencySummary summary = compute.summary();
// docs: end latency-recorder

  const cjfc::Rusage faults = after - before;

  const float cost = *function.output<float>(cjfc::kOutCost);
  const double grad_norm = *function.output<double>(cjfc::kOutGradNorm);
  const std::int32_t iterations_used =
      *function.output<std::int32_t>(cjfc::kOutIterationsUsed);
  const std::int32_t backtracks_used =
      *function.output<std::int32_t>(cjfc::kOutBacktracksUsed);

  int exit_code = cjfc::kExitOk;
  if (step_errors != 0 || !finite_outputs) {
    // A wrong answer outranks an allocation gate: the gate describes how the
    // answer was produced, and there is no point grading that first.
    exit_code = cjfc::kExitCorrectness;
  } else {
    exit_code = cjfc::alloc_gate_exit_code(guard, gate, require_guard);
  }

  if (!quiet) {
    std::printf("%s\n", runtime.describe().c_str());
    std::printf("artifact:   %s (%s)\n", artifact.c_str(),
                function.load_detail().c_str());
    std::printf("load:       runtime %.1f ms, function %.1f ms\n", runtime_ms,
                load_ms);
    std::printf("first call: %.1f us\n", first_call_us);

    cjfc::print_summary("steady state", summary);
    guard.report(stdout, iterations);

    std::printf("\n=== run ===\n");
    std::printf("  faults (%s): minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld\n",
                cjfc::Rusage::scope(), faults.minflt, faults.majflt,
                faults.nvcsw, faults.nivcsw);
    if (audit_values) {
      std::printf("  checks: step_counter_ok=%d finite_outputs=%d\n",
                  step_errors == 0 ? 1 : 0, finite_outputs ? 1 : 0);
    } else {
      std::printf(
          "  checks: step_counter_ok=%d finite_outputs=skipped (--no-check)\n",
          step_errors == 0 ? 1 : 0);
    }
    std::printf(
        "  result: cost=%.6f grad_norm=%.6e iterations_used=%d "
        "backtracks_used=%d\n",
        static_cast<double>(cost), grad_norm,
        static_cast<int>(iterations_used),
        static_cast<int>(backtracks_used));
  }

  if (!samples_path.empty() && !compute.write_samples(samples_path.c_str())) {
    throw std::runtime_error("cannot write samples to " + samples_path);
  }

  if (!json_path.empty()) {
    pjrt::HistogramBin bins[40];
    const std::size_t nbins =
        compute.histogram(bins, sizeof bins / sizeof bins[0]);

    cjfc::json report;
    report["schema"] = 1;
    report["example"] = "02_trajopt";
    report["artifact"] = artifact;
    report["config"] = cjfc::json{
        {"iterations", iterations},
        {"warmup", warmup},
        {"threads", threads},
        {"synchronous", synchronous},
    };
    report["host"] = cjfc::host_json(env);
    report["runtime"] = cjfc::runtime_json(runtime, function, runtime_ms,
                                           load_ms, first_call_us);
    // The shared block nests these; the schema names them at the top of the
    // runtime object, so they appear in both places rather than a reader
    // having to know which example wrote the file.
    report["runtime"]["load_kind"] =
        cjfc::load_kind_name(function.load_kind());
    report["runtime"]["runtime_ms"] = runtime_ms;
    report["runtime"]["load_ms"] = load_ms;
    report["runtime"]["first_call_us"] = first_call_us;
    report["compute_us"] = cjfc::summary_json(summary);
    report["compute_us"]["histogram"] = cjfc::histogram_json(bins, nbins);
    report["rusage"] = cjfc::json{
        {"minflt", faults.minflt},
        {"majflt", faults.majflt},
        {"nvcsw", faults.nvcsw},
        {"nivcsw", faults.nivcsw},
    };
    report["allocations"] = cjfc::alloc_json(guard, iterations);
    report["checks"] = cjfc::json{
        {"step_counter_ok", step_errors == 0},
        {"step_errors", step_errors},
        {"finite_outputs", finite_outputs},
    };
    report["exit_code"] = exit_code;
    cjfc::write_json(json_path, report);
  }

  return exit_code;
}

}  // namespace

/// @brief Run the example, turning any exception into exit code 1.
int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "example_02_trajopt: %s\n", error.what());
    return cjfc::kExitError;
  }
}
