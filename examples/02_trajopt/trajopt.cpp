/**
 * @file trajopt.cpp
 * @brief Time the trajectory-optimisation artifact, and check that the fast
 *        answer is also the right one.
 *
 * This is the example to quote numbers from.  It loads the artifact
 * `examples/02_trajopt/export.py` writes, drives it the way a receding-horizon
 * controller would -- reference in, controls out, outputs fed back into the
 * next call's inputs -- and reports the tail of the call latency rather than
 * its average.  The flags, the finite-output audit and the reports are in
 * `support.hpp`.
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
 *     over warm-up would fold in the initialization warm-up exists to pay for.
 *   - **Nothing in the timed loop allocates, locks, logs or flushes.**
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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

#include "common/cli.hpp"
#include "common/rt_env.hpp"
#include "common/workload.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/runtime.hpp"
#include "support.hpp"

namespace {

int run(int argc, char** argv) {
  const cjfc::Cli cli = trajopt::make_cli(argc, argv);
  if (cli.help()) {
    return cjfc::kExitOk;
  }
  const trajopt::Options options = trajopt::parse_options(cli);

  // Read before anything is measured: a latency number from a busy machine is
  // not noisy, it is wrong.
  const cjfc::HostEnv env = cjfc::detect_host_env();
  trajopt::warn_if_busy(env);

  pjrt::RuntimeOptions runtime_options;
  runtime_options.synchronous = options.synchronous;
  runtime_options.worker_threads = static_cast<int>(options.threads);

  trajopt::Outcome outcome;
  const auto t_start = trajopt::Clock::now();
  pjrt::Runtime runtime(runtime_options);
  const auto t_runtime = trajopt::Clock::now();

  // The harness warms up, so that the first call below is genuinely the first
  // one and can be reported as such.
  pjrt::FunctionOptions function_options;
  function_options.warmup_calls = 0;
  pjrt::Function function(runtime, options.artifact, function_options);
  const auto t_loaded = trajopt::Clock::now();
  outcome.runtime_ms = trajopt::millis(t_start, t_runtime);
  outcome.load_ms = trajopt::millis(t_runtime, t_loaded);

  const cjfc::workload::Dims dims = cjfc::workload::check_signature(function);
  cjfc::workload::init_inputs(function, dims);

  // Resolved once.  Everything the loop touches is a pointer or an integer by
  // the time the stopwatch starts.
  double* const x_ref = function.input<double>(cjfc::workload::kInXRef);
  const std::vector<trajopt::FloatArena> audited =
      options.audit_values ? trajopt::float_outputs(function)
                           : std::vector<trajopt::FloatArena>();
  const trajopt::FaultInjector injector(options.inject_fault, /*cycle=*/1);

  std::int64_t cycle = 0;

  // Recirculate one cycle's outputs into the next cycle's inputs, and check
  // what came back.  Run for every call, warm-up included: a step counter that
  // goes wrong during warm-up is exactly as broken as one that goes wrong
  // later.  The step check is one integer comparison and always runs;
  // --no-check drops only the value audit, which is the part that walks every
  // element of every float arena.
  const auto finish_cycle = [&](std::int64_t k) {
    if (!cjfc::workload::feedback(function, dims, k)) {
      ++outcome.step_errors;
    }
    injector.apply(function, audited, k);
    if (options.audit_values && !trajopt::all_finite(audited)) {
      outcome.finite_outputs = false;
    }
  };

  // docs: begin trajopt-run
  // The cold call, timed alone, then the warm-up: the same loop body, on the
  // same arenas, recording nothing.
  // cjfc = call_jax_from_cpp helpers
  cjfc::workload::write_reference(x_ref, dims, cycle);
  const auto t_call = trajopt::Clock::now();
  function.call();
  outcome.first_call_us = trajopt::micros(t_call, trajopt::Clock::now());
  finish_cycle(cycle);
  ++cycle;

  for (std::size_t i = 0; i < options.warmup; ++i) {
    cjfc::workload::write_reference(x_ref, dims, cycle);
    function.call();
    finish_cycle(cycle);
    ++cycle;
  }

  pjrt::AllocGuard guard;

  // docs: begin latency-recorder
  // Capacity is reserved once, here: the recorder drops rather than grows,
  // because growing would allocate in the middle of the run being measured.
  pjrt::LatencyRecorder compute(options.iterations);

  const cjfc::Rusage before = cjfc::Rusage::now();
  {
    pjrt::AllocGuardScope armed(guard);
    for (std::size_t i = 0; i < options.iterations; ++i) {
      // Fresh reference for this cycle, written straight into the input arena
      // XLA will read.  Between calls, never during one.
      cjfc::workload::write_reference(x_ref, dims, cycle);
      {
        pjrt::ScopedLatency sample(compute);
        function.call();
      }
      finish_cycle(cycle);
      ++cycle;
    }
  }
  const cjfc::Rusage after = cjfc::Rusage::now();

  outcome.compute = compute.summary();
  // docs: end latency-recorder
  // docs: end trajopt-run

  outcome.faults = after - before;
  trajopt::read_solution(function, outcome);

  // A wrong answer outranks an allocation gate: the gate describes how the
  // answer was produced, and there is no point grading that first.
  outcome.exit_code =
      outcome.step_errors != 0 || !outcome.finite_outputs
          ? cjfc::kExitCorrectness
          : cjfc::alloc_gate_exit_code(guard, options.alloc_gate,
                                       options.require_guard);

  trajopt::print_report(options, runtime, function, outcome, guard);
  if (!options.samples_path.empty()) {
    trajopt::write_samples(options, compute);
  }
  if (!options.json_path.empty()) {
    trajopt::write_report(options, env, runtime, function, compute, outcome,
                          guard);
  }
  return outcome.exit_code;
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
