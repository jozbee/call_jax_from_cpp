/**
 * @file realtime.cpp
 * @brief A periodic control loop: the startup sequence, a loop body that
 *        allocates nothing, and the numbers that decide whether it is fit to
 *        fly.
 *
 * Everything expensive happens before the loop; the loop itself is sleep until
 * the next deadline, write the inputs, `call()`, feed the outputs back, record.
 * It runs the artifact `examples/02_trajopt/export.py` writes.  Example 03 is
 * the same loop with nothing else in it; the flags, the host audit and the
 * reports are in `support.hpp`.
 */
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

#include "common/cli.hpp"
#include "common/periodic.hpp"
#include "common/rt_env.hpp"
#include "common/workload.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/runtime.hpp"
#include "support.hpp"

namespace {

/// Everything the loop touches, resolved before it starts.  Times are
/// `std::int64_t` nanoseconds: deadline arithmetic is where a forgotten
/// `timespec` renormalization hides.
struct LoopState {
  pjrt::Function* function = nullptr;
  cjfc::workload::Dims dims;
  double* x_ref = nullptr;
  rt::Recorders* rec = nullptr;

  std::int64_t period_ns = 0;
  std::int64_t target_ns = 0;     ///< When this cycle was scheduled to wake.
  std::int64_t prev_wake_ns = 0;  ///< When the previous cycle actually woke.
  bool have_prev = false;

  std::int64_t k = 0;  ///< Cycle counter; continues across warm-up.
  rt::Deadlines counters;
};

/// Run @p count cycles (or until the stop flag when @p forever), recording
/// when @p record.  One body for warm-up and the measured window, so the two
/// are identical; every statistic is gated on @p record, so the counters, the
/// recorders and the guard describe the same cycles.  Returns cycles completed.
// docs: begin rt-loop
std::size_t run_cycles(LoopState& s, std::size_t count, bool forever,
                       bool record) {
  std::size_t done = 0;
  // cjfc = call_jax_from_cpp helpers
  for (std::size_t i = 0; (forever || i < count) && !cjfc::stopping(); ++i) {
    // A target already in the past returns at once: the loop catches up.
    s.target_ns += s.period_ns;
    while (cjfc::sleep_until(s.target_ns) == EINTR) {
      if (cjfc::stopping()) {
        return done;
      }
    }
    const std::int64_t wake_ns = cjfc::now_ns();
    if (record) {
      s.rec->wake.record(wake_ns - s.target_ns);
      // Signed: waking early is as much a defect as waking late.
      if (s.have_prev) {
        s.rec->jitter.record(wake_ns - s.prev_wake_ns - s.period_ns);
      }
    }
    s.prev_wake_ns = wake_ns;
    s.have_prev = true;

    cjfc::workload::write_reference(s.x_ref, s.dims, s.k);
    const std::int64_t call_start_ns = cjfc::now_ns();
    s.function->call();
    const std::int64_t call_end_ns = cjfc::now_ns();
    const bool step_ok = cjfc::workload::feedback(*s.function, s.dims, s.k);
    const std::int64_t end_ns = cjfc::now_ns();

    if (record) {
      s.rec->compute.record(call_end_ns - call_start_ns);
      s.rec->cycle.record(end_ns - wake_ns);
      s.counters.step_errors += step_ok ? 0 : 1;
      // The deadline is the *next* wake-up, one period after this one.
      s.counters.observe(end_ns - s.target_ns - s.period_ns);
    }
    ++s.k;
    ++done;
  }
  return done;
}
// docs: end rt-loop

}  // namespace

int main(int argc, char** argv) {
  try {
    // Flags first: a typo should fail before the signal handler is installed.
    const cjfc::Cli cli = rt::make_cli(argc, argv);
    if (cli.help()) {
      return cjfc::kExitOk;
    }
    const rt::Options options = rt::parse_options(cli);
    cjfc::install_stop_handlers();

    const cjfc::HostEnv env = cjfc::detect_host_env();
    rt::print_host(options, env);
    rt::warn_if_busy(env);

    // Creating the Runtime starts XLA's pools; the corral must come after.
    rt::Timing timing;
    const auto runtime_start = rt::Clock::now();
    pjrt::RuntimeOptions runtime_options;
    runtime_options.synchronous = true;
    runtime_options.cpu_device_count = 1;
    runtime_options.worker_threads = options.threads;
    pjrt::Runtime runtime(runtime_options);
    timing.runtime_ms = rt::ms_since(runtime_start);
    rt::print_runtime(options, runtime);

    // warmup_calls = 0, so the cold call below can be timed on its own.
    const auto load_start = rt::Clock::now();
    pjrt::FunctionOptions function_options;
    function_options.warmup_calls = 0;
    pjrt::Function function(runtime, options.artifact, function_options);
    timing.load_ms = rt::ms_since(load_start);

    const cjfc::workload::Dims dims = cjfc::workload::check_signature(function);
    cjfc::workload::init_inputs(function, dims);

    // docs: begin rt-harden
    // In the one safe order: allocator first, priority last.  Nothing here is
    // fatal: an unprivileged run reports what it did not get and continues.
    cjfc::DmaLatencyHold dma;  // holds the C-state constraint for the run
    int cpu_chosen = -1;
    const std::vector<cjfc::Step> steps =
        cjfc::apply_hardening(env, options.hardening, dma, &cpu_chosen);
    rt::print_steps(options, steps);
    // docs: end rt-harden

    // Everything the loop touches, allocated here and never again.
    rt::Recorders recorders(options.iterations != 0 ? options.iterations
                                                    : rt::kUnboundedCapacity);
    pjrt::AllocGuard guard;

    LoopState state;
    state.function = &function;
    state.dims = dims;
    state.x_ref = function.input<double>(cjfc::workload::kInXRef);
    state.rec = &recorders;
    state.period_ns = static_cast<std::int64_t>(options.period_us) * 1000;

    // The cold call alone: it carries the faults and lazy initialization a
    // steady-state number must not contain.  Then warm-up, recording nothing.
    const auto cold_start = rt::Clock::now();
    function.call();
    timing.first_call_us = rt::us_since(cold_start);
    rt::print_cold_start(options, timing, function.load_detail());

    state.target_ns = cjfc::now_ns();
    run_cycles(state, options.warmup, /*forever=*/false, /*record=*/false);

    // docs: begin alloc-guard
    // The measured window: rusage and the census cover exactly the cycles the
    // recorders do.  Arming over warm-up would fold in what warm-up pays for.
    const cjfc::Rusage rusage_before = cjfc::Rusage::now();
    guard.arm();
    const std::size_t completed =
        run_cycles(state, options.iterations,
                   /*forever=*/options.iterations == 0, /*record=*/true);
    guard.disarm();
    const cjfc::Rusage rusage_after = cjfc::Rusage::now();
    const cjfc::Rusage rusage = rusage_after - rusage_before;
    // docs: end alloc-guard

    // docs: begin rt-report
    // The loop is over: from here on, allocating and writing files is allowed.
    rt::Results results(recorders, completed, state.counters, rusage,
                        static_cast<double>(options.period_us));

    // A wrong answer outranks the allocation gate.
    results.exit_code =
        options.check && !results.step_counter_ok()
            ? cjfc::kExitCorrectness
            : cjfc::alloc_gate_exit_code(guard, options.alloc_gate,
                                         options.require_guard);
    // docs: end rt-report

    if (options.quiet) {
      rt::print_quiet_line(options, results, guard);
    } else {
      rt::print_report(options, results, guard);
    }
    if (!options.json_path.empty()) {
      rt::write_report(options, env, steps, runtime, function, timing, results,
                       guard, cpu_chosen);
    }

    if (!options.samples_path.empty()) {
      rt::write_samples(options.samples_path, recorders);
    }
    return results.exit_code;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return cjfc::kExitError;
  }
}
