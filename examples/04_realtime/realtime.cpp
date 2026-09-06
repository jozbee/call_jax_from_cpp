/**
 * @file realtime.cpp
 * @brief A periodic control loop: the startup sequence, a loop body that
 *        allocates nothing, and the numbers that decide whether it is fit to
 *        fly.
 *
 * Everything expensive happens before the loop -- the plugin, the client, the
 * executable, the arenas, the hardening, the recorders -- and the loop itself
 * is: sleep until the next deadline, write the inputs, `call()`, feed the
 * outputs back, record.  It runs the artifact `examples/02_trajopt/export.py`
 * writes, so there is no export script here; the flags, the host audit and the
 * two reports are in `support.hpp`.
 *
 * Example 03 is this loop with nothing else in it; everything here that 03
 * does not have is measurement and reporting, and optional.
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

/**
 * @brief Everything the loop touches, resolved before it starts.
 *
 * Passed by reference into `run_cycles` so that the warm-up and the timed
 * window run the *same* code over the *same* state.  Every time is an
 * `std::int64_t` of nanoseconds: arithmetic on deadlines is where a forgotten
 * `timespec` renormalization hides.
 */
struct LoopState {
  pjrt::Function* function = nullptr;
  cjfc::workload::Dims dims;
  double* x_ref = nullptr;  ///< Resolved once; arenas do not move.
  rt::Recorders* rec = nullptr;

  std::int64_t period_ns = 0;
  std::int64_t target_ns = 0;     ///< When this cycle was scheduled to wake.
  std::int64_t prev_wake_ns = 0;  ///< When the previous cycle actually woke.
  bool have_prev = false;

  std::int64_t k = 0;  ///< Cycle counter; continues across warm-up.
  rt::Deadlines counters;
};

/**
 * @brief Run @p count cycles (or until the stop flag when @p forever), timing
 *        them into the recorders when @p record.
 *
 * Allocation-free, lock-free, single-threaded, silent.  The one branch in it is
 * on @p record, which is constant for the whole call; keeping it here rather
 * than duplicating the body is what makes the warm-up provably identical to the
 * measured loop, and gating every statistic on it is what makes the counters,
 * the four distributions and the allocation guard describe the same window.
 *
 * @return Cycles actually completed.
 */
// docs: begin rt-loop
std::size_t run_cycles(LoopState& s, std::size_t count, bool forever,
                       bool record) {
  std::size_t done = 0;
  // cjfc = call_jax_from_cpp helpers
  for (std::size_t i = 0; (forever || i < count) && !cjfc::stopping(); ++i) {
    // A target already in the past returns immediately, which is how the loop
    // catches up after an overrun instead of skipping a cycle.
    s.target_ns += s.period_ns;
    while (cjfc::sleep_until(s.target_ns) == EINTR) {
      if (cjfc::stopping()) {
        return done;
      }
    }
    const std::int64_t wake_ns = cjfc::now_ns();
    if (record) {
      s.rec->wake.record(wake_ns - s.target_ns);
      // Signed on purpose: waking early is as much a scheduling defect as
      // waking late, and clamping would hide half of them.
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
    // 1. Flags first, then the signal handler: a run that is going to fail on
    //    a typo should fail before it installs anything.
    const cjfc::Cli cli = rt::make_cli(argc, argv);
    if (cli.help()) {
      return cjfc::kExitOk;
    }
    const rt::Options options = rt::parse_options(cli);
    cjfc::install_stop_handlers();

    // 2. What the host is willing to give this loop.
    const cjfc::HostEnv env = cjfc::detect_host_env();
    rt::print_host(options, env);
    rt::warn_if_busy(env);

    // 3. The client, once.  Creating it starts XLA's pools, which is why the
    //    corral step below comes after this point.
    rt::Timing timing;
    const auto runtime_start = rt::Clock::now();
    pjrt::RuntimeOptions runtime_options;
    runtime_options.synchronous = true;
    runtime_options.cpu_device_count = 1;
    runtime_options.worker_threads = options.threads;
    pjrt::Runtime runtime(runtime_options);
    timing.runtime_ms = rt::ms_since(runtime_start);
    rt::print_runtime(options, runtime);

    // 4. The executable and its arenas.  warmup_calls is 0 so that the first
    //    call can be timed on its own below.
    const auto load_start = rt::Clock::now();
    pjrt::FunctionOptions function_options;
    function_options.warmup_calls = 0;
    pjrt::Function function(runtime, options.artifact, function_options);
    timing.load_ms = rt::ms_since(load_start);

    const cjfc::workload::Dims dims = cjfc::workload::check_signature(function);
    cjfc::workload::init_inputs(function, dims);

    // docs: begin rt-harden
    // 5. Ask the operating system for everything it will give, in the order
    //    that cannot shoot the process in the foot: the allocator before the
    //    heap grows, priority last so that loading and warm-up do not run at
    //    SCHED_FIFO.  Nothing here is fatal -- an unprivileged run reports what
    //    it did not get and continues, which is the expected result on a
    //    developer's machine.
    cjfc::DmaLatencyHold dma;  // holds the C-state constraint for the run
    int cpu_chosen = -1;
    const std::vector<cjfc::Step> steps =
        cjfc::apply_hardening(env, options.hardening, dma, &cpu_chosen);
    rt::print_steps(options, steps);
    // docs: end rt-harden

    // 6. Everything the loop will touch, allocated here and never again.
    rt::Recorders recorders(options.iterations != 0 ? options.iterations
                                                    : rt::kUnboundedCapacity);
    pjrt::AllocGuard guard;

    LoopState state;
    state.function = &function;
    state.dims = dims;
    state.x_ref = function.input<double>(cjfc::workload::kInXRef);
    state.rec = &recorders;
    state.period_ns = static_cast<std::int64_t>(options.period_us) * 1000;

    // 7. The cold call, alone: it carries the page faults and the lazy
    //    initialization a steady-state number must not contain.  The warm-up
    //    then runs the real loop body, on the real period, recording nothing.
    const auto cold_start = rt::Clock::now();
    function.call();
    timing.first_call_us = rt::us_since(cold_start);
    rt::print_cold_start(options, timing, function.load_detail());

    state.target_ns = cjfc::now_ns();
    state.prev_wake_ns = state.target_ns;
    run_cycles(state, options.warmup, /*forever=*/false, /*record=*/false);

    // docs: begin alloc-guard
    // 8. The measured window, and nothing else inside it: the rusage snapshots
    //    and the allocation census cover exactly the cycles the recorders do.
    //    Arming over the warm-up instead would fold in the faults and lazy
    //    initialization the warm-up exists to pay for.
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
    // 9. Everything from here on is allowed to allocate, take locks and write
    //    files: the loop is over.  Summarizing after the run rather than
    //    narrating during it is the whole reason the recorder exists.
    rt::Results results(recorders, completed, state.counters, rusage,
                        static_cast<double>(options.period_us));

    // Correctness before the allocation gate: a loop that produced the wrong
    // answer without allocating is still broken, and the exit code should say
    // which failure to look at first.
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
