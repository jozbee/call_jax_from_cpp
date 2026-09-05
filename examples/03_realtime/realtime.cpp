/**
 * @file realtime.cpp
 * @brief A periodic control loop: the whole startup sequence, a loop body that
 *        allocates nothing, and the numbers that decide whether it is fit to
 *        fly.
 *
 * This is the shape a control process actually takes.  Everything expensive
 * happens before the loop -- the plugin, the client, the executable, the
 * arenas, the hardening, the recorders -- and the loop itself is: sleep until
 * the next deadline, write the inputs, `call()`, feed the outputs back, record.
 * Nothing in it allocates, locks, logs or flushes.
 *
 * It runs the artifact `examples/02_trajopt/export.py` writes, so there is no
 * export script here; the recirculation rule (this cycle's outputs become the
 * next cycle's inputs) lives in `common/trajopt_signature.hpp` because the
 * benchmark uses the same one.
 *
 * ### What is measured, and why it is four distributions rather than one
 *
 * | | question |
 * |---|---|
 * | compute | how long `call()` took |
 * | cycle | how long the whole cycle took, from wake-up to the end of the work |
 * | wake latency | how late the kernel returned from the sleep |
 * | period jitter | how far this wake-up was from one period after the last one -- **signed**, because waking early is as much a defect as waking late |
 *
 * Confusing the first with the last is the usual mistake.  A perfect `call()`
 * inside a loop that wakes 400 us late every third cycle is not a working
 * control loop, and only the jitter distribution says so.
 *
 * ### Two honest limitations
 *
 * **A missed deadline is late data, not a cancelled call.**  PJRT cannot cancel
 * a running CPU computation; there is no such entry point in the C API, and
 * XLA:CPU has no preemption point to cancel at.  A watchdog can drop the answer
 * once it arrives, but the core stays busy until the computation finishes.
 * Design for a *bounded* computation instead.
 *
 * **The loop never skips a period.**  On an overrun, the next absolute target
 * is already in the past, `clock_nanosleep` returns immediately, and the loop
 * catches up -- so a single overrun shows up as an inflated wake latency on the
 * *following* cycle rather than as a silently dropped cycle.  That is
 * deliberate: for a controller the cycle counter is the plant's clock, and a
 * loop that quietly renumbers its cycles after a miss produces a control signal
 * that no longer lines up with the state it was computed from.
 *
 * Usage:
 *   example_03_realtime [--artifact artifacts/trajopt] [--period-us 10000]
 *                       [--iterations 3000] [--warmup 200] ...
 *
 * `--help` prints the authoritative flag list.
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "common/report.hpp"
#include "common/rt_env.hpp"
#include "common/trajopt_signature.hpp"
#include "pjrt_exec/alloc_guard.hpp"
#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/runtime.hpp"

#if defined(__linux__)
#define CJFC_HAVE_CLOCK_NANOSLEEP 1
#else
#define CJFC_HAVE_CLOCK_NANOSLEEP 0
#endif

namespace {

/// Nanoseconds in a second, spelled once.
constexpr std::int64_t kNsPerSec = 1000 * 1000 * 1000;

/// Sample budget for a run that was told to go until SIGINT.  A million cycles
/// is 2.8 hours at 100 Hz; past that the recorders count drops rather than
/// growing, because growing would allocate inside the window being measured.
constexpr std::size_t kUnboundedCapacity = 1000000;

/**
 * Set by the signal handler, read by the loop.
 *
 * A handler may do exactly one thing here: set this flag.  Everything else --
 * printing, summarizing, writing the report -- happens on the way out of the
 * loop, on the normal thread, where it is allowed to allocate and take locks.
 * `std::atomic<bool>` rather than `volatile sig_atomic_t` because it is
 * lock-free (asserted below, so a platform where it is not fails to compile
 * rather than calling into the allocator from a signal handler).
 */
std::atomic<bool> g_stop{false};

static_assert(std::atomic<bool>::is_always_lock_free,
              "a signal handler may not touch a lock-based atomic");

extern "C" void handle_stop_signal(int) {
  g_stop.store(true, std::memory_order_relaxed);
}

/**
 * Install the handler for SIGINT and SIGTERM.
 *
 * `sa_flags` deliberately omits `SA_RESTART`: the loop wants the sleep to
 * return `EINTR` so it can notice the flag, and an automatically restarted
 * `clock_nanosleep` would hold it in the kernel until the next period.
 * SIGTERM as well as SIGINT, because a containerized control process is
 * stopped with the former and should still print its report.
 */
void install_signal_handlers() {
  struct sigaction action {};
  action.sa_handler = handle_stop_signal;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);
}

/// Whether the loop has been asked to stop.  Relaxed: this is one flag with no
/// other state ordered against it.
bool stopping() { return g_stop.load(std::memory_order_relaxed); }

/// @brief The clock the whole loop is expressed in, read into @p t.
void now(timespec& t) { clock_gettime(CLOCK_MONOTONIC, &t); }

/// @brief `timespec` as nanoseconds since the clock's epoch.
std::int64_t to_ns(const timespec& t) {
  return static_cast<std::int64_t>(t.tv_sec) * kNsPerSec +
         static_cast<std::int64_t>(t.tv_nsec);
}

/// @brief Advance @p t by @p ns, renormalizing `tv_nsec` into `[0, 1e9)`.
void add_ns(timespec& t, std::int64_t ns) {
  std::int64_t nsec = static_cast<std::int64_t>(t.tv_nsec) + ns;
  t.tv_sec += static_cast<time_t>(nsec / kNsPerSec);
  nsec %= kNsPerSec;
  if (nsec < 0) {
    nsec += kNsPerSec;
    t.tv_sec -= 1;
  }
  t.tv_nsec = static_cast<long>(nsec);
}

/**
 * @brief Sleep until the absolute time @p target on `CLOCK_MONOTONIC`.
 *
 * Absolute, not relative: a relative sleep of one period accumulates every
 * cycle's wake-up latency into the phase, so the loop drifts away from its
 * schedule by exactly the quantity it is trying to measure.  With an absolute
 * target, lateness is bounded by the last cycle rather than by the whole run.
 *
 * @return 0, or `EINTR` when a signal arrived first.  `clock_nanosleep`
 *         returns the error number rather than setting `errno`.
 */
int sleep_until(const timespec& target) {
#if CJFC_HAVE_CLOCK_NANOSLEEP
  return clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr);
#else
  // No absolute monotonic sleep here (macOS): compute the remainder and sleep
  // relatively.  Good enough to run the example; not good enough to quote a
  // jitter number from, which is why the report prints the platform.
  timespec current;
  now(current);
  const std::int64_t remaining_ns = to_ns(target) - to_ns(current);
  if (remaining_ns <= 0) {
    return 0;
  }
  timespec relative;
  relative.tv_sec = static_cast<time_t>(remaining_ns / kNsPerSec);
  relative.tv_nsec = static_cast<long>(remaining_ns % kNsPerSec);
  return nanosleep(&relative, nullptr) == 0 ? 0 : errno;
#endif
}

// ---------------------------------------------------------------- the flags

/// Everything the command line can say, resolved once at startup.
struct Options {
  std::string artifact = "artifacts/trajopt";
  std::size_t period_us = 10000;
  std::size_t iterations = 3000;  ///< 0 runs until SIGINT.
  std::size_t warmup = 200;
  std::string cpu = "auto";
  int rt_priority = 80;  ///< 0 disables the SCHED_FIFO request.
  bool malloc_tune = true;
  bool mlock = true;
  bool corral = true;
  bool dma_latency = false;
  int threads = 1;
  std::string json_path;
  std::string samples_path;
  std::string alloc_gate = "self";
  bool require_guard = false;
  bool check = true;
  bool quiet = false;
};

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
    "  --threads N         XLA worker threads; 0 is XLA's default (default: 1)\n"
    "  --json PATH         write the report as JSON as well as printing it\n"
    "  --samples PATH      write every raw sample as CSV (one file per series)\n"
    "  --alloc-gate off|self|all  which allocations fail the run (default: self)\n"
    "  --require-guard     fail when malloc_guard.so was not preloaded\n"
    "  --no-check          do not fail on a step-counter mismatch\n"
    "  --quiet             print one summary line instead of the full report\n"
    "  -h, --help          print this and exit\n"
    "\n"
    "exit: 0 ok, 1 error, 2 wrong answer, 3 the loop allocated,\n"
    "      4 an allocation gate was required but nothing was measured\n";

/// One of @p allowed, or a `std::runtime_error` naming the flag and what it
/// takes.  A mistyped `--alloc-gate slef` that silently disabled the gate would
/// be a green run that checked nothing.
std::string enum_flag(const cjfc::Cli& cli, const char* name,
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
Options parse_options(const cjfc::Cli& cli) {
  Options options;
  options.artifact = cli.get("artifact", options.artifact);
  options.period_us = cli.get_size("period-us", options.period_us);
  options.iterations = cli.get_size("iterations", options.iterations);
  options.warmup = cli.get_size("warmup", options.warmup);
  options.cpu = cli.get("cpu", options.cpu);
  options.rt_priority =
      static_cast<int>(cli.get_long("rt-priority", options.rt_priority));
  options.malloc_tune = !cli.flag("no-malloc-tune");
  options.mlock = !cli.flag("no-mlock");
  options.corral = !cli.flag("no-corral");
  options.dma_latency =
      enum_flag(cli, "dma-latency", "off", {"auto", "off"}) == "auto";
  options.threads = static_cast<int>(cli.get_long("threads", options.threads));
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
  if (options.rt_priority < 0 || options.rt_priority > 99) {
    throw std::runtime_error("'--rt-priority' must be between 0 and 99, got " +
                             std::to_string(options.rt_priority));
  }
  if (options.threads < 0) {
    throw std::runtime_error("'--threads' must be 0 or more");
  }
  return options;
}

// --------------------------------------------------------------- the report

/// @brief A CPU set the way the kernel spells one: `2-5,8`, or `(none)`.
std::string cpulist(const std::vector<int>& cpus) {
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
std::string rlimit_text(long value) {
  return value < 0 ? std::string("unlimited") : std::to_string(value);
}

/// @brief Print the host audit: the settings that decide whether any number
///        below it is worth reading.
void print_host(const cjfc::HostEnv& env) {
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
  std::printf("  limits        rtprio %s, memlock %s, sched_rt_runtime_us %ld\n",
              rlimit_text(env.rlimit_rtprio).c_str(),
              rlimit_text(env.rlimit_memlock).c_str(), env.rt_runtime_us);
  std::printf("  c-states      /dev/cpu_dma_latency %s\n",
              env.cpu_dma_latency_writable ? "writable" : "not writable");
  std::printf("  loadavg       %.2f %.2f %.2f\n", env.loadavg1, env.loadavg5,
              env.loadavg15);
}

// ------------------------------------------------------------- the loop

/**
 * @brief Everything the loop touches, reserved before it starts.
 *
 * Passed by reference into `run_cycles` so that the warm-up and the timed
 * window run the *same* code over the *same* state: a warm-up that differs
 * from the loop it is warming up for is warming up something else.
 */
struct LoopState {
  pjrt::Function* function = nullptr;
  cjfc::Dims dims;
  double* x_ref = nullptr;  ///< Resolved once; arenas do not move.

  std::int64_t period_ns = 0;

  pjrt::LatencyRecorder* compute = nullptr;
  pjrt::LatencyRecorder* cycle = nullptr;
  pjrt::LatencyRecorder* wake = nullptr;
  pjrt::LatencyRecorder* jitter = nullptr;

  timespec target{};     ///< The absolute time this cycle was scheduled for.
  timespec prev_wake{};  ///< The previous cycle's actual wake-up.
  bool have_prev = false;

  std::int64_t k = 0;  ///< Cycle counter; continues across warm-up.

  std::size_t missed = 0;
  std::int64_t worst_overrun_ns = 0;
  std::size_t step_errors = 0;
};

/**
 * @brief Run @p count cycles (or until the stop flag when @p forever), timing
 *        them into the recorders when @p record.
 *
 * Allocation-free, lock-free, single-threaded, silent.  The one branch in it is
 * on @p record, which is constant for the whole call and therefore free after
 * the first iteration; keeping it here rather than duplicating the body is what
 * makes the warm-up provably identical to the measured loop.
 *
 * Every statistic is gated on @p record, so the deadline counters, the
 * step-counter errors and the four distributions all describe exactly the same
 * window -- the same one the allocation guard is armed over.
 *
 * @return Cycles actually completed.
 */
std::size_t run_cycles(LoopState& s, std::size_t count, bool forever,
                       bool record) {
  std::size_t done = 0;
  for (std::size_t i = 0; forever || i < count; ++i) {
    if (stopping()) {
      break;
    }

    add_ns(s.target, s.period_ns);

    // A target already in the past returns immediately, which is how the loop
    // catches up after an overrun instead of skipping a cycle.
    // Any return other than EINTR leaves the cycle running on a late
    // schedule rather than aborting it; the wake-latency distribution is
    // where that shows up.
    while (sleep_until(s.target) == EINTR) {
      if (stopping()) {
        return done;
      }
    }

    timespec woke;
    now(woke);
    const std::int64_t wake_ns = to_ns(woke);

    if (record) {
      s.wake->record(wake_ns - to_ns(s.target));
      if (s.have_prev) {
        // Signed on purpose: a cycle that wakes early is as much a scheduling
        // defect as one that wakes late, and clamping would hide half of them.
        s.jitter->record(wake_ns - to_ns(s.prev_wake) - s.period_ns);
      }
    }
    s.prev_wake = woke;
    s.have_prev = true;

    cjfc::write_reference(s.x_ref, s.dims, s.k);

    timespec call_start;
    now(call_start);
    s.function->call();
    timespec call_end;
    now(call_end);

    const bool step_ok = cjfc::feedback(*s.function, s.dims, s.k);

    timespec cycle_end;
    now(cycle_end);
    const std::int64_t end_ns = to_ns(cycle_end);

    if (record) {
      s.compute->record(to_ns(call_end) - to_ns(call_start));
      s.cycle->record(end_ns - wake_ns);
      if (!step_ok) {
        ++s.step_errors;
      }
      // The deadline is the *next* wake-up, one period after this one.
      const std::int64_t overrun_ns = end_ns - to_ns(s.target) - s.period_ns;
      if (overrun_ns > 0) {
        ++s.missed;
        if (overrun_ns > s.worst_overrun_ns) {
          s.worst_overrun_ns = overrun_ns;
        }
      }
    }

    ++s.k;
    ++done;
  }
  return done;
}

/// @brief `raw.csv` + `cycle` -> `raw_cycle.csv`; a path with no extension just
///        gets the suffix appended.
std::string with_suffix(const std::string& path, const char* suffix) {
  const std::size_t dot = path.find_last_of('.');
  const std::size_t slash = path.find_last_of('/');
  if (dot == std::string::npos ||
      (slash != std::string::npos && dot < slash)) {
    return path + "_" + suffix;
  }
  return path.substr(0, dot) + "_" + suffix + path.substr(dot);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // 1. Flags first, then the signal handler: a run that is going to fail on
    //    a typo should fail before it installs anything.
    const cjfc::Cli cli(argc, argv,
                        {"artifact", "period-us", "iterations", "warmup", "cpu",
                         "rt-priority", "no-malloc-tune", "no-mlock",
                         "no-corral", "dma-latency", "threads", "json",
                         "samples", "alloc-gate", "require-guard", "no-check",
                         "quiet"},
                        kUsage);
    if (cli.help()) {
      return cjfc::kExitOk;
    }
    const Options options = parse_options(cli);
    install_signal_handlers();

    // 2. What the host is willing to give this loop.
    const cjfc::HostEnv env = cjfc::detect_host_env();
    if (!options.quiet) {
      print_host(env);
    }
    if (env.busy) {
      // stderr, so it survives `--json` piping and `--quiet` both: a tail
      // number from a busy machine is not noisy, it is wrong.
      std::fprintf(stderr,
                   "WARNING: loadavg1=%.2f > 1.0 -- numbers from a busy "
                   "machine are wrong, not noisy\n",
                   env.loadavg1);
    }

    const auto clock_now = std::chrono::steady_clock::now;
    const auto ms_since = [](std::chrono::steady_clock::time_point t0) {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
          .count();
    };

    // 3. The client, once.  Creating it starts XLA's pools, which is why the
    //    corral step below has to come after this point.
    const auto runtime_start = clock_now();
    pjrt::RuntimeOptions runtime_options;
    runtime_options.synchronous = true;
    runtime_options.cpu_device_count = 1;
    runtime_options.worker_threads = options.threads;
    pjrt::Runtime runtime(runtime_options);
    const double runtime_ms = ms_since(runtime_start);
    if (!options.quiet) {
      std::printf("\n=== runtime ===\n  %s\n", runtime.describe().c_str());
    }

    // 4. The executable and its arenas.  warmup_calls is 0 so that the first
    //    call can be timed on its own below; the Function must not quietly
    //    spend it first.
    const auto load_start = clock_now();
    pjrt::FunctionOptions function_options;
    function_options.warmup_calls = 0;
    pjrt::Function function(runtime, options.artifact, function_options);
    const double load_ms = ms_since(load_start);

    const cjfc::Dims dims = cjfc::check_signature(function);
    cjfc::init_inputs(function, dims);

    // docs: begin rt-harden
    // 5. Ask the operating system for everything it will give, in the order
    //    that cannot shoot the process in the foot: the allocator before the
    //    heap grows, priority last so that loading and warm-up do not run at
    //    SCHED_FIFO.  Nothing here is fatal -- an unprivileged run reports what
    //    it did not get and continues, which is the expected result on a
    //    developer's machine.
    cjfc::HardeningOptions hardening;
    hardening.malloc_tune = options.malloc_tune;
    hardening.mlock = options.mlock;
    hardening.corral = options.corral;
    hardening.cpu = options.cpu;
    hardening.rt_priority = options.rt_priority;
    hardening.dma_latency = options.dma_latency;

    cjfc::DmaLatencyHold dma;  // holds the C-state constraint for the run
    int cpu_chosen = -1;
    const std::vector<cjfc::Step> steps =
        cjfc::apply_hardening(env, hardening, dma, &cpu_chosen);
    if (!options.quiet) {
      std::printf("\n=== hardening ===\n");
      for (const cjfc::Step& step : steps) {
        cjfc::print_step(step);
      }
    }
    // docs: end rt-harden

    // 6. Everything the loop will touch, allocated here and never again.
    const std::size_t capacity =
        options.iterations != 0 ? options.iterations : kUnboundedCapacity;
    pjrt::LatencyRecorder compute(capacity);
    pjrt::LatencyRecorder cycle(capacity);
    pjrt::LatencyRecorder wake(capacity);
    pjrt::LatencyRecorder jitter(capacity);
    pjrt::AllocGuard guard;

    LoopState state;
    state.function = &function;
    state.dims = dims;
    state.x_ref = function.input<double>(cjfc::kInXRef);
    state.period_ns = static_cast<std::int64_t>(options.period_us) * 1000;
    state.compute = &compute;
    state.cycle = &cycle;
    state.wake = &wake;
    state.jitter = &jitter;

    // 7. The cold call, alone: it carries the page faults and the lazy
    //    initialization that a steady-state number must not contain.  Then the
    //    warm-up runs the real loop body, on the real period, recording
    //    nothing.
    const auto cold_start = clock_now();
    function.call();
    const double first_call_us =
        std::chrono::duration<double, std::micro>(clock_now() - cold_start)
            .count();
    if (!options.quiet) {
      std::printf("\n=== cold start ===\n");
      std::printf("  runtime     %10.1f ms   (plugin, client, XLA pools)\n",
                  runtime_ms);
      std::printf("  load        %10.1f ms   (%s)\n", load_ms,
                  function.load_detail().c_str());
      std::printf("  first call  %10.1f us   (excluded from every number "
                  "below)\n",
                  first_call_us);
    }

    now(state.target);
    state.prev_wake = state.target;
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
    const pjrt::LatencySummary compute_summary = compute.summary();
    const pjrt::LatencySummary cycle_summary = cycle.summary();
    const pjrt::LatencySummary wake_summary = wake.summary();
    const pjrt::LatencySummary jitter_summary = jitter.summary();

    const double period_us = static_cast<double>(options.period_us);
    const double miss_rate =
        completed > 0 ? static_cast<double>(state.missed) /
                            static_cast<double>(completed)
                      : 0.0;
    const double worst_overrun_us =
        static_cast<double>(state.worst_overrun_ns) * 1e-3;
    const double utilization_p50 = cycle_summary.p50_us / period_us;
    const double utilization_max = cycle_summary.max_us / period_us;
    const bool step_counter_ok = state.step_errors == 0;

    // Correctness before the allocation gate: a loop that produced the wrong
    // answer without allocating is still broken, and the exit code should say
    // which failure to look at first.
    int exit_code = cjfc::kExitOk;
    if (options.check && !step_counter_ok) {
      exit_code = cjfc::kExitCorrectness;
    } else {
      exit_code = cjfc::alloc_gate_exit_code(guard, options.alloc_gate,
                                             options.require_guard);
    }

    if (!options.quiet) {
      cjfc::print_summary("call latency (compute)", compute_summary);
      cjfc::print_summary("cycle time (wake to done)", cycle_summary);
      cjfc::print_summary("wake-up latency (late by)", wake_summary);
      cjfc::print_summary("period jitter (signed)", jitter_summary,
                          /*signed_samples=*/true);

      std::printf("\n=== periodic behaviour ===\n");
      std::printf(
          "  deadlines: %zu missed of %zu (%.3f%%), worst overrun %.1f us, "
          "utilization p50 %.3f max %.3f\n",
          state.missed, completed, miss_rate * 100.0, worst_overrun_us,
          utilization_p50, utilization_max);
      std::printf(
          "  rusage:    minflt %ld, majflt %ld, nvcsw %ld, nivcsw %ld (%s "
          "scope)\n",
          rusage.minflt, rusage.majflt, rusage.nvcsw, rusage.nivcsw,
          cjfc::Rusage::scope());
      guard.report(stdout, completed);
      std::printf("\n=== checks ===\n");
      std::printf("  step counter %s (%zu mismatches of %zu%s)\n",
                  step_counter_ok ? "ok" : "FAILED", state.step_errors,
                  completed, options.check ? "" : ", not gated: --no-check");
      std::printf("  exit code    %d\n", exit_code);
    }

    if (!options.json_path.empty()) {
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
            {"cpu", options.cpu},
            {"cpu_chosen", cpu_chosen},
            {"rt_priority", options.rt_priority},
            {"threads", options.threads},
            {"synchronous", runtime_options.synchronous}}},
          {"host", cjfc::host_json(env)},
          {"hardening", hardening_json},
          {"runtime", cjfc::runtime_json(runtime, function, runtime_ms, load_ms,
                                         first_call_us)},
          {"compute_us", cjfc::summary_json(compute_summary)},
          {"cycle_us", cjfc::summary_json(cycle_summary)},
          {"wake_latency_us", cjfc::summary_json(wake_summary)},
          {"period_jitter_us", cjfc::summary_json(jitter_summary)},
          {"deadlines",
           {{"period_us", period_us},
            {"missed", state.missed},
            {"miss_rate", miss_rate},
            {"worst_overrun_us", worst_overrun_us},
            {"utilization_p50", utilization_p50},
            {"utilization_max", utilization_max}}},
          {"rusage",
           {{"minflt", rusage.minflt},
            {"majflt", rusage.majflt},
            {"nvcsw", rusage.nvcsw},
            {"nivcsw", rusage.nivcsw}}},
          {"allocations", cjfc::alloc_json(guard, completed)},
          {"checks",
           {{"step_counter_ok", step_counter_ok},
            {"step_errors", state.step_errors}}},
          {"exit_code", exit_code},
      };
      cjfc::write_json(options.json_path, report);
    }
    // docs: end rt-report

    if (!options.samples_path.empty()) {
      // Four series, four files: a summary cannot say *when* an outlier
      // happened, and a spike on cycle 3 and a spike on cycle 30,000 have the
      // same p99.9 and completely different causes.
      compute.write_samples(options.samples_path.c_str());
      cycle.write_samples(with_suffix(options.samples_path, "cycle").c_str());
      wake.write_samples(with_suffix(options.samples_path, "wake").c_str());
      jitter.write_samples(with_suffix(options.samples_path, "jitter").c_str());
    }

    if (options.quiet) {
      std::printf(
          "03_realtime n=%zu period=%zuus compute p50=%.1f p99.9=%.1f "
          "max=%.1f jitter p99.9=%.1f missed=%zu self_allocs=%lu exit=%d\n",
          completed, options.period_us, compute_summary.p50_us,
          compute_summary.p999_us, compute_summary.max_us,
          jitter_summary.p999_us, state.missed, guard.allocs_self(), exit_code);
    }
    return exit_code;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return cjfc::kExitError;
  }
}
