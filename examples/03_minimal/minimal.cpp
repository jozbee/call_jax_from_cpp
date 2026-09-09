/**
 * @file minimal.cpp
 * @brief The smallest real-time loop worth running: hardening, a period, a
 *        call, two recorders.
 *
 * Self-contained: the library and nothing else, so the file can be copied
 * without a helper layer.  Example 04 is this loop with the measurement
 * attached.  It runs the artifact `examples/01_basic/export.py` writes, whose
 * residual output is what makes a wrong answer visible here.
 *
 *     example_03_minimal [artifact] [period_us] [cycles] [cpu]
 */
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <thread>
#include <vector>

#include "pjrt_exec/latency.hpp"
#include "pjrt_exec/rt.hpp"
#include "pjrt_exec/runtime.hpp"

namespace {

constexpr std::int64_t kNsPerSec = 1000 * 1000 * 1000;

/// `CLOCK_MONOTONIC` as nanoseconds: NTP cannot move it mid-run.
std::int64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<std::int64_t>(t.tv_sec) * kNsPerSec +
         static_cast<std::int64_t>(t.tv_nsec);
}

/// Sleep until the absolute time @p target_ns; returns 0 or `EINTR`.  Absolute,
/// because a relative sleep adds each wake-up's lateness to the next period; a
/// target already in the past returns at once, so the loop catches up.
int sleep_until(std::int64_t target_ns) {
  timespec target;
  target.tv_sec = static_cast<time_t>(target_ns / kNsPerSec);
  target.tv_nsec = static_cast<long>(target_ns % kNsPerSec);
#ifdef __linux__
  return clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr);
#else
  // No absolute monotonic sleep here (macOS): fine to run, not to quote from.
  const std::int64_t remaining_ns = target_ns - now_ns();
  if (remaining_ns <= 0) {
    return 0;
  }
  timespec relative;
  relative.tv_sec = static_cast<time_t>(remaining_ns / kNsPerSec);
  relative.tv_nsec = static_cast<long>(remaining_ns % kNsPerSec);
  return nanosleep(&relative, nullptr) == 0 ? 0 : errno;
#endif
}

/// Every online CPU except @p cpu: where XLA's pools are sent.
std::vector<int> cpus_except(int cpu) {
  std::vector<int> cpus;
  const int online = static_cast<int>(std::thread::hardware_concurrency());
  for (int i = 0; i < online; ++i) {
    if (i != cpu) {
      cpus.push_back(i);
    }
  }
  return cpus;
}

/// One line per hardening step: what was asked for, and what came of it.
void print(const char* name, const pjrt::rt::Status& status) {
  std::printf("  [%s] %s: %s\n", status.ok ? "ok  " : "skip", name,
              status.detail.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const char* artifact = argc > 1 ? argv[1] : "artifacts/basic";
  const std::int64_t period_ns = (argc > 2 ? std::atoll(argv[2]) : 1000) * 1000;
  const std::size_t cycles =
      argc > 3 ? static_cast<std::size_t>(std::atoll(argv[3])) : 2000;
  const int cpu = argc > 4 ? std::atoi(argv[4]) : -1;

  try {
    // docs: begin minimal-setup
    // Before the Runtime, so the heap startup grows is the hardened one.
    print("harden_malloc", pjrt::rt::harden_malloc());

    pjrt::RuntimeOptions runtime_options;
    runtime_options.synchronous = true;  // call() computes on this thread
    runtime_options.cpu_device_count = 1;
    runtime_options.worker_threads = 1;
    pjrt::Runtime runtime(runtime_options);
    pjrt::Function function(runtime, artifact);

    // After the load, so what it prefaults is the memory the loop will touch.
    print("lock_memory", pjrt::rt::lock_memory());
    if (cpu >= 0) {
      print("pin_current_thread", pjrt::rt::pin_current_thread(cpu));
      // XLA's pools do not exist until the Runtime does, so not earlier.
      print("corral_xla_threads",
            pjrt::rt::corral_xla_threads(cpus_except(cpu)));
    }
    // Not held here: /dev/cpu_dma_latency needs root; see cjfc::DmaLatencyHold.
    // Priority last: loading and warm-up must not run at SCHED_FIFO.
    print("set_realtime_priority", pjrt::rt::set_realtime_priority(80));
    // docs: end minimal-setup

    // docs: begin minimal-loop
    pjrt::LatencyRecorder wake(cycles), call(cycles);
    double* A = function.input<double>(0);  // the arenas XLA reads,
    double* b = function.input<double>(1);
    const double* x = function.output<double>(0);  // and the ones it writes
    const double* r = function.output<double>(1);

    const std::size_t n = function.input_numel(1);
    for (std::size_t i = 0; i < n; ++i) {  // set once: A does not change
      for (std::size_t j = 0; j < n; ++j) {
        A[i * n + j] = i == j ? 4.0 : (j == i + 1 ? 0.5 : 0.0);
      }
    }

    double max_residual = 0.0;
    std::int64_t target_ns = now_ns();
    for (std::size_t k = 0; k < cycles; ++k) {
      target_ns += period_ns;  // the deadline, never "now plus a period"
      while (sleep_until(target_ns) == EINTR) {
      }
      wake.record(now_ns() - target_ns);
      for (std::size_t i = 0; i < n; ++i) {  // between calls, never during one
        b[i] = std::sin(0.01 * static_cast<double>(k) + static_cast<double>(i));
      }
      const std::int64_t call_start_ns = now_ns();
      function.call();
      call.record(now_ns() - call_start_ns);
      max_residual = *r > max_residual ? *r : max_residual;
    }
    // docs: end minimal-loop

    wake.report(stdout, "wake-up latency");
    call.report(stdout, "call latency");
    std::printf("\n%s\n", pjrt::rt::describe_environment().c_str());
    std::printf("max_residual=%g x[0]=%g\n", max_residual, x[0]);

    // Exit 2 is a wrong answer, as in examples/common/report.hpp.
    return max_residual > 1e-9 ? 2 : 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "example_03_minimal: %s\n", error.what());
    return 1;
  }
}
