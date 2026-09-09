/**
 * @file rt.hpp
 * @brief Optional real-time hardening for the thread that runs the control
 *        loop.
 *
 * Nothing here changes what is computed, only how reliably the operating
 * system lets the computation finish on time.  Each function is independent,
 * reports whether it took effect, and is a no-op returning `false` on a
 * platform that does not provide it.
 *
 * A control process calls these once, around loading its `Function`:
 *
 * @code
 *   pjrt::rt::harden_malloc();            // before the Runtime exists
 *   pjrt::Runtime runtime;
 *   pjrt::Function f(runtime, "artifacts/trajopt");
 *   pjrt::rt::lock_memory();
 *   pjrt::rt::pin_current_thread(2);
 *   pjrt::rt::corral_xla_threads({3});    // after: keep XLA's pools off cpu 2
 *   pjrt::rt::set_realtime_priority(80);  // last: setup never runs real-time
 * @endcode
 *
 * The host has to cooperate as well -- isolated cores, a performance
 * governor, deep C-states disabled.  `tools/rt_check.sh` audits that side.
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pjrt::rt {

/// Whether a hardening step took effect, and why not when it did not.
struct Status {
  bool ok = false;
  std::string detail;

  explicit operator bool() const { return ok; }
};

/**
 * @brief Keep the process resident: no page of it may be swapped out.
 *
 * Also grows and touches the heap and stack once, so that the first calls do
 * not pay for faulting in pages the allocator lazily reserved.
 */
Status lock_memory(std::size_t prefault_bytes = 64 * 1024 * 1024);

/**
 * @brief Stop the allocator from returning memory to the kernel.
 *
 * A trimmed heap has to be faulted back in by the *next* allocation, which
 * turns a routine call into an outlier.  Also caps the arena count: a control
 * loop is single-threaded and gains nothing from per-thread arenas.
 */
Status harden_malloc();

/// Bind the calling thread to one CPU, so it stops migrating between caches.
Status pin_current_thread(int cpu);

/// Bind the calling thread to a set of CPUs.
Status pin_current_thread(const std::vector<int>& cpus);

/**
 * @brief Run the calling thread under `SCHED_FIFO` at `priority`.
 *
 * Needs `CAP_SYS_NICE` or a raised `RLIMIT_RTPRIO`.  Opt-in because a
 * real-time thread that spins forever starves the machine.
 */
Status set_realtime_priority(int priority = 80);

/**
 * @brief Move XLA's worker threads onto `cpus`, away from the caller's core.
 *
 * Finds them by walking `/proc/self/task` for a thread name containing "XLA",
 * so call it after the `Runtime` exists.  The calling thread is never moved,
 * nor are threads the plugin leaves unnamed, which are indistinguishable from
 * the caller's own; success means "every pool thread that could be
 * identified".
 */
Status corral_xla_threads(const std::vector<int>& cpus);

/// Human-readable summary of what is and is not in effect, for logs.
std::string describe_environment();

}  // namespace pjrt::rt
