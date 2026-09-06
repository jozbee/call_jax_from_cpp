/**
 * @file rt.hpp
 * @brief Optional real-time hardening for the thread that runs the control
 *        loop.
 *
 * None of this changes what is computed; it changes how reliably the operating
 * system lets the computation finish on time.  Each function is independent,
 * reports whether it took effect, and is a no-op returning `false` on
 * platforms that do not provide it (macOS, mainly, where these are development
 * conveniences rather than a deployment target).
 *
 * A typical control process calls, once, after loading its `Function` and
 * before entering the loop:
 *
 * @code
 *   pjrt::rt::harden_malloc();
 *   pjrt::rt::lock_memory();
 *   pjrt::rt::pin_current_thread(2);
 *   pjrt::rt::corral_xla_threads({3});   // keep XLA's pools off cpu 2
 *   pjrt::rt::set_realtime_priority(80);
 * @endcode
 *
 * The host itself has to cooperate as well -- isolated cores, a performance
 * governor, disabled deep C-states.  `tools/rt_check.sh` audits that side.
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
 * Trimming the heap means the *next* allocation has to fault it back in, which
 * turns a routine call into an outlier.  Also caps the number of arenas, since
 * a control loop is single-threaded and does not benefit from per-thread ones.
 */
Status harden_malloc();

/// Bind the calling thread to one CPU, so it stops migrating between caches.
Status pin_current_thread(int cpu);

/// Bind the calling thread to a set of CPUs.
Status pin_current_thread(const std::vector<int>& cpus);

/**
 * @brief Run the calling thread under `SCHED_FIFO` at `priority`.
 *
 * Requires `CAP_SYS_NICE` (or a raised `RLIMIT_RTPRIO`); in a container, run
 * with `--cap-add=SYS_NICE --ulimit rtprio=99`.  A real-time thread that spins
 * forever will starve the machine, which is why this is opt-in and why the
 * computation being bounded matters.
 */
Status set_realtime_priority(int priority = 80);

/**
 * @brief Move XLA's worker threads onto `cpus`, away from the caller's core.
 *
 * XLA starts its pools when the client is created and puts "XLA" in the names
 * it gives those threads (`tf_XLAEigen…` at the pinned version), so they can
 * be found afterwards by walking `/proc/self/task` for a name containing it.
 * Call this after the `Runtime` exists.  With inline execution the pools
 * should be idle, and this keeps them from waking up on the core the control
 * loop is using.
 *
 * The calling thread is never moved, and threads XLA leaves unnamed -- they
 * inherit the executable's name and are indistinguishable from the caller's
 * own -- are not found, so a successful result is "every pool thread that
 * could be identified", not "every thread the plugin started".
 */
Status corral_xla_threads(const std::vector<int>& cpus);

/// Human-readable summary of what is and is not in effect, for logs.
std::string describe_environment();

}  // namespace pjrt::rt
