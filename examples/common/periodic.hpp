/**
 * @file periodic.hpp
 * @brief The parts of a periodic loop that are not the work: a monotonic
 *        clock, an absolute sleep, and a stop flag a signal handler may set.
 *
 * Two programs here need these -- the real-time example and anything else that
 * grows a fixed-period loop -- and both need them spelled the same way, because
 * the difference between an absolute and a relative sleep is the difference
 * between a loop that holds its phase and one that drifts by exactly the
 * quantity it is trying to measure.
 *
 * Everything is in nanoseconds on `CLOCK_MONOTONIC`.  A `timespec` is built
 * only where the kernel insists on one: an `std::int64_t` is what arithmetic on
 * deadlines wants, and carrying two representations through a loop body is how
 * a renormalization gets forgotten.
 */
#pragma once

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <ctime>

#if defined(__linux__)
#define CJFC_HAVE_CLOCK_NANOSLEEP 1
#else
#define CJFC_HAVE_CLOCK_NANOSLEEP 0
#endif

namespace cjfc {

/// Nanoseconds in a second, spelled once.
inline constexpr std::int64_t kNsPerSec = 1000 * 1000 * 1000;

/// @brief `CLOCK_MONOTONIC` as nanoseconds since the clock's epoch.
///
/// Monotonic rather than the wall clock: an NTP step during a run would
/// otherwise show up as a spectacular outlier that never happened.
inline std::int64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<std::int64_t>(t.tv_sec) * kNsPerSec +
         static_cast<std::int64_t>(t.tv_nsec);
}

// docs: begin sleep-until
/**
 * @brief Sleep until the absolute time @p target_ns on `CLOCK_MONOTONIC`.
 *
 * Absolute, not relative: a relative sleep of one period accumulates every
 * cycle's wake-up latency into the phase, so the loop drifts away from its
 * schedule by exactly the quantity it is trying to measure.  With an absolute
 * target, lateness is bounded by the last cycle rather than by the whole run,
 * and a target already in the past returns immediately -- which is how a loop
 * catches up after an overrun instead of skipping a cycle.
 *
 * @return 0, or `EINTR` when a signal arrived first.  `clock_nanosleep`
 *         returns the error number rather than setting `errno`.
 */
inline int sleep_until(std::int64_t target_ns) {
  timespec target;
  target.tv_sec = static_cast<time_t>(target_ns / kNsPerSec);
  target.tv_nsec = static_cast<long>(target_ns % kNsPerSec);
#if CJFC_HAVE_CLOCK_NANOSLEEP
  return clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr);
#else
  // No absolute monotonic sleep here (macOS): compute the remainder and sleep
  // relatively.  Good enough to run the example; not good enough to quote a
  // jitter number from, which is why the report prints the platform.
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
// docs: end sleep-until

namespace detail {

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
inline std::atomic<bool> g_stop{false};

static_assert(std::atomic<bool>::is_always_lock_free,
              "a signal handler may not touch a lock-based atomic");

/// C linkage, and a prefixed name because that linkage makes it global.
extern "C" inline void cjfc_stop_signal_handler(int) {
  g_stop.store(true, std::memory_order_relaxed);
}

}  // namespace detail

/**
 * @brief Install the handler for SIGINT and SIGTERM.
 *
 * `sa_flags` deliberately omits `SA_RESTART`: a periodic loop wants the sleep
 * to return `EINTR` so it can notice the flag, and an automatically restarted
 * `clock_nanosleep` would hold it in the kernel until the next period.
 * SIGTERM as well as SIGINT, because a containerized control process is
 * stopped with the former and should still print its report.
 */
inline void install_stop_handlers() {
  struct sigaction action{};
  action.sa_handler = detail::cjfc_stop_signal_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);
}

/// @brief Whether a stop signal has arrived.  Relaxed: this is one flag with no
///        other state ordered against it.
inline bool stopping() {
  return detail::g_stop.load(std::memory_order_relaxed);
}

}  // namespace cjfc
