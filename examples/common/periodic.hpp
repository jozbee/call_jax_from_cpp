/**
 * @file periodic.hpp
 * @brief The parts of a periodic loop that are not the work: a monotonic
 *        clock, an absolute sleep, and a stop flag a signal handler may set.
 *
 * Shared so that every loop spells them the same way: the difference between
 * an absolute and a relative sleep is the difference between a loop that holds
 * its phase and one that drifts by exactly the quantity it is measuring.
 *
 * Everything is nanoseconds on `CLOCK_MONOTONIC`.  A `timespec` is built only
 * where the kernel insists on one; carrying two representations through a loop
 * body is how a renormalization gets forgotten.
 */
#pragma once

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <ctime>

// call_jax_from_cpp: helpers the examples share; not the library
namespace cjfc {

/// Nanoseconds in a second.
inline constexpr std::int64_t kNsPerSec = 1000 * 1000 * 1000;

/// @brief `CLOCK_MONOTONIC` as nanoseconds since the clock's epoch.  Not the
///        wall clock: an NTP step mid-run would show up as an outlier that
///        never happened.
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
 * Absolute, not relative, so wake-up lateness does not accumulate into the
 * phase.  A target already in the past returns at once, which is how a loop
 * catches up after an overrun instead of skipping a cycle.
 *
 * @return 0, or `EINTR` when a signal arrived first.  `clock_nanosleep`
 *         returns the error number rather than setting `errno`.
 */
inline int sleep_until(std::int64_t target_ns) {
  timespec target;
  target.tv_sec = static_cast<time_t>(target_ns / kNsPerSec);
  target.tv_nsec = static_cast<long>(target_ns % kNsPerSec);
#if defined(__linux__)
  return clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, nullptr);
#else
  // macOS has no absolute monotonic sleep.  Good enough to run the example,
  // not to quote a jitter number from.
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

/// Set by the signal handler, read by the loop.  A handler does exactly one
/// thing here, set this flag; the report is written on the way out of the
/// loop, where allocating and locking are allowed.
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
 * `sa_flags` omits `SA_RESTART`: the sleep must return `EINTR` so the loop
 * notices the flag, and a restarted `clock_nanosleep` would hold it in the
 * kernel until the next period.  SIGTERM as well, because a containerized
 * process is stopped with it and should still print its report.
 */
inline void install_stop_handlers() {
  struct sigaction action{};
  action.sa_handler = detail::cjfc_stop_signal_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);
}

/// @brief Whether a stop signal has arrived.  Relaxed: one flag, with no other
///        state ordered against it.
inline bool stopping() {
  return detail::g_stop.load(std::memory_order_relaxed);
}

}  // namespace cjfc
