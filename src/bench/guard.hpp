/**
 * @file guard.hpp
 * @brief Optional hook into the preloaded allocation counter.
 *
 * The benchmark does not link against `malloc_guard.c`.  It looks the markers
 * up at run time and no-ops when they are absent, so the same binary runs both
 * with and without the preload.  See `tests/support/malloc_guard.c`.
 */
#pragma once

#include <dlfcn.h>

#include <cstdio>

namespace bench {

/// Counts allocations made between `arm()` and `disarm()`, when preloaded.
class AllocGuard {
 public:
  AllocGuard() {
    arm_ = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "pjrt_guard_arm"));
    disarm_ =
        reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "pjrt_guard_disarm"));
    allocs_ = reinterpret_cast<unsigned long (*)()>(
        dlsym(RTLD_DEFAULT, "pjrt_guard_alloc_count"));
    frees_ = reinterpret_cast<unsigned long (*)()>(
        dlsym(RTLD_DEFAULT, "pjrt_guard_free_count"));
    total_ = reinterpret_cast<unsigned long (*)()>(
        dlsym(RTLD_DEFAULT, "pjrt_guard_total_alloc_count"));
  }

  bool present() const { return arm_ != nullptr && allocs_ != nullptr; }

  void arm() const {
    if (arm_ != nullptr) {
      arm_();
    }
  }

  void disarm() const {
    if (disarm_ != nullptr) {
      disarm_();
    }
  }

  unsigned long allocs() const { return allocs_ != nullptr ? allocs_() : 0; }
  unsigned long frees() const { return frees_ != nullptr ? frees_() : 0; }
  unsigned long total() const { return total_ != nullptr ? total_() : 0; }

  /// Report the armed-window census; `iterations` gives the per-call rate.
  void report(std::size_t iterations) const {
    if (!present()) {
      std::printf(
          "\n=== allocations ===\n  guard not preloaded (run `make "
          "test_alloc`)\n");
      return;
    }
    const unsigned long a = allocs();
    const double per_call =
        iterations > 0 ? static_cast<double>(a) / static_cast<double>(iterations)
                       : 0.0;
    std::printf("\n=== allocations ===\n");
    std::printf("  armed window: %lu allocs, %lu frees\n", a, frees());
    std::printf("  per call:     %.2f\n", per_call);
    // A zero armed count only means something if the interposer is live.
    std::printf("  process total: %lu (proves the guard is live)\n", total());
  }

 private:
  void (*arm_)() = nullptr;
  void (*disarm_)() = nullptr;
  unsigned long (*allocs_)() = nullptr;
  unsigned long (*frees_)() = nullptr;
  unsigned long (*total_)() = nullptr;
};

}  // namespace bench
