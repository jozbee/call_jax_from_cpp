/**
 * @file alloc_guard.hpp
 * @brief Optional hook into the preloaded allocation counter.
 *
 * The counter is `tests/support/malloc_guard.c`, an allocator interposer
 * that is preloaded and never linked.  The guard resolves its markers with
 * `dlsym(RTLD_DEFAULT, ...)` and degrades to no-ops returning 0 when they
 * are absent, so one binary runs both with and without the preload:
 *
 * @code
 *   LD_PRELOAD=build/lib/malloc_guard.so ./build/bin/bench ...
 * @endcode
 *
 * The number that must stay at zero is `allocs_self()`, the wrapper's own
 * allocations in the steady-state call path.  A whole-process zero is out of
 * reach: XLA's thunk runtime allocates inside the plugin on every call.
 */
#pragma once

#include <cstddef>
#include <cstdio>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#define PJRT_ALLOC_GUARD_SUPPORTED 1
#else
#define PJRT_ALLOC_GUARD_SUPPORTED 0
#endif

namespace pjrt {

#if PJRT_ALLOC_GUARD_SUPPORTED

/**
 * @brief Counts allocations made between arm() and disarm(), when preloaded.
 *
 * Without the preload every accessor returns 0 and `present()` is false.
 */
class AllocGuard {
 public:
  /// Allocations attributed to the main executable and libpjrt_exec.
  static constexpr int kClassSelf = 0;
  /// Allocations attributed to the PJRT CPU plugin.
  static constexpr int kClassPlugin = 1;
  /// Everything else: libc, libstdc++, LAPACK, the thread pool.
  static constexpr int kClassRuntime = 2;

  /// @brief Resolve the interposer's markers, if it was preloaded.
  AllocGuard()
      : arm_(resolve<void (*)()>("pjrt_guard_arm")),
        disarm_(resolve<void (*)()>("pjrt_guard_disarm")),
        allocs_(resolve<unsigned long (*)()>("pjrt_guard_alloc_count")),
        frees_(resolve<unsigned long (*)()>("pjrt_guard_free_count")),
        total_(resolve<unsigned long (*)()>("pjrt_guard_total_alloc_count")),
        class_(resolve<unsigned long (*)(int)>("pjrt_guard_alloc_count_class")),
        classified_(resolve<int (*)()>("pjrt_guard_classified")) {}

  /// @brief Whether the interposer is live in this process.
  bool present() const { return arm_ != nullptr && allocs_ != nullptr; }

  /**
   * @brief Whether the per-module split is meaningful.
   *
   * False on macOS, where the classifier's module ranges are not built; the
   * totals are still correct there.
   */
  bool classified() const {
    return class_ != nullptr && classified_ != nullptr && classified_() != 0;
  }

  /// @brief Reset the counters and start counting.
  void arm() {
    if (arm_ != nullptr) {
      arm_();
    }
  }

  /// @brief Stop counting; the counters keep their values for reporting.
  void disarm() {
    if (disarm_ != nullptr) {
      disarm_();
    }
  }

  /// @brief Allocations made while armed.
  unsigned long allocs() const { return allocs_ != nullptr ? allocs_() : 0; }
  /// @brief Frees made while armed.
  unsigned long frees() const { return frees_ != nullptr ? frees_() : 0; }
  /**
   * @brief Allocations made by the whole process since it started.
   *
   * Counted whether armed or not.  Thousands here with zero while armed means
   * the path is clean; zero here means the preload never took effect.
   */
  unsigned long total() const { return total_ != nullptr ? total_() : 0; }

  /// @brief Armed allocations from the main executable or libpjrt_exec.
  unsigned long allocs_self() const { return class_count(kClassSelf); }
  /// @brief Armed allocations from inside the PJRT CPU plugin.
  unsigned long allocs_plugin() const { return class_count(kClassPlugin); }
  /// @brief Armed allocations from libc, libstdc++, LAPACK and everything else.
  unsigned long allocs_runtime() const { return class_count(kClassRuntime); }

  /**
   * @brief Print the armed-window census to @p out.
   *
   * @param out        Where to print; a null stream prints nothing.
   * @param iterations Calls made while armed, so the totals can be quoted
   *                   per call.
   */
  void report(std::FILE* out, std::size_t iterations) const {
    if (out == nullptr) {
      return;
    }
    std::fprintf(out, "\n=== allocations ===\n");
    if (!present()) {
      std::fprintf(out, "  guard not preloaded; run with %s=%s\n",
                   preload_variable(), "build/lib/malloc_guard.so");
      return;
    }
    const double n = iterations > 0 ? static_cast<double>(iterations) : 1.0;
    const unsigned long a = allocs();
    std::fprintf(out, "  armed window:  %lu allocs, %lu frees over %zu calls\n",
                 a, frees(), iterations);
    std::fprintf(out, "  per call:      %.2f\n", static_cast<double>(a) / n);
    if (classified()) {
      std::fprintf(out, "  self:          %lu (%.2f/call)   <- must be zero\n",
                   allocs_self(), static_cast<double>(allocs_self()) / n);
      std::fprintf(out, "  plugin:        %lu (%.2f/call)\n", allocs_plugin(),
                   static_cast<double>(allocs_plugin()) / n);
      std::fprintf(out, "  runtime:       %lu (%.2f/call)\n", allocs_runtime(),
                   static_cast<double>(allocs_runtime()) / n);
    } else {
      std::fprintf(out, "  by caller:     unavailable on this platform\n");
    }
    std::fprintf(out,
                 "  process total: %lu (nonzero proves the guard is live)\n",
                 total());
  }

 private:
  template <class Fn>
  static Fn resolve(const char* symbol) {
    return reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, symbol));
  }

  unsigned long class_count(int cls) const {
    return class_ != nullptr ? class_(cls) : 0;
  }

  static const char* preload_variable() {
#if defined(__APPLE__)
    return "DYLD_INSERT_LIBRARIES";
#else
    return "LD_PRELOAD";
#endif
  }

  void (*arm_)();
  void (*disarm_)();
  unsigned long (*allocs_)();
  unsigned long (*frees_)();
  unsigned long (*total_)();
  unsigned long (*class_)(int);
  int (*classified_)();
};

#else  // No dynamic-symbol lookup: keep the API, do nothing.

/// @brief Inert stand-in on platforms without `dlsym`.
class AllocGuard {
 public:
  static constexpr int kClassSelf = 0;
  static constexpr int kClassPlugin = 1;
  static constexpr int kClassRuntime = 2;

  AllocGuard() = default;
  bool present() const { return false; }
  bool classified() const { return false; }
  void arm() {}
  void disarm() {}
  unsigned long allocs() const { return 0; }
  unsigned long frees() const { return 0; }
  unsigned long total() const { return 0; }
  unsigned long allocs_self() const { return 0; }
  unsigned long allocs_plugin() const { return 0; }
  unsigned long allocs_runtime() const { return 0; }
  void report(std::FILE* out, std::size_t iterations) const {
    (void)iterations;
    if (out != nullptr) {
      std::fprintf(out,
                   "\n=== allocations ===\n"
                   "  allocation guard unavailable on this platform\n");
    }
  }
};

#endif  // PJRT_ALLOC_GUARD_SUPPORTED

// docs: begin alloc_guard_scope
/**
 * @brief Arms the guard for one scope and disarms it on the way out.
 *
 * Wrap exactly the region being claimed clean: the steady-state calls, after
 * warm-up, whose page faults and lazy initialization would otherwise be
 * counted against the path.
 *
 * @code
 *   {
 *     pjrt::AllocGuardScope armed(guard);
 *     for (std::size_t i = 0; i < iterations; ++i) fn.call();
 *   }
 *   guard.report(stdout, iterations);
 * @endcode
 */
class AllocGuardScope {
 public:
  /// @brief Reset the counters and start counting.
  explicit AllocGuardScope(AllocGuard& guard) : guard_(guard) { guard_.arm(); }

  ~AllocGuardScope() { guard_.disarm(); }

  AllocGuardScope(const AllocGuardScope&) = delete;
  AllocGuardScope& operator=(const AllocGuardScope&) = delete;

 private:
  AllocGuard& guard_;
};
// docs: end alloc_guard_scope

}  // namespace pjrt
