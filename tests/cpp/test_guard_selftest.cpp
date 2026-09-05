/**
 * @file test_guard_selftest.cpp
 * @brief Prove that the preloaded allocation counter sees allocations and
 *        charges them to the module that made them.
 *
 * The allocation gate this project runs in CI is a claim about a number:
 * `self` must be zero over the steady-state calls.  A counter that is not
 * actually interposed also reports zero, and so does one whose classifier puts
 * everything in `runtime`.  Both would turn the gate into a green light that
 * measures nothing, which is worse than not having it.
 *
 * So this makes three allocations that could not be anything but `self` -- a
 * `malloc`/`free` pair, a `new`/`delete` pair, and a `std::vector` growing --
 * inside an armed window, and prints what the guard says it saw:
 *
 *     present=1 classified=1 self=12 plugin=0 runtime=0 frees=12
 *
 * All three are this binary's own code.  `std::vector` earns its place: its
 * growth happens in a header template instantiated here, so it reaches
 * `operator new` from an address inside this executable, and it is the case
 * that would be misattributed to libstdc++ by a classifier that only looked at
 * which library defines the allocator.
 *
 * Run it without the preload as well.  Then `present=0` and every count is 0,
 * which is what the accessors are specified to do when the markers are absent,
 * and it is how the caller tells "nothing allocated" apart from "nothing was
 * measured".
 *
 * This program judges nothing; it reports.  Whether `self` is allowed to be
 * zero on this run depends on whether the preload was in the environment, and
 * only the caller knows that.
 */
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pjrt_exec/alloc_guard.hpp"

namespace {

/// Does nothing, and the optimizer is not allowed to know that.
void consume(const void*) {}

/**
 * @brief Where every scripted allocation is handed to, so that none of them
 *        can be optimized away.
 *
 * Both compilers replace a `malloc`/`free` pair whose pointer never escapes
 * with a stack slot, and the standard explicitly lets them do the same to
 * `new`/`delete`.  Storing the pointer into a `volatile` is not enough -- it
 * keeps the *store*, not the allocation -- and measured here at `-O2`, that
 * version reported 11 allocations where 13 were made: both pairs had
 * evaporated and the vector was carrying the whole test.
 *
 * A call through a volatile function pointer has to be made, and the compiler
 * cannot see where it goes, so the pointer genuinely escapes.  Verified with
 * clang++ and g++.
 */
void (*volatile g_consume)(const void*) = consume;

/// Read out of the vector so that its growth cannot be elided either.
volatile int g_total = 0;

/// Elements pushed one at a time. Enough reallocations that the count cannot
/// be mistaken for noise, and small enough to stay in the allocator's fast
/// path.
constexpr int kPushes = 1024;

}  // namespace

int main() {
  // Constructed before arming: it resolves seven symbols with dlsym, and dlsym
  // is entitled to allocate.
  pjrt::AllocGuard guard;

  {
    pjrt::AllocGuardScope armed(guard);

    void* block = std::malloc(64);
    if (block == nullptr) {
      std::fprintf(stderr, "test_guard_selftest: malloc(64) failed\n");
      return 1;
    }
    static_cast<unsigned char*>(block)[0] = 1;
    g_consume(block);
    std::free(block);

    int* one = new int(7);
    g_total = *one;
    g_consume(one);
    delete one;

    // Scoped so the vector is destroyed inside the armed window and its free
    // is counted with the rest.
    {
      std::vector<int> growing;
      for (int i = 0; i < kPushes; ++i) {
        growing.push_back(i);
      }
      g_consume(growing.data());
      g_total = growing[static_cast<std::size_t>(kPushes) - 1];
    }
  }

  std::printf(
      "present=%d classified=%d self=%lu plugin=%lu runtime=%lu "
      "frees=%lu\n",
      guard.present() ? 1 : 0, guard.classified() ? 1 : 0, guard.allocs_self(),
      guard.allocs_plugin(), guard.allocs_runtime(), guard.frees());
  return 0;
}
