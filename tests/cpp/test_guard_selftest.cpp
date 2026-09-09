/**
 * @file test_guard_selftest.cpp
 * @brief Prove that the preloaded allocation counter sees allocations and
 *        charges them to the module that made them.
 *
 * The allocation gate is a claim about a number: `self` must be zero over
 * the steady-state calls.  A counter that is not actually interposed also
 * reports zero, and so does one whose classifier puts everything in
 * `runtime`.  So this makes allocations that could not be anything but
 * `self` -- a `malloc`/`free` pair, a `new`/`delete` pair, and a
 * `std::vector` growing -- inside an armed window, and prints what the guard
 * saw:
 *
 *     present=1 classified=1 self=13 plugin=0 runtime=0 frees=13
 *
 * The exact count is the standard library's business; the test asserts a
 * floor.  `std::vector` earns its place: its growth happens in a template
 * instantiated here, so it reaches `operator new` from an address inside
 * this executable, which is the case a classifier keyed on which library
 * defines the allocator would misattribute to libstdc++.
 *
 * Run it without the preload as well: then `present=0` and every count is 0,
 * which is how the caller tells "nothing allocated" apart from "nothing was
 * measured".  This program judges nothing; only the caller knows whether the
 * preload was in the environment.
 */
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pjrt_exec/alloc_guard.hpp"

namespace {

void consume(const void*) {}

/// Both compilers replace a `malloc`/`free` pair whose pointer never escapes
/// with a stack slot, and the standard lets them do the same to
/// `new`/`delete`.  Storing the pointer into a `volatile` keeps the *store*,
/// not the allocation: measured at `-O2`, that version counted 11 where 13
/// were made.  A call through a volatile function pointer cannot be seen
/// through, so the pointer genuinely escapes.  Verified with clang++ and g++.
void (*volatile g_consume)(const void*) = consume;

/// Read out of the vector so that its growth cannot be elided either.
volatile int g_total = 0;

/// Enough reallocations that the count cannot be mistaken for noise.
constexpr int kPushes = 1024;

}  // namespace

int main() {
  // Before arming: the constructor resolves its markers with dlsym, which is
  // entitled to allocate.
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

    // Scoped so the vector's free is counted inside the armed window.
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
