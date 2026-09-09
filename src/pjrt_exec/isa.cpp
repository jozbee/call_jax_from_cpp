#include "isa.hpp"

#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

#if defined(__aarch64__)
#include <sys/auxv.h>

// glibc leaves the AT_HWCAP bits to the kernel headers, which need not be
// installed.  The SVE bit is architectural and has never moved.
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
#endif

namespace pjrt::internal {
namespace {

// The levels of one family, weakest first: the x86-64 psABI levels, and SVE
// or not on aarch64.  A level on no ladder -- "unknown", or a family this file
// has not heard of -- compares against nothing, so an aarch64 host is never
// talked into loading an x86-64 executable.
constexpr const char* kX86Ladder[] = {"x86-64-v1", "x86-64-v2", "x86-64-v3",
                                      "x86-64-v4"};
constexpr const char* kArmLadder[] = {"aarch64", "aarch64+sve"};

/// Position of `level` in `ladder`, or -1 when it is not on this ladder.
template <std::size_t N>
int rung(const char* const (&ladder)[N], const std::string& level) {
  for (std::size_t i = 0; i < N; ++i) {
    if (level == ladder[i]) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

#if defined(__x86_64__) || defined(_M_X64)

// `__builtin_cpu_supports` rejects a name it does not know at compile time,
// and clang 18 (Ubuntu 24.04, so the CI container) rejects "lzcnt" and
// "movbe".  Both bits are one CPUID leaf away, so the v3 test stays faithful to
// the psABI's list.
bool cpuid_ecx_bit(unsigned int leaf, unsigned int bit) {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (__get_cpuid(leaf, &eax, &ebx, &ecx, &edx) == 0) {
    return false;
  }
  return (ecx & (1u << bit)) != 0;
}

#endif

}  // namespace

std::string host_isa_level() {
#if defined(__x86_64__) || defined(_M_X64)
  // Required before the first `__builtin_cpu_supports` in a translation unit
  // that may run ahead of the compiler's own constructor for the feature table.
  __builtin_cpu_init();

  // Strongest first.  A part with the AVX-512 set always carries the v3 set,
  // so no rung re-tests the rungs below it.
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512cd") &&
      __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512vl")) {
    return "x86-64-v4";
  }
  if (__builtin_cpu_supports("avx") && __builtin_cpu_supports("avx2") &&
      __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") &&
      __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") &&
      cpuid_ecx_bit(0x80000001u, 5) &&  // ABM/LZCNT
      cpuid_ecx_bit(1u, 22)) {          // MOVBE
    return "x86-64-v3";
  }
  if (__builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("ssse3") &&
      __builtin_cpu_supports("popcnt")) {
    return "x86-64-v2";
  }
  // Anything running this binary at all is at least the baseline.
  return "x86-64-v1";
#elif defined(__aarch64__)
  // aarch64 has no microarchitecture levels; SVE is the one difference wide
  // enough to change what a compiler emits, so it is the one recorded.
  return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0 ? "aarch64+sve" : "aarch64";
#else
  return "unknown";
#endif
}

bool isa_at_least(const std::string& host, const std::string& required) {
  const int host_x86 = rung(kX86Ladder, host);
  const int required_x86 = rung(kX86Ladder, required);
  if (host_x86 >= 0 && required_x86 >= 0) {
    return host_x86 >= required_x86;
  }

  const int host_arm = rung(kArmLadder, host);
  const int required_arm = rung(kArmLadder, required);
  if (host_arm >= 0 && required_arm >= 0) {
    return host_arm >= required_arm;
  }

  return false;
}

}  // namespace pjrt::internal
