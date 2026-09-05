#include "isa.hpp"

#include <cstddef>

#if defined(__aarch64__)
#include <sys/auxv.h>

// glibc puts AT_HWCAP in <sys/auxv.h> but leaves the bit definitions to the
// kernel headers, which are not guaranteed to be installed.  The SVE bit is
// architectural and has never moved, so define it rather than fail the build
// on a machine missing <asm/hwcap.h>.
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
#endif

namespace pjrt::internal {
namespace {

// The levels of one architecture family, weakest first.  A level that appears
// in no ladder -- "unknown", or a family this file has never heard of --
// compares against nothing, which is the whole point: an aarch64 host must not
// be talked into loading an x86-64 executable by a comparison that happens to
// return true.
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

}  // namespace

std::string host_isa_level() {
#if defined(__x86_64__) || defined(_M_X64)
  // Required before the first __builtin_cpu_supports in any translation unit
  // whose code might run ahead of the compiler's own constructor for the CPU
  // feature table.  Idempotent, and it costs one CPUID at load.
  __builtin_cpu_init();

  // Tested strongest first.  A part carrying the AVX-512 set always carries
  // the v3 set too, so the cascade agrees with the psABI without each rung
  // re-testing the rungs below it.
  if (__builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512cd") &&
      __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512vl")) {
    return "x86-64-v4";
  }
  if (__builtin_cpu_supports("avx") && __builtin_cpu_supports("avx2") &&
      __builtin_cpu_supports("bmi") && __builtin_cpu_supports("bmi2") &&
      __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") &&
      __builtin_cpu_supports("lzcnt") && __builtin_cpu_supports("movbe")) {
    return "x86-64-v3";
  }
  if (__builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("ssse3") &&
      __builtin_cpu_supports("popcnt")) {
    return "x86-64-v2";
  }
  // Anything running this binary at all is at least the baseline.
  return "x86-64-v1";
#elif defined(__aarch64__)
  // aarch64 has no microarchitecture levels to speak of; SVE is the one
  // difference wide enough to change what a compiler emits, so it is the one
  // distinction the sidecar records.
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
