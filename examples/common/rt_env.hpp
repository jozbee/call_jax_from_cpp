/**
 * @file rt_env.hpp
 * @brief What the host is willing to give a real-time loop, and the steps that
 *        ask for it.
 *
 * `pjrt::rt` provides the hardening primitives; this is the layer above them:
 * it reads the settings the host was booted with, decides which CPU the loop
 * should run on, applies the primitives in an order that will not shoot the
 * process in the foot, and reports every step instead of failing.  A control
 * process that refuses to start because it could not get `SCHED_FIFO` is worse
 * than one that starts and says so.
 *
 * The audit is deliberately the same vocabulary as `tools/rt_check.sh`, which
 * asks the same questions from the shell: isolated CPUs, `nohz_full`, the
 * scaling governor, transparent hugepages, SMT, `RLIMIT_RTPRIO`,
 * `RLIMIT_MEMLOCK`, `/dev/cpu_dma_latency`.  A report from a run and a report
 * from the script should be readable side by side.
 *
 * Nothing here is required for correct results.  It is required for *believable
 * latency numbers*: the first thing to check when a p99.9 looks wrong is
 * whether any of this took effect, which is why every example prints it.
 *
 * Linux only in substance; elsewhere every field reads "unavailable" and every
 * step skips, so the examples still build and run on a development machine.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pjrt_exec/rt.hpp"

#if defined(__linux__)
#include <fcntl.h>
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#define CJFC_HAVE_RUSAGE 1
#else
#define CJFC_HAVE_RUSAGE 0
#endif

namespace cjfc {

/**
 * @brief The host settings that decide whether a bounded computation finishes
 *        on time.
 *
 * Read-only: detecting this changes nothing.  Unreadable values are
 * `"unavailable"` or -1 rather than an exception, because a container that
 * hides half of `/sys` is a normal place to run, not an error.
 */
struct HostEnv {
  /// `uname -r`, or "unavailable".
  std::string kernel;

  /// Whether this is a PREEMPT_RT kernel, from `/sys/kernel/realtime` or the
  /// `uname` version string.
  bool preempt_rt = false;

  /// `/.dockerenv` exists, or `$CJFC_IN_CONTAINER` is set.  Worth reporting
  /// because the capability and rlimit answers below usually differ inside one.
  bool in_container = false;

  /// Online CPUs, or -1.
  int cpus_online = -1;

  /// `isolcpus=` as the kernel reports it in
  /// `/sys/devices/system/cpu/isolated`.  Empty when the host was booted
  /// without it, which is the common case and the reason `choose_cpu` has to
  /// explain itself.
  std::vector<int> isolated;

  /// `/sys/devices/system/cpu/nohz_full`: the CPUs the timer tick leaves alone.
  std::vector<int> nohz_full;

  /// This thread's affinity mask.  The intersection with `isolated` is what can
  /// actually be pinned to; a mask narrowed by `taskset` or a container's
  /// cpuset makes an otherwise reasonable `--cpu 3` impossible.
  std::vector<int> affinity;

  /// `scaling_governor` for cpu0.  Anything but "performance" means the clock
  /// can change under the measurement.
  std::string governor;

  /// The selected transparent-hugepage mode ("always", "madvise", "never"),
  /// unbracketed.  `khugepaged` stalls faulting threads under "always".
  std::string thp;

  /// `/sys/devices/system/cpu/smt/control`: "on", "off", "notsupported".  A
  /// sibling hyperthread steals from the loop's core.
  std::string smt;

  /// `RLIMIT_RTPRIO` soft limit; -1 for unlimited.  0 means `SCHED_FIFO` is not
  /// available to this process.
  long rlimit_rtprio = -1;

  /// `RLIMIT_MEMLOCK` soft limit in bytes; -1 for unlimited, which is what
  /// `mlockall` wants.
  long rlimit_memlock = -1;

  /// `/proc/sys/kernel/sched_rt_runtime_us`: the microseconds per period a
  /// real-time thread may run before the kernel throttles it.  -1 is both the
  /// "throttling disabled" value the file itself holds and what is reported
  /// when it cannot be read.
  long rt_runtime_us = -1;

  /// Whether `/dev/cpu_dma_latency` can be opened for writing, i.e. whether
  /// this process can keep the cores out of deep C-states.
  bool cpu_dma_latency_writable = false;

  double loadavg1 = -1.0;   ///< One-minute load average, or -1.
  double loadavg5 = -1.0;   ///< Five-minute load average, or -1.
  double loadavg15 = -1.0;  ///< Fifteen-minute load average, or -1.

  /// `loadavg1 > 1.0`.  A latency number measured while this is true is not
  /// noisy, it is wrong: the same configuration measured during a build
  /// reported p50 2.4x high and max/p50 4.4 instead of 1.1.
  bool busy = false;
};

/**
 * @brief Expand a kernel CPU list -- `"2-5,8"` -> `{2,3,4,5,8}`.
 *
 * Handles the two ways these files say "nothing": empty, and the literal
 * `"(null)"` that `nohz_full` prints when it is unset.  Anything unparseable is
 * skipped rather than throwing; this is diagnostic input, and a strange
 * `/sys` file should not take down a control process.
 */
inline std::vector<int> parse_cpulist(std::string text) {
  std::vector<int> cpus;
  if (text == "(null)") {
    return cpus;
  }
  std::size_t pos = 0;
  while (pos < text.size()) {
    const std::size_t comma = text.find(',', pos);
    std::string item =
        text.substr(pos, comma == std::string::npos ? std::string::npos
                                                    : comma - pos);
    pos = comma == std::string::npos ? text.size() : comma + 1;

    const std::size_t dash = item.find('-');
    try {
      if (dash == std::string::npos) {
        cpus.push_back(std::stoi(item));
      } else {
        const int lo = std::stoi(item.substr(0, dash));
        const int hi = std::stoi(item.substr(dash + 1));
        // A malformed range must not turn into a multi-gigabyte vector.
        if (hi >= lo && hi - lo < 4096) {
          for (int cpu = lo; cpu <= hi; ++cpu) {
            cpus.push_back(cpu);
          }
        }
      }
    } catch (const std::exception&) {
      continue;  // not a number: this file is not what we thought it was
    }
  }
  return cpus;
}

/// @brief Whether @p cpu appears in @p cpus.
inline bool contains_cpu(const std::vector<int>& cpus, int cpu) {
  return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
}

/// @brief Every CPU in @p mask except @p cpu -- the set to corral XLA's worker
///        threads onto once the loop has claimed one core.
inline std::vector<int> cpus_except(const std::vector<int>& mask, int cpu) {
  std::vector<int> rest;
  rest.reserve(mask.size());
  for (int c : mask) {
    if (c != cpu) {
      rest.push_back(c);
    }
  }
  return rest;
}

/**
 * @brief Pick the CPU the loop should run on, and say why.
 *
 * @param spec  `"auto"` to choose one, `"none"` to stay unpinned, or a CPU
 *              number.
 * @param why   Filled in with the reason, in every case including success.  It
 *              is the detail line of the pinning step, and the answer to "why
 *              is this thing not pinned" without a second run.
 * @return The CPU to pin to, or -1 to leave the thread where the scheduler puts
 *         it.
 *
 * `"auto"` prefers a CPU that is both isolated and `nohz_full` -- isolation
 * keeps other work off it, `nohz_full` keeps the timer tick off it, and the
 * combination is what a tick-free core actually requires.  It falls back to
 * merely isolated, and then gives up: pinning to a CPU the rest of the system
 * is also using trades one source of jitter for another, so an unhardened host
 * is left alone unless the caller insists with an explicit number.
 */
inline int choose_cpu(const HostEnv& env, const std::string& spec,
                      std::string& why) {
  if (spec == "none" || spec.empty()) {
    why = "pinning disabled (--cpu none)";
    return -1;
  }

  if (spec == "auto") {
    for (int cpu : env.isolated) {
      if (contains_cpu(env.nohz_full, cpu) && contains_cpu(env.affinity, cpu)) {
        why = "isolated and nohz_full, and in this thread's affinity mask";
        return cpu;
      }
    }
    for (int cpu : env.isolated) {
      if (contains_cpu(env.affinity, cpu)) {
        why = "isolated (but not nohz_full: the timer tick still fires here)";
        return cpu;
      }
    }
    why =
        "no isolated cpus available to this thread; running unpinned. "
        "Boot with isolcpus=/nohz_full=, or pass --cpu N to pin anyway";
    return -1;
  }

  int cpu = -1;
  try {
    cpu = std::stoi(spec);
  } catch (const std::exception&) {
    why = "cpu spec '" + spec + "' is not 'auto', 'none' or a cpu number";
    return -1;
  }
  if (!env.affinity.empty() && !contains_cpu(env.affinity, cpu)) {
    why = "cpu " + std::to_string(cpu) +
          " is not in this thread's affinity mask; running unpinned";
    return -1;
  }
  why = "requested explicitly (--cpu " + std::to_string(cpu) + ")";
  return cpu;
}

/// @brief One hardening step and what became of it.
struct Step {
  std::string name;   ///< The helper that was asked, e.g. "lock_memory".
  bool ok = false;    ///< Whether it took effect.
  std::string detail; ///< What it did, or why it did not.
};

/// @brief Print one step as `  [ok  ] name: detail`, or `  [skip] ...` when it
///        did not take effect.
inline void print_step(const Step& step) {
  std::printf("  [%s] %s%s%s\n", step.ok ? "ok  " : "skip", step.name.c_str(),
              step.detail.empty() ? "" : ": ", step.detail.c_str());
}

/**
 * @brief Holds `/dev/cpu_dma_latency` open at 0 microseconds.
 *
 * The kernel applies the constraint for exactly as long as the file descriptor
 * stays open and drops it the moment it closes -- writing 0 and closing the
 * file achieves nothing, which is a mistake that is very easy to make and
 * impossible to see in the numbers except as the multi-millisecond outlier it
 * was supposed to prevent.  So this object owns the descriptor and the caller
 * keeps it alive for the whole run.
 *
 * Needs write access to the device, which normally means root.  Without it,
 * `acquire()` returns a skipped step and the run continues.
 */
class DmaLatencyHold {
 public:
  DmaLatencyHold() = default;

  /// Releases the constraint: deep C-states become available again.
  ~DmaLatencyHold() { release(); }

  DmaLatencyHold(const DmaLatencyHold&) = delete;
  DmaLatencyHold& operator=(const DmaLatencyHold&) = delete;

  /// @brief Open the device and write a 0 microsecond latency target, keeping
  ///        the descriptor.
  Step acquire() {
#if defined(__linux__)
    if (fd_ >= 0) {
      return Step{"cpu_dma_latency", true, "already held at 0 us"};
    }
    fd_ = ::open("/dev/cpu_dma_latency", O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
      return Step{"cpu_dma_latency", false,
                  std::string("open(/dev/cpu_dma_latency): ") +
                      std::strerror(errno) + " (needs root)"};
    }
    const std::int32_t target_us = 0;
    if (::write(fd_, &target_us, sizeof(target_us)) !=
        static_cast<ssize_t>(sizeof(target_us))) {
      const std::string reason = std::strerror(errno);
      release();
      return Step{"cpu_dma_latency", false, "write: " + reason};
    }
    return Step{"cpu_dma_latency", true,
                "holding /dev/cpu_dma_latency at 0 us (deep C-states off)"};
#else
    return Step{"cpu_dma_latency", false, "not supported on this platform"};
#endif
  }

  /// @brief Whether the constraint is currently held.
  bool held() const { return fd_ >= 0; }

  /// @brief Close the descriptor, releasing the constraint.
  void release() {
#if defined(__linux__)
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
#endif
  }

 private:
  int fd_ = -1;
};

/**
 * @brief Page faults and context switches, for the thread where that is
 *        available.
 *
 * The counters that matter in a loop that claims to be allocation-free: a
 * major fault is a disk read in the middle of a control cycle, and an
 * involuntary context switch is the scheduler taking the core away.  Take one
 * before the loop and one after, and subtract.
 */
struct Rusage {
  long minflt = -1;  ///< Minor faults: a page mapped without touching disk.
  long majflt = -1;  ///< Major faults: a page that had to be read in.
  long nvcsw = -1;   ///< Voluntary context switches: the thread waited.
  long nivcsw = -1;  ///< Involuntary ones: the thread was preempted.

  /**
   * @brief Sample the counters now.
   *
   * Scoped to the calling thread where `RUSAGE_THREAD` exists, so XLA's pool
   * threads do not contribute; `scope()` says which was used, and the numbers
   * mean something different in each case.
   */
  static Rusage now() noexcept {
    Rusage r;
#if CJFC_HAVE_RUSAGE
    rusage ru{};
    if (getrusage(kWho, &ru) == 0) {
      r.minflt = static_cast<long>(ru.ru_minflt);
      r.majflt = static_cast<long>(ru.ru_majflt);
      r.nvcsw = static_cast<long>(ru.ru_nvcsw);
      r.nivcsw = static_cast<long>(ru.ru_nivcsw);
    }
#endif
    return r;
  }

  /// @brief "thread", "process" or "unavailable": what `now()` measures here.
  static const char* scope() noexcept {
#if CJFC_HAVE_RUSAGE && defined(RUSAGE_THREAD)
    return "thread";
#elif CJFC_HAVE_RUSAGE
    return "process";
#else
    return "unavailable";
#endif
  }

  /// @brief Field-wise difference, for reporting what one loop cost.
  Rusage operator-(const Rusage& before) const noexcept {
    Rusage d;
    d.minflt = minflt - before.minflt;
    d.majflt = majflt - before.majflt;
    d.nvcsw = nvcsw - before.nvcsw;
    d.nivcsw = nivcsw - before.nivcsw;
    return d;
  }

#if CJFC_HAVE_RUSAGE
 private:
#if defined(RUSAGE_THREAD)
  static constexpr int kWho = RUSAGE_THREAD;
#else
  static constexpr int kWho = RUSAGE_SELF;
#endif
#endif
};

#if defined(__linux__)
namespace detail {

/// First line of @p path, or an empty string when it cannot be read.
inline std::string read_first_line(const char* path) {
  std::ifstream file(path);
  std::string line;
  if (file && std::getline(file, line)) {
    return line;
  }
  return std::string();
}

/// First line of @p path, or "unavailable" -- the spelling the report uses for
/// a value this host does not expose.
inline std::string read_line_or_unavailable(const char* path) {
  const std::string line = read_first_line(path);
  return line.empty() ? std::string("unavailable") : line;
}

/// Soft limit of @p resource in its own units; -1 for unlimited.
inline long rlimit_soft(int resource) {
  rlimit limit{};
  if (getrlimit(resource, &limit) != 0 || limit.rlim_cur == RLIM_INFINITY) {
    return -1;
  }
  return static_cast<long>(limit.rlim_cur);
}

/// The bracketed choice out of a sysfs multiple-choice line -- "always
/// [madvise] never" is the mode plus the menu, and only the mode is a fact
/// about this host.
inline std::string bracketed_choice(const std::string& line) {
  const std::size_t open = line.find('[');
  const std::size_t close = line.find(']', open + 1);
  if (open == std::string::npos || close == std::string::npos) {
    return line;
  }
  return line.substr(open + 1, close - open - 1);
}

}  // namespace detail

/// @brief Read every setting in `HostEnv` from this host.
inline HostEnv detect_host_env() {
  HostEnv env;

  utsname uts{};
  if (uname(&uts) == 0) {
    env.kernel = uts.release;
    // Mainline stamps PREEMPT_RT into the version string; the sysfs file below
    // is the authority when it exists.
    env.preempt_rt = std::string(uts.version).find("PREEMPT_RT") !=
                     std::string::npos;
  } else {
    env.kernel = "unavailable";
  }
  const std::string realtime = detail::read_first_line("/sys/kernel/realtime");
  if (realtime == "1") {
    env.preempt_rt = true;
  }

  env.in_container = ::access("/.dockerenv", F_OK) == 0 ||
                     std::getenv("CJFC_IN_CONTAINER") != nullptr;

  const long online = sysconf(_SC_NPROCESSORS_ONLN);
  env.cpus_online = online > 0 ? static_cast<int>(online) : -1;

  env.isolated = parse_cpulist(
      detail::read_first_line("/sys/devices/system/cpu/isolated"));
  env.nohz_full = parse_cpulist(
      detail::read_first_line("/sys/devices/system/cpu/nohz_full"));

  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &set)) {
        env.affinity.push_back(cpu);
      }
    }
  }

  env.governor = detail::read_line_or_unavailable(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  const std::string thp =
      detail::read_first_line("/sys/kernel/mm/transparent_hugepage/enabled");
  env.thp = thp.empty() ? "unavailable" : detail::bracketed_choice(thp);
  env.smt =
      detail::read_line_or_unavailable("/sys/devices/system/cpu/smt/control");

  env.rlimit_rtprio = detail::rlimit_soft(RLIMIT_RTPRIO);
  env.rlimit_memlock = detail::rlimit_soft(RLIMIT_MEMLOCK);

  const std::string rt_runtime =
      detail::read_first_line("/proc/sys/kernel/sched_rt_runtime_us");
  if (!rt_runtime.empty()) {
    try {
      env.rt_runtime_us = std::stol(rt_runtime);
    } catch (const std::exception&) {
      env.rt_runtime_us = -1;
    }
  }

  // Opening it changes nothing; only writing to a descriptor that stays open
  // does.  @see DmaLatencyHold.
  const int dma_fd = ::open("/dev/cpu_dma_latency", O_RDWR | O_CLOEXEC);
  if (dma_fd >= 0) {
    env.cpu_dma_latency_writable = true;
    ::close(dma_fd);
  }

  std::ifstream loadavg("/proc/loadavg");
  if (loadavg >> env.loadavg1 >> env.loadavg5 >> env.loadavg15) {
    env.busy = env.loadavg1 > 1.0;
  } else {
    env.loadavg1 = env.loadavg5 = env.loadavg15 = -1.0;
  }
  return env;
}

#else  // Not Linux: report honestly rather than approximately.

/// @brief Everything unavailable: none of these settings exist on this
///        platform, and reporting a plausible-looking value would be worse than
///        reporting none.
inline HostEnv detect_host_env() {
  HostEnv env;
  env.kernel = "unavailable";
  env.governor = "unavailable";
  env.thp = "unavailable";
  env.smt = "unavailable";
  return env;
}

#endif  // __linux__

/**
 * @brief What a program wants asked for on its behalf.
 *
 * `rt_priority` is 0 by default -- `SCHED_FIFO` is opt-in because a real-time
 * thread that misbehaves takes the machine with it, and because a run that
 * quietly got it and a run that quietly did not are indistinguishable in the
 * output unless the request was explicit.
 */
struct HardeningOptions {
  bool malloc_tune = true;   ///< `pjrt::rt::harden_malloc`.
  bool mlock = true;         ///< `pjrt::rt::lock_memory`.
  bool corral = true;        ///< Move XLA's pools off the loop's CPU.
  std::string cpu = "auto";  ///< `choose_cpu` spec: "auto", "none" or a number.
  int rt_priority = 0;       ///< `SCHED_FIFO` priority; 0 leaves it alone.
  bool dma_latency = false;  ///< Hold `/dev/cpu_dma_latency` at 0 (needs root).
};

// docs: begin rt-harden
/**
 * @brief Apply the hardening steps, in the one order that is safe, and report
 *        each.
 *
 * Never fails: every step that does not take effect becomes a skipped `Step`
 * with the reason in it, and the program runs anyway.  An unprivileged run on a
 * stock kernel is expected to skip most of this and is still a correct run --
 * it is just not a run to quote tail numbers from.
 *
 * The order is not arbitrary:
 *
 *   1. `harden_malloc` before anything allocates in bulk, so the heap it
 *      configures is the heap the rest of startup grows.
 *   2. `lock_memory`, which prefaults and locks what exists by then.
 *   3. `pin_current_thread`, so the loop owns one core.
 *   4. `corral_xla_threads`, which needs the `Runtime` to exist -- XLA's pools
 *      are created with the client, so call this after loading, not before.
 *   5. `cpu_dma_latency`, held for the process lifetime by @p dma.
 *   6. `set_realtime_priority` **last**, so that loading, warm-up and every
 *      allocation that comes with them do not run at real-time priority where
 *      a long operation would starve the rest of the machine.
 *
 * @param dma          Kept alive by the caller; closing it releases the C-state
 *                     constraint.
 * @param chosen_cpu   Set to the CPU that was pinned to, or -1.  Optional.
 * @return One `Step` per helper, in the order above.
 */
inline std::vector<Step> apply_hardening(const HostEnv& env,
                                         const HardeningOptions& options,
                                         DmaLatencyHold& dma,
                                         int* chosen_cpu) {
  auto from_status = [](const char* name, const pjrt::rt::Status& status) {
    return Step{name, status.ok, status.detail};
  };

  std::vector<Step> steps;
  steps.reserve(6);

  steps.push_back(options.malloc_tune
                      ? from_status("harden_malloc", pjrt::rt::harden_malloc())
                      : Step{"harden_malloc", false, "not requested"});

  steps.push_back(options.mlock
                      ? from_status("lock_memory", pjrt::rt::lock_memory())
                      : Step{"lock_memory", false, "not requested"});

  std::string why;
  const int cpu = choose_cpu(env, options.cpu, why);
  if (chosen_cpu != nullptr) {
    *chosen_cpu = cpu;
  }
  steps.push_back(cpu >= 0 ? from_status("pin_current_thread",
                                         pjrt::rt::pin_current_thread(cpu))
                           : Step{"pin_current_thread", false, why});

  const std::vector<int> others = cpus_except(env.affinity, cpu);
  if (!options.corral) {
    steps.push_back(Step{"corral_xla_threads", false, "not requested"});
  } else if (others.empty()) {
    steps.push_back(Step{"corral_xla_threads", false,
                         "no cpu left to move them to (affinity mask is one "
                         "cpu wide)"});
  } else {
    steps.push_back(from_status("corral_xla_threads",
                                pjrt::rt::corral_xla_threads(others)));
  }

  steps.push_back(options.dma_latency
                      ? dma.acquire()
                      : Step{"cpu_dma_latency", false, "not requested"});

  steps.push_back(
      options.rt_priority > 0
          ? from_status("set_realtime_priority",
                        pjrt::rt::set_realtime_priority(options.rt_priority))
          : Step{"set_realtime_priority", false, "not requested"});

  return steps;
}
// docs: end rt-harden

}  // namespace cjfc
