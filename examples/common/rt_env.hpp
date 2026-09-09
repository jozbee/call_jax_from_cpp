/**
 * @file rt_env.hpp
 * @brief What the host is willing to give a real-time loop, and the steps that
 *        ask for it.
 *
 * `pjrt::rt` provides the hardening primitives; this is the layer above them.
 * It reads the settings the host was booted with, chooses the loop's CPU,
 * applies the primitives in the one safe order, and reports every step instead
 * of failing: a control process that refuses to start because it could not get
 * `SCHED_FIFO` is worse than one that starts and says so.
 *
 * The audit uses the same vocabulary as `tools/rt_check.sh`, so a report from
 * a run and one from the script read side by side.  None of it is required
 * for correct results, only for believable latency numbers, which is why every
 * example prints it.
 *
 * Linux only in substance; elsewhere every field reads "unavailable" and every
 * step skips, so the examples still build on a development machine.
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

// call_jax_from_cpp: helpers the examples share; not the library
namespace cjfc {

/**
 * @brief The host settings that decide whether a bounded computation finishes
 *        on time.
 *
 * Unreadable values are `"unavailable"` or -1 rather than an exception: a
 * container that hides half of `/sys` is a normal place to run, not an error.
 */
struct HostEnv {
  /// `uname -r`, or "unavailable".
  std::string kernel;

  /// PREEMPT_RT, from `/sys/kernel/realtime` or the `uname` version string.
  bool preempt_rt = false;

  /// `/.dockerenv` exists, or `$CJFC_IN_CONTAINER` is set.  The capability and
  /// rlimit answers below usually differ inside one.
  bool in_container = false;

  /// Online CPUs, or -1.
  int cpus_online = -1;

  /// `isolcpus=` as `/sys/devices/system/cpu/isolated` reports it.  Empty on a
  /// host booted without it, which is why `choose_cpu` explains itself.
  std::vector<int> isolated;

  /// `/sys/devices/system/cpu/nohz_full`: the CPUs the timer tick leaves alone.
  std::vector<int> nohz_full;

  /// This thread's affinity mask.  Its intersection with `isolated` is what can
  /// be pinned to; `taskset` or a container's cpuset narrows it.
  std::vector<int> affinity;

  /// `scaling_governor` for cpu0.  Anything but "performance" means the clock
  /// can change under the measurement.
  std::string governor;

  /// The selected transparent-hugepage mode, unbracketed.  `khugepaged` stalls
  /// faulting threads under "always".
  std::string thp;

  /// `/sys/devices/system/cpu/smt/control`: "on", "off", "notsupported".
  std::string smt;

  /// `RLIMIT_RTPRIO` soft limit; -1 for unlimited.  0 means no `SCHED_FIFO`.
  long rlimit_rtprio = -1;

  /// `RLIMIT_MEMLOCK` soft limit in bytes; -1 for unlimited, which is what
  /// `mlockall` wants.
  long rlimit_memlock = -1;

  /// `/proc/sys/kernel/sched_rt_runtime_us`: microseconds per period a
  /// real-time thread may run before it is throttled.  -1 is both the file's
  /// own "disabled" value and what an unreadable file reports.
  long rt_runtime_us = -1;

  /// Whether `/dev/cpu_dma_latency` can be opened for writing, i.e. whether
  /// this process can keep the cores out of deep C-states.
  bool cpu_dma_latency_writable = false;

  double loadavg1 = -1.0;   ///< One-minute load average, or -1.
  double loadavg5 = -1.0;   ///< Five-minute load average, or -1.
  double loadavg15 = -1.0;  ///< Fifteen-minute load average, or -1.

  /// `loadavg1 > 1.0`.  A number measured while this is true is not noisy, it
  /// is wrong; `docs/developer/measurement.md` has the figures.
  bool busy = false;
};

/**
 * @brief Expand a kernel CPU list -- `"2-5,8"` -> `{2,3,4,5,8}`.
 *
 * Empty and the literal `"(null)"` that `nohz_full` prints when unset both
 * mean "nothing".  Anything unparseable is skipped rather than thrown: this is
 * diagnostic input, and a strange `/sys` file should not take down a control
 * process.
 */
inline std::vector<int> parse_cpulist(const std::string& text) {
  std::vector<int> cpus;
  if (text == "(null)") {
    return cpus;
  }
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t end = text.find(',', pos);
    if (end == std::string::npos) {
      end = text.size();
    }
    const std::string item = text.substr(pos, end - pos);
    pos = end + 1;

    const std::size_t dash = item.find('-');
    try {
      if (dash == std::string::npos) {
        cpus.push_back(std::stoi(item));
        continue;
      }
      const int lo = std::stoi(item.substr(0, dash));
      const int hi = std::stoi(item.substr(dash + 1));
      // A malformed range must not turn into a multi-gigabyte vector.
      if (hi >= lo && hi - lo < 4096) {
        for (int cpu = lo; cpu <= hi; ++cpu) {
          cpus.push_back(cpu);
        }
      }
    } catch (const std::exception&) {
      continue;
    }
  }
  return cpus;
}

/// @brief Whether @p cpu appears in @p cpus.
inline bool contains_cpu(const std::vector<int>& cpus, int cpu) {
  return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
}

/// @brief Every CPU in @p mask except @p cpu: where XLA's worker threads go
///        once the loop has claimed one core.
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
 * `"auto"` prefers a CPU that is both isolated and `nohz_full`, falls back to
 * merely isolated, and otherwise stays unpinned: pinning to a CPU the rest of
 * the system also uses trades one source of jitter for another, so an
 * unhardened host is left alone unless the caller insists with a number.
 *
 * @param env   The host audit: which CPUs are isolated and `nohz_full`.
 * @param spec  `"auto"`, `"none"`, or a CPU number.
 * @param why   The reason, filled in on success too; it becomes the detail
 *              line of the pinning step.
 * @return The CPU to pin to, or -1 to leave the thread where it is.
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
    // A single-CPU affinity mask means somebody pinned this process before it
    // started (`taskset`, a cpuset): report the CPU rather than a skip.
    if (env.affinity.size() == 1) {
      why = "already pinned before launch (affinity is a single cpu)";
      return env.affinity.front();
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
  std::string name;    ///< The helper that was asked, e.g. "lock_memory".
  bool ok = false;     ///< Whether it took effect.
  std::string detail;  ///< What it did, or why it did not.
};

/// @brief Print one step as `  [ok  ] name: detail` or `  [skip] name: detail`.
inline void print_step(const Step& step) {
  std::printf("  [%s] %s%s%s\n", step.ok ? "ok  " : "skip", step.name.c_str(),
              step.detail.empty() ? "" : ": ", step.detail.c_str());
}

/**
 * @brief Holds `/dev/cpu_dma_latency` open at 0 microseconds.
 *
 * The kernel applies the constraint only while the descriptor is open, so
 * writing 0 and closing the file achieves nothing.  This object owns the
 * descriptor and the caller keeps it alive for the whole run.  Needs write
 * access to the device, normally root; without it `acquire()` returns a
 * skipped step and the run continues.
 */
class DmaLatencyHold {
 public:
  DmaLatencyHold() = default;

  /// Releases the constraint.
  ~DmaLatencyHold() { release(); }

  DmaLatencyHold(const DmaLatencyHold&) = delete;
  DmaLatencyHold& operator=(const DmaLatencyHold&) = delete;

  /// @brief Open the device and write a 0 microsecond target, keeping the
  ///        descriptor.
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
 * @brief Page faults and context switches, per thread where that exists.
 *
 * A major fault is a disk read in the middle of a control cycle; an
 * involuntary context switch is the scheduler taking the core away.  Take one
 * before the loop and one after, and subtract.
 */
struct Rusage {
  long minflt = -1;  ///< Minor faults: a page mapped without touching disk.
  long majflt = -1;  ///< Major faults: a page that had to be read in.
  long nvcsw = -1;   ///< Voluntary context switches: the thread waited.
  long nivcsw = -1;  ///< Involuntary ones: the thread was preempted.

  /// @brief Sample the counters now.  Scoped to the calling thread where
  ///        `RUSAGE_THREAD` exists, so XLA's pool threads do not contribute;
  ///        `scope()` says which.
  static Rusage now() noexcept {
    Rusage r;
#if CJFC_HAVE_RUSAGE
#if defined(RUSAGE_THREAD)
    constexpr int who = RUSAGE_THREAD;
#else
    constexpr int who = RUSAGE_SELF;
#endif
    rusage ru{};
    if (getrusage(who, &ru) == 0) {
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

  /// @brief Field-wise difference: what one loop cost.
  Rusage operator-(const Rusage& before) const noexcept {
    Rusage d;
    d.minflt = minflt - before.minflt;
    d.majflt = majflt - before.majflt;
    d.nvcsw = nvcsw - before.nvcsw;
    d.nivcsw = nivcsw - before.nivcsw;
    return d;
  }
};

#if defined(__linux__)
namespace detail {

/// First line of @p path, or empty when it cannot be read.
inline std::string read_first_line(const char* path) {
  std::ifstream file(path);
  std::string line;
  if (file && std::getline(file, line)) {
    return line;
  }
  return std::string();
}

/// First line of @p path, or "unavailable", the report's spelling for it.
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

/// The bracketed choice out of a sysfs menu line: "always [madvise] never" is
/// the mode plus the menu, and only the mode is a fact about this host.
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
    env.preempt_rt =
        std::string(uts.version).find("PREEMPT_RT") != std::string::npos;
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

  // Opening changes nothing; only a write to a descriptor that stays open
  // does (see DmaLatencyHold).
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

#else  // Not Linux.

/// @brief Everything unavailable: a plausible-looking value would be worse
///        than none.
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
 * `rt_priority` defaults to 0: `SCHED_FIFO` is opt-in because a real-time
 * thread that misbehaves takes the machine with it, and because a run that
 * quietly got it and one that quietly did not are indistinguishable in the
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

// docs: begin rt-harden-impl
/**
 * @brief Apply the hardening steps in the one order that is safe, and report
 *        each.
 *
 * Never fails: a step that does not take effect becomes a skipped `Step` with
 * the reason in it, and the program runs anyway.  The order matters:
 * `harden_malloc` before anything allocates in bulk, `corral_xla_threads`
 * after the `Runtime` exists because XLA's pools are created with the client,
 * and `set_realtime_priority` last so that loading and warm-up never run at
 * real-time priority.  `docs/developer/realtime-notes.md` has each step.
 *
 * @param env         The host audit; a step whose precondition is absent here
 *                    is reported as skipped rather than attempted.
 * @param options     Which of the six steps to run.
 * @param dma         Kept alive by the caller; closing it releases the C-state
 *                    constraint.
 * @param chosen_cpu  Set to the CPU pinned to, or -1.  Optional.
 * @return One `Step` per helper, in the order applied.
 */
inline std::vector<Step> apply_hardening(const HostEnv& env,
                                         const HardeningOptions& options,
                                         DmaLatencyHold& dma, int* chosen_cpu) {
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
// docs: end rt-harden-impl

}  // namespace cjfc
