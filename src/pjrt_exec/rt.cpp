#include "pjrt_exec/rt.hpp"

#include <cstring>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <fstream>
#endif

namespace pjrt::rt {
namespace {

#if defined(__linux__)
Status errno_failure(const char* what) {
  return Status{false, std::string(what) + ": " + std::strerror(errno)};
}
#else
Status unsupported() { return Status{false, "not supported on this platform"}; }
#endif

}  // namespace

#if defined(__linux__)

Status lock_memory(std::size_t prefault_bytes) {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    return errno_failure("mlockall");
  }
  // Grow the heap once and touch every page, so the loop's arena is already
  // resident and already owned by this process.
  if (prefault_bytes > 0) {
    void* block = std::malloc(prefault_bytes);
    if (block != nullptr) {
      std::memset(block, 0, prefault_bytes);
      std::free(block);
    }
  }
  return Status{true, "mlockall(MCL_CURRENT|MCL_FUTURE)"};
}

Status harden_malloc() {
  // -1 disables trimming outright: never hand the top of the heap back.
  const bool trim = mallopt(M_TRIM_THRESHOLD, -1) == 1;
  // Serve every size from the heap rather than fresh mmap/munmap pairs, whose
  // page faults land inside whichever call happens to trigger them.
  const bool mmap_off = mallopt(M_MMAP_MAX, 0) == 1;
  // One arena: this path is single-threaded by design.
  const bool arenas = mallopt(M_ARENA_MAX, 1) == 1;
  if (!trim && !mmap_off && !arenas) {
    return Status{false, "mallopt had no effect (non-glibc allocator?)"};
  }
  return Status{true, "M_TRIM_THRESHOLD=-1 M_MMAP_MAX=0 M_ARENA_MAX=1"};
}

Status pin_current_thread(const std::vector<int>& cpus) {
  if (cpus.empty()) {
    return Status{false, "no cpus given"};
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  std::string listed;
  for (int cpu : cpus) {
    CPU_SET(cpu, &set);
    listed += (listed.empty() ? "" : ",") + std::to_string(cpu);
  }
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    return errno_failure("pthread_setaffinity_np");
  }
  return Status{true, "pinned to cpu " + listed};
}

Status pin_current_thread(int cpu) {
  return pin_current_thread(std::vector<int>{cpu});
}

Status set_realtime_priority(int priority) {
  sched_param param = {};
  param.sched_priority = priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
    return errno_failure(
        "pthread_setschedparam(SCHED_FIFO) -- needs CAP_SYS_NICE");
  }
  return Status{true, "SCHED_FIFO priority " + std::to_string(priority)};
}

Status corral_xla_threads(const std::vector<int>& cpus) {
  if (cpus.empty()) {
    return Status{false, "no cpus given"};
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu : cpus) {
    CPU_SET(cpu, &set);
  }

  DIR* dir = opendir("/proc/self/task");
  if (dir == nullptr) {
    return errno_failure("opendir(/proc/self/task)");
  }

  // Skip the caller's own thread, by id rather than by name: it was just
  // pinned where it wants to be, and a name test cannot tell it from a pool
  // thread when the executable's own name carries "XLA".
  const long self_tid = syscall(SYS_gettid);

  int moved = 0;
  int seen = 0;
  while (const dirent* entry = readdir(dir)) {
    if (entry->d_name[0] == '.') {
      continue;
    }
    const std::string tid = entry->d_name;
    if (std::atol(tid.c_str()) == self_tid) {
      continue;
    }
    std::ifstream comm_file("/proc/self/task/" + tid + "/comm");
    std::string comm;
    if (!comm_file || !std::getline(comm_file, comm)) {
      continue;
    }
    // A substring test, not a prefix: TSL prefixes the names it sets, so the
    // pool threads are "tf_XLAEigen" at the pinned version and were "XLAEigen"
    // before it.  Verified against /proc/<pid>/task/*/comm of example 04.
    if (comm.find("XLA") == std::string::npos) {
      continue;
    }
    ++seen;
    // sched_setaffinity takes a kernel tid, which is what these entries are.
    if (sched_setaffinity(std::atoi(tid.c_str()), sizeof(set), &set) == 0) {
      ++moved;
    }
  }
  closedir(dir);

  if (seen == 0) {
    return Status{false, "no XLA worker threads found (client not created?)"};
  }
  return Status{moved == seen, "moved " + std::to_string(moved) + " of " +
                                   std::to_string(seen) + " XLA threads"};
}

std::string describe_environment() {
  const auto setting = [](const char* label, const char* path) {
    std::ifstream file(path);
    std::string value;
    if (!file || !std::getline(file, value)) {
      value = "unavailable";
    }
    return std::string(label) + ": " + value + "\n";
  };

  std::string out;
  out += setting("governor",
                 "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  out += setting("isolated cpus", "/sys/devices/system/cpu/isolated");
  out += setting("nohz_full", "/sys/devices/system/cpu/nohz_full");
  out += setting("transparent hugepages",
                 "/sys/kernel/mm/transparent_hugepage/enabled");

  rlimit limit = {};
  if (getrlimit(RLIMIT_RTPRIO, &limit) == 0) {
    out += "RLIMIT_RTPRIO: " + std::to_string(limit.rlim_cur) + "\n";
  }
  return out;
}

#else  // not Linux

Status lock_memory(std::size_t) { return unsupported(); }
Status harden_malloc() { return unsupported(); }
Status pin_current_thread(int) { return unsupported(); }
Status pin_current_thread(const std::vector<int>&) { return unsupported(); }
Status set_realtime_priority(int) { return unsupported(); }
Status corral_xla_threads(const std::vector<int>&) { return unsupported(); }

std::string describe_environment() {
  return "real-time hardening is Linux-only; this platform is for development "
         "and correctness, not latency numbers\n";
}

#endif

}  // namespace pjrt::rt
