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
#include <cstdio>
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
Status unsupported() {
  return Status{false, "not supported on this platform"};
}
#endif

}  // namespace

#if defined(__linux__)

Status lock_memory(std::size_t prefault_bytes) {
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    return errno_failure("mlockall");
  }
  // Grow the heap once and touch every page, so the arena the loop will use is
  // already resident and already owned by this process.
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

  // The caller's own thread is never a candidate: it has just been pinned to
  // the core it wants, and corralling it onto `cpus` would undo that.  Skipped
  // by id rather than by name, because a name test cannot tell the loop thread
  // from a pool thread if the executable is itself called something with "XLA"
  // in it.
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
    // Any thread whose name carries "XLA" is one of XLA's; anything else here
    // is ours.  This was a prefix match on "XLAEigen"/"XLAPjRtCpuClient", and
    // it rotted silently: TSL prefixes the name it sets, so at the pinned XLA
    // version the pool threads are called "tf_XLAEigen" and the prefix match
    // found nothing on any host.  The step then reported "no XLA worker
    // threads found (client not created?)", which reads as "there was nothing
    // to move" rather than as a helper that had stopped working.  A substring
    // test survives that rename and still matches the un-prefixed names older
    // versions used.  Verified by reading /proc/<pid>/task/*/comm of a running
    // example 04: `--threads N` yields exactly N `tf_XLAEigen` threads,
    // under inline and asynchronous dispatch alike.
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
  auto read_first_line = [](const char* path) -> std::string {
    std::ifstream f(path);
    std::string line;
    if (f && std::getline(f, line)) {
      return line;
    }
    return "unavailable";
  };

  std::string out;
  out += "governor: " +
         read_first_line(
             "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") +
         "\n";
  out += "isolated cpus: " +
         read_first_line("/sys/devices/system/cpu/isolated") + "\n";
  out += "nohz_full: " + read_first_line("/sys/devices/system/cpu/nohz_full") +
         "\n";
  out += "transparent hugepages: " +
         read_first_line("/sys/kernel/mm/transparent_hugepage/enabled") + "\n";

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
