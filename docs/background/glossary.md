# Glossary

*Two or three lines per term and one link each. A guide links a term on its
first use, so no page has to define it inline.*

```{glossary}
absolute sleep
  Sleeping until a point in time on `CLOCK_MONOTONIC` rather than for a
  duration, so one wake-up's lateness does not feed into the next period.
  `clock_nanosleep` with `TIMER_ABSTIME`; {doc}`realtime-linux`.

allocation census
  A count of every allocation made while a guard was armed, attributed to the
  module that made it, kept by a preloaded interposer. The wrapper's own count
  must be zero; XLA's is reported. {doc}`/api/cpp/alloc-guard`.

arena
  The 64-byte-aligned buffer a `Function` owns for one input or output,
  wrapped once as a zero-copy PJRT buffer so that a call transfers nothing.
  {doc}`/guides/calling`.

artifact
  The three files an export writes under one base path: the compiled
  executable (`.binpb`), the StableHLO bytecode (`.mlirbc`) and the sidecar
  (`.json`). {doc}`/guides/exporting`.

C-state
  An idle state of a core. Deeper states save more power and take longer to
  leave; a sleep between calls can put the core into one. {doc}`realtime-linux`.

context switch
  The core moving from one thread to another: *voluntary* when the thread
  blocked, *involuntary* when it was preempted. On a pinned real-time thread,
  an involuntary one means something else wanted the core. {doc}`realtime-linux`.

cpu_dma_latency
  `/dev/cpu_dma_latency`, a PM QoS file. Held open with a zero written to it,
  it keeps every core out of deep C-states until it is closed. Needs root.
  {doc}`realtime-linux`.

cycle time
  From waking to being ready to sleep again: the call and everything around it
  in the loop body. {doc}`latency`.

deadline miss
  A cycle whose work ended after the next release; the *overrun* is by how
  much. PJRT cannot cancel a running computation, so an overrun is late data,
  never a cancelled call. {doc}`latency`.

feedback
  Copying one cycle's outputs into the next cycle's inputs, between calls: the
  receding-horizon step of a controller. In the examples,
  `cjfc::workload::feedback`. {doc}`/api/cpp/examples`.

governor
  The cpufreq policy that decides a core's clock. `performance` holds it up;
  `powersave` lets it fall between calls. {doc}`realtime-linux`.

inline execution
  Running the computation on the calling thread instead of dispatching it to a
  pool. A plugin option the runtime asks for; `synchronous_mode()` says whether
  it was granted. {doc}`xla-and-pjrt`.

isolated CPU
  A core removed from the scheduler's load balancing (`isolcpus=`), usually
  also without the timer tick (`nohz_full=`), so that only what is pinned there
  runs there. {doc}`realtime-linux`.

memory locking
  `mlockall`: pinning every page of the process in RAM so that none is swapped
  out and none has to be faulted in. Needs `RLIMIT_MEMLOCK`. {doc}`realtime-linux`.

nohz_full
  The kernel parameter that stops the periodic timer tick on a core with a
  single runnable task. {doc}`realtime-linux`.

page fault
  The kernel mapping a page on its first touch (*minor*) or reading it back
  from disk (*major*). Either one inside a call is an outlier. {doc}`realtime-linux`.

period jitter
  This wake-up minus the previous one, minus the period. Signed: waking early
  is as much a defect as waking late. {doc}`latency`.

PJRT
  The C API through which JAX, and this project, drive a compiler and runtime:
  clients, devices, buffers, executables, events. {doc}`xla-and-pjrt`.

PJRT plugin
  A shared object implementing PJRT for one backend. This project `dlopen`s
  the CPU plugin at run time; nothing links against it.
  {doc}`/getting-started/installation`.

PREEMPT_RT
  The kernel configuration that makes almost all kernel code preemptible,
  bounding how long a real-time thread waits on the kernel. {doc}`realtime-linux`.

rlimits
  `RLIMIT_RTPRIO`, whether this process may ask for `SCHED_FIFO`, and
  `RLIMIT_MEMLOCK`, how much it may lock. In a container, `ulimits`.
  {doc}`/guides/realtime`.

SCHED_FIFO
  The real-time scheduling class: a thread runs ahead of every normal-class
  thread until it blocks. `sched(7)`; {doc}`realtime-linux`.

sidecar
  The `.json` beside the executable: the signature, the dtypes and shapes, the
  fingerprint. The only description of the inputs that exists, so it is
  cross-checked at load. {doc}`/api/artifact-format`.

StableHLO
  The portable operation set JAX lowers to and XLA compiles. The `.mlirbc`
  file is its bytecode, and the fallback when the compiled executable cannot
  be loaded. {doc}`xla-and-pjrt`.

tail
  The bad calls: `p99.9`, `max`, and their ratios to `p50`. The figure of
  merit for this project. {doc}`latency`.

thread pool
  XLA's worker threads, started when the client is created and named after
  XLA. With inline execution they should be idle, and they can still wake on
  the loop's core unless moved. {doc}`/guides/realtime`.

wake-up latency
  How late the sleep returned relative to the deadline it was given: the
  scheduler's and the idle state's contribution. {doc}`latency`.

warm-up
  Calls made after loading and before measuring, so that page faults and lazy
  initialization are paid once. The first call after a load is always the
  slowest. {doc}`/guides/calling`.

XLA
  The compiler behind JAX. Its CPU backend emits machine code for the
  exporting host; its thunk runtime walks the compiled program at run time and
  allocates as it goes, inside the plugin. {doc}`xla-and-pjrt`.

zero-copy buffer
  A PJRT buffer that aliases the caller's memory instead of copying it. A
  `Function` wraps each arena once, and writes made between calls are seen by
  the next call. {doc}`/guides/calling`.
```
