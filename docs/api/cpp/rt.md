# rt

`pjrt::rt` is the optional real-time hardening layer for the thread that runs
the control loop. Nothing here changes what is computed; it changes how
reliably the operating system lets the computation finish on time. Each
function is independent, reports whether it took effect, and is a no-op
returning `false` on platforms that do not provide it — macOS, mainly, where
these are development conveniences rather than a deployment target.

Nothing fails the program. A {cpp:struct}`~pjrt::rt::Status` comes back with
`ok` false and a `detail`
saying why, because the useful behaviour for a process that could not raise its
own priority is to run anyway and say so in its startup log, not to refuse to
start. That is also why these are separate calls rather than one
`harden_everything()`: `mlockall` and `SCHED_FIFO` need privileges that
`sched_setaffinity` does not, and a container that grants one may not grant the
other.

The mistake people make is expecting these to be sufficient. They are the
process's half of the bargain; the host has to cooperate as well, with a
performance governor, isolated cores, and deep C-states disabled.
`tools/rt_check.sh` audits that side, read-only, and its output belongs
alongside any latency number you record — a p99.9 means little without
knowing whether the governor was on `powersave` at the time.
{doc}`/guides/realtime` maps each helper to the problem it removes;
{doc}`/background/realtime-linux` explains the mechanisms.

## Order of operations

A control process calls these once, around loading its `Function` and before
entering the loop. The order matters: {cpp:func}`~pjrt::rt::harden_malloc`
before the `Runtime`, so the heap it configures is the one startup grows;
{cpp:func}`~pjrt::rt::corral_xla_threads` after it, because it walks
`/proc/self/task` looking for threads XLA names when the client is created;
priority last, so the setup work itself never runs on a real-time thread.

Example 03's setup, which is the whole of it:

```{literalinclude} ../../../examples/03_minimal/minimal.cpp
:language: cpp
:start-after: docs: begin minimal-setup
:end-before: docs: end minimal-setup
```

## Status

```{doxygenstruct} pjrt::rt::Status
```

Two fields and an explicit conversion, so a call site reads as a condition:

```cpp
struct Status {
  bool ok = false;
  std::string detail;
  explicit operator bool() const;
};

if (const auto s = pjrt::rt::set_realtime_priority(80); !s) {
  std::fprintf(stderr, "not real-time: %s\n", s.detail.c_str());
}
```

`detail` is filled in on success as well as on failure, and it is what
{cpp:func}`~pjrt::rt::describe_environment` summarizes.

## Memory

```{doxygenfunction} pjrt::rt::lock_memory
```

```{doxygenfunction} pjrt::rt::harden_malloc
```

`lock_memory` needs `RLIMIT_MEMLOCK` raised — in a container,
`--ulimit memlock=-1`. `harden_malloc` needs nothing, and is the cheaper half
of the pair: trimming the heap means the *next* allocation has to fault it back
in from the kernel, which turns a routine call into an outlier.

## Placement

```{doxygenfunction} pjrt::rt::pin_current_thread(int)
```

```{doxygenfunction} pjrt::rt::pin_current_thread(const std::vector<int>&)
```

```{doxygenfunction} pjrt::rt::corral_xla_threads
```

XLA starts its pools when the client is created and names those threads
(`tf_XLAEigen…` at the pinned XLA version — the prefix is TSL's), which is
what makes them findable afterwards by walking `/proc/self/task`. With inline execution the pools should be idle;
corralling them keeps them from waking up on the core the control loop is
using. Neither call needs a privilege beyond permission to change the calling
process's own affinity.

## Priority

```{doxygenfunction} pjrt::rt::set_realtime_priority
```

This one needs `CAP_SYS_NICE`, or a raised `RLIMIT_RTPRIO`; in a container, run
with `--cap-add=SYS_NICE --ulimit rtprio=99`. A real-time thread that spins
forever will starve the machine, which is why it is opt-in and why it matters
that the computation being run is bounded.

## Reporting

```{doxygenfunction} pjrt::rt::describe_environment
```

Log it once, next to {cpp:func}`~pjrt::Runtime::describe`. Between them they
answer most of the questions a surprising latency number raises before anyone
has to reproduce it.
