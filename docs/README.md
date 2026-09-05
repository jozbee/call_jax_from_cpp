# Primer for agents working on this repo

Read this directory before changing anything under `src/`, `tools/`, or
`third_party/xla`. It records what was established by measurement rather than
by reading documentation, and the mistakes that were actually made getting
there. The top-level `README.md` is the user-facing document; these notes are
the engineering context behind it.

Read in order:

| | |
|---|---|
| [01-orientation.md](01-orientation.md) | What the project is for, how the pieces fit, where to look |
| [02-measurement.md](02-measurement.md) | **Read before producing any latency number.** The traps invalidate results silently |
| [03-runtime-internals.md](03-runtime-internals.md) | Verified PJRT/XLA:CPU behaviour and why the hot path is shaped the way it is |
| [04-xla-fork.md](04-xla-fork.md) | The two patches carried in the fork, and what a rebase must preserve |
| [05-results.md](05-results.md) | The numbers, what moved them, and what conclusively did not |
| [06-open-threads.md](06-open-threads.md) | Unfinished work, environment caveats, things that will bite you |

## The one-paragraph version

A JAX function is exported ahead of time to a serialized PJRT executable
(`.binpb`) plus a JSON sidecar, and called from C++ through the PJRT **C** API.
The application is nonlinear MPC in a control loop, so the figure of merit is
the **worst call in a million, not the average**. XLA:CPU is not the source of
the jitter that motivated this work — the old C++ wrapper was, by rebuilding
device buffers on every call. `pjrt::Runtime` / `pjrt::Function`
(`src/pjrt_exec/runtime.hpp`) is the fix: load once, call many, allocate
nothing in the steady state.

## Rules of thumb that hold across this codebase

1. **Do not report a latency number measured on a busy machine.** It is not
   noisy, it is wrong. See [02](02-measurement.md).
2. **Prove behaviour against the plugin, don't infer it from headers.** Several
   plausible-sounding PJRT facts turned out false here (and several
   implausible ones true). The pattern that works: write a probe, run it.
3. **Unknown PJRT create options are silently ignored.** That is what makes the
   fork patch safe: dropping it costs performance, never correctness.
4. **`.binpb` artifacts are locked to the exporting architecture.** Loading
   relinks embedded machine code; it never recompiles.
5. **The steady-state call path must not allocate, lock, log, or flush.** All
   four were present in the original wrapper.
