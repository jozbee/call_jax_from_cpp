# CLAUDE.md

**Read `docs/` before changing anything here.** It is a primer written for
agents: what was established by measurement rather than documentation, and the
mistakes made getting there. Start with `docs/README.md`.

Two rules that override normal instincts on this codebase:

1. **Never report a latency number measured on a busy machine.** A concurrent
   build does not add noise, it invalidates the result (measured: p50 2.4x
   high, max/p50 4.4 instead of 1.1). Check `/proc/loadavg` first.
   `docs/02-measurement.md` has the rest of the traps.
2. **The goal is the tail, not the mean** — p99.9/p50 and rare spikes. Do not
   substitute an average-latency objective.

Use `pjrt::Runtime` / `pjrt::Function` (`src/pjrt_exec/runtime.hpp`) for
anything in a call path. The legacy `Client`/`Buffer`/`AOTComputation` wrappers
allocate per call and exist only as a benchmark baseline until `eng/comp`
migrates.
