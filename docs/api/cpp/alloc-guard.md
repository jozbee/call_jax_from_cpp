# AllocGuard

Grepping the source for `malloc` proves nothing about what a linked binary
does at run time. OpenBLAS, libm and the C++ runtime all allocate behind the
caller's back, and an inlined `std::vector` growth is invisible to any static
check. The only trustworthy answer comes from interposing the allocator in the
real process, which is what `tests/support/malloc_guard.c` does.

Nothing links against that interposer. `pjrt::AllocGuard` resolves its markers
with `dlsym(RTLD_DEFAULT, ...)` and degrades to no-ops returning 0 when they
are absent, so one binary runs identically with and without the preload and a
benchmark decides at run time whether it has an allocation census to report.
`present()` is that decision; `total()` — allocations by the whole process
since it started, counted whether armed or not — is what makes a zero armed
count believable, because thousands there with zero while armed means the path
is clean, while zero there means the preload never took effect at all.

Be exact about what the gate is. A whole-process "zero allocations" claim is
not achievable here: XLA's thunk runtime allocates roughly **thousands per call times per
call**, inside the plugin, roughly one per StableHLO op, and none of that is
reachable through the PJRT C API. So each allocation made while armed is
attributed to the module it came from, and the number that must stay at zero is
`allocs_self()` — the wrapper's own allocations in the steady-state call path.
`make test-alloc` is that gate.

## Running the census

```console
$ make test-alloc
```

or, by hand, preloading the interposer into any binary that uses the guard:

```console
$ LD_PRELOAD=build/lib/malloc_guard.so ./build/bin/bench --iterations 200
```

On macOS the variable is `DYLD_INSERT_LIBRARIES`; the guard reports the right
one in its own "not preloaded" message.

## The guard

```{doxygenclass} pjrt::AllocGuard
:members: kClassSelf, kClassPlugin, kClassRuntime, AllocGuard, present, classified, arm, disarm, allocs, frees, total, allocs_self, allocs_plugin, allocs_runtime, report
```

`classified()` is false on macOS, where interposition works but the module
ranges the classifier needs are not built. The totals are still correct there;
only the per-module split is missing, which is why a census that matters runs
on Linux.

The three classes are exhaustive and mean what they say:

| Class | Counts allocations from |
|---|---|
| `kClassSelf` | the main executable and `libpjrt_exec` — **must be zero** |
| `kClassPlugin` | inside the PJRT CPU plugin |
| `kClassRuntime` | libc, libstdc++, LAPACK, the thread pool, everything else |

## Arming a scope

```{doxygenclass} pjrt::AllocGuardScope
:members: AllocGuardScope
```

Wrap exactly the region being claimed clean — the steady-state calls, after
warm-up. Including warm-up would fold in the page faults and the lazy
initialization that warm-up exists to pay for, and turn a clean path into a few
thousand allocations.

`report(stdout, iterations)` then prints the armed window as per-call figures,
which is the form the numbers in this project are quoted in.
