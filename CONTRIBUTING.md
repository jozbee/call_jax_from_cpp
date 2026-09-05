# Contributing

Thanks for looking. This is a small project with a narrow goal, so the bar for
a change is mostly about evidence rather than style.

## Before you start

Read the [developer guide](https://jozbee.github.io/call_jax_from_cpp/developer/index.html).
Most of what is surprising about this codebase was established by measurement
rather than by reading documentation, and that guide records both the finding
and the mistake that produced it. `AGENTS.md` is the same material compressed
to a page.

## Setting up

```console
uv sync --extra docs      # Python environment; uv comes from mise, and is not
                          # on PATH in a non-interactive shell
make plugin               # the prebuilt PJRT CPU plugin, sha256-verified
make && make test         # build and run the fast tests
```

Nothing links against the PJRT plugin — it is `dlopen`-ed at run time — so a
plain `make` never touches bazel or the network. `make plugin` is a separate
errand that only the *run* targets need, and a plugin supplied through
`$PJRT_CPU_PLUGIN` is just as good as one in `build/plugin`.

Everything also works in the container, which already has doxygen and bazel:

```console
docker compose -f docker/compose.yml run --rm dev
```

Never hand-type a version. `versions.env` is the single source of truth, the
Makefile and CMake and CI all read it, and every documentation page can use it
as a MyST substitution.

## Formatting

```console
make format
```

That runs `clang-format` over the C++ and the interposer (Google style, via
`--fallback-style`; there is no `.clang-format` in the tree, and adding one
later changes nothing about this command), then `ruff format` and
`ruff check --fix` over the Python. Lines are 80 columns on both sides. If
`clang-format` is not installed the C++ half is skipped with a note rather than
failing.

## Running the tests

```console
make test          # the fast suite: builds the examples, the C++ tests and the
                   # guard, then runs pytest
make test-slow     # the same, including the long campaigns (--runslow)
make test-rt       # the real-time gates; only meaningful on a tuned, idle host
make test-alloc    # the zero-wrapper-allocations gate
```

`make test-alloc` is the one to run for any change to the call path. It
preloads an allocator interposer and gates on the wrapper's own allocations
being zero. It is not a whole-process claim: XLA's thunk runtime allocates
thousands of times per call inside the plugin, which is not reachable through the
PJRT C API, and the census reports that number rather than hiding it.

## Changing the call path

Anything in `include/pjrt_exec` or `src/pjrt_exec` that runs per call has to
keep the steady state free of allocation, locks, logging and flushing.

Behaviour of the PJRT plugin should be **proved with a probe, not inferred from
the headers**. Several plausible-sounding facts about it turned out to be false
here, and several implausible ones true. Make sure the probe is sensitive to
what it perturbs: one early probe reported a confident negative because the
input it perturbed did not reach the output it checked.

## Measuring

Read
[the measurement guide](https://jozbee.github.io/call_jax_from_cpp/guides/measuring.html)
before producing a number. Two rules override normal instincts:

1. **Never report a latency number measured on a busy machine.** A concurrent
   build does not add noise to the result, it invalidates it: the same
   configuration measured during a bazel build reported a p50 2.4x high and a
   max/p50 of 4.4 instead of 1.1. Check `/proc/loadavg` first, and discard a
   contaminated run rather than correcting it.
2. **The goal is the tail, not the mean.** p99.9/p50, max/p50, and the worst
   call in a long campaign. An average-latency improvement is not the objective
   and should not be offered as one. A claim about how *often* spikes happen
   needs 20,000+ calls per configuration, interleaved in short rounds — a >2x
   outlier appears about once per 20,000 calls, and a sequential A/B drifts with
   CPU temperature by the same order as the effect being measured.

Keep `tools/rt_check.sh` output next to any numbers you record.

## Documentation

The site builds with warnings as errors, so a broken cross-reference or an
orphan page fails CI:

```console
make docs                 # needs doxygen on PATH
make docs-live            # rebuild and reload while editing
make docs-linkcheck       # verify every external link
```

Code in the documentation comes from marked regions in the real sources —
`// docs: begin <name>` … `// docs: end <name>`, pulled in with
`literalinclude` — never pasted. A pasted snippet is a copy that rots silently.
On the C++ reference pages, name every member explicitly: an undocumented
public member does not reach the Doxygen XML at all, so a directive that names
one is a build failure rather than an empty box.

## Submitting

Open an issue first for anything large. For a pull request: one idea per
commit, and a message that explains *why*.

- [ ] `make test` passes, and `make test-alloc` too if the call path changed.
- [ ] `make format` has been run.
- [ ] `make docs` builds clean under `-W`.
- [ ] Any snippet added to the docs comes from a marked region, not a paste.
- [ ] No version was hand-typed; `versions.env` is still the only place.
- [ ] If the change is meant to move a number: the before and the after, taken
      on an idle machine, with the `tools/rt_check.sh` output and the
      `/proc/loadavg` reading beside them.
- [ ] If it touches the XLA fork: the branch is pushed, the patch is
      regenerated under `third_party/patches/`, and the hash in `versions.env`
      matches both.
