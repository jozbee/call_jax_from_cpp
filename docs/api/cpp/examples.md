# Example helpers (`cjfc`)

`cjfc` — call_jax_from_cpp — is the layer the examples share, under
`examples/common/`. It is not part of the library's API or ABI: nothing in
`include/pjrt_exec` depends on it, it is header-only, and it is written to be
copied into a program that wants the same thing. Five headers are generic —
the flag parser, the absolute-time sleep and stop flag, the host audit and the
hardening order, the JSON report, the enum names — and one is the trajopt
{ref}`workload's I/O contract <cjfc-workload>`, which is what examples 02 and
04 and the benchmark run and what a program with its own function replaces.

Members are listed explicitly, as on every reference page. A name shown in an
example that is missing here is a build failure, not an omission.

## `cli.hpp`

Flag parsing for the example programs: known flags, typed getters, `--help`,
and a refusal on anything unknown. Example 03 takes positional arguments and
needs none of it.

```{doxygenfile} cli.hpp
:sections: briefdescription detaileddescription
```

```{doxygenclass} cjfc::Cli
:members: Cli, help, print_usage, flag, get, get_long, get_size
```

## `periodic.hpp`

The absolute sleep and the stop flag a periodic loop needs. Example 03 inlines
the same two functions so that it stays self-contained; example 04 and the
benchmark include this.

```{doxygenfile} periodic.hpp
:sections: briefdescription detaileddescription
```

```{doxygenvariable} cjfc::kNsPerSec
```

```{doxygenfunction} cjfc::now_ns
```

```{doxygenfunction} cjfc::sleep_until
```

```{doxygenfunction} cjfc::install_stop_handlers
```

```{doxygenfunction} cjfc::stopping
```

## `rt_env.hpp`

What the host is willing to give a real-time loop, and the steps that ask for
it. {cpp:func}`~cjfc::apply_hardening` is the six steps in the one safe order,
each reported as a {cpp:struct}`~cjfc::Step`; {doc}`/guides/realtime` says
what each buys.

```{doxygenfile} rt_env.hpp
:sections: briefdescription detaileddescription
```

```{doxygenstruct} cjfc::HostEnv
:members: kernel, preempt_rt, in_container, cpus_online, isolated, nohz_full, affinity, governor, thp, smt, rlimit_rtprio, rlimit_memlock, rt_runtime_us, cpu_dma_latency_writable, loadavg1, loadavg5
```

```{doxygenstruct} cjfc::Step
:members: name, ok, detail
```

```{doxygenstruct} cjfc::HardeningOptions
:members: malloc_tune, mlock, corral, cpu, rt_priority, dma_latency
```

```{doxygenstruct} cjfc::Rusage
:members: minflt, majflt, nvcsw, nivcsw, now, scope
```

```{doxygenclass} cjfc::DmaLatencyHold
:members: acquire, held, release
```

```{doxygenfunction} cjfc::detect_host_env
```

```{doxygenfunction} cjfc::choose_cpu
```

```{doxygenfunction} cjfc::apply_hardening
```

```{doxygenfunction} cjfc::print_step
```

```{doxygenfunction} cjfc::parse_cpulist
```

```{doxygenfunction} cjfc::contains_cpu
```

```{doxygenfunction} cjfc::cpus_except
```

(cjfc-workload)=

## `workload.hpp`

The I/O contract of the function `examples/02_trajopt/export.py` exports:
which input is which, how to start, how to feed one cycle's outputs into the
next cycle's inputs. A program with its own function replaces this header and
keeps the rest of the layer.

```{doxygenfile} workload.hpp
:sections: briefdescription detaileddescription
```

```{doxygenenum} cjfc::workload::In
```

```{doxygenenum} cjfc::workload::Out
```

```{doxygenvariable} cjfc::workload::kNominalParams
```

```{doxygenvariable} cjfc::workload::kWeights
```

```{doxygenstruct} cjfc::workload::Dims
:members: nx, nu, h, np, n_iters
```

```{doxygenfunction} cjfc::workload::check_signature
```

```{doxygenfunction} cjfc::workload::init_inputs
```

```{doxygenfunction} cjfc::workload::write_reference
```

```{doxygenfunction} cjfc::workload::feedback
```

## `report.hpp`

The JSON report and the shared exit codes — the table every example and the
benchmark agree on.

```{doxygenfile} report.hpp
:sections: briefdescription detaileddescription
```

```{doxygentypedef} cjfc::json
```

```{doxygenvariable} cjfc::kExitOk
```

```{doxygenvariable} cjfc::kExitError
```

```{doxygenvariable} cjfc::kExitCorrectness
```

```{doxygenvariable} cjfc::kExitAllocGate
```

```{doxygenvariable} cjfc::kExitGuardMissing
```

```{doxygenfunction} cjfc::summary_json
```

```{doxygenfunction} cjfc::histogram_json
```

```{doxygenfunction} cjfc::alloc_json
```

```{doxygenfunction} cjfc::host_json
```

```{doxygenfunction} cjfc::steps_json
```

```{doxygenfunction} cjfc::spec_json
```

```{doxygenfunction} cjfc::runtime_json
```

```{doxygenfunction} cjfc::print_summary
```

```{doxygenfunction} cjfc::write_json
```

```{doxygenfunction} cjfc::alloc_gate_exit_code
```

## `names.hpp`

Enum spellings the reports use.

```{doxygenfile} names.hpp
:sections: briefdescription detaileddescription
```

```{doxygenfunction} cjfc::sync_mode_name
```

```{doxygenfunction} cjfc::load_kind_name
```
