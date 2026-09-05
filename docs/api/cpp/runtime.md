# Runtime

`pjrt::Runtime` owns three things for the life of the process: the `dlopen`-ed
PJRT plugin, the `PJRT_Api` function table it exports, and one client with one
addressable device. Every `pjrt::Function` loads its executable into that
client and issues its executions to that device.

Create exactly one and keep it alive. Creating a client starts XLA's thread
pools and its lazily-initialized statics, so destroying and recreating one
mid-run is a guaranteed latency spike; destroying one while a `Function` still
holds an executable is undefined. `Runtime` is also the object to share between
threads: it is read-only after construction, while a `Function` owns fixed
per-call storage and belongs to a single thread. The plugin itself is opened
`RTLD_NOW | RTLD_LOCAL` and never closed, because XLA leaves statics behind it
that outlive any client.

The mistake to avoid is assuming that `RuntimeOptions::synchronous` took
effect. Inline execution is reachable only through the `asynchronous` create
option — `PJRT_ExecuteOptions` has no execution-mode field — and as of the
pinned XLA commit the CPU plugin **validates create option names and fails
creation with `InvalidArgument` on one it does not recognize**, where older
versions ignored unknown options silently. So construction queries
`PJRT_Plugin_Attributes` first, which needs no client, sends only the options
this plugin admits to understanding, and retries without an option the plugin
still refuses. What actually happened is reported by
{cpp:func}`~pjrt::Runtime::synchronous_mode`, and a rejected option costs
latency rather than correctness. Log {cpp:func}`~pjrt::Runtime::describe` once
at startup; it is the first thing to ask for when a latency number looks wrong.

## Options

The plugin is looked for in three places, in order: `plugin_path`, the
`PJRT_CPU_PLUGIN` environment variable, and `default_plugin_path()` — the
location `make plugin` writes to, compiled in at build time.

```{doxygenstruct} pjrt::RuntimeOptions
:members: plugin_path, synchronous, cpu_device_count, worker_threads, max_inflight_computations, allow_async_fallback
```

`worker_threads` is applied with `setenv("PJRT_NPROC", ...)` before the client
is created, because that is where XLA's `DefaultThreadPoolSize()` reads it.
That is a process environment variable, so it is visible to any client created
afterwards, and it is why thread-pool sizing needs no patch to the plugin.

## Execution mode

```{doxygenenum} pjrt::SyncMode
```

The four values answer a different question than a boolean would. `Inline` is
the plugin confirming, through its `supports_synchronous_execution` attribute,
that computations run on the calling thread. `Accepted` means the option was
sent and creation succeeded, but the plugin advertises no marker, so nothing
stronger can honestly be said. `Rejected` means the option was refused and the
client was created without it — execution is asynchronous despite the request,
which is exactly the case
{cpp:member}`RuntimeOptions::allow_async_fallback <pjrt::RuntimeOptions::allow_async_fallback>`
exists to make visible or fatal. `Async` means `synchronous = false` was asked
for and the option was never sent.

{cpp:func}`~pjrt::Runtime::synchronous_supported` collapses the first two into
one boolean, which is what a benchmark label wants.

## Plugin self-description

```{doxygenstruct} pjrt::PluginInfo
:members: path, api_major, api_minor, platform_name, platform_version, attributes, advertises_synchronous_execution, advertises_max_inflight
```

`attributes` is kept whole and in the plugin's own order because it is the only
self-description a plugin offers, and it belongs verbatim in a bug report. The
two `advertises_*` flags are the ones the constructor acts on.

## The runtime

```{doxygenclass} pjrt::Runtime
:members: Runtime, api, client, device, options, plugin, synchronous_mode, synchronous_supported, describe
```

`options()` returns the options **as given**, including any the plugin then
declined; `plugin()` and `synchronous_mode()` are where you find out what was
actually negotiated.

## Free functions

```{doxygenfunction} pjrt::default_plugin_path
```

```{doxygenfunction} pjrt::version
```

```{doxygenfunction} pjrt::vendored_pjrt_api_minor
```

The vendored PJRT C API version is {{ pjrt_api_major }}.{{ pjrt_api_minor }}.
Compare it with `PluginInfo::api_minor` when a plugin behaves unlike the header
says it should: a plugin older than the vendored header may not know the
trailing fields of the `Args` structs this code fills in.
