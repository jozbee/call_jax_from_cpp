/**
 * @file runtime.hpp
 * @brief Load once, call many: the low-jitter path for calling a compiled JAX
 *        function from a real-time loop.
 *
 * A control loop cares about the worst call in a million, not the average one,
 * so everything that can happen before the loop starts happens at load time:
 *
 *   - the JSON sidecar is read and cross-checked against the executable, so a
 *     stale artifact is a `LoadError` at startup rather than a silent overwrite
 *     past the end of an arena on call ten thousand;
 *   - every input and output gets its own 64-byte-aligned arena, owned by the
 *     `Function`.  Alignment is not cosmetic here: below `xla::cpu::MinAlign()`
 *     XLA falls back to copying the buffer and says nothing, and the entire
 *     benefit disappears without a diagnostic;
 *   - each input arena is wrapped once in a zero-copy `PJRT_Buffer`, so a call
 *     transfers nothing -- XLA reads the arena where it lies;
 *   - every per-call array is sized, and the warm-up calls fault in the pages
 *     and warm the runtime's own lazy state.
 *
 * A call is then: execute, one await, and one `memcpy` per output read straight
 * out of device memory.  Nothing on that path allocates, locks, logs or
 * flushes.  (thousands of allocations per call still happen inside XLA's
 * thunk runtime, about one per StableHLO op.  Those are not reachable from
 * here; the ones this wrapper used to add are gone.)
 *
 * @code
 *   pjrt::Runtime rt;                             // one per process
 *   pjrt::Function f(rt, "artifacts/trajopt");    // load once
 *
 *   const std::size_t x0 = *f.find_input("x0");   // resolve names at startup
 *   double* state = f.input<double>(x0);
 *
 *   for (;;) {
 *     std::memcpy(state, measured, f.input_nbytes(x0));
 *     f.call();
 *     const double* u = f.output<double>(0);
 *     // ... command the actuators from u ...
 *   }
 * @endcode
 *
 * Two rules this API cannot enforce for you:
 *
 * **Write the inputs between calls, never during one.** The PJRT buffers alias
 * the arenas for the life of the `Function`, and XLA is reading them while
 * `call()` runs; writing from another thread or a signal handler mid-call is a
 * data race on the computation's own operands.  Between calls it is safe --
 * nothing is in flight -- and that is the property this design rests on.  It is
 * a deliberate, measured bend of the PJRT host-buffer contract, which speaks
 * only about mutation while a transfer is outstanding.
 *
 * **One `Function` per thread.** A `Function` owns fixed storage, so sharing
 * one between threads means sharing those arenas and the executable's per-call
 * state.  `Runtime` is the object to share: create it once, then give each
 * thread its own `Function` loaded from the same artifact.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pjrt/pjrt_c_api.h"
#include "pjrt_exec/dtype.hpp"

namespace pjrt {

/**
 * @brief An error returned by the PJRT C API.
 *
 * Constructing one **consumes** `error`: it copies out the message and the
 * status code, then calls `PJRT_Error_Destroy`.  A `PJRT_Error*` therefore
 * never needs freeing at the call site -- hand it to `check_error` and forget
 * it.
 */
class Error : public std::runtime_error {
 public:
  /// Take the message and code from `error`, then destroy it.
  Error(const PJRT_Api* api, PJRT_Error* error);

  /// The abseil-style status code the plugin reported.
  PJRT_Error_Code code() const { return code_; }

 private:
  PJRT_Error_Code code_;
};

/**
 * @brief An artifact could not be loaded.
 *
 * Distinct from `Error` because these are the failures a caller can act on: a
 * missing plugin, a sidecar that disagrees with its executable, an element type
 * this project does not support, a `.binpb` built for a wider instruction set
 * than this host has.  All of them happen at load, none during a call.
 */
class LoadError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Throw `Error` when `error` is non-null; do nothing (and allocate nothing)
/// otherwise.  This is the only place a `PJRT_Error*` should be inspected.
void check_error(const PJRT_Api* api, PJRT_Error* error);

/// Client-wide settings, fixed when the client is created.
struct RuntimeOptions {
  /**
   * Plugin shared object to load.  Empty means `$PJRT_CPU_PLUGIN`, and failing
   * that `default_plugin_path()`, the location `make plugin` writes to.
   *
   * The library is opened `RTLD_NOW | RTLD_LOCAL` and never closed: XLA leaves
   * statics behind it that live as long as the process.
   */
  std::string plugin_path;

  /**
   * Ask for computations to run inline on the calling thread rather than being
   * handed to the client's dispatch pool.  Removing that hand-off is the single
   * biggest structural cut to tail latency, and it is reachable only through
   * the `asynchronous` create option -- `PJRT_ExecuteOptions` has no
   * execution-mode field.
   *
   * Whether it took effect is reported by `Runtime::synchronous_mode()`; a
   * plugin that rejects the option leaves the computation asynchronous, which
   * costs latency and never correctness.
   */
  bool synchronous = true;

  /// Logical CPU devices to create.  One is all a control loop needs, and it
  /// also bounds the size of the runtime's thread pools.
  int cpu_device_count = 1;

  /**
   * Threads for XLA's intra-op and dispatch pools.  0 leaves XLA's default of
   * one thread per core, which oversubscribes a machine that is doing anything
   * else.
   *
   * Applied by `setenv("PJRT_NPROC", ...)` before the client is created,
   * because that is where `DefaultThreadPoolSize()` reads it.  It is a process
   * environment variable, so it is visible to any client created afterwards.
   */
  int worker_threads = 1;

  /// Concurrent executions the client will admit.  0 leaves the plugin's
  /// default.  Sent only when the plugin advertises
  /// `supports_max_inflight_computations`, since unknown create options are now
  /// rejected rather than ignored.
  int max_inflight_computations = 0;

  /// On a create failure naming an option the plugin does not know, drop that
  /// option and retry instead of throwing.  Turn it off to find out, loudly,
  /// that you are not running the plugin you thought you were.
  bool allow_async_fallback = true;
};

/// What became of `RuntimeOptions::synchronous`.
enum class SyncMode {
  /// The plugin advertises `supports_synchronous_execution` and the option was
  /// accepted: computations run on the calling thread.
  Inline,
  /// The option was sent and client creation succeeded, but the plugin does not
  /// advertise the marker attribute, so this is the best that can be said.
  Accepted,
  /// The plugin rejected the option and the client was created without it.
  /// Execution is asynchronous despite the request.
  Rejected,
  /// `synchronous = false` was asked for; the option was never sent.
  Async,
};

/**
 * @brief What the loaded plugin says about itself.
 *
 * The attributes and the API version are readable before any client exists,
 * which is exactly what makes them usable for deciding which create options are
 * safe to send.  The platform strings need a client and are filled in once it
 * is up.
 */
struct PluginInfo {
  /// The shared object that was actually opened, after the
  /// option/environment/default search.
  std::string path;

  /// PJRT C API version the plugin reports.  Compare against
  /// `vendored_pjrt_api_minor()`: a plugin older than the vendored header may
  /// not know the trailing fields of the `Args` structs this code fills in.
  int api_major = 0;
  int api_minor = 0;  ///< Minor half of the same version.  @see api_major

  /// `PJRT_Client_PlatformName` / `..._PlatformVersion`, e.g. "cpu" and the
  /// XLA build it came from.  Empty until a client exists.
  std::string platform_name;
  std::string platform_version;  ///< The XLA build behind that platform.

  /// Every attribute from `PJRT_Plugin_Attributes`, stringified, in the order
  /// the plugin listed them.  Kept whole because it is the only
  /// self-description a plugin offers, and it belongs in a bug report.
  std::vector<std::pair<std::string, std::string>> attributes;

  /// Whether the `supports_synchronous_execution` attribute is present, i.e.
  /// whether this plugin carries the create-option patch.
  bool advertises_synchronous_execution = false;

  /// Whether `supports_max_inflight_computations` is present.  The option is
  /// withheld when it is not, because the CPU plugin now returns
  /// `InvalidArgument` for a create option it does not recognize.
  bool advertises_max_inflight = false;
};

/**
 * @brief The plugin, its API table, and one client, owned for the process
 *        lifetime.
 *
 * Creating a client starts XLA's thread pools and its lazily-initialized
 * statics, so create exactly one and keep it alive: destroying and recreating a
 * client mid-run is a guaranteed latency spike, and destroying one while a
 * `Function` still holds an executable is undefined.
 *
 * Construction queries `PJRT_Plugin_Attributes` first -- it needs no client --
 * and sends only the create options this plugin admits to understanding.  That
 * ordering matters: as of XLA `dcf304bc` the CPU plugin validates option names
 * and fails creation with `InvalidArgument` on an unknown one, where it used to
 * ignore them silently.
 */
class Runtime {
 public:
  /// Load the plugin, read its attributes, and create the client.
  /// @throws LoadError when no plugin can be found or opened.
  /// @throws Error when the plugin refuses to initialize or create a client.
  explicit Runtime(const RuntimeOptions& options = {});

  /// Destroys the client.  The plugin itself stays loaded.
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  /// The plugin's function table.  Valid for the lifetime of the process.
  const PJRT_Api* api() const { return api_; }

  /// The client every `Function` loads its executable into.
  PJRT_Client* client() const { return client_; }

  /// The single addressable device executions are issued to.
  PJRT_Device* device() const { return device_; }

  /// The options as given, including any that the plugin then declined.
  const RuntimeOptions& options() const { return options_; }

  /// What the plugin reported about itself.
  const PluginInfo& plugin() const { return plugin_; }

  /// Whether `RuntimeOptions::synchronous` took effect, and how confidently.
  SyncMode synchronous_mode() const { return sync_mode_; }

  /// True when execution is inline, or as inline as the plugin will confirm.
  /// False means the dispatch hand-off is still in the call path.
  bool synchronous_supported() const {
    return sync_mode_ == SyncMode::Inline || sync_mode_ == SyncMode::Accepted;
  }

  /// One paragraph naming the plugin, platform, API version, execution mode and
  /// thread configuration -- the line worth logging once at startup, and the
  /// first thing to ask for when a latency number looks wrong.
  std::string describe() const;

 private:
  /// `dlopen` the plugin and resolve `GetPjrtApi`, then
  /// `PJRT_Plugin_Initialize`.  Sets `dl_handle_`, `api_` and `plugin_.path`.
  void load_plugin(const std::string& path);

  /// Fill `plugin_` from `PJRT_Plugin_Attributes` and the API version.  Must
  /// run before `create_client`, which is the point: it decides which create
  /// options are safe to send.
  void query_attributes();

  /// Apply `PJRT_NPROC`, create the client, pick the device, and settle
  /// `sync_mode_` -- retrying without a rejected option when
  /// `allow_async_fallback` permits it.
  void create_client();

  /// Destroy the client if one exists, swallowing any error. Shared by the
  /// destructor and by the constructor's failure path, which has no
  /// destructor to fall back on.
  void destroy_client() noexcept;

  RuntimeOptions options_;
  void* dl_handle_ = nullptr;
  const PJRT_Api* api_ = nullptr;
  PJRT_Client* client_ = nullptr;
  PJRT_Device* device_ = nullptr;
  PluginInfo plugin_;
  SyncMode sync_mode_ = SyncMode::Async;
};

/// How a `Function`'s executable came to exist.
enum class LoadKind {
  /// Relinked from the `.binpb` by `PJRT_Executable_DeserializeAndLoad`.  The
  /// machine code was generated by the exporting machine and is only relocated
  /// here, which is why a `.binpb` is architecture- and ISA-locked.
  Deserialized,
  /// Compiled in this process from the `.mlirbc` StableHLO.  Portable, and
  /// costs seconds rather than milliseconds at load.
  Compiled,
};

/// Which artifact a `Function` is allowed to load from.
enum class LoadPolicy {
  /// Prefer the `.binpb`; fall back to compiling the `.mlirbc` when the binary
  /// is absent, or when `isa_guard` finds it was built for a wider instruction
  /// set than this host implements.
  Auto,
  /// Deserialize the `.binpb` or fail.  What a deployment wants: a fallback
  /// that quietly compiles for seconds is not a fallback in a control loop.
  BinaryOnly,
  /// Compile the `.mlirbc` even when a `.binpb` is present.  Useful for
  /// reproducing a compilation, and for running an artifact exported elsewhere.
  CompileOnly,
};

/// One input or output of a loaded function, exactly as the sidecar describes
/// it and cross-checked against the executable.
struct ArraySpec {
  /// The JAX argument or result name, or `arg<i>`/`out<i>` for a v1 sidecar
  /// that carried no names.
  std::string name;

  /// Element type, as the sidecar names it.  The initializer exists only so a
  /// default-constructed spec is determinate; the loader always overwrites it.
  DType dtype = DType::Float64;

  /// The exact JAX shape, row-major.  Empty for a scalar.
  std::vector<std::int64_t> shape;

  /// Product of `shape`; 1 for a scalar.
  std::size_t numel = 0;

  /// `numel * itemsize(dtype)`, the size of the arena and the length of the
  /// per-call `memcpy`.
  std::size_t nbytes = 0;

  /// Whether the export asked for this argument to be donated.
  ///
  /// Reporting only. The runtime pins every input as non-donatable regardless,
  /// because a donated buffer is consumed by the execution it is passed to and
  /// these buffers are created once and reused by every call. Honouring
  /// donation would mean rebuilding input buffers per call, which is the
  /// per-call allocation this API exists to avoid.
  bool donated = false;
};

/// Per-function settings, fixed at load time.
struct FunctionOptions {
  /// Calls made at load and discarded, to fault in the arenas and warm the
  /// runtime's lazy state.  The first call after a load is always the slowest;
  /// this is where it gets spent.
  int warmup_calls = 3;

  /// Cross-check the sidecar against `PJRT_Executable_NumOutputs`,
  /// `..._OutputElementTypes` and `..._OutputDimensions`.  A sidecar that has
  /// gone stale relative to its executable is otherwise a buffer-overrun class
  /// of bug, discovered as corrupted output.
  bool check_metadata = true;

  /// Per-call checking: bounds and dtype in `input<T>()`/`output<T>()`, and
  /// re-entrancy in `call()`.  Costs a predictable branch on a member, so it is
  /// cheap enough to leave on outside the loop and worth turning off inside it.
  bool debug = false;

  /// Audit arena contents before and after every call: no nan or inf in a
  /// floating arena, nothing but 0 or 1 in a bool arena.  This walks every
  /// element of every arena, so it is a development and acceptance-test tool,
  /// not something to run in a control loop.
  bool check_values = false;

  /// Which artifact to load from.
  LoadPolicy load_policy = LoadPolicy::Auto;

  /// Under `LoadPolicy::Auto`, skip a `.binpb` whose sidecar records an
  /// `isa_level` this host does not implement.  Without it the failure is an
  /// illegal instruction somewhere inside the executable, with no hint that the
  /// artifact came from a newer machine.
  bool isa_guard = true;

  /// Serialized `CompileOptionsProto` for the `.mlirbc` path.  Empty means the
  /// compiler's defaults.  Ignored when the `.binpb` is used, which never
  /// recompiles.
  std::string compile_options;
};

/**
 * @brief One compiled JAX function, loaded once and called repeatedly.
 *
 * Write the arguments into `input<T>(i)`, call `call()`, read the results from
 * `output<T>(i)`.  Both point into storage this object owns: the input arenas
 * live as long as the `Function` and are the memory XLA actually reads, and the
 * output arenas are overwritten by the next call.
 *
 * Indices are the sidecar's, which are JAX's argument and result order.
 * Resolve names to indices once with `find_input` / `find_output` at startup
 * rather than in the loop.
 *
 * Not thread-safe, and not thread-safe by design rather than by omission: see
 * the two rules at the top of this file.
 */
class Function {
 public:
  /**
   * @brief Load `<base_path>.json` and the executable it names.
   *
   * @param runtime The client to load into. It must outlive this `Function`,
   *                and may be shared with other `Function`s.
   * @param base_path Path without an extension: `artifacts/trajopt` reads
   *                  `artifacts/trajopt.json` and then `trajopt.binpb` or
   *                  `trajopt.mlirbc` from the same directory.
   * @param options Load-time and debug settings, all fixed for the lifetime of
   *                the `Function`.
   * @throws LoadError when an artifact is missing, unreadable, internally
   *         inconsistent, or built for a machine unlike this one.
   * @throws Error when the plugin refuses the executable.
   */
  Function(Runtime& runtime, const std::string& base_path,
           const FunctionOptions& options = {});

  /// Destroys the executable and the input buffers, and frees the arenas.
  ~Function();

  Function(const Function&) = delete;
  Function& operator=(const Function&) = delete;

  /// The function's name from the sidecar, used in every error message.
  const std::string& name() const { return name_; }

  /// Arguments the function takes, in JAX's argument order.
  std::size_t num_inputs() const { return inputs_.size(); }

  /// Results it returns, flattened into JAX's result order.
  std::size_t num_outputs() const { return outputs_.size(); }

  /// Full description of input `i`.
  /// @throws std::out_of_range always checked, in every build: this is startup
  ///         code, and a wrong index here is worth catching even in a release
  ///         build.
  const ArraySpec& input_spec(std::size_t i) const {
    check_input_index(i);
    return inputs_[i];
  }

  /// Full description of output `i`.
  /// @throws std::out_of_range as `input_spec`.
  const ArraySpec& output_spec(std::size_t i) const {
    check_output_index(i);
    return outputs_[i];
  }

  /// Index of the input named `name`, or `std::nullopt`.  A linear scan over
  /// the names; call it at startup, not in the loop.
  std::optional<std::size_t> find_input(std::string_view name) const;

  /// Index of the output named `name`, or `std::nullopt`.
  std::optional<std::size_t> find_output(std::string_view name) const;

  /// Element type of input `i` -- the `T` that `input<T>(i)` must ask for.
  DType input_dtype(std::size_t i) const { return input_spec(i).dtype; }

  /// Element type of output `i`.
  DType output_dtype(std::size_t i) const { return output_spec(i).dtype; }

  /// Shape of input `i`, row-major, empty for a scalar.
  const std::vector<std::int64_t>& input_shape(std::size_t i) const {
    return input_spec(i).shape;
  }

  /// Shape of output `i`, row-major, empty for a scalar.
  const std::vector<std::int64_t>& output_shape(std::size_t i) const {
    return output_spec(i).shape;
  }

  /// Dimensions of input `i`; 0 for a scalar.
  std::size_t input_rank(std::size_t i) const {
    return input_spec(i).shape.size();
  }

  /// Dimensions of output `i`; 0 for a scalar.
  std::size_t output_rank(std::size_t i) const {
    return output_spec(i).shape.size();
  }

  /// Elements in input `i`; 1 for a scalar.
  std::size_t input_numel(std::size_t i) const { return input_spec(i).numel; }

  /// Elements in output `i`; 1 for a scalar.
  std::size_t output_numel(std::size_t i) const { return output_spec(i).numel; }

  /// Bytes in input `i`'s arena -- the length of the `memcpy` that fills it.
  std::size_t input_nbytes(std::size_t i) const { return input_spec(i).nbytes; }

  /// Bytes in output `i`'s arena.
  std::size_t output_nbytes(std::size_t i) const {
    return output_spec(i).nbytes;
  }

  /// Whether the executable was deserialized or compiled here.
  LoadKind load_kind() const { return load_kind_; }

  /// Which file was loaded and why, in a form fit for a log line -- including
  /// the reason a `.binpb` was passed over, when one was.
  const std::string& load_detail() const { return load_detail_; }

  /// `PJRT_Executable_Fingerprint`, empty when the plugin does not implement
  /// it.  Two processes reporting the same fingerprint are running the same
  /// compiled program, which is the cheap way to confirm that a benchmark and a
  /// deployment agree.
  const std::string& fingerprint() const { return fingerprint_; }

  /// The options this function was loaded with.
  const FunctionOptions& options() const { return options_; }

  /// Untyped writable arena for input `i`, `input_nbytes(i)` bytes, 64-byte
  /// aligned, valid for the life of the `Function`.  Bounds-checked only when
  /// `FunctionOptions::debug` is set.
  void* input_raw(std::size_t i) {
    if (debug_) {
      check_input_index(i);
    }
    return input_arenas_[i];
  }

  /// Untyped results of the most recent `call()`, valid until the next one and
  /// bounds-checked on the same terms as `input_raw`.
  const void* output_raw(std::size_t i) const {
    if (debug_) {
      check_output_index(i);
    }
    return output_arenas_[i];
  }

  /**
   * @brief Writable arena for input `i`, typed.
   *
   * With `FunctionOptions::debug` off this is one indexed load and a cast: the
   * branch is on a member that is always in cache and always predicted.  `T`
   * must be one of the types `%dtype_of` names -- anything else is a compile
   * error naming the type -- and with `debug` on it must also agree with the
   * dtype the sidecar declares, which is what catches an artifact that was
   * re-exported as float32 under calling code that still says `double`.
   *
   * @throws std::out_of_range when `i` is not a valid index, only when
   *         `FunctionOptions::debug` is set.
   * @throws std::invalid_argument when `T` disagrees with the declared dtype,
   *         only when `FunctionOptions::debug` is set.
   */
  template <class T>
  T* input(std::size_t i) {
    if (debug_) {
      check_input_access(i, dtype_of_v<T>);
    }
    return static_cast<T*>(input_arenas_[i]);
  }

  /// Results of the most recent `call()` for output `i`, typed.  Valid until
  /// the next call.  Checked exactly as `input<T>()` is.
  template <class T>
  const T* output(std::size_t i) const {
    if (debug_) {
      check_output_access(i, dtype_of_v<T>);
    }
    return static_cast<const T*>(output_arenas_[i]);
  }

  /**
   * @brief Execute once and block until the outputs are in their arenas.
   *
   * Allocation-free, lock-free and silent.  Reads whatever is in the input
   * arenas at entry, so write them before, never during.
   *
   * @throws Error when the execution itself fails.
   * @throws std::logic_error on a re-entrant call, when
   *         `FunctionOptions::debug` is set.
   * @throws std::domain_error on a nan, an inf or a bad bool, when
   *         `FunctionOptions::check_values` is set.
   */
  void call();

 private:
  /// Holds `in_call_` true for the duration of a call, including one that
  /// leaves through an exception, so the re-entrancy flag can never be left
  /// stuck on.  Two stores on a member that is already in cache -- cheap enough
  /// to run unconditionally rather than only in debug builds.
  class CallGuard {
   public:
    explicit CallGuard(bool& flag) noexcept : flag_(flag) { flag_ = true; }
    ~CallGuard() { flag_ = false; }

    CallGuard(const CallGuard&) = delete;
    CallGuard& operator=(const CallGuard&) = delete;

   private:
    bool& flag_;
  };

  /// Read `<base_path>.json` into `name_`, `inputs_` and `outputs_`, accepting
  /// schema 1 and 2 and rejecting anything newer.
  void load_sidecar(const std::string& base_path);

  /// Choose an artifact under `options_.load_policy` and the ISA guard, load or
  /// compile it, and record `load_kind_`, `load_detail_` and `fingerprint_`.
  void load_executable(const std::string& base_path);

  /// Cross-check the sidecar's outputs against what the executable reports.
  /// Inputs cannot be checked this way -- the PJRT C API has no parameter-shape
  /// query -- so an input mismatch surfaces as a warm-up call failure instead.
  void validate_metadata();

  /// `posix_memalign(64, ...)` one arena per input and per output.  64 is
  /// `xla::cpu::Align()`; anything less risks the silent copy fallback.
  void allocate_arenas();

  /// Wrap each input arena in a zero-copy `PJRT_Buffer`, once, for the life of
  /// the `Function`.
  void wrap_inputs();

  /// `check_values` audits, run around each call so that a bad value can be
  /// attributed to the caller or to the computation.
  void check_values_before();
  void check_values_after();

  // Out of line on purpose: these build their message strings, and inlining
  // that into the typed accessors would put a stack frame and a string
  // literal's worth of code on the path that has to stay a load and a cast.
  void check_input_index(std::size_t i) const;
  void check_output_index(std::size_t i) const;
  void check_input_access(std::size_t i, DType accessed_as) const;
  void check_output_access(std::size_t i, DType accessed_as) const;

  Runtime& runtime_;
  std::string name_;
  FunctionOptions options_;

  std::vector<ArraySpec> inputs_;
  std::vector<ArraySpec> outputs_;

  // Storage the caller reads and writes, owned here and 64-byte aligned so XLA
  // aliases the inputs instead of silently copying them.  `void*` because the
  // element type is whatever the artifact declares; the typed accessors are
  // where that becomes a C++ type again.
  std::vector<void*> input_arenas_;
  std::vector<void*> output_arenas_;

  // Created once at load and reused by every call.  Their contents are the
  // arenas themselves, not copies of them.
  std::vector<PJRT_Buffer*> input_buffers_;

  // Sized at load, refilled by each call, so `call()` never grows a vector.
  std::vector<PJRT_Buffer*> output_buffers_;

  // Inputs XLA must not donate, pointed at by
  // `execute_options_.non_donatable_input_indices` for the life of the
  // `Function`.  EVERY input belongs here, donated or not: the buffers are
  // created once and reused, so any donation -- asked for by the export or
  // inferred by the compiler from a may-alias -- would destroy a buffer the
  // next call still needs.
  std::vector<std::int64_t> non_donatable_;

  // Filled in at load and reused unchanged by every call.  Value-initialized
  // here and assigned field by field rather than built with a designated
  // initializer, because this header version grew four trailing fields that a
  // designated initializer would silently leave uninitialized.
  PJRT_ExecuteOptions execute_options_{};

  PJRT_LoadedExecutable* executable_ = nullptr;

  LoadKind load_kind_ = LoadKind::Deserialized;
  std::string load_detail_;
  std::string fingerprint_;

  // Copies of the two `options_` flags the call path reads, kept as plain
  // members so a hot-path check is one load from this object rather than a walk
  // into a struct that also holds strings.
  bool debug_ = false;
  bool check_values_ = false;

  // Set for the duration of `call()`; mutable so the guard can clear it from
  // any context.
  mutable bool in_call_ = false;
};

/// Where `make plugin` puts the CPU plugin, compiled in at build time.  The
/// last place `RuntimeOptions::plugin_path` looks.
const char* default_plugin_path();

/// This project's version, as it appears in reports and bug threads.
const char* version();

/// Minor version of the PJRT C API header vendored under `third_party/pjrt`.
/// Compare with `PluginInfo::api_minor` when a plugin behaves unlike the header
/// says it should.
int vendored_pjrt_api_minor();

}  // namespace pjrt
