/**
 * @file runtime.hpp
 * @brief Load once, call many: a low-jitter path for calling a compiled JAX
 *        function from a real-time loop.
 *
 * The original `pjrt_exec` API creates device buffers, events and shared
 * pointers on every call.  That is fine for a benchmark and bad for a control
 * loop, where the interesting number is not the average but the worst call in
 * a million.  This API moves all of that work to load time:
 *
 *   - inputs live in 64-byte-aligned arenas the runtime owns, wrapped once in
 *     zero-copy PJRT buffers, so a call costs no host-to-device copy;
 *   - outputs are read straight out of device memory;
 *   - nothing in the steady-state path allocates, locks, or logs.
 *
 * Usage:
 * @code
 *   pjrt::Runtime rt;                              // one per process
 *   pjrt::Function f(rt, "artifacts/mpc_solver");  // load once
 *
 *   std::memcpy(f.input(0), acc_ref, 3 * sizeof(double));
 *   f.call();
 *   const double* u = f.output(0);
 * @endcode
 *
 * A `Function` is deliberately not thread-safe: it owns fixed input and output
 * storage, so sharing one across threads would mean sharing those buffers.
 * Use one `Function` per thread (they can share a `Runtime`).
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "src/xla/pjrt_c_api.h"
#include "src/xla/pjrt_c_api_cpu.h"

namespace pjrt {

/// Client-wide settings. These are fixed when the client is created.
struct RuntimeOptions {
  /**
   * Run computations inline on the calling thread instead of dispatching them
   * to the client's thread pool.  Removing that hand-off is the single biggest
   * structural cut to tail latency, but it is only available from a plugin
   * built with the `asynchronous` create option (see README).  Stock plugins
   * ignore unknown create options, so this degrades to asynchronous dispatch
   * rather than failing.
   */
  bool synchronous = true;

  /// Logical CPU devices. One is all a single control loop needs, and it also
  /// bounds the size of the runtime's thread pools.
  int cpu_device_count = 1;

  /// Threads for XLA's intra-op and dispatch pools, applied via `PJRT_NPROC`.
  /// 0 leaves XLA's default (one thread per core), which oversubscribes a
  /// machine that is doing anything else.
  int worker_threads = 1;
};

/// Per-function settings, fixed at load time.
struct FunctionOptions {
  /// Calls made at load, discarded, to fault in pages and warm the runtime.
  int warmup_calls = 3;

  /// Cross-check the `.json` sidecar against the compiled executable.
  bool check_metadata = true;
};

/**
 * @brief A PJRT client plus the plugin API, owned for the process lifetime.
 *
 * Creating a client starts XLA's thread pools, so create exactly one and keep
 * it alive; destroying and recreating it mid-run is a guaranteed latency
 * spike.
 */
class Runtime {
 public:
  explicit Runtime(const RuntimeOptions& options = {});
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  PJRT_Client* client() const { return client_; }
  PJRT_Device* device() const { return device_; }
  const RuntimeOptions& options() const { return options_; }

  /// True when the plugin accepted the `asynchronous` create option, i.e. when
  /// `RuntimeOptions::synchronous` actually took effect.
  bool synchronous_supported() const { return synchronous_supported_; }

 private:
  RuntimeOptions options_;
  PJRT_Client* client_ = nullptr;
  PJRT_Device* device_ = nullptr;
  bool synchronous_supported_ = false;
};

/**
 * @brief One compiled JAX function, ready to be called repeatedly.
 *
 * Write the inputs into `input(i)`, call `call()`, read the results from
 * `output(i)`.  Both point at storage the `Function` owns; output storage is
 * overwritten by the next call.
 */
class Function {
 public:
  Function(Runtime& runtime, const std::string& base_name,
           const FunctionOptions& options = {});
  ~Function();

  Function(const Function&) = delete;
  Function& operator=(const Function&) = delete;

  std::size_t num_inputs() const { return input_sizes_.size(); }
  std::size_t num_outputs() const { return output_sizes_.size(); }
  std::size_t input_size(std::size_t i) const { return input_sizes_[i]; }
  std::size_t output_size(std::size_t i) const { return output_sizes_[i]; }

  /// Writable input arena for argument `i`, valid for the function's lifetime.
  double* input(std::size_t i) { return input_arenas_[i]; }

  /// Results of the most recent `call()`, valid until the next one.
  const double* output(std::size_t i) const { return output_arenas_[i]; }

  /// Execute once and block until the results are in the output arenas.
  void call();

 private:
  void load_metadata(const std::string& base_name);
  void load_executable(const std::string& base_name);
  void allocate_arenas();
  void wrap_inputs();

  Runtime& runtime_;
  std::vector<std::size_t> input_sizes_;
  std::vector<std::size_t> output_sizes_;

  // Storage the caller reads and writes. Owned here, 64-byte aligned so that
  // XLA accepts the inputs zero-copy instead of silently copying them.
  std::vector<double*> input_arenas_;
  std::vector<double*> output_arenas_;

  // Created once at load and reused by every call.
  std::vector<PJRT_Buffer*> input_buffers_;

  // Per-call scratch, sized at load so that `call()` allocates nothing.
  std::vector<PJRT_Buffer*> output_buffers_;
  PJRT_ExecuteOptions execute_options_ = {};

  PJRT_LoadedExecutable* executable_ = nullptr;
};

}  // namespace pjrt
