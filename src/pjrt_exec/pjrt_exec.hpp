/**
 * @file
 * @brief Using the pjrt api, load a compiled program and execute it.
 *
 * We provide cpp wrappers for the pjrt api for loading and executing a compiled
 * executable.
 * The basic workflow is as follows:
 * 1. Write a program in jax and compile it to the target platform.
 * 2. Load the compiled program using the pjrt api.
 * 3. Execute the program using the pjrt api.
 *
 * We only depend on the pjrt api header files, and we depend on a corresponding
 * shared library for the target platform.
 * We do not dynamically load the shared library, but we assume that it is
 * linked with the executable.
 *
 * Some helpful links and projects that provided inspiration:
 * - https://github.com/jax-ml/jax/discussions/22184
 * - https://github.com/joaospinto/call_jax_from_cpp/tree/main
 * -
 * https://github.com/LeelaChessZero/lc0/tree/97817028ae513cddd779abf606675c0808c353b2/src/neural/backends/xla
 * - https://github.com/gomlx/gopjrt/tree/main
 *
 * @remark Most classes should be constructed as shared pointers.
 * We are wrapping the pjrt api, and it would be nice for the cpp destructors to
 * destroy the pjrt objects.
 * But the pjrt objects should be destroyed at most (only) once, so shared
 * pointers will perform the desired reference counting.
 */
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/xla/pjrt_c_api.h"
#include "src/xla/pjrt_c_api_cpu.h"

namespace pjrt {
/**
 * @brief Get the initialized PJRT api.
 *
 * The plugin is initialized on the first call and never again.  This has to be
 * a function rather than a namespace-scope variable in this header: a `static`
 * definition here would give every translation unit its own copy, each
 * initializing the plugin separately at unordered static-initialization time.
 */
const PJRT_Api* api();

/**
 * @brief Error class for PJRT errors.
 * Should only be envoked by `check_error`.
 * It grabs the appropriate error message and code from the PJRT api, and it
 * gracefully destroys the error struct.
 */
class Error : public std::runtime_error {
 public:
  explicit Error(PJRT_Error* error);

 private:
  size_t code_;
  std::string message_;
};

void check_error(PJRT_Error* error);

class Device;

/**
 * @brief Client class for PJRT.
 * This class is a wrapper around the PJRT_Client struct.
 * It provides a simple interface for creating and destroying clients.
 * It also provides a way to get the platform name and version.
 */
class Client {
 public:
  Client();
  ~Client();

  std::vector<std::shared_ptr<Device>> get_devices() const;

  /// Underlying handle, for code that needs the C API directly.
  PJRT_Client* raw() const { return client_; }

 private:
  PJRT_Client* client_;

  // need `client_` access
  friend class Buffer;
  friend class AOTComputation;
};

/**
 * @brief Buffer class for PJRT. Should only be created from `Client`.
 */
class Device {
 public:
  Device(PJRT_Device* device);

  /// Underlying handle, for code that needs the C API directly.
  PJRT_Device* raw() const { return device_; }

  // We don't need a special destructor, I guess...
  // I think that the device is owned by the client, so destroying the client
  //  will destroy the device
  // ~Device();

 private:
  PJRT_Device* device_;
  friend class Buffer;
};

/**
 * @brief PJRT events to await, e.g., transfering data and executing functions.
 */
class Event {
 public:
  Event(PJRT_Event* event);
  ~Event();
  void await();

 private:
  PJRT_Event* event_;
};

/**
 * @brief Buffer class for PJRT.
 * Used to transfer data between host and device, and manage the buffer
 * lifecycle (creation and destruction).
 * Note that a new buffer is created, both when copying to from host to device
 * and when copying to from device to host.
 * @note We promise not to modify the buffe contents while the buffer is alive,
 * which shouldn't be a problem if the user only uses one buffer per call.
 */
class Buffer {
 public:
  Buffer(PJRT_Buffer* buffer);
  ~Buffer();
  static std::tuple<std::shared_ptr<Buffer>, std::shared_ptr<Event>> to_device(
      const double* data, size_t size, std::shared_ptr<Client> client,
      std::shared_ptr<Device> device);
  static std::shared_ptr<Buffer> to_device_blocking(
      const double* data, size_t size, std::shared_ptr<Client> client,
      std::shared_ptr<Device> device);
  /**
   * @brief Wrap host memory without copying it.
   *
   * The buffer aliases `data`, so `data` must outlive the buffer, must stay
   * 64-byte aligned, and must not be written while a computation using it is
   * in flight.  Writing it *between* calls is what makes a persistent input
   * arena possible.
   */
  static std::shared_ptr<Buffer> to_device_zero_copy(
      const double* data, size_t size, std::shared_ptr<Client> client,
      std::shared_ptr<Device> device);
  std::shared_ptr<Event> to_host(double* data, size_t size);
  void to_host_blocking(double* data, size_t size);
  std::vector<std::size_t> get_dims() const;

  /// Underlying handle, for code that needs the C API directly.
  PJRT_Buffer* raw() const { return buffer_; }

 private:
  PJRT_Buffer* buffer_;
  friend class AOTComputation;
};

/**
 * @brief Compiled executable class for PJRT.
 * We assume that an executable is already compiled and serialized.
 * We load the file, and allow compuatation.
 */
class AOTComputation {
 public:
  AOTComputation(const std::string& base_name, std::shared_ptr<Client> client);
  ~AOTComputation();
  std::tuple<std::vector<std::shared_ptr<Buffer>>, std::shared_ptr<Event>>
  execute(std::vector<std::shared_ptr<Buffer>> input);
  std::vector<std::shared_ptr<Buffer>> execute_blocking(
      std::vector<std::shared_ptr<Buffer>> input);

 private:
  std::vector<std::size_t> input_sizes_;
  std::vector<std::size_t> output_sizes_;
  PJRT_LoadedExecutable* loaded_executable_;
  std::shared_ptr<Client> client_;
};

}  // namespace pjrt
