/**
 * @file plugin_probe.cpp
 * @brief Report what a PJRT CPU plugin is and what it accepts.
 *
 * The create-option surface of a PJRT plugin is not discoverable by
 * experiment: this XLA version validates option names and fails client
 * creation on anything it does not recognise, so "try it and see" is not
 * available. `PJRT_Plugin_Attributes` is what a caller can ask before
 * committing, and this probe prints the answer.
 *
 * Use it after building or downloading a plugin, and as the verification step
 * in docs/developer/bumping-jax.md. A plugin built from this project's XLA
 * fork advertises `supports_synchronous_execution`; a stock one does not, and
 * the runtime then reports SyncMode::Accepted rather than Inline.
 *
 *   build/bin/plugin_probe [path/to/libpjrt_c_api_cpu_plugin.so] [--view]
 *
 * `--view` additionally probes PJRT_Client_CreateViewOfDeviceBuffer, whose
 * availability on CPU has already changed once. Documentation that says what a
 * plugin does is worth exactly as much as the probe that re-checks it.
 *
 * Exits 0 when a client could be created, 1 otherwise.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "pjrt_exec/runtime.hpp"

/// Probe PJRT_Client_CreateViewOfDeviceBuffer, which the CPU plugin was once
/// documented as not implementing. It does implement it: the view aliases the
/// caller's pointer and observes writes made after it exists. Note the header
/// calls `on_delete_callback` optional while the implementation throws
/// `std::bad_function_call` on a null one.
void probe_view(pjrt::Runtime& runtime) {
  const PJRT_Api* api = runtime.api();
  std::printf("view_supported=%d\n",
              api->PJRT_Client_CreateViewOfDeviceBuffer != nullptr ? 1 : 0);
  if (api->PJRT_Client_CreateViewOfDeviceBuffer == nullptr) {
    return;
  }

  void* memory = nullptr;
  if (posix_memalign(&memory, 64, 64 * sizeof(double)) != 0) {
    std::printf("view_probe=alloc-failed\n");
    return;
  }
  double* caller = static_cast<double*>(memory);
  for (int i = 0; i < 64; ++i) {
    caller[i] = 1.5 * i;
  }

  std::int64_t dims[1] = {64};
  PJRT_Client_CreateViewOfDeviceBuffer_Args args{};
  args.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  args.client = runtime.client();
  args.device_buffer_ptr = caller;
  args.dims = dims;
  args.num_dims = 1;
  args.element_type = PJRT_Buffer_Type_F64;
  args.device = runtime.device();
  args.on_delete_callback = [](void*, void*) {};

  if (api->PJRT_Client_CreateViewOfDeviceBuffer(&args) != nullptr) {
    std::printf("view_created=0\n");
    std::free(memory);
    return;
  }
  std::printf("view_created=1\n");

  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args pointer{};
  pointer.struct_size =
      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
  pointer.buffer = args.buffer;
  if (api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&pointer) == nullptr) {
    const double* seen = static_cast<const double*>(pointer.device_memory_ptr);
    std::printf("view_aliases=%d\n",
                pointer.device_memory_ptr == caller ? 1 : 0);
    caller[7] = 99.25;
    std::printf("view_sees_later_write=%d\n", seen[7] == 99.25 ? 1 : 0);
  }

  PJRT_Buffer_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  destroy.buffer = args.buffer;
  (void)api->PJRT_Buffer_Destroy(&destroy);
  std::free(memory);
}

int main(int argc, char** argv) {
  pjrt::RuntimeOptions options;
  bool want_view = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--view") {
      want_view = true;
    } else if (argument.rfind("--", 0) == 0) {
      std::fprintf(stderr, "plugin_probe: unknown flag %s\n", argv[i]);
      return 1;
    } else {
      options.plugin_path = argument;
    }
  }

  try {
    pjrt::Runtime runtime(options);
    const pjrt::PluginInfo& plugin = runtime.plugin();

    std::printf("plugin_path=%s\n", plugin.path.c_str());
    std::printf("api_version=%d.%d\n", plugin.api_major, plugin.api_minor);
    std::printf("vendored_api_minor=%d\n", pjrt::vendored_pjrt_api_minor());
    std::printf("platform_name=%s\n", plugin.platform_name.c_str());
    std::printf("platform_version=%s\n", plugin.platform_version.c_str());

    const char* mode = "unknown";
    switch (runtime.synchronous_mode()) {
      case pjrt::SyncMode::Inline:   mode = "inline";   break;
      case pjrt::SyncMode::Accepted: mode = "accepted"; break;
      case pjrt::SyncMode::Rejected: mode = "rejected"; break;
      case pjrt::SyncMode::Async:    mode = "async";    break;
    }
    std::printf("sync_mode=%s\n", mode);
    std::printf("synchronous_supported=%d\n",
                runtime.synchronous_supported() ? 1 : 0);
    std::printf("advertises_synchronous_execution=%d\n",
                plugin.advertises_synchronous_execution ? 1 : 0);
    std::printf("advertises_max_inflight=%d\n",
                plugin.advertises_max_inflight ? 1 : 0);

    std::printf("num_attributes=%zu\n", plugin.attributes.size());
    for (const auto& attribute : plugin.attributes) {
      std::printf("attribute[%s]=%s\n", attribute.first.c_str(),
                  attribute.second.c_str());
    }

    if (want_view) {
      probe_view(runtime);
    }

    std::printf("\n%s\n", runtime.describe().c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "plugin_probe: %s\n", error.what());
    return 1;
  }
}
