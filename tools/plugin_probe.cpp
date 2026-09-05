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
 *   build/bin/plugin_probe [path/to/libpjrt_c_api_cpu_plugin.so]
 *
 * Exits 0 when a client could be created, 1 otherwise.
 */
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "pjrt_exec/runtime.hpp"

int main(int argc, char** argv) {
  pjrt::RuntimeOptions options;
  if (argc > 1) {
    options.plugin_path = argv[1];
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

    std::printf("\n%s\n", runtime.describe().c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "plugin_probe: %s\n", error.what());
    return 1;
  }
}
