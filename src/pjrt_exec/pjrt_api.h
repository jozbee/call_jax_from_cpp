/**
 * @file
 * @brief Global api variable and common client class.
 *
 * See `pjrt_exec.h` for usage of this file.
 * We separated `pjrt_exec.h` and `pjrt_api.h` to make it more clear that there
 * should be a single client for all exectable classes.
 * Namely, `pjrt_exec.h` defines a global client variable.
 * Because getting a client can cause an error, we also define Error here.
 */
#pragma once

#include <stdexcept>
#include <string>

#include "src/xla/pjrt_c_api.h"
#include "src/xla/pjrt_c_api_cpu.h"

namespace pjrt {}  // namespace pjrt
