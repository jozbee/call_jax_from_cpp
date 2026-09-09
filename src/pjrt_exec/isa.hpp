/**
 * @file isa.hpp
 * @brief Which instruction-set level this host implements, and whether that is
 *        enough to run code built for another one.
 *
 * `PJRT_Executable_DeserializeAndLoad` relinks the machine code in a `.binpb`
 * and never checks whether this CPU has the instructions it contains, so a
 * `.binpb` from an AVX-512 host is a SIGILL deep inside the executable on a
 * machine without it.  The exporter records its level in the sidecar; the
 * loader compares it against this host's before opening the file, and compiles
 * the `.mlirbc` instead when the comparison does not come out in its favour.
 *
 * Internal to the loader: nothing about the levels is stable enough to promise.
 *
 * @note The rules are duplicated, in full, in `python/jax2exec/_isa.py`.  The
 *       comparison only means anything if both ends compute the level the same
 *       way, so a change to either file is a change to both.
 */
#pragma once

#include <string>

namespace pjrt::internal {

/// This host's instruction-set level: `"x86-64-v1"` through `"x86-64-v4"`,
/// `"aarch64"` or `"aarch64+sve"`, or `"unknown"`.
std::string host_isa_level();

/// Whether a host at level `host` can run code built for `required`: true
/// only when both are on one ladder and `host` is at or above `required`.
/// Anything it cannot compare -- different families, "unknown", a spelling it
/// does not recognize -- is false rather than the Python counterpart's
/// tri-state, because the loader's only answer to "cannot tell" is to compile.
bool isa_at_least(const std::string& host, const std::string& required);

}  // namespace pjrt::internal
