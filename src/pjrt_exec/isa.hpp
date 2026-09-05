/**
 * @file isa.hpp
 * @brief Which instruction-set level this host implements, and whether that is
 *        enough to run code built for another one.
 *
 * A serialized executable carries machine code generated for the exporting
 * machine.  `PJRT_Executable_DeserializeAndLoad` relinks it; it never
 * recompiles it and never inspects whether this CPU has the instructions the
 * code contains.  Load a `.binpb` exported on an AVX-512 host onto a machine
 * without AVX-512 and the failure arrives as SIGILL somewhere deep inside the
 * executable, with a backtrace that says nothing about artifacts.
 *
 * The exporter therefore records its own level in the sidecar and the loader
 * compares it against this host's before it opens the file, compiling the
 * `.mlirbc` instead when the comparison does not come out in its favour.
 *
 * Internal to the loader.  These are not part of the public API: nothing about
 * the levels is stable enough to promise, and a caller who wants this
 * information wants it from the sidecar, not from us.
 *
 * @note The rules below are duplicated, deliberately and in full, in
 *       `python/jax2exec/_isa.py`.  The comparison only means anything if both
 *       ends compute the level the same way, so a change to either file is a
 *       change to both.  The levels are the x86-64 psABI microarchitecture
 *       levels, cut down to the flags that actually gate XLA's codegen.
 */
#pragma once

#include <string>

namespace pjrt::internal {

/**
 * @brief This host's instruction-set level.
 *
 * @return `"x86-64-v1"` through `"x86-64-v4"` on x86_64, `"aarch64"` or
 *         `"aarch64+sve"` on 64-bit Arm, and `"unknown"` on anything else --
 *         where admitting ignorance is worth more than a level nobody can act
 *         on.
 */
std::string host_isa_level();

/**
 * @brief Whether a host at level `host` can run code built for `required`.
 *
 * @param host     A level as `host_isa_level()` spells it.
 * @param required The `isa_level` a sidecar recorded.
 * @return True only when the two are in the same family and `host` sits at or
 *         above `required`.  Different families, an unknown level on either
 *         side, or a spelling this function does not recognize all return
 *         false: the question is "can this be proven safe", and anything it
 *         cannot compare it cannot prove.  Note that this is *not* the
 *         tri-state its Python counterpart returns -- the loader has only one
 *         thing to do with "cannot tell", which is to compile instead.
 */
bool isa_at_least(const std::string& host, const std::string& required);

}  // namespace pjrt::internal
