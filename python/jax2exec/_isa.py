"""Which instruction-set level this host implements.

A serialized executable embeds machine code for the machine that produced it,
so a ``.binpb`` exported on an AVX-512 box does not run on a host without it
-- and the failure arrives as a load error or an illegal instruction, not as a
polite refusal.  The sidecar therefore records the exporting host's level, and
the C++ loader compares it against its own before it even opens the file,
falling back to compiling the ``.mlirbc`` when the host is weaker.

That comparison only means something if both sides compute the level the same
way, so the rules below are duplicated -- deliberately, in full -- in
``src/pjrt_exec/isa.cpp``.  Change one and you must change the other; the
levels are the microarchitecture levels from the x86-64 psABI, cut down to the
flags that actually gate XLA's codegen.
"""

from __future__ import annotations

import functools
import platform

__all__ = ["cpu_model", "host_isa_level", "isa_supports"]

_CPUINFO = "/proc/cpuinfo"

# The psABI microarchitecture levels, each expressed as the flags
# /proc/cpuinfo prints for it.  LZCNT is spelled "abm" on AMD parts and
# "lzcnt" on some Intel ones, so it is checked separately.
_X86_V2 = frozenset({"sse4_2", "ssse3", "popcnt"})
_X86_V3 = frozenset({"avx", "avx2", "bmi1", "bmi2", "fma", "f16c", "movbe"})
_X86_V4 = frozenset({"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"})

# Levels within one architecture family, weakest first.  Anything absent from
# every row -- "unknown", or a family this file does not know -- compares
# against nothing.
_LADDERS = (
    ("x86-64-v1", "x86-64-v2", "x86-64-v3", "x86-64-v4"),
    ("aarch64", "aarch64+sve"),
)


def _cpuinfo_lines() -> list[str]:
    """Read ``/proc/cpuinfo``, or return nothing where it does not exist."""
    try:
        with open(_CPUINFO, encoding="utf-8", errors="replace") as handle:
            return handle.readlines()
    except OSError:
        return []


@functools.cache
def _cpu_flags() -> frozenset[str]:
    """Return the flags of the first core.

    Notes
    -----
    x86 kernels print them under ``flags``, aarch64 kernels under
    ``Features``.  The set is cached: instruction sets do not change under a
    running process, and both callers of this module are on a load path.
    """
    for line in _cpuinfo_lines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in ("flags", "features"):
            return frozenset(value.split())
    return frozenset()


def cpu_model() -> str | None:
    """Return the CPU model string, or ``None`` when the host does not say.

    Returns
    -------
    str or None
        The ``model name`` field of ``/proc/cpuinfo``.  aarch64 kernels
        usually omit it, and so does every non-Linux host, in which case the
        sidecar simply records ``null``: it is a comment for a human reading
        an artifact, never something the loader acts on.
    """
    for line in _cpuinfo_lines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "model name":
            model = value.strip()
            if model:
                return model
    return None


def host_isa_level() -> str:
    """Return this host's instruction-set level.

    Returns
    -------
    str
        ``"x86-64-v1"`` through ``"x86-64-v4"`` on x86_64, ``"aarch64"`` or
        ``"aarch64+sve"`` on 64-bit Arm, and ``"unknown"`` anywhere else --
        including on a host whose flags could not be read, where claiming a
        level would be worse than admitting ignorance.

    Notes
    -----
    Levels are tested strongest first, exactly as ``src/pjrt_exec/isa.cpp``
    does.  A part carrying the AVX-512 set always carries the v3 set as well,
    so the cascade needs no cumulative check to agree with the psABI.
    """
    machine = platform.machine()
    flags = _cpu_flags()

    if machine in ("x86_64", "AMD64"):
        if _X86_V4 <= flags:
            return "x86-64-v4"
        has_lzcnt = "abm" in flags or "lzcnt" in flags
        if _X86_V3 <= flags and has_lzcnt:
            return "x86-64-v3"
        if _X86_V2 <= flags:
            return "x86-64-v2"
        return "x86-64-v1"

    if machine in ("aarch64", "arm64"):
        return "aarch64+sve" if "sve" in flags else "aarch64"

    return "unknown"


def isa_supports(host: str, required: str) -> bool | None:
    """Say whether a host at level ``host`` can run code built for ``required``.

    Parameters
    ----------
    host : str
        A level as returned by :func:`host_isa_level`.
    required : str
        The ``isa_level`` recorded in a sidecar.

    Returns
    -------
    bool or None
        True when ``host`` is at least ``required``, False when it is weaker,
        and ``None`` when the two cannot be compared at all -- a different
        architecture family, or a level this file does not know.  ``None`` is
        not "probably fine": an aarch64 host cannot run an x86-64 ``.binpb``.
    """
    for ladder in _LADDERS:
        if host in ladder and required in ladder:
            return ladder.index(host) >= ladder.index(required)
    return None
