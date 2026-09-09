"""Which instruction-set level this host implements.

The sidecar records the exporting host's level and the C++ loader compares
it with its own before opening a ``.binpb``.  The rules here are duplicated,
in full, in ``src/pjrt_exec/isa.cpp``: change one and you must change the
other.
"""

from __future__ import annotations

import functools
import platform

__all__ = ["cpu_model", "host_isa_level", "isa_supports"]

_CPUINFO = "/proc/cpuinfo"

# The psABI levels as /proc/cpuinfo spells them, cut down to the flags that
# gate XLA's codegen.  LZCNT is "abm" on AMD parts and "lzcnt" on some Intel
# ones, so it is checked on its own.
_X86_V2 = frozenset({"sse4_2", "ssse3", "popcnt"})
_X86_V3 = frozenset({"avx", "avx2", "bmi1", "bmi2", "fma", "f16c", "movbe"})
_X86_V4 = frozenset({"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"})

# Levels within one architecture family, weakest first.  A level absent from
# every row compares against nothing.
_LADDERS = (
    ("x86-64-v1", "x86-64-v2", "x86-64-v3", "x86-64-v4"),
    ("aarch64", "aarch64+sve"),
)


def _cpuinfo_field(*keys: str) -> str:
    """Return the first ``/proc/cpuinfo`` field named by one of ``keys``.

    The first match is the first core's.
    """
    try:
        with open(_CPUINFO, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, sep, value = line.partition(":")
                if sep and key.strip().lower() in keys:
                    return value.strip()
    except OSError:
        return ""
    return ""


@functools.cache
def _cpu_flags() -> frozenset[str]:
    """Return the first core's flags (``flags`` on x86, ``Features`` on arm)."""
    return frozenset(_cpuinfo_field("flags", "features").split())


def cpu_model() -> str | None:
    """Return the CPU model string, or ``None`` when the host does not say.

    Returns
    -------
    str or None
        The ``model name`` field of ``/proc/cpuinfo``, which aarch64 kernels
        and non-Linux hosts omit.  The sidecar records it for a human reader;
        the loader never acts on it.
    """
    return _cpuinfo_field("model name") or None


def host_isa_level() -> str:
    """Return this host's instruction-set level.

    Returns
    -------
    str
        ``"x86-64-v1"`` through ``"x86-64-v4"`` on x86_64, ``"aarch64"`` or
        ``"aarch64+sve"`` on 64-bit Arm, and ``"unknown"`` anywhere else.

    Notes
    -----
    Levels are tested strongest first, as ``src/pjrt_exec/isa.cpp`` does.  A
    part with the AVX-512 set always has the v3 set as well, so the cascade
    needs no cumulative check to agree with the psABI.

    An unreadable ``/proc/cpuinfo`` gives the weakest level of the family,
    not ``"unknown"``, so the sidecar under-claims and the loader's guard
    cannot catch it; see ``docs/developer/open-threads.md``.
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
        and ``None`` when the two cannot be compared: a different family, or
        a level this file does not know.  ``None`` is not "probably fine".
    """
    for ladder in _LADDERS:
        if host in ladder and required in ladder:
            return ladder.index(host) >= ladder.index(required)
    return None
