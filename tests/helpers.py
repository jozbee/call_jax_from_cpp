"""Small utilities the integration tests share, on top of ``conftest``: a
per-session cache for runs several tests read, the environment knobs only
the integration tests honour, and two predicates."""

from __future__ import annotations

import itertools
import os

#: Results of the expensive runs, keyed by name.
_CACHE: dict[str, object] = {}


def cached(key, factory):
    """Call @p factory once per session and hand every caller its result.

    A module-scoped fixture cannot request the function-scoped ``run`` and
    ``load_json`` -- that is a collection error -- so the value is cached
    instead, whatever the scope of the fixture asking.
    """
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def preload(guard_so):
    """The environment change that arms the allocation census.

    Absolute: the loader resolves a bare ``LD_PRELOAD`` name against the
    library search path, so a relative one silently does nothing and the
    census then reads exactly like zero allocations.
    """
    return {"LD_PRELOAD": str(guard_so.resolve())}


def alloc_strict():
    """Whether ``$CJFC_ALLOC_STRICT=1`` asked for the stricter gate.

    ``self`` is always gated; ``runtime`` only under this flag, so a
    libstdc++ or a plugin attributing an allocation differently does not fail
    an otherwise clean run.
    """
    return os.environ.get("CJFC_ALLOC_STRICT", "") == "1"


def rt_iterations():
    """``$CJFC_RT_ITERATIONS``, or 2000 when it is unset or unusable."""
    default = 2000
    try:
        value = int(os.environ.get("CJFC_RT_ITERATIONS", "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def is_root():
    """Whether this process could write ``/dev/cpu_dma_latency``."""
    return hasattr(os, "geteuid") and os.geteuid() == 0


def ascending(values):
    """Whether @p values is non-decreasing, for percentile ordering."""
    return all(a <= b for a, b in itertools.pairwise(values))
