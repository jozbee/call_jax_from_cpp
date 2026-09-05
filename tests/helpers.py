"""Small utilities the integration tests share, on top of ``conftest``.

``tests/conftest.py`` owns everything substantial: the build, the plugin, the
exported artifacts, the damaged copies, ``run``/``load_json``/
``parse_kv_lines`` and the two conditional markers.  What is left here is the
handful of things those do not cover -- caching a run that several tests
read, the environment knobs that only the integration tests honour, and two
predicates that would otherwise be spelled slightly differently in six files.
"""

from __future__ import annotations

import itertools
import os

#: Results of the expensive runs, keyed by name.
_CACHE: dict[str, object] = {}


def cached(key, factory):
    """Call @p factory once per session and hand every caller its result.

    A module-scoped fixture is the usual way to share a twenty-second run
    between the six tests that assert on it.  It is not available here:
    ``run`` and ``load_json`` are function-scoped fixtures, and a
    module-scoped fixture requesting one is a collection error rather than a
    slow test.  Caching the value works whatever their scope is.
    """
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def preload(guard_so):
    """The environment change that arms the allocation census.

    Absolute, because the loader resolves a bare ``LD_PRELOAD`` name against
    the library search path rather than the working directory -- a relative
    one silently does nothing, and the census then reports an absence that
    reads exactly like zero allocations.
    """
    return {"LD_PRELOAD": str(guard_so.resolve())}


def alloc_strict():
    """Whether ``$CJFC_ALLOC_STRICT=1`` asked for the stricter gate.

    ``self`` -- the calling binary's own allocations -- is always gated.
    ``runtime``, the wrapper library's, is zero on this host too but is only
    asserted under this flag, so that a libstdc++ or plugin attributing an
    allocation differently does not fail an otherwise clean run.
    """
    return os.environ.get("CJFC_ALLOC_STRICT", "") == "1"


def rt_iterations(default=2000):
    """``$CJFC_RT_ITERATIONS``, or @p default when unset or unusable."""
    raw = os.environ.get("CJFC_RT_ITERATIONS", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def is_root():
    """Whether this process could write ``/dev/cpu_dma_latency``."""
    return hasattr(os, "geteuid") and os.geteuid() == 0


def ascending(values):
    """Whether @p values is non-decreasing, for percentile ordering."""
    return all(a <= b for a, b in itertools.pairwise(values))
