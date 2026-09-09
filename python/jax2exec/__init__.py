"""Export JAX functions as artifacts a C++ real-time caller can load.

``export`` writes the executable, its StableHLO fallback and the sidecar;
``write_reference_cases`` freezes what the function returns; and
``python -m jax2exec check <base>`` inspects an artifact set without JAX,
which the machine running the C++ caller usually does not have.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

from ._dtypes import SUPPORTED_DTYPES
from ._sidecar import SCHEMA_VERSION, TOOL_VERSION

if TYPE_CHECKING:  # pragma: no cover - for type checkers and readers
    from .export import (
        SUPPORTED_JAX,
        ExportError,
        ExportResult,
        export,
        jax2exec,
    )
    from .reference import write_reference_cases

#: Version of this package, recorded in every sidecar's ``generator`` block.
__version__ = TOOL_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_DTYPES",
    "SUPPORTED_JAX",
    "ExportError",
    "ExportResult",
    "__version__",
    "export",
    "jax2exec",
    "write_reference_cases",
]

# Attribute -> module that defines it.  Both modules import JAX, which the
# machine running `check` may not have.
_LAZY = {
    "SUPPORTED_JAX": "export",
    "ExportError": "export",
    "ExportResult": "export",
    "export": "export",
    "jax2exec": "export",
    "write_reference_cases": "reference",
}


def __getattr__(name: str) -> Any:
    """Import the module defining ``name`` the first time it is asked for.

    ``export`` is both a submodule and the function it defines.  Importing
    the submodule for any reason -- ``reference`` imports it -- binds
    ``jax2exec.export`` to the module, and a name bound in ``globals()``
    never reaches this function again.  Rebinding every loaded lazy name puts
    the function back on top whichever name was asked for first.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    importlib.import_module(f".{module_name}", __name__)

    for lazy_name, lazy_module in _LAZY.items():
        loaded = sys.modules.get(f"{__name__}.{lazy_module}")
        if loaded is not None and hasattr(loaded, lazy_name):
            globals()[lazy_name] = getattr(loaded, lazy_name)

    try:
        return globals()[name]
    except KeyError:  # pragma: no cover - a _LAZY entry naming a missing symbol
        raise AttributeError(
            f"module {__name__!r} maps {name!r} to {module_name!r}, "
            f"which does not define it"
        ) from None


def __dir__() -> list[str]:
    """List the lazy names too, so tab completion finds them."""
    return sorted(__all__)
