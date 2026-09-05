"""Export JAX functions as artifacts a C++ real-time caller can load.

``export(fun, args, directory, name)`` compiles a function ahead of time and
writes three files: the serialized PJRT executable the C++ loader
deserializes, StableHLO bytecode it can compile instead when that executable
was built for a different machine, and a JSON sidecar describing every input
and output -- which is the only description of the parameters that exists,
because the PJRT C API cannot be asked.

``write_reference_cases`` freezes what the function returns, so the C++ side
can be checked against JAX rather than merely observed to run.

``python -m jax2exec check <base>`` describes an artifact set and says whether
it will run on the host it is being inspected from.  That command deliberately
needs no JAX: the machine running the C++ caller usually has none, so the
imports that pull JAX in are deferred until something actually asks for them.
"""

from __future__ import annotations

import importlib
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

# Attribute -> module that defines it. Both modules import JAX, which costs a
# second or two and, on a deployment machine, may not be installed at all.
_LAZY = {
    "SUPPORTED_JAX": "export",
    "ExportError": "export",
    "ExportResult": "export",
    "export": "export",
    "jax2exec": "export",
    "write_reference_cases": "reference",
}


def __getattr__(name: str) -> Any:
    """Import the module defining ``name`` the first time it is asked for."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # subsequent lookups skip this function entirely
    return value


def __dir__() -> list[str]:
    """List the lazy names too, so tab completion finds them."""
    return sorted(__all__)
