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
    """Import the module defining ``name`` the first time it is asked for.

    Every public name the imported submodule provides is bound, not just the
    one that was asked for. That matters because ``export`` is both a submodule
    and the function it defines: importing the submodule for any reason makes
    the import system bind ``jax2exec.export`` to the *module*, which then
    shadows this function forever. Binding the whole group puts the function
    back on top, so ``from jax2exec import ExportError, export`` behaves the
    same as ``from jax2exec import export, ExportError``. Before this, the
    first spelling handed the caller a module and the second a function, and
    an import sorter reordering those names was enough to break a working
    program with ``TypeError: 'module' object is not callable``.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    importlib.import_module(f".{module_name}", __name__)

    # Bind every lazy name whose module is now loaded, not just the ones from
    # the module just asked for: `reference` imports `export`, so fetching
    # `write_reference_cases` is enough to leave the module shadowing the
    # function. Subsequent lookups then skip this function entirely.
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
