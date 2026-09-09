"""The dtype table that the exporter, the sidecar and the C++ loader agree on.

The types XLA has and this project does not (float16, bfloat16, complex,
float8, sub-byte integers, JAX's extended dtypes) are absent on purpose: the
exporter rejects them by name rather than hand C++ bytes it cannot spell.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "SUPPORTED_DTYPES",
    "SUPPORTED_SUMMARY",
    "DTypeInfo",
    "dtype_info",
    "dtype_name",
    "unsupported_dtype_message",
]


# docs: begin dtype-table
@dataclasses.dataclass(frozen=True)
class DTypeInfo:
    """How one element type is spelled by each consumer of an artifact.

    Attributes
    ----------
    name : str
        The NumPy dtype name, which is what the sidecar stores.
    pjrt : str
        The ``PJRT_Buffer_Type`` enumerator, without its prefix.
    cxx : str
        The C++ element type of the arena a caller reads or writes.
    itemsize : int
        Width in bytes; ``nbytes`` in the sidecar is ``numel * itemsize``.
    needs_x64 : bool
        True for the 64-bit types, which JAX narrows to 32 bits unless
        ``jax_enable_x64`` is set before tracing.
    """

    name: str
    pjrt: str
    cxx: str
    itemsize: int
    needs_x64: bool


#: Every element type an exported function may take or return, keyed by NumPy
#: dtype name.
SUPPORTED_DTYPES: Mapping[str, DTypeInfo] = MappingProxyType(
    {
        "bool": DTypeInfo("bool", "PRED", "bool", 1, False),
        "int8": DTypeInfo("int8", "S8", "std::int8_t", 1, False),
        "int16": DTypeInfo("int16", "S16", "std::int16_t", 2, False),
        "int32": DTypeInfo("int32", "S32", "std::int32_t", 4, False),
        "int64": DTypeInfo("int64", "S64", "std::int64_t", 8, True),
        "uint8": DTypeInfo("uint8", "U8", "std::uint8_t", 1, False),
        "uint16": DTypeInfo("uint16", "U16", "std::uint16_t", 2, False),
        "uint32": DTypeInfo("uint32", "U32", "std::uint32_t", 4, False),
        "uint64": DTypeInfo("uint64", "U64", "std::uint64_t", 8, True),
        "float32": DTypeInfo("float32", "F32", "float", 4, False),
        "float64": DTypeInfo("float64", "F64", "double", 8, True),
    }
)
# docs: end dtype-table


#: How an error message lists the table's keys.  Typed out, not generated,
#: so the message stays short.
SUPPORTED_SUMMARY = "bool, int8/16/32/64, uint8/16/32/64, float32, float64"


def dtype_name(dt: Any) -> str:
    """Return the NumPy name of ``dt``.

    Parameters
    ----------
    dt : Any
        Anything ``numpy.dtype`` accepts, or a JAX extended dtype.

    Returns
    -------
    str
        ``numpy.dtype(dt).name``.  JAX's extended dtypes (PRNG keys,
        ``float0``) are not NumPy dtypes, so those return their own name,
        which is enough for the error message about to be raised.
    """
    try:
        return np.dtype(dt).name
    except (TypeError, ValueError):
        return str(getattr(dt, "name", dt))


def dtype_info(dt: Any) -> DTypeInfo:
    """Look ``dt`` up in :data:`SUPPORTED_DTYPES`.

    Parameters
    ----------
    dt : Any
        A dtype or anything convertible to one.

    Returns
    -------
    DTypeInfo
        The table row for ``dt``.

    Raises
    ------
    ValueError
        If ``dt`` is not a supported element type.  A caller with a position
        to report should use :func:`unsupported_dtype_message` instead.
    """
    name = dtype_name(dt)
    info = SUPPORTED_DTYPES.get(name)
    if info is None:
        raise ValueError(
            f"dtype {name} is not supported by jax2exec "
            f"(supported: {SUPPORTED_SUMMARY}); cast inside the function"
        )
    return info


def unsupported_dtype_message(kind: str, index: int, name: str, dt: Any) -> str:
    """Build the message for an argument whose dtype has no table row.

    Parameters
    ----------
    kind : str
        ``"input"`` or ``"output"``.
    index : int
        Position in the flattened input or output list.
    name : str
        The name the sidecar would have given it.
    dt : Any
        The offending dtype.

    Returns
    -------
    str
        A message naming the position, the dtype and the way out.
    """
    return (
        f"{kind} {index} ('{name}') has dtype {dtype_name(dt)}, which "
        f"jax2exec does not support (supported: {SUPPORTED_SUMMARY}); "
        "cast inside the function"
    )
