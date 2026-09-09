"""Build and write the JSON sidecar that travels with an executable.

The PJRT C API has no query for parameter shapes, so the sidecar is the
loader's only description of the inputs.  Nothing here imports JAX: this is
what ``python -m jax2exec check`` runs on a machine that has only artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._dtypes import dtype_info, dtype_name
from ._isa import cpu_model, host_isa_level

__all__ = [
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "TOOL_VERSION",
    "array_entry",
    "atomic_write_bytes",
    "build_sidecar",
    "load_sidecar",
    "normalize_arrays",
    "sha256_hex",
    "write_sidecar",
]

#: Sidecar layout this package writes.  The C++ loader accepts 1 and 2 and
#: refuses anything newer, so this moves only when the loader moves with it.
SCHEMA_VERSION = 2

#: Recorded in ``generator``.  The version must match ``pyproject.toml``.
TOOL_NAME = "jax2exec"
TOOL_VERSION = "0.2.0"


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``.

    Parameters
    ----------
    data : bytes
        Artifact bytes.

    Returns
    -------
    str
        Lowercase hex digest, which the sidecar records per artifact.
    """
    return hashlib.sha256(data).hexdigest()


def array_entry(
    index: int,
    name: str,
    dtype: Any,
    shape: Sequence[int],
    *,
    donated: bool | None = None,
) -> dict[str, Any]:
    """Describe one input or output array for the sidecar.

    Parameters
    ----------
    index : int
        Position in the flattened input or output list.
    name : str
        Name the C++ side can look the array up by.
    dtype : Any
        A supported dtype.
    shape : Sequence[int]
        The exact JAX shape, row-major; empty for a scalar.
    donated : bool or None, optional
        Recorded for inputs only; outputs omit the key.

    Returns
    -------
    dict
        ``index``, ``name``, ``dtype``, ``shape``, ``numel``, ``nbytes`` and,
        for inputs, ``donated``.  ``nbytes`` is what the loader allocates.
    """
    dims = [int(d) for d in shape]
    numel = math.prod(dims)  # math.prod(()) == 1, which is the scalar case
    info = dtype_info(dtype)
    entry: dict[str, Any] = {
        "index": int(index),
        "name": name,
        "dtype": info.name,
        "shape": dims,
        "numel": numel,
        "nbytes": numel * info.itemsize,
    }
    if donated is not None:
        entry["donated"] = bool(donated)
    return entry


def build_sidecar(
    *,
    name: str,
    inputs: Sequence[dict[str, Any]],
    outputs: Sequence[dict[str, Any]],
    jax_version: str,
    jaxlib_version: str,
    x64_enabled: bool,
    executable: str,
    executable_sha256: str,
    mlir: str | None = None,
    mlir_sha256: str | None = None,
    calling_convention_version: int | None = None,
    executable_source: str | None = None,
    donate_argnums: Iterable[int] = (),
) -> dict[str, Any]:
    """Assemble the schema 2 sidecar.

    Parameters
    ----------
    name : str
        Artifact base name, without extension.
    inputs, outputs : Sequence[dict]
        Entries from :func:`array_entry`, in executable order.
    jax_version, jaxlib_version : str
        The versions that produced the executable.
    x64_enabled : bool
        Whether ``jax_enable_x64`` was set while tracing.
    executable, executable_sha256 : str
        File name and digest of the serialized executable.
    mlir, mlir_sha256 : str or None, optional
        File name and digest of the StableHLO bytecode, when it was written.
    calling_convention_version : int or None, optional
        ``jax.export``'s calling convention version for that bytecode.
    executable_source : str or None, optional
        How the PJRT bytes were got out of jaxlib: ``"ifrt-unwrapped"`` or
        ``"as-is"`` (see ``jax2exec._ifrt``).  A JAX bump that changes the
        envelope shows up here first.
    donate_argnums : Iterable[int], optional
        As passed to ``jax.jit``.

    Returns
    -------
    dict
        The sidecar, ready for :func:`write_sidecar`.

    Notes
    -----
    The ``mlir`` keys are omitted rather than written as ``null`` when there
    is no bytecode, so a reader can test for the fallback by presence alone.
    """
    artifacts: dict[str, Any] = {
        "executable": executable,
        "executable_sha256": executable_sha256,
    }
    if executable_source is not None:
        artifacts["executable_source"] = executable_source
    if mlir is not None:
        artifacts["mlir"] = mlir
        artifacts["mlir_sha256"] = mlir_sha256
        if calling_convention_version is not None:
            artifacts["stablehlo_calling_convention_version"] = int(
                calling_convention_version
            )

    return {
        "schema": SCHEMA_VERSION,
        "name": name,
        "generator": {"tool": TOOL_NAME, "version": TOOL_VERSION},
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "platform": "cpu",  # the exporter lowers for CPU and nothing else
        "export": {
            "time_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "x64_enabled": bool(x64_enabled),
            "host": {
                "os": platform.system().lower(),
                "arch": platform.machine(),
                "isa_level": host_isa_level(),
                "python": platform.python_version(),
                "cpu_model": cpu_model(),
            },
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        },
        "artifacts": artifacts,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "donation": {"donate_argnums": [int(i) for i in donate_argnums]},
    }


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so a reader sees all of it or none of it.

    Parameters
    ----------
    path : Path
        Destination.  Its parent directory must exist.
    data : bytes
        Contents.

    Notes
    -----
    A temporary file in the same directory is written, fsynced and renamed
    over the destination, so a half-written ``.binpb`` is never visible under
    its real name.
    """
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def write_sidecar(path: Path, sidecar: dict[str, Any]) -> None:
    """Write ``sidecar`` as JSON, atomically.

    Parameters
    ----------
    path : Path
        Destination ``.json`` file.
    sidecar : dict
        As built by :func:`build_sidecar`.

    Notes
    -----
    Key order is the insertion order of :func:`build_sidecar`, not sorted, so
    a human opening the file reads the identifying fields first.
    """
    text = json.dumps(sidecar, indent=2, sort_keys=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def load_sidecar(path: Path) -> dict[str, Any]:
    """Read a sidecar from disk.

    Parameters
    ----------
    path : Path
        The ``.json`` file.

    Returns
    -------
    dict
        The parsed sidecar, unvalidated: ``check`` has to report on a v1
        sidecar and on one this package is too old to understand.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_arrays(sidecar: dict[str, Any]) -> tuple[list, list]:
    """Return ``(inputs, outputs)`` for a v1 or v2 sidecar.

    Parameters
    ----------
    sidecar : dict
        A parsed sidecar of either schema.

    Returns
    -------
    tuple of list
        Entries in the v2 shape.  A v1 sidecar records only sizes and a single
        float64 dtype, with 0 meaning a scalar; it is widened to the view the
        loader takes of it, with names ``arg<i>``/``out<i>``.
    """
    if int(sidecar.get("schema", 1)) >= 2:
        return list(sidecar.get("inputs", [])), list(sidecar.get("outputs", []))

    def widen(info: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
        dtype = dtype_name(info.get("dtype", "float64"))
        entries = []
        for index, size in enumerate(info.get("sizes", [])):
            shape = [] if int(size) == 0 else [int(size)]
            entries.append(array_entry(index, f"{prefix}{index}", dtype, shape))
        return entries

    return (
        widen(sidecar.get("args_info", {}), "arg"),
        widen(sidecar.get("out_info", {}), "out"),
    )
