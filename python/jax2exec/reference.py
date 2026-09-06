"""Freeze what a function returns, so the C++ caller can be checked against it.

A C++ call path that runs is not the same as a C++ call path that is right.
These files are the ground truth the C++ tests compare against: for each case,
every input and every output as raw bytes in call order, plus a manifest
saying what those bytes are and how close a match has to be.

The layout is deliberately the dumbest thing that works -- no header, no
padding, no length prefixes -- because the reader is a C++ test that already
knows every shape and dtype from the manifest, and anything cleverer is one
more thing that can disagree between the two languages.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._dtypes import SUPPORTED_DTYPES, dtype_name, unsupported_dtype_message
from ._sidecar import SCHEMA_VERSION, atomic_write_bytes
from .export import (
    ExportError,
    _prepare_directory,
    default_input_names,
    default_output_names,
)

__all__ = ["DEFAULT_TOLERANCE", "write_reference_cases"]

#: Maximum relative error a C++ result may show against the frozen one.
#: Float32 gets more room because XLA is free to fuse and reassociate, and the
#: two paths do not have to reassociate the same way.
DEFAULT_TOLERANCE: Mapping[str, float] = {"float64": 1e-6, "float32": 1e-4}


def _leaves_as_numpy(tree: Any) -> list[np.ndarray]:
    """Flatten a pytree to NumPy arrays, forcing any pending computation."""
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _describe(arrays: Sequence[np.ndarray], names: Sequence[str]) -> list[dict]:
    """Describe each array the way the manifest records it."""
    return [
        {
            "name": names[index],
            "dtype": dtype_name(array.dtype),
            "shape": [int(d) for d in array.shape],
        }
        for index, array in enumerate(arrays)
    ]


def _check_dtypes(
    arrays: Sequence[np.ndarray], names: Sequence[str], kind: str
) -> None:
    """Reject an element type the C++ side has no arena for."""
    for index, array in enumerate(arrays):
        if dtype_name(array.dtype) not in SUPPORTED_DTYPES:
            raise ExportError(
                unsupported_dtype_message(
                    kind, index, names[index], array.dtype
                )
            )


def _check_finite(
    arrays: Sequence[np.ndarray], names: Sequence[str], case: int
) -> None:
    """Refuse to freeze a NaN or an infinity.

    A reference case containing a NaN passes a max-relative-error comparison
    silently -- every comparison against NaN is false, so nothing exceeds the
    tolerance -- which turns the strongest test in the suite into one that
    cannot fail.
    """
    for index, array in enumerate(arrays):
        if array.dtype.kind != "f":
            continue  # integers and bools have no non-finite values
        if not np.isfinite(array).all():
            raise ExportError(
                f"case {case} output {index} ('{names[index]}') is not "
                "finite; a NaN reference silently passes every relative-error "
                "comparison, so fix the case or the function"
            )


def _check_consistent(
    described: Sequence[dict], first: Sequence[dict], kind: str, case: int
) -> None:
    """Require every case to agree: one executable runs all of them."""
    if len(described) != len(first):
        raise ExportError(
            f"case {case} has {len(described)} {kind}s but case 0 has "
            f"{len(first)}"
        )
    for index, (got, want) in enumerate(zip(described, first)):
        if got["dtype"] != want["dtype"] or got["shape"] != want["shape"]:
            raise ExportError(
                f"case {case} {kind} {index} ('{want['name']}') is "
                f"{got['dtype']}{got['shape']} but case 0 is "
                f"{want['dtype']}{want['shape']}; every case runs through the "
                "same executable, so they must agree"
            )


def write_reference_cases(
    fun: Any,
    arg_tuples: Iterable[Sequence[Any]],
    directory: str | Path,
    name: str,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    tolerance: Mapping[str, float] | None = None,
) -> Path:
    """Run ``fun`` on each argument tuple and freeze the results.

    Parameters
    ----------
    fun : callable
        The same function that was exported.  It is jitted here, so the values
        frozen are the ones XLA computes, not the ones an eager interpreter
        would.
    arg_tuples : Iterable of Sequence
        One entry per case, each a full argument tuple of concrete values.
    directory : str or Path
        Where ``{name}_cases.json`` and ``{name}_case<i>.bin`` go.
    name : str
        Base name, matching the exported artifacts.
    input_names, output_names : Sequence of str or None, optional
        Names for the manifest.  Default to the same names
        :func:`jax2exec.export` would give, so the two files line up.
    tolerance : Mapping of str to float, or None, optional
        Per-dtype maximum relative error, merged over
        :data:`~jax2exec.reference.DEFAULT_TOLERANCE`.

    Returns
    -------
    Path
        The written manifest.

    Raises
    ------
    ExportError
        If there are no cases, if a case disagrees with case 0 about shapes or
        dtypes, if an element type is unsupported, or if an output is not
        finite.

    Notes
    -----
    Each ``.bin`` holds every input followed by every output, in call order,
    C order, native width, with no header and no padding; a scalar occupies
    exactly one element.  Values are written as JAX actually traced them, so a
    float64 argument passed without ``jax_enable_x64`` is frozen as the
    float32 the executable will really be handed.
    """
    # The C++ reader memcpys these straight into its arenas, so a big-endian
    # host would need a byte-swapping reader that nothing here provides.
    assert sys.byteorder == "little", "reference cases are little-endian"

    out_dir = _prepare_directory(directory)
    jit_fun = jax.jit(fun)

    case_files: list[str] = []
    payloads: list[bytes] = []
    in_names: list[str] = []
    out_names: list[str] = []
    in_described: list[dict] = []
    out_described: list[dict] = []

    for case, arg_tuple in enumerate(arg_tuples):
        if isinstance(arg_tuple, (str, bytes)) or not isinstance(
            arg_tuple, Sequence
        ):
            raise ExportError(
                f"case {case} is a {type(arg_tuple).__name__}; each case must "
                "be a sequence holding one full argument tuple"
            )

        # Converting first records exactly what the executable will see: with
        # x64 disabled a float64 argument becomes float32 here, as it does in
        # the export.
        converted = jax.tree_util.tree_map(jnp.asarray, tuple(arg_tuple))
        results = jit_fun(*converted)

        inputs = _leaves_as_numpy(converted)
        outputs = _leaves_as_numpy(results)

        if case == 0:
            in_names = default_input_names(fun, len(inputs), input_names)
            out_names = default_output_names(
                results, len(outputs), output_names
            )
            _check_dtypes(inputs, in_names, "input")
            _check_dtypes(outputs, out_names, "output")
            in_described = _describe(inputs, in_names)
            out_described = _describe(outputs, out_names)
        else:
            _check_consistent(
                _describe(inputs, in_names), in_described, "input", case
            )
            _check_consistent(
                _describe(outputs, out_names), out_described, "output", case
            )

        _check_finite(outputs, out_names, case)

        payloads.append(
            b"".join(
                np.ascontiguousarray(array).tobytes()
                for array in (*inputs, *outputs)
            )
        )
        case_files.append(f"{name}_case{case}.bin")

    if not case_files:
        raise ExportError("no cases: arg_tuples is empty")

    manifest = {
        "schema": SCHEMA_VERSION,
        "name": name,
        "inputs": in_described,
        "outputs": out_described,
        "cases": case_files,
        "tolerance": {**DEFAULT_TOLERANCE, **(tolerance or {})},
    }

    # Cases first, manifest last: the manifest is what a reader opens, and it
    # must never name a file that is not there yet.
    for file_name, payload in zip(case_files, payloads):
        atomic_write_bytes(out_dir / file_name, payload)

    manifest_path = out_dir / f"{name}_cases.json"
    atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=False) + "\n").encode(),
    )
    return manifest_path
