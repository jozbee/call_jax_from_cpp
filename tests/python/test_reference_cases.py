"""The frozen results the C++ side is checked against.

A C++ call path that runs is not the same as a C++ call path that is right.
``write_reference_cases`` runs the function under JAX and writes, per case,
every input and every output as raw bytes in call order, plus a manifest
saying what those bytes are.  The C++ test then memcpys them into its arenas
and compares -- so if the manifest and the bytes ever disagree, the strongest
test in the suite starts comparing the wrong things.

These tests read the bytes back the way that C++ reader does: from the
manifest alone, with no header, no padding and no length prefixes, because
that is exactly what is on disk.
"""

from __future__ import annotations

import json
import math

import pytest

jax = pytest.importorskip("jax", reason="freezing a reference case runs JAX")

# Process-global, and set before anything traces: the reference bytes have to
# be the widths the executable will really be handed.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax2exec import SCHEMA_VERSION

# From the submodules rather than the package: see the note in
# tests/python/test_exporter.py about the lazy loader's shadowed `export`.
from jax2exec.export import ExportError
from jax2exec.reference import write_reference_cases


def mixed(state, step, use_terminal):
    """Three dtypes in, three dtypes out.

    float64 for the quantity, int32 for a counter and bool for a mode flag:
    the same mix the trajopt fixture has, small enough to check by hand.
    """
    scaled = state * jnp.where(use_terminal, 2.0, 0.5)
    return {
        "total": scaled.sum(),
        "next_step": step + jnp.int32(1),
        "narrowed": scaled.astype(jnp.float32),
    }


#: Two cases, differing in every argument, so a reader that ignored the case
#: index or reused a buffer would produce identical bytes and be caught.
CASES = (
    (np.arange(6.0).reshape(2, 3), np.int32(7), np.bool_(True)),
    (np.linspace(-1.0, 1.0, 6).reshape(2, 3), np.int32(12345), np.bool_(False)),
)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """Write the cases once, and return ``(directory, manifest)``."""
    directory = tmp_path_factory.mktemp("cases")
    manifest_path = write_reference_cases(mixed, CASES, directory, "mixed")
    assert manifest_path == directory / "mixed_cases.json"
    return directory, json.loads(manifest_path.read_text(encoding="utf-8"))


def test_manifest_describes_the_layout(frozen):
    """Schema, names, dtypes, shapes and tolerances, in call order."""
    _directory, manifest = frozen
    assert manifest["schema"] == SCHEMA_VERSION == 2
    assert manifest["name"] == "mixed"
    assert manifest["cases"] == ["mixed_case0.bin", "mixed_case1.bin"]

    assert [entry["name"] for entry in manifest["inputs"]] == [
        "state",
        "step",
        "use_terminal",
    ]
    assert [entry["dtype"] for entry in manifest["inputs"]] == [
        "float64",
        "int32",
        "bool",
    ]
    assert [entry["shape"] for entry in manifest["inputs"]] == [[2, 3], [], []]

    # A dict result names its own leaves, in the order JAX flattens them.
    assert [entry["name"] for entry in manifest["outputs"]] == [
        "narrowed",
        "next_step",
        "total",
    ]
    assert [entry["dtype"] for entry in manifest["outputs"]] == [
        "float32",
        "int32",
        "float64",
    ]

    # Float32 gets more room than float64 because XLA is free to fuse and
    # reassociate, and two paths need not reassociate the same way.
    assert manifest["tolerance"]["float32"] > manifest["tolerance"]["float64"]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_case_bytes_are_what_jax_computed(frozen, case):
    """Read each case the way the C++ reader does, and recompute it.

    Every array is read at the offset the manifest implies -- no header, no
    padding -- and the file has to end exactly where the last output ends.
    """
    directory, manifest = frozen
    blob = (directory / manifest["cases"][case]).read_bytes()

    arrays, offset = _read_case(blob, manifest)
    assert offset == len(blob), "the file is exactly its arrays, in order"

    arguments = CASES[case]
    converted = jax.tree_util.tree_map(jnp.asarray, tuple(arguments))
    expected = [
        np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(converted)
    ] + [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(jax.jit(mixed)(*converted))
    ]

    entries = [*manifest["inputs"], *manifest["outputs"]]
    tolerance = manifest["tolerance"]
    for entry, got, want in zip(entries, arrays, expected, strict=True):
        assert got.dtype == want.dtype, entry["name"]
        assert got.shape == want.shape, entry["name"]
        if entry["dtype"] in tolerance:
            np.testing.assert_allclose(
                got,
                want,
                rtol=tolerance[entry["dtype"]],
                atol=0.0,
                err_msg=entry["name"],
            )
        else:
            # Integers and bools have no tolerance to speak of: a counter that
            # is one out is not a rounding difference.
            np.testing.assert_array_equal(got, want, err_msg=entry["name"])


def test_cases_differ(frozen):
    """Two cases, two different files.

    Cheap, and it catches the failure mode a round trip cannot: a writer that
    froze case 0 twice would satisfy every assertion above.
    """
    directory, manifest = frozen
    payloads = {(directory / name).read_bytes() for name in manifest["cases"]}
    assert len(payloads) == len(manifest["cases"])


def _read_case(blob: bytes, manifest: dict) -> tuple[list[np.ndarray], int]:
    """Split one case file into arrays, using nothing but the manifest."""
    arrays: list[np.ndarray] = []
    offset = 0
    for entry in [*manifest["inputs"], *manifest["outputs"]]:
        dtype = np.dtype(entry["dtype"])
        shape = tuple(entry["shape"])
        count = math.prod(shape)  # math.prod(()) == 1: the scalar case
        arrays.append(
            np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
            .reshape(shape)
            .copy()
        )
        offset += count * dtype.itemsize
    return arrays, offset


def test_a_non_finite_output_is_refused(tmp_path):
    """A NaN reference silently passes every relative-error comparison.

    Every comparison against NaN is false, so nothing exceeds the tolerance and
    the C++ comparison becomes a test that cannot fail.  Refusing to freeze one
    is what keeps that from happening quietly.
    """
    with pytest.raises(ExportError, match="not finite"):
        write_reference_cases(
            lambda x: jnp.log(x),
            [(np.array([0.0, 1.0]),)],
            tmp_path,
            "nonfinite",
        )
    # Cases are written after every case has been checked, and the manifest
    # last, so a refusal leaves nothing for a reader to find.
    assert list(tmp_path.iterdir()) == []


def test_no_cases_is_refused(tmp_path):
    """An empty case list would write a manifest naming nothing."""
    with pytest.raises(ExportError, match="no cases"):
        write_reference_cases(mixed, [], tmp_path, "empty")
    assert list(tmp_path.iterdir()) == []
