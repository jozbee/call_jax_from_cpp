"""What ``jax2exec.export`` writes, and what it refuses to write.

The sidecar is the only description of an executable's parameters that
exists: the PJRT C API answers ``PJRT_Executable_NumOutputs`` and the output
types, but has no query for the inputs at all.  Everything the C++ loader
believes about the input arenas -- how many, how wide, what to call them --
comes from this file's subject, so these tests assert on the sidecar field by
field rather than on "an export happened".

The refusals get the same weight as the successes.  Each one exists because
the alternative is a C++ caller writing into an arena of the wrong width and
discovering it as corrupted output somewhere else, which is why every
rejection is also checked to leave nothing on disk.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax", reason="the exporter is JAX")
jaxlib = pytest.importorskip("jaxlib")

# Before anything is traced, and process-global: without it JAX narrows every
# float64 to float32 silently, which is the trap `test_x64_narrowing_*` below
# exercises -- in a subprocess, because flipping this back is not possible.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax2exec import SCHEMA_VERSION, SUPPORTED_DTYPES
from jax2exec._dtypes import SUPPORTED_SUMMARY
from jax2exec._sidecar import TOOL_NAME, TOOL_VERSION

# Imported from the submodule rather than from the package, deliberately:
# `from jax2exec import ExportError, export` binds `export` to the *module*
# of that name, not to the function, whenever another lazily loaded attribute
# is touched first.  That is a live defect in the package's lazy loader --
# `test_export_survives_a_lazy_sibling_import` below records it -- and this
# import route is unambiguous either way.
from jax2exec.export import ExportError, export

#: The repository root; this file is ``tests/python/test_exporter.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Element types XLA has and this project deliberately does not.  A C++ caller
#: has no way to spell them, so the exporter names them rather than letting
#: bytes be reinterpreted downstream.
UNSUPPORTED = ("float16", "bfloat16", "complex64")


#: A fresh interpreter importing the public API the way an editor's import
#: sorter writes it: names within one ``from`` statement come out
#: alphabetically, with ``export`` last.
_PUBLIC_IMPORT = """
from jax2exec import ExportError, export

print(f"kind={type(export).__name__}")
print(f"callable={callable(export)}")
"""


def test_export_survives_a_lazy_sibling_import(run):
    """``from jax2exec import ExportError, export`` must give the function.

    Run in a subprocess because the answer depends on which lazy attribute the
    process touched first, and by the time this test runs the session has
    touched several.  The examples in this repository happen to import
    ``export`` first and so are unaffected; a caller who does not, or whose
    formatter sorts the names, gets ``TypeError: 'module' object is not
    callable`` from the documented public API.
    """
    completed = run(
        [sys.executable, "-c", _PUBLIC_IMPORT],
        env={"PYTHONPATH": str(REPO_ROOT / "python")},
        timeout=600,
    )
    assert "kind=function" in completed.stdout
    assert "callable=True" in completed.stdout


def two_in_two_out(matrix, vector):
    """A matrix, a vector, and a result that depends on both.

    Every input has to reach an output or XLA drops it from the executable and
    the exporter refuses, so the smallest honest example still multiplies.
    """
    scaled = matrix @ vector
    return {"scaled": scaled, "total": scaled.sum()}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One export of :func:`two_in_two_out`, shared by the sidecar tests."""
    directory = tmp_path_factory.mktemp("sidecar")
    return export(
        two_in_two_out,
        (
            jax.ShapeDtypeStruct((3, 4), jnp.float64),
            jax.ShapeDtypeStruct((4,), jnp.float64),
        ),
        directory=directory,
        name="shape",
    )


# ------------------------------------------------------------- schema 2 shape


def test_sidecar_identifies_itself(exported):
    """Schema, generator and versions: what a stale artifact is diagnosed by."""
    sidecar = exported.metadata
    assert sidecar["schema"] == SCHEMA_VERSION == 2
    assert sidecar["name"] == "shape"
    assert sidecar["generator"] == {"tool": TOOL_NAME, "version": TOOL_VERSION}
    assert sidecar["jax_version"] == jax.__version__
    assert sidecar["jaxlib_version"] == jaxlib.__version__
    assert sidecar["platform"] == "cpu"


def test_sidecar_records_the_exporting_host(exported):
    """The host block is what the loader compares against before it opens a
    file: a ``.binpb`` embeds machine code for the machine that produced it."""
    export_block = exported.metadata["export"]
    assert export_block["x64_enabled"] is True
    host = export_block["host"]
    assert host["arch"] == platform.machine()
    assert host["os"] == platform.system().lower()
    assert host["python"] == platform.python_version()
    # Never empty: "unknown" is the answer for a host whose flags could not be
    # read, and claiming a level would be worse than admitting ignorance.
    assert host["isa_level"]


def test_sidecar_describes_every_array(exported):
    """Each entry carries the four numbers the C++ arena allocator needs."""
    sidecar = exported.metadata
    assert [entry["name"] for entry in sidecar["inputs"]] == [
        "matrix",
        "vector",
    ]
    # A dict result names its own leaves, in the order JAX flattens them.
    assert [entry["name"] for entry in sidecar["outputs"]] == [
        "scaled",
        "total",
    ]
    assert [entry["shape"] for entry in sidecar["inputs"]] == [[3, 4], [4]]
    assert [entry["shape"] for entry in sidecar["outputs"]] == [[3], []]

    for kind in ("inputs", "outputs"):
        for position, entry in enumerate(sidecar[kind]):
            assert entry["index"] == position, f"{kind}[{position}] index"
            info = SUPPORTED_DTYPES[entry["dtype"]]
            assert entry["numel"] == max(
                1, int(np.prod(entry["shape"], dtype=np.int64))
            )
            # nbytes is what the loader allocates and what a caller memcpys
            # into; a disagreement here is a heap overrun, not a wrong answer.
            assert entry["nbytes"] == entry["numel"] * info.itemsize

    # Inputs record donation, outputs do not have the key at all.
    assert all("donated" in entry for entry in sidecar["inputs"])
    assert all("donated" not in entry for entry in sidecar["outputs"])
    assert sidecar["donation"] == {"donate_argnums": []}


def test_sidecar_digests_match_the_files(exported):
    """The digests are how a truncated or swapped artifact is told apart from
    a matching one, so they have to be of what actually landed."""
    artifacts = exported.metadata["artifacts"]
    assert artifacts["executable"] == exported.executable.name
    assert artifacts["executable_sha256"] == _sha256(exported.executable)
    assert artifacts["mlir"] == exported.mlir.name
    assert artifacts["mlir_sha256"] == _sha256(exported.mlir)


def test_sidecar_records_how_the_pjrt_bytes_were_obtained(exported):
    """``executable_source`` is where a JAX bump that changes jaxlib's IFRT
    envelope shows up first; see ``jax2exec._ifrt``."""
    source = exported.metadata["artifacts"]["executable_source"]
    assert source in {"ifrt-unwrapped", "as-is"}


def test_sidecar_on_disk_is_the_returned_metadata(exported, load_json):
    """``ExportResult.metadata`` is the file, not a description of it."""
    assert load_json(exported.sidecar) == exported.metadata


def test_mlir_bytecode_is_written_and_not_empty(exported):
    """The ``.mlirbc`` is the answer to the architecture lock: it is what the
    loader compiles in-process when the ``.binpb`` was built elsewhere."""
    assert exported.mlir is not None
    assert exported.mlir.is_file()
    assert exported.mlir.stat().st_size > 0
    assert exported.executable.stat().st_size > 0


def _sha256(path: Path) -> str:
    """Hex digest of a file, the way the sidecar records it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------- rank-N


def test_rank_n_shapes_survive_the_round_trip(tmp_path):
    """Rank is carried through untouched, in both directions.

    A rank-1 example cannot detect a layout disagreement -- every element still
    lands where it was expected to -- so the shapes that matter are the ones
    with more than one axis.
    """

    def reshape(block):
        return jnp.broadcast_to(block.sum(), (2, 2, 2)) * 1.0

    result = export(
        reshape,
        (jax.ShapeDtypeStruct((3, 4), jnp.float64),),
        directory=tmp_path,
        name="rankn",
    )
    (only_input,) = result.metadata["inputs"]
    (only_output,) = result.metadata["outputs"]
    assert only_input["shape"] == [3, 4]
    assert only_input["numel"] == 12
    assert only_input["nbytes"] == 12 * 8
    assert only_output["shape"] == [2, 2, 2]
    assert only_output["numel"] == 8
    assert only_output["nbytes"] == 8 * 8


# --------------------------------------------------------------------- dtypes


@pytest.mark.parametrize("dtype_name", sorted(SUPPORTED_DTYPES))
def test_every_supported_dtype_exports(dtype_name, tmp_path):
    """One export per row of the dtype table.

    The table is the contract four consumers have to agree on -- NumPy, PJRT,
    C++ and the sidecar -- and a row that was added to it without being
    exportable is exactly the half-added dtype the table exists to prevent.
    """
    info = SUPPORTED_DTYPES[dtype_name]
    dtype = jnp.dtype(dtype_name)

    def double(x):
        # Bool has no arithmetic to speak of; the point is only that the
        # element type survives from the argument to the result.
        return jnp.logical_not(x) if dtype_name == "bool" else x + x

    result = export(
        double,
        (jax.ShapeDtypeStruct((3,), dtype),),
        directory=tmp_path,
        name=f"dtype_{dtype_name}",
    )
    (only_input,) = result.metadata["inputs"]
    (only_output,) = result.metadata["outputs"]
    assert only_input["dtype"] == dtype_name
    assert only_output["dtype"] == dtype_name
    assert only_input["nbytes"] == 3 * info.itemsize
    assert result.executable.is_file()


# ----------------------------------------------------------------- rejections


@pytest.mark.parametrize("dtype_name", UNSUPPORTED)
def test_unsupported_dtypes_are_named_not_reinterpreted(dtype_name, tmp_path):
    """The refusal names the offending argument and the whole supported set."""
    dtype = jnp.dtype(dtype_name)
    with pytest.raises(ExportError) as raised:
        export(
            lambda x: x + x,
            (jax.ShapeDtypeStruct((3,), dtype),),
            directory=tmp_path,
            name="unsupported",
        )
    message = str(raised.value)
    assert dtype_name in message
    assert SUPPORTED_SUMMARY in message
    assert list(tmp_path.iterdir()) == []


def test_keyword_arguments_are_rejected(tmp_path):
    """A C++ call is positional, so the arguments have to be too.

    ``export`` only ever calls ``lower(*args)``, which is what makes its
    internal keyword-argument guard unreachable from here; the reachable half
    of the same rule is this one, and a caller reaching for keywords reaches
    for a mapping.
    """
    with pytest.raises(ExportError) as raised:
        export(
            two_in_two_out,
            {
                "matrix": jax.ShapeDtypeStruct((3, 4), jnp.float64),
                "vector": jax.ShapeDtypeStruct((4,), jnp.float64),
            },
            directory=tmp_path,
            name="kwargs",
        )
    assert "sequence" in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_zero_element_arrays_are_rejected(tmp_path):
    """Nothing establishes what a zero-byte PJRT buffer does on this path, so
    the exporter refuses rather than finding out in a control loop."""
    with pytest.raises(ExportError, match="no elements"):
        export(
            lambda x: x + x,
            (jax.ShapeDtypeStruct((0,), jnp.float64),),
            directory=tmp_path,
            name="empty",
        )
    assert list(tmp_path.iterdir()) == []


def test_a_rejected_export_writes_nothing_at_all(tmp_path):
    """The whole directory stays empty, not just the artifacts of this name.

    The exporter this replaced wrote the executable first and asserted
    afterwards, so a rejected function left a stale ``.binpb`` beside a sidecar
    describing something else -- which the C++ side then loaded, and which
    failed a long way from the cause.
    """
    target = tmp_path / "nothing-should-land-here"
    target.mkdir()
    with pytest.raises(ExportError):
        export(
            lambda x: x + x,
            (jax.ShapeDtypeStruct((3,), jnp.float16),),
            directory=target,
            name="rejected",
        )
    assert list(target.iterdir()) == []


# ------------------------------------------------------------------ the x64 trap

#: Run in a subprocess: ``jax_enable_x64`` is process-global and cannot be
#: turned back on for the tests that follow, so a test that needs it off needs
#: an interpreter of its own.  Exits 0 only when the export was refused.
_X64_TRAP = """
import pathlib
import sys

import jax
import jax.numpy as jnp

import jax2exec

out = pathlib.Path(sys.argv[1])
if jax.config.jax_enable_x64:
    print("x64_unexpectedly_enabled=1")
    raise SystemExit(2)

try:
    jax2exec.export(
        lambda x: x + x,
        (jax.ShapeDtypeStruct((4,), jnp.float64),),
        directory=out,
        name="x64trap",
    )
except jax2exec.ExportError as exc:
    print(f"wrote={len(list(out.iterdir()))}")
    print(f"message={exc}")
    raise SystemExit(0)

print("wrote=" + str(len(list(out.iterdir()))))
print("export_was_not_refused=1")
raise SystemExit(1)
"""


def test_x64_narrowing_is_refused_not_recorded(run, tmp_path):
    """float64 without ``jax_enable_x64`` must be an error, not a float32
    sidecar.

    JAX narrows the argument and says nothing; the export then succeeds and the
    sidecar honestly records float32, and a C++ caller writing doubles into a
    4-byte-per-element arena walks off the end of it.  The message has to name
    the switch, because that is the whole fix.
    """
    completed = run(
        [sys.executable, "-c", _X64_TRAP, str(tmp_path)],
        env={
            "PYTHONPATH": str(REPO_ROOT / "python"),
            # Explicit: an outer JAX_ENABLE_X64 would make the child exit 2.
            "JAX_ENABLE_X64": "0",
        },
        timeout=600,
    )
    assert "jax.config.update" in completed.stdout
    assert "float64" in completed.stdout and "float32" in completed.stdout
    assert "wrote=0" in completed.stdout
    assert list(tmp_path.iterdir()) == []
