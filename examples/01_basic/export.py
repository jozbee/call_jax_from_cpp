"""Export the smallest function still worth calling from C++.

``fun(A, b) -> (x, r)`` solves a dense 4x4 system and returns the residual
JAX computed for it.  ``A`` is rank 2 so that the row-major layout path is
exercised on both sides of the boundary; a layout disagreement is invisible
in a rank-1 example.  ``jnp.linalg.inv`` lowers to a LAPACK FFI custom call
that a stock PJRT CPU plugin cannot load, so a wrong plugin fails here, at
``pjrt::Function`` construction, rather than later in a control loop (see
``docs/developer/xla-fork.md``).

    uv run python examples/01_basic/export.py --out artifacts
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax2exec import export


def _out_dir() -> str:
    """The directory the three files are written to."""
    parser = argparse.ArgumentParser(
        description="Export fun(A, b) -> (x, r) for examples/01_basic."
    )
    parser.add_argument(
        "--out",
        default="artifacts",
        help="directory for the .binpb, .mlirbc and .json "
        "(default: %(default)s)",
    )
    return parser.parse_args().out


def describe(result) -> None:
    """Print what was written, and the signature the sidecar declares."""
    print(f"executable {result.executable}")
    print(f"mlir       {result.mlir}")
    print(f"sidecar    {result.sidecar}")
    for role in ("inputs", "outputs"):
        for entry in result.metadata[role]:
            print(
                f"{role[:-1]:7} {entry['index']} {entry['name']}: "
                f"{entry['dtype']}{entry['shape']} "
                f"({entry['nbytes']} bytes)"
            )


# docs: begin export
jax.config.update("jax_enable_x64", True)  # before anything is traced

#: Order of the system; 4 so the solution fits on one line of output.
N = 4


def fun(A, b):
    """Solve ``A x = b``; ``r`` is the residual, so C++ can check twice."""
    x = jnp.linalg.inv(A) @ b
    r = jnp.linalg.norm(A @ x - b)
    return x, r


if __name__ == "__main__":
    result = export(
        fun,
        (
            jax.ShapeDtypeStruct((N, N), jnp.float64),  # A
            jax.ShapeDtypeStruct((N,), jnp.float64),  # b
        ),
        directory=_out_dir(),  # "artifacts"
        name="basic",
    )
    # docs: end export
    describe(result)
