"""Export the smallest function still worth calling from C++.

``fun(A, b) -> (x, r)`` solves a dense 4x4 linear system: ``x`` solves
``A x = b``, and ``r`` is the residual norm JAX computed for that solution.
``A`` is rank 2 rather than a flattened vector on purpose.  A matrix argument
is the first thing that exercises the row-major layout path on both sides of
the boundary, and a disagreement about layout is invisible in a rank-1
example -- every element still lands where it was expected to.

It is also a deliberate test of the plugin.  ``jnp.linalg.inv`` lowers to a
LAPACK FFI custom call, and a stock PJRT CPU plugin registers no LAPACK FFI
handlers: jaxlib registers those from Python on import, and the C++ process
imports nothing.  Loading this artifact against a stock plugin therefore fails
at ``pjrt::Function`` construction with::

    No FFI handler registered for lapack_dgetrf_ffi on a platform Host

which is the point.  A function without a custom call would load happily
against the wrong plugin and leave that discovery for a control loop to make
later, in the field.  The plugin ``make plugin`` provides carries those
handlers; see ``docs/developer/xla-fork.md``.

Run it directly, or through ``make export``::

    uv run python examples/01_basic/export.py --out artifacts --name basic
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax2exec import export

#: Order of the system.  Small enough that the whole solution fits on one line
#: of the example's output, which is the only reason it is 4 and not 400.
N = 4


def fun(A, b):
    """Solve ``A x = b`` and report how far off the solution actually is.

    Returning the residual as well as the solution is what lets the C++ side
    check itself twice over: once against a residual it recomputes from its
    own arenas, and once against this number, which XLA produced inside the
    same executable.  The two disagreeing means the C++ side is not reading
    the arrays the way the exporter described them.
    """
    x = jnp.linalg.inv(A) @ b
    r = jnp.linalg.norm(A @ x - b)
    return x, r


def main() -> int:
    """Parse the flags, export, and describe what was written."""
    parser = argparse.ArgumentParser(
        description="Export fun(A, b) -> (x, r) for examples/01_basic."
    )
    parser.add_argument(
        "--out",
        default="artifacts",
        help="directory for the .binpb, .mlirbc and .json (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        default="basic",
        help="base name of the three artifacts (default: %(default)s)",
    )
    args = parser.parse_args()

    # docs: begin export
    jax.config.update("jax_enable_x64", True)  # before anything is traced

    result = export(
        fun,
        (
            jax.ShapeDtypeStruct((N, N), jnp.float64),  # A
            jax.ShapeDtypeStruct((N,), jnp.float64),  # b
        ),
        directory=args.out,  # "artifacts"
        name=args.name,  # "basic"
    )
    # docs: end export

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
