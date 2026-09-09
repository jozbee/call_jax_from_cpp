"""Export the resolved-rate step a ``ros2_control`` controller calls.

``step(q, t, dt) -> q_cmd`` moves a two-link planar arm one control period
along a circle: a damped least-squares solve of ``J qdot = GAIN * e``, with
``J`` from ``jax.jacfwd``.  The solve lowers to a LAPACK custom call, so the
artifact needs the fork's plugin.

Run it directly, or through ``make export``::

    uv run python examples/05_ros2_control/export.py --out artifacts
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax2exec import export


def _out_dir() -> str:
    """The directory the three files are written to."""
    parser = argparse.ArgumentParser(
        description="Export step(q, t, dt) -> q_cmd for example 05."
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


# docs: begin arm-export
jax.config.update("jax_enable_x64", True)  # before anything is traced

L1, L2 = 1.0, 0.8  # link lengths, metres
GAIN, DAMPING = 5.0, 1e-3


def fk(q):
    """End-effector position of the two-link planar arm."""
    return jnp.stack(
        [
            L1 * jnp.cos(q[0]) + L2 * jnp.cos(q[0] + q[1]),
            L1 * jnp.sin(q[0]) + L2 * jnp.sin(q[0] + q[1]),
        ]
    )


def target(t):
    """Where the end-effector should be at time ``t``: a circle."""
    return jnp.stack(
        [1.0 + 0.3 * jnp.cos(0.5 * t), 0.4 + 0.3 * jnp.sin(0.5 * t)]
    )


def step(q, t, dt):
    """One control period of resolved-rate control, in joint space."""
    e = target(t) - fk(q)
    J = jax.jacfwd(fk)(q)
    qdot = jnp.linalg.solve(J.T @ J + DAMPING * jnp.eye(2), J.T @ (GAIN * e))
    return q + dt * qdot


if __name__ == "__main__":
    result = export(
        step,
        (
            jax.ShapeDtypeStruct((2,), jnp.float64),  # q
            jax.ShapeDtypeStruct((), jnp.float64),  # t
            jax.ShapeDtypeStruct((), jnp.float64),  # dt
        ),
        directory=_out_dir(),  # "artifacts"
        name="arm",
    )
    # docs: end arm-export
    describe(result)
