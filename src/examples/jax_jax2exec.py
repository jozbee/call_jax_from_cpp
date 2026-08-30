"""Example usage of jax2exec."""

import jax
import jax.numpy as jnp

from jax2exec.jax2exec import jax2exec

jax.config.update("jax_enable_x64", True)


def fun(A: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Example function to be compiled."""
    A = A.reshape(4, 4)
    return jnp.linalg.inv(A) @ b


if __name__ == "__main__":
    # dummy input for tracing
    dummy_in = (
        jax.ShapeDtypeStruct(shape=(16,), dtype=jax.numpy.float64),
        jax.ShapeDtypeStruct(shape=(4,), dtype=jax.numpy.float64),
    )

    # directory to save the compiled executable and metadata
    directory = "./artifacts"
    fun_name = "jax_jax2exec"

    # compile the function and save the executable
    jax2exec(fun, dummy_in, directory, fun_name)
