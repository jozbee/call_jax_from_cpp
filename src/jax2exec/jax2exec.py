"""From jax, export a seriealized executable and meta data.

Given a jax function that can be `jax.jit`-ed, we export a serialized executanle
that is compatible with the xla pjrt runtime.
Cf. https://openxla.org/xla/pjrt.
See the examples in `call_jax_from_cpp` for cpp usage.
"""

import json
import typing as tp

import jax
import jax.extend
import jax.numpy as jnp

# if not set, jax will only compile to float32 (not float64)
jax.config.update("jax_enable_x64", True)

array_t: tp.TypeAlias = tp.Union[jax.Array, jax.ShapeDtypeStruct]

def jax2exec(
    fun: tp.Callable,
    args: tuple[array_t, ...],
    directory: str,
    fun_name: str,
) -> None:
    """Export a jax function as a serialized executable (for cpu).

    Save executable and metadata in {fun_name}.binpb and {fun_name}.json,
    respectively, in the specified directory.

    Parameters
    ----------
    fun :
        A jax function that can be `jax.jit`-ed.
    args :
        A tuple of jax arrays that are used to trace the function.
        These should preferably be instances of `jax.ShapeDtypeStruct` to avoid
        unnecessary computation.
    directory :
        The directory where the serialized executable and meta data will be
        saved.
    fun_name :
        The name of the function.
        This will be used to name the files.
    """
    # compile
    jit_fun = jax.jit(fun)
    lower_fun = jit_fun.lower(*args)
    cpu = jax.extend.backend.get_backend(platform="cpu")

    # technically, exec is a "loaded executable", which is the type that pjrt
    #  natively uses
    exec = cpu.compile(str(lower_fun.compiler_ir("stablehlo")))

    serialized_exec = cpu.serialize_executable(exec)
    with open(f"{directory}/{fun_name}.binpb", "wb") as f:
        f.write(serialized_exec)

    # check if inputs and outputs are compatible with our cpp implementation of
    #  `pjrt_exec`
    # (note that the serialized executable is written before the checking...)
    args_info = lower_fun.args_info
    out_info = lower_fun.out_info

    assert len(lower_fun.args_info[1]) == 0, "Input cannot have keywords"
    assert all(len(info.shape) <= 1 for info in args_info[0]), (
        "Input shapes must be 1D"
    )
    assert all(jnp.isdtype(info.dtype, jnp.float64) for info in args_info[0]), (
        "Input shapes must be float64"
    )

    assert all(len(info.shape) <= 1 for info in out_info), (
        "Output shapes must be 1D"
    )
    assert all(jnp.isdtype(info.dtype, jnp.float64) for info in out_info), (
        "Output dtypes must be float64"
    )

    # really, we are only interested in the sizes of the inputs (for now)
    args_sizes = []
    for info in args_info[0]:
        if len(info.shape) == 1:
            args_sizes.append(info.shape[0])
        else:  # len(info.shape) == 0
            args_sizes.append(0)
    out_sizes = []
    for info in out_info:
        if len(info.shape) == 1:
            out_sizes.append(info.shape[0])
        else:  # len(info.shape) == 0
            out_sizes.append(0)
    meta_data = {
        "args_info": {
            "sizes": args_sizes,
            "dtype": "float64",
        },
        "out_info": {
            "sizes": out_sizes,
            "dtype": "float64",
        },
    }
    with open(f"{directory}/{fun_name}.json", "w") as f:
        json.dump(meta_data, f, indent=2)
