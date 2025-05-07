import jax
jax.config.update("jax_enable_x64", True)

def fun(x):
    return x ** 2

if __name__ == "__main__":
    dummy_in = [
        jax.ShapeDtypeStruct(shape=(), dtype=jax.numpy.float64),
    ]

    # TODO(jozbee): but is this compiled?
    jit_fun = jax.jit(fun)
    lower_fun = jit_fun.lower(*dummy_in)
    hlo_fun = lower_fun.compiler_ir("hlo")
    compiled_fun = hlo_fun.as_serialized_hlo_module_proto()

    with open("artifacts/jax_example.binpb", "wb") as f:
        f.write(compiled_fun)

    with open("artifacts/jax_example.hlo", "w") as f:
        f.write(hlo_fun.as_hlo_text())
