import hlo_pb2
import jax
import jax.extend
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

    with open("artifacts/jax_example.pb", "w") as f:
        f.write(str(hlo_pb2.HloModuleProto.FromString(compiled_fun)))

    with open("artifacts/jax_example.hlo", "w") as f:
        f.write(hlo_fun.as_hlo_text())

    # serialized compile options
    comp_opt = hlo_pb2.CompileOptionsProto()
    comp_opt.executable_build_options.num_replicas = 1
    comp_opt.executable_build_options.num_partitions = 1
    comp_opt.executable_build_options.device_assignment.replica_count = 1
    comp_opt.executable_build_options.device_assignment.computation_count = 1
    cd = hlo_pb2.XlaDeviceAssignmentProto.ComputationDevice()
    cd.replica_device_ids.append(0)
    comp_opt.executable_build_options.device_assignment.computation_devices.append(cd)
    with open("artifacts/jax_example_comp_opt.binpb", "wb") as f:
        f.write(comp_opt.SerializeToString())
    with open("artifacts/jax_example_comp_opt.pb", "w") as f:
        f.write(str(comp_opt))

    # serialize executable
    cpu = jax.extend.backend.get_backend(platform="cpu")
    exec = cpu.compile(str(lower_fun.compiler_ir("stablehlo")))
    with open("artifacts/jax_example_exec.binpb", "wb") as f:
        f.write(cpu.serialize_executable(exec))
