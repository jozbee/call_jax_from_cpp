/**
 * @file probe_zerocopy.cpp
 * @brief Answer the two open questions the fast path depends on.
 *
 * EQ-3a  Does `BufferFromHostBuffer` with zero-copy semantics actually alias
 *        the caller's memory on CPU, and does a write to that memory between
 *        executions reach the next execution?  If it does, input buffers can
 *        be created once at load and the per-call host->device copy disappears.
 *
 * EQ-3b  Is `PJRT_Buffer_UnsafePointer` implemented for CPU buffers?  If it
 *        is, outputs can be read straight out of device memory instead of
 *        being copied through `PJRT_Buffer_ToHostBuffer`.
 *
 * This is a throwaway diagnostic, but the answers decide the shape of the
 * runtime, so it lives in the tree rather than in a scratch directory.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/bench/fixture.hpp"
#include "src/pjrt_exec/pjrt_exec.hpp"

namespace {

/// Allocate `n` doubles aligned to XLA's preferred 64 bytes.
double* aligned_doubles(std::size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(double)) != 0) {
    throw std::runtime_error("posix_memalign failed");
  }
  std::memset(p, 0, n * sizeof(double));
  return static_cast<double*>(p);
}

const char* semantics_name(PJRT_HostBufferSemantics s) {
  switch (s) {
    case PJRT_HostBufferSemantics_kImmutableOnlyDuringCall:
      return "kImmutableOnlyDuringCall";
    case PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes:
      return "kImmutableUntilTransferCompletes";
    case PJRT_HostBufferSemantics_kImmutableZeroCopy:
      return "kImmutableZeroCopy";
    case PJRT_HostBufferSemantics_kMutableZeroCopy:
      return "kMutableZeroCopy";
  }
  return "?";
}

/// Create one buffer with the given semantics and report whether the device
/// pointer is the host pointer -- i.e. whether the copy was actually elided.
void probe_aliasing(PJRT_Client* client, PJRT_Device* device,
                    PJRT_HostBufferSemantics semantics) {
  const std::size_t n = 1200;
  double* host = aligned_doubles(n);
  const std::array<int64_t, 1> dims = {static_cast<int64_t>(n)};

  PJRT_Client_BufferFromHostBuffer_Args args = {};
  args.struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
  args.client = client;
  args.data = host;
  args.type = PJRT_Buffer_Type_F64;
  args.dims = dims.data();
  args.num_dims = 1;
  args.host_buffer_semantics = semantics;
  args.device = device;

  PJRT_Error* err = pjrt::api()->PJRT_Client_BufferFromHostBuffer(&args);
  if (err != nullptr) {
    std::printf("  %-34s REJECTED\n", semantics_name(semantics));
    free(host);
    return;
  }

  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args ptr_args = {};
  ptr_args.struct_size =
      sizeof(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args);
  ptr_args.buffer = args.buffer;
  PJRT_Error* ptr_err =
      pjrt::api()->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&ptr_args);

  if (ptr_err != nullptr) {
    std::printf("  %-34s accepted, device pointer UNAVAILABLE\n",
                semantics_name(semantics));
  } else {
    const bool aliased = ptr_args.device_memory_ptr == host;
    std::printf("  %-34s accepted, %s (host=%p device=%p)\n",
                semantics_name(semantics),
                aliased ? "ALIASED (zero copy)" : "COPIED",
                static_cast<void*>(host), ptr_args.device_memory_ptr);
  }

  PJRT_Buffer_Destroy_Args destroy = {};
  destroy.struct_size = sizeof(PJRT_Buffer_Destroy_Args);
  destroy.buffer = args.buffer;
  pjrt::api()->PJRT_Buffer_Destroy(&destroy);
  free(host);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string fixture_name = argc > 1 ? argv[1] : "mpc_solver";
    const bench::Fixture fixture("tests/assets/mpc", fixture_name);
    const bench::Case& ref = fixture.cases()[0];

    auto client = std::make_shared<pjrt::Client>();
    auto device = client->get_devices()[0];
    pjrt::AOTComputation comp("artifacts/" + fixture_name, client);

    std::printf("=== EQ-3a: host buffer semantics on CPU ===\n");
    probe_aliasing(client->raw(), device->raw(),
                   PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes);
    probe_aliasing(client->raw(), device->raw(),
                   PJRT_HostBufferSemantics_kImmutableZeroCopy);
    probe_aliasing(client->raw(), device->raw(),
                   PJRT_HostBufferSemantics_kMutableZeroCopy);

    // Misaligned on purpose: XLA refuses zero copy below `cpu::MinAlign()`,
    // so the runtime must own its input arenas rather than accept any pointer.
    {
      double* base = aligned_doubles(1300);
      double* misaligned = base + 1;  // 8-byte aligned only
      const std::array<int64_t, 1> dims = {1200};
      PJRT_Client_BufferFromHostBuffer_Args args = {};
      args.struct_size = sizeof(PJRT_Client_BufferFromHostBuffer_Args);
      args.client = client->raw();
      args.data = misaligned;
      args.type = PJRT_Buffer_Type_F64;
      args.dims = dims.data();
      args.num_dims = 1;
      args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableZeroCopy;
      args.device = device->raw();
      PJRT_Error* err = pjrt::api()->PJRT_Client_BufferFromHostBuffer(&args);
      if (err == nullptr) {
        PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args p = {};
        p.struct_size = sizeof(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args);
        p.buffer = args.buffer;
        pjrt::api()->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&p);
        std::printf("  %-34s misaligned input -> %s\n", "kImmutableZeroCopy",
                    p.device_memory_ptr == misaligned ? "ALIASED" : "COPIED");
        PJRT_Buffer_Destroy_Args d = {};
        d.struct_size = sizeof(PJRT_Buffer_Destroy_Args);
        d.buffer = args.buffer;
        pjrt::api()->PJRT_Buffer_Destroy(&d);
      }
      free(base);
    }

    // --- EQ-3a part 2: does a write between executions take effect? ---
    std::printf("\n=== EQ-3a: reuse a zero-copy buffer across calls ===\n");
    std::vector<double*> arenas;
    std::vector<std::shared_ptr<pjrt::Buffer>> inputs;
    for (std::size_t i = 0; i < fixture.input_sizes().size(); ++i) {
      const std::size_t n =
          fixture.input_sizes()[i] == 0 ? 1 : fixture.input_sizes()[i];
      double* arena = aligned_doubles(n);
      std::memcpy(arena, ref.inputs[i].data(), n * sizeof(double));
      arenas.push_back(arena);
      inputs.push_back(pjrt::Buffer::to_device_zero_copy(
          arena, fixture.input_sizes()[i], client, device));
    }

    // Checksum every output, so a change anywhere is visible rather than
    // relying on one element to be sensitive to one input.
    auto run = [&]() {
      auto outs = comp.execute_blocking(inputs);
      double sum = 0.0;
      for (std::size_t i = 0; i < outs.size(); ++i) {
        const std::size_t n = fixture.output_sizes()[i] == 0
                                  ? 1
                                  : fixture.output_sizes()[i];
        std::vector<double> v(n);
        outs[i]->to_host_blocking(v.data(), fixture.output_sizes()[i]);
        for (double x : v) {
          sum += x * x;
        }
      }
      return sum;
    };

    const double before = run();
    // Mutate the arena in place, exactly as a control loop would between
    // steps, and re-execute the same buffers.  `last_control` is the
    // 1200-element decision variable, so perturbing it must move the result.
    for (std::size_t j = 0; j < 1200; ++j) {
      arenas[5][j] += 0.05;
    }
    const double after = run();
    for (std::size_t j = 0; j < 1200; ++j) {
      arenas[5][j] -= 0.05;
    }
    const double restored = run();

    std::printf("  checksum before=%.12g  mutated=%.12g  restored=%.12g\n",
                before, after, restored);
    std::printf("  mutation observed: %s\n",
                before != after ? "YES (buffers are reusable)" : "NO");
    std::printf("  restore observed:  %s\n",
                before == restored ? "YES (deterministic)" : "NO");

    // Control: build fresh copying buffers from the same mutated arena. If
    // these move but the reused zero-copy ones did not, the reuse is stale.
    {
      for (std::size_t j = 0; j < 1200; ++j) {
        arenas[5][j] += 0.05;
      }
      std::vector<std::shared_ptr<pjrt::Buffer>> fresh;
      for (std::size_t i = 0; i < fixture.input_sizes().size(); ++i) {
        fresh.push_back(pjrt::Buffer::to_device_blocking(
            arenas[i], fixture.input_sizes()[i], client, device));
      }
      auto outs = comp.execute_blocking(fresh);
      double sum = 0.0;
      for (std::size_t i = 0; i < outs.size(); ++i) {
        const std::size_t n = fixture.output_sizes()[i] == 0
                                  ? 1
                                  : fixture.output_sizes()[i];
        std::vector<double> v(n);
        outs[i]->to_host_blocking(v.data(), fixture.output_sizes()[i]);
        for (double x : v) {
          sum += x * x;
        }
      }
      for (std::size_t j = 0; j < 1200; ++j) {
        arenas[5][j] -= 0.05;
      }
      std::printf("  control (fresh copying buffers, mutated) = %.12g\n", sum);
      std::printf("  => mutation is real: %s\n",
                  sum != before ? "YES" : "NO");
    }

    // --- EQ-3b: is UnsafePointer implemented for CPU outputs? ---
    std::printf("\n=== EQ-3b: reading outputs without a copy ===\n");
    {
      auto outs = comp.execute_blocking(inputs);
      PJRT_Buffer_UnsafePointer_Args up = {};
      up.struct_size = sizeof(PJRT_Buffer_UnsafePointer_Args);
      up.buffer = outs[0]->raw();
      PJRT_Error* err = pjrt::api()->PJRT_Buffer_UnsafePointer(&up);
      if (err != nullptr) {
        std::printf("  PJRT_Buffer_UnsafePointer: UNIMPLEMENTED\n");
      } else {
        const double* p = reinterpret_cast<const double*>(up.buffer_pointer);
        std::printf("  PJRT_Buffer_UnsafePointer: OK, out0[0]=%.12g\n", p[0]);
      }

      PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args op = {};
      op.struct_size = sizeof(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args);
      op.buffer = outs[0]->raw();
      PJRT_Error* oerr =
          pjrt::api()->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&op);
      if (oerr != nullptr) {
        std::printf("  OpaqueDeviceMemoryDataPointer: UNIMPLEMENTED\n");
      } else {
        const double* p =
            reinterpret_cast<const double*>(op.device_memory_ptr);
        std::printf("  OpaqueDeviceMemoryDataPointer: OK, out0[0]=%.12g\n",
                    p[0]);
      }
    }

    for (double* a : arenas) {
      free(a);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
