"""Export benchmark fixtures: the MPC acceptance workload and a synthetic twin.

Two kernels are exported, both with the same 16-input / 14-output float64
signature so that they exercise an identical buffer path in the runtime:

`mpc_solver`
    One control step of the Stewart-platform MPC controller.  This is the
    real acceptance workload.  Its L-BFGS `while_loop`s exit early and one
    `cond` switches on after ~50 calls, so part of its latency spread is
    *algorithmic* rather than system jitter.

`synth_solver`
    A fixed-trip-count `scan` of comparable cost with no data-dependent
    control flow at all.  Any latency spread it shows is system jitter,
    which is what makes it the control for the MPC measurements.

Serialized executables embed target machine code, so this must be run on
(or for) the machine that will execute them.

Usage
-----
    python3 tools/export_fixture.py --target all
"""

import argparse
import functools
import os
import sys
import time

import numpy as np

# Prefer this working copy of `jax2exec` over any installed/submodule copy.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from _fixture_common import (  # noqa: E402
    MPC_INPUT_NAMES,
    MPC_INTEGRAL_OUTPUT,
)
from jax2exec.jax2exec import jax2exec  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Input sizes of `mpc_solver`, in call order.  The synthetic twin reuses them.
SIZES: tuple[int, ...] = (
    1, 1, 1, 3, 3, 1200, 21, 8, 15, 15, 24, 6, 2, 8, 1, 1
)
OUT_SIZES: tuple[int, ...] = (
    3, 1, 3, 1200, 21, 8, 15, 15, 24, 6, 2, 8, 1, 1
)

# Synthetic kernel shape.  `HORIZON` matches the MPC prediction horizon and is
# fixed by the 1200-element control vector (200 steps x 6 channels).  Cost is
# tuned with `DEPTH` rather than `WIDTH` so the kernel stays a long serial
# dependency chain like the MPC, instead of turning into a parallel BLAS
# workload with completely different scheduling behaviour.
HORIZON = 200
WIDTH = 96
DEPTH = 7


def dummy_args() -> tuple[jax.ShapeDtypeStruct, ...]:
    return tuple(
        jax.ShapeDtypeStruct(shape=(n,), dtype=jnp.float64) for n in SIZES
    )


###########
# synthetic
###########


def _synth_rollout(control: jax.Array, seed: jax.Array) -> jax.Array:
    """Fixed-cost sequential rollout: `HORIZON` steps, no early exit."""
    key = jnp.arange(WIDTH, dtype=jnp.float64) / WIDTH
    # A fixed, well-conditioned mixing matrix (traced as a constant).
    idx = jnp.arange(WIDTH, dtype=jnp.float64)
    mix = jnp.cos(idx[:, None] * 0.37 + idx[None, :] * 0.11) / WIDTH
    mix = mix + jnp.eye(WIDTH) * 0.9

    u = control.reshape(HORIZON, 6)

    def step(carry, u_t):
        drive = jnp.sum(u_t) * key + seed
        # unrolled: fixed trip count, no lax control flow of any kind
        for _ in range(DEPTH):
            carry = jnp.tanh(mix @ carry + drive)
        return carry, jnp.sum(carry * carry)

    carry0 = key + seed
    _, costs = jax.lax.scan(step, carry0, u)
    return jnp.sum(costs)


def synth_solver(*args: jax.Array) -> tuple[jax.Array, ...]:
    """Synthetic twin of `mpc_solver`: same signature, fixed cost."""
    control = args[5]
    iter_ = args[15]

    # Every input has to feed the result: XLA drops parameters the computation
    # never reads, which would leave the executable with a different arity than
    # the MPC and defeat the point of matching its buffer path.
    seed = sum(jnp.sum(a) for i, a in enumerate(args) if i != 5) * 1e-6

    # value-and-grad through the rollout, mirroring the MPC's AD structure
    val, grad = jax.value_and_grad(_synth_rollout)(control, seed)

    def sized(n: int, fill: jax.Array) -> jax.Array:
        return jnp.full((n,), jnp.squeeze(fill), dtype=jnp.float64)

    outs = []
    for i, n in enumerate(OUT_SIZES):
        if n == 1200:
            outs.append(grad)
        elif i == MPC_INTEGRAL_OUTPUT:
            outs.append(iter_ + 1.0)  # integral, free exactness check
        else:
            outs.append(sized(n, val * (i + 1) * 1e-9))
    return tuple(outs)


#####
# mpc
#####


def build_mpc(comp_repo: str):
    """Import the MPC solver from the `comp` repository and bind its spec."""
    sys.path.insert(0, comp_repo)
    sys.path.insert(0, os.path.join(comp_repo, "cpp", "src"))
    import mpc_export  # type: ignore  # noqa: E402
    from exp_mpc.stewart_min import mpc_spec, opt  # type: ignore  # noqa: E402

    limits = mpc_spec.MPCLimits()
    spec = mpc_spec.MPCSpec.init_weight_margins(
        mpc_export.WeightMode.constant_weights,
        limits,
        max_iter=2,
        max_ls=1,
        use_terminal=True,
        init_norm=1e-1,
    )
    zero_ts = opt.TrainState.zero_init(spec)
    modes = (
        mpc_export.PersonelMode.init_spec(spec),
        mpc_export.PredictionMode.init_spec(spec),
        mpc_export.WeightMode.init_spec(spec),
    )
    return functools.partial(mpc_export.mpc_solver, spec, zero_ts, modes)


##########
# fixtures
##########


def reference_cases(
    fun, inputs: list[list[np.ndarray]], out_path: str
) -> None:
    """Run `fun` on each input set and save inputs+outputs as npz."""
    jit_fun = jax.jit(fun)
    for i, case in enumerate(inputs):
        outs = jit_fun(*[jnp.asarray(a) for a in case])
        outs = jax.tree_util.tree_leaves(outs)
        data = {
            f"in_{name}": np.asarray(a, dtype=np.float64)
            for name, a in zip(MPC_INPUT_NAMES, case)
        }
        data.update(
            {
                f"out_{j:02d}": np.asarray(o, dtype=np.float64)
                for j, o in enumerate(outs)
            }
        )
        path = f"{out_path}_case{i}.npz"
        np.savez(path, **data)
        print(f"wrote {path}")


def default_inputs(source_npz: list[str]) -> list[list[np.ndarray]]:
    """Load input sets from reference npz files, else synthesize valid ones."""
    cases: list[list[np.ndarray]] = []
    for path in source_npz:
        if not os.path.exists(path):
            continue
        npz = np.load(path)
        keys = [f"in_{n}" for n in MPC_INPUT_NAMES]
        if all(k in npz for k in keys):
            cases.append([np.asarray(npz[k], dtype=np.float64) for k in keys])
    if cases:
        return cases

    # Fall back to synthesized inputs: modes must be valid enum indices and
    # `iter` must be integral, so they cannot simply be random.
    rng = np.random.default_rng(0)
    out = []
    for modes in ((0.0, 0.0, 0.0), (5.0, 1.0, 0.0)):
        case = []
        for i, n in enumerate(SIZES):
            if i < 3:
                case.append(np.array([modes[i]], dtype=np.float64))
            elif i == 15:
                case.append(np.array([100.0], dtype=np.float64))
            elif i == 14:
                case.append(np.array([0.1], dtype=np.float64))
            else:
                case.append(rng.standard_normal(n) * 1e-3)
        out.append(case)
    return out


def timed(fun, args, n: int = 20) -> float:
    """Median wall-clock seconds per call, for sanity/tuning."""
    jit_fun = jax.jit(fun)
    outs = jit_fun(*args)
    jax.block_until_ready(outs)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fun(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("mpc", "synthetic", "all"), default="all"
    )
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--assets-dir", default="tests/assets/mpc")
    parser.add_argument(
        "--comp-repo",
        default=os.environ.get("COMP_REPO", "/Users/jozbee/work/eng/comp"),
    )
    parser.add_argument(
        "--source-npz",
        nargs="*",
        default=[],
        help="reference npz files to take input values from",
    )
    parser.add_argument("--time", action="store_true", help="report timing")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.assets_dir, exist_ok=True)
    dummy = dummy_args()

    if args.target in ("mpc", "all"):
        try:
            fun = build_mpc(args.comp_repo)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP mpc: cannot import from {args.comp_repo}: {exc}")
        else:
            jax2exec(fun, dummy, args.out_dir, "mpc_solver")
            print(f"wrote {args.out_dir}/mpc_solver.binpb")
            cases = default_inputs(args.source_npz)
            reference_cases(
                fun, cases, os.path.join(args.assets_dir, "mpc_solver_ref")
            )
            if args.time:
                a = [jnp.asarray(x) for x in cases[0]]
                print(f"mpc_solver: {timed(fun, a) * 1e3:.3f} ms/call")

    if args.target in ("synthetic", "all"):
        jax2exec(synth_solver, dummy, args.out_dir, "synth_solver")
        print(f"wrote {args.out_dir}/synth_solver.binpb")
        cases = default_inputs(args.source_npz)
        reference_cases(
            synth_solver, cases,
            os.path.join(args.assets_dir, "synth_solver_ref"),
        )
        if args.time:
            a = [jnp.asarray(x) for x in cases[0]]
            print(f"synth_solver: {timed(synth_solver, a) * 1e3:.3f} ms/call")


if __name__ == "__main__":
    main()
