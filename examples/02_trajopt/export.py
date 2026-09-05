"""Export the trajectory-optimisation workload the timing examples call.

This is the public stand-in for a control workload.  The motivation for the
whole project is nonlinear model-predictive control -- a solver that has to
finish inside a fixed cycle, where the worst call in a million is the number
that matters -- but no MPC code ships here.  What ships is a function with the
*shape* of one: a nonlinear plant, a rollout over a horizon, a warm start, and
an iteration count fixed at trace time.

The plant is a chain of ``NQ`` masses coupled by cubic springs, with viscous
damping, a sine gravity-like restoring term, and a dense ``tanh`` coupling that
makes every mass feel every other one.  It is integrated with RK4 over an
``H``-step horizon by ``lax.scan``, and the controls are improved by
``N_ITERS`` warm-started gradient-descent iterations, each choosing a step from
``N_TRIALS`` backtracking candidates by an Armijo test.

Nothing about it is physically meaningful.  It exists to be *representative* --
float64, dense linear algebra, transcendental functions, a scan, and a
reverse-mode gradient through all of it -- and to take a millisecond or so per
call, which is the regime a 1 kHz control loop lives in.

Fixed trip counts, everywhere
-----------------------------
Every loop is unrolled or scanned with a count known at trace time.  There is
no ``while_loop``, no convergence test that ends the solve early, and no branch
that skips work.  That is deliberate: this function is the ruler the C++ side
is measured with, and a workload whose *cost* depends on its data contributes
algorithmic spread to ``max/p50`` that then cannot be told apart from system
jitter.  Only the *results* -- the step actually chosen, ``backtracks_used``,
``iterations_used`` -- are data-dependent.

Keeping it in tune
------------------
Two numbers say whether the workload is still the right size, and
``--time --stats`` prints both:

* the **median call** should land between **1 and 5 ms**.  Too fast: raise
  ``N_ITERS`` first (it scales the gradient work without changing the memory
  footprint), then ``H``.  Too slow: lower ``H``.
* the **StableHLO op count** should be **at least 2000**.  Below that the
  executable is small enough that per-call runtime overhead, not the
  computation, is what the benchmark measures.

And one number says whether the optimiser is still sane: ``cost_history`` for
case 0 must be **non-increasing**.  If it is not, halve ``ALPHA0`` -- the line
search is starting outside the region where the descent direction helps.

Anything changed here has to be re-exported before it is called: a ``.binpb``
embeds machine code for the host that produced it, and the C++ side reads its
shapes out of the sidecar written beside it.

Usage
-----
``python examples/02_trajopt/export.py --out artifacts --cases 4``

``--time`` and ``--stats`` add the two tuning measurements; ``--no-cases``
writes the artifacts without the reference cases, which is what a re-export on
a machine that only needs to *run* the function wants.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, NamedTuple

import jax
import numpy as np

# Before anything traces: without x64 JAX narrows every float64 argument to
# float32 and says nothing, and a C++ caller writing doubles into a
# 4-byte-per-element arena then walks off the end of it.  jax2exec.export
# catches that and refuses, but the switch belongs here, where anyone reading
# the file to find out what width it exports will look.
jax.config.update("jax_enable_x64", True)

# Below the switch on purpose rather than with the imports above: everything
# past this line either builds an array or traces one, and keeping the switch
# visibly first is what stops a later edit from adding a constant that is
# quietly narrowed.
import jax.numpy as jnp
from jax import lax
from jax2exec import export, write_reference_cases

# docs: begin trajopt-constants
NQ = 24  #: Masses in the chain.
NU = 6  #: Actuators.
H = 50  #: Horizon steps.
NP = 8  #: Plant parameters.
NX = 2 * NQ  #: State dimension: positions then velocities.
N_ITERS = 5  #: Gradient-descent iterations, Python-unrolled.
N_TRIALS = 4  #: Line-search candidates per iteration.
DT = 0.02  #: Nominal integration step, scaled by ``params[7]``.
ALPHA0 = 0.05  #: Largest line-search step.
GRAD_TOL = 1e-3  #: Gradient norm below which an iteration counts as converged.
# docs: end trajopt-constants

#: Plant parameters in the order the kernel unpacks them: mass, k_lin, k_cub,
#: damp, grav, k_couple, u_gain, dt_scale.  Kept identical to
#: ``cjfc::kNominalParams`` in ``examples/common/trajopt_signature.hpp``; the
#: C++ examples fill their ``params`` arena from that table.
NOMINAL_PARAMS = np.array(
    [1.0, 4.0, 1.0, 0.5, 2.0, 0.5, 1.0, 1.0], dtype=np.float64
)

#: Cost weights: tracking, effort, terminal, smoothing.  float32 because the
#: kernel takes them that way -- a knob, not a quantity the answer's accuracy
#: depends on.  Identical to ``cjfc::kWeights``.
WEIGHTS = np.array([1.0, 0.01, 5.0, 0.1], dtype=np.float32)

#: The values ``step`` is drawn from for the random cases.  Three magnitudes,
#: so a C++ reader that truncated the counter to 8 or 16 bits fails on one of
#: them rather than on none.
STEP_CHOICES = (7, 100, 12345)

# Dense, deterministic, and traced as a constant rather than passed in: it is
# a property of the plant, not of the problem instance, and folding it into the
# executable is what makes the coupling term a single matmul.
_INDEX = np.arange(NQ, dtype=np.float64)
W_COUPLE = np.cos(0.37 * _INDEX[:, None] + 0.11 * _INDEX[None, :]) / NQ

# One actuator every few masses, so the controls cannot address the chain
# directly and the optimiser has to work through the dynamics.
B_ACT = np.zeros((NQ, NU), dtype=np.float64)
B_ACT[np.rint(np.linspace(0, NQ - 1, NU)).astype(int), np.arange(NU)] = 1.0


class Solution(NamedTuple):
    """What one solve returns, in the order the C++ side indexes it.

    A ``NamedTuple`` rather than a tuple or a dict on purpose: the exporter
    reads output names out of the pytree, a plain tuple carries none (they
    would become ``out_0``..``out_7``), and a dict flattens in *sorted* key
    order, which would silently permute the results.
    """

    u_opt: jax.Array  #: float64 ``[H, NU]``: the optimised controls.
    x_pred: jax.Array  #: float64 ``[H, NX]``: the predicted states.
    cost: jax.Array  #: float32: the final objective value.
    grad_norm: jax.Array  #: float64: gradient norm at the last iterate.
    cost_history: jax.Array  #: float64 ``[N_ITERS]``: objective per iteration.
    iterations_used: jax.Array  #: int32: iterations whose gradient was large.
    backtracks_used: jax.Array  #: int32: line-search backtracks taken.
    step_next: jax.Array  #: int32: ``step + 1``, the exactness check.


def dynamics(x, u, p):
    """Continuous-time acceleration of the chain.

    The spring force between neighbours is cubic, so the plant is genuinely
    nonlinear rather than a linear system wearing a costume; ``tanh`` of a
    dense mixing matrix couples every mass to every other one, which is what
    keeps the reverse-mode gradient from collapsing into something sparse and
    cheap.
    """
    mass = p[0]
    k_lin = p[1]
    k_cub = p[2]
    damp = p[3]
    grav = p[4]
    k_couple = p[5]
    u_gain = p[6]

    q = x[:NQ]
    qd = x[NQ:]
    dq = q[1:] - q[:-1]
    fs = k_lin * dq + k_cub * dq**3

    # Each spring pushes its two ends in opposite directions; the scatter-add
    # is the chain's incidence matrix written without materialising it.
    f = jnp.zeros(NQ, dtype=x.dtype).at[:-1].add(fs).at[1:].add(-fs)
    f = (
        f
        - damp * qd
        - grav * jnp.sin(q)
        - k_couple * jnp.tanh(W_COUPLE @ q)
        + u_gain * (B_ACT @ u)
    )
    return jnp.concatenate([qd, f / mass])


def rk4(x, u, p):
    """One RK4 step with a zero-order hold on ``u``.

    RK4 rather than Euler because the four stages put four dynamics
    evaluations -- and four transposed ones in the gradient -- into every
    horizon step, which is where the arithmetic that makes this workload worth
    timing comes from.
    """
    dt = DT * p[7]
    k1 = dynamics(x, u, p)
    k2 = dynamics(x + 0.5 * dt * k1, u, p)
    k3 = dynamics(x + 0.5 * dt * k2, u, p)
    k4 = dynamics(x + dt * k3, u, p)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(x0, controls, p):
    """Integrate the plant forward under ``controls``; returns ``[H, NX]``."""

    def advance(x, u):
        nxt = rk4(x, u, p)
        return nxt, nxt

    return lax.scan(advance, x0, controls)[1]


def total_cost(controls, x0, x_ref, p, w, use_terminal):
    """Tracking, effort, smoothing, and an optional terminal term.

    ``use_terminal`` selects with ``jnp.where`` rather than with a Python
    ``if``: it is a traced value, and both arms are evaluated either way, so
    the cost of a call does not depend on it.
    """
    err = rollout(x0, controls, p) - x_ref
    cost = (
        w[0] * jnp.sum(err**2)
        + w[1] * jnp.sum(controls**2)
        + w[3] * jnp.sum(jnp.diff(controls, axis=0) ** 2)
    )
    terminal = w[2] * jnp.sum(err[-1] ** 2)
    return cost + jnp.where(use_terminal, terminal, 0.0)


def line_search(objective, controls, grad, alphas):
    """Evaluate the objective at every candidate step, in one scan.

    A scan rather than a Python loop keeps the four evaluations as one
    ``while`` in the executable instead of four copies of the rollout, which is
    the difference between an op count that describes the workload and one that
    describes the unroller.
    """

    def trial(carry, alpha):
        return carry, objective(controls - alpha * grad)

    return lax.scan(trial, None, alphas)[1]


def solve(x0, x_ref, params, u_warm, weights, use_terminal, step):
    """Improve ``u_warm`` for the horizon starting at ``x0``.

    Every input reaches an output, which is not a stylistic point: XLA prunes a
    parameter that reaches nothing, the executable then takes fewer arguments
    than the sidecar declares, and the C++ side fails at the first call with
    "Execution supplied 7 buffers but compiled program expected 4".  Inputs 0-5
    feed the cost; ``step`` feeds nothing numerical, so it feeds
    ``step_next = step + 1`` -- which is also the exact, integer check a C++
    loop uses to prove the executable read the inputs it just wrote.

    The iteration count, the number of line-search candidates and the horizon
    are all fixed at trace time, so two calls on different data cost the same.
    """
    # float64 throughout the solve; the weights arrive as float32 because they
    # are a tuning knob, and mixing widths inside the gradient is not.
    w = weights.astype(jnp.float64)

    def objective(controls):
        return total_cost(controls, x0, x_ref, params, w, use_terminal)

    controls = u_warm
    alphas = ALPHA0 * 0.5 ** jnp.arange(N_TRIALS, dtype=jnp.float64)
    backtracks = jnp.int32(0)
    costs = []
    gnorms = []

    for _ in range(N_ITERS):  # Python-unrolled: the trip count is a constant
        cost, grad = jax.value_and_grad(objective)(controls)
        gg = jnp.sum(grad * grad)
        trials = line_search(objective, controls, grad, alphas)

        # Armijo: accept the first step that buys a fraction of the decrease
        # the gradient promised, else take the least bad one.
        armijo = trials <= cost - 1e-4 * alphas * gg
        chosen = jnp.where(
            jnp.any(armijo), jnp.argmax(armijo), jnp.argmin(trials)
        ).astype(jnp.int32)

        # Never accept an increase.  This is what makes cost_history
        # non-increasing, which is the invariant the export checks.
        controls = jnp.where(
            trials[chosen] < cost,
            controls - alphas[chosen] * grad,
            controls,
        )
        backtracks = backtracks + chosen
        costs.append(cost)
        gnorms.append(jnp.sqrt(gg))

    # An iteration counts as used when its gradient was still large: the
    # arithmetic ran regardless, so this reports convergence without ever
    # having skipped work for it.
    used = jnp.sum(jnp.stack(gnorms) > GRAD_TOL).astype(jnp.int32)

    return Solution(
        u_opt=controls,
        x_pred=rollout(x0, controls, params),
        cost=objective(controls).astype(jnp.float32),
        # The gradient at the last iterate a gradient was taken at.  Taking one
        # more would cost a sixth backward pass to report a diagnostic.
        grad_norm=gnorms[-1],
        cost_history=jnp.stack(costs),
        iterations_used=used,
        backtracks_used=backtracks,
        step_next=(step + 1).astype(jnp.int32),
    )


#: The argument shapes the function is traced with, in call order.  This table
#: and ``examples/common/trajopt_signature.hpp`` must agree; the C++ side
#: checks dtypes and ranks against it at load and refuses to run otherwise.
SPECS = (
    jax.ShapeDtypeStruct((NX,), jnp.float64),  # x0
    jax.ShapeDtypeStruct((H, NX), jnp.float64),  # x_ref
    jax.ShapeDtypeStruct((NP,), jnp.float64),  # params
    jax.ShapeDtypeStruct((H, NU), jnp.float64),  # u_warm
    jax.ShapeDtypeStruct((4,), jnp.float32),  # weights
    jax.ShapeDtypeStruct((), jnp.bool_),  # use_terminal
    jax.ShapeDtypeStruct((), jnp.int32),  # step
)


def reference_trajectory(k: int) -> np.ndarray:
    """The reference the C++ examples feed at cycle ``k``.

    The NumPy twin of ``cjfc::write_reference`` in
    ``examples/common/trajopt_signature.hpp``, and it has to stay identical:
    case 0 is what makes a C++ run and a Python run comparable, and the two
    are only comparable if they solved the same problem.  A slow sine sweeping
    along the positions, velocities left at zero.
    """
    t = np.arange(H, dtype=np.float64)[:, None]
    j = np.arange(NQ, dtype=np.float64)[None, :]
    positions = 0.3 * np.sin(0.05 * (k + t) + 0.2 * j)
    return np.concatenate([positions, np.zeros((H, NQ))], axis=1)


def build_cases(count: int, seed: int) -> list[tuple[Any, ...]]:
    """Build ``count`` argument tuples: case 0 nominal, the rest perturbed.

    Case 0 is the one the C++ tests key on, so it is fixed: zero state, the
    reference at cycle 0, nominal plant, cold start, terminal cost on, step 0.
    The rest perturb the state, the plant and the warm start, alternate the
    terminal-cost branch and vary the step counter, so a case sweep exercises
    both branches of the kernel and several magnitudes of the counter.

    Their references are the same sinusoid at a different cycle rather than
    noise: a random ``x_ref`` gives the solver nothing to track, and a case
    that is unsolvable is a weak test of a solver.
    """
    cases: list[tuple[Any, ...]] = [
        (
            np.zeros(NX, dtype=np.float64),
            reference_trajectory(0),
            NOMINAL_PARAMS.copy(),
            np.zeros((H, NU), dtype=np.float64),
            WEIGHTS.copy(),
            np.bool_(True),
            np.int32(0),
        )
    ]

    rng = np.random.default_rng(seed)
    for case in range(1, count):
        cases.append(
            (
                rng.normal(0.0, 0.3, NX),
                reference_trajectory(37 * case),
                NOMINAL_PARAMS * rng.uniform(0.95, 1.05, NP),
                rng.normal(0.0, 0.05, (H, NU)),
                WEIGHTS.copy(),
                np.bool_(case % 2 == 0),
                np.int32(rng.choice(STEP_CHOICES)),
            )
        )
    return cases


def _fail(message: str) -> SystemExit:
    """Stop with a message rather than a traceback; nothing is written yet."""
    return SystemExit(f"export.py: {message}")


def _validate(case: int, args: tuple[Any, ...], result: Solution) -> None:
    """Refuse to freeze a case that is not finite or not self-consistent.

    A NaN reference passes every relative-error comparison silently -- each
    comparison against NaN is false, so nothing exceeds the tolerance -- which
    would turn the strongest test in the suite into one that cannot fail.  The
    step check is the same one the C++ loop makes, done here so that a broken
    export is caught before it reaches a benchmark.
    """
    for name, value in zip(Solution._fields, result):
        array = np.asarray(value)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise _fail(
                f"case {case} output '{name}' is not finite; the workload or "
                "the case is broken, and freezing it would hide that"
            )

    step_next = int(np.asarray(result.step_next))
    want = int(np.asarray(args[6])) + 1
    if step_next != want:
        raise _fail(
            f"case {case} returned step_next={step_next}, expected {want}"
        )


def _op_count(text: str) -> int:
    """StableHLO operations in a lowering, counted the crude way.

    Counting ``stablehlo.`` occurrences over-counts slightly (attribute names
    carry the dialect prefix too) and never under-counts, which is the right
    direction for a floor.
    """
    return text.count("stablehlo.")


def _median_call_ms(compiled: Any, args: tuple[Any, ...], calls: int) -> float:
    """Median wall clock of ``calls`` calls, after a warm-up call.

    The median, not the mean: this is a tuning aid measured on whatever
    machine the export happens to run on, and one descheduled call should not
    move it.  It is *not* a latency result -- that is what the C++ example is
    for, and it reports the tail rather than the middle.
    """
    jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(calls):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append(time.perf_counter() - start)
    return 1e3 * float(np.median(samples))


def _tuning_note(median_ms: float | None, ops: int | None) -> list[str]:
    """The maintenance rule, printed only when the workload has drifted."""
    too_fast = median_ms is not None and median_ms < 1.0
    too_slow = median_ms is not None and median_ms > 5.0
    too_small = ops is not None and ops < 2000

    if not (too_fast or too_slow or too_small):
        return []

    lines = ["", "tuning note:"]
    if too_fast:
        lines.append(
            f"  median {median_ms:.2f} ms is under 1 ms: raise N_ITERS first, "
            "then H."
        )
    if too_slow:
        lines.append(f"  median {median_ms:.2f} ms is over 5 ms: lower H.")
    if too_small:
        lines.append(
            f"  {ops} StableHLO ops is under 2000: the benchmark would be "
            "measuring per-call overhead rather than the computation. Raise "
            "N_ITERS first, then H."
        )
    lines.append(
        "  cost_history for case 0 must also stay non-increasing; halve "
        "ALPHA0 if it does not."
    )
    return lines


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The flags, which are the same ones the Makefile and CMake pass."""
    parser = argparse.ArgumentParser(
        description=(
            "Export the 02_trajopt workload and freeze its reference cases."
        )
    )
    parser.add_argument(
        "--out", default="artifacts", help="directory for the artifacts"
    )
    parser.add_argument(
        "--name", default="trajopt", help="base name for the artifact set"
    )
    parser.add_argument(
        "--cases", type=int, default=4, help="reference cases to write"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the perturbed cases"
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="report the median wall clock per call (a tuning aid)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="report the StableHLO op count of the lowering",
    )
    parser.add_argument(
        "--no-cases",
        action="store_true",
        help="write the artifacts without the reference cases",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Export, measure, validate, freeze -- in that order."""
    args = _parse_args(argv)
    if not args.no_cases and args.cases < 1:
        raise _fail("--cases must be at least 1, or pass --no-cases")

    result = export(solve, SPECS, directory=args.out, name=args.name)
    print(f"wrote {result.executable}")
    if result.mlir is not None:
        print(f"wrote {result.mlir}")
    print(f"wrote {result.sidecar}")

    ops = None
    if args.stats:
        ops = _op_count(jax.jit(solve).lower(*SPECS).as_text())
        print(f"stablehlo ops: {ops}")

    cases = build_cases(max(args.cases, 1), args.seed)

    median_ms = None
    if args.time:
        median_ms = _median_call_ms(result.compiled, cases[0], 20)
        print(f"median call:   {median_ms:.3f} ms (case 0, in-process)")

    # Run every case through the compiled function before anything is frozen:
    # write_reference_cases would happily record a broken answer.
    solutions = [result.compiled(*case_args) for case_args in cases]
    for index, (case_args, solution) in enumerate(zip(cases, solutions)):
        _validate(index, case_args, solution)

    # The line search never accepts an increase, so this holds by
    # construction; when it stops holding, the step is starting outside the
    # region where the descent direction helps.
    history = np.asarray(solutions[0].cost_history)
    if np.any(np.diff(history) > 0.0):
        print(
            "warning: cost_history for case 0 increases somewhere "
            f"({history.tolist()}); halve ALPHA0",
            file=sys.stderr,
        )

    if not args.no_cases:
        manifest = write_reference_cases(
            solve, cases, directory=args.out, name=args.name
        )
        plural = "" if len(cases) == 1 else "s"
        print(f"wrote {manifest} ({len(cases)} case{plural})")

    for line in _tuning_note(median_ms, ops):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
