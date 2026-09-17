"""Time one JAX callable under whatever ``XLA_FLAGS`` this process inherited.

The only module of the package that imports JAX for a measurement.  It is a
separate process because ``XLA_FLAGS`` is read once, when the backend is
initialised, so a flag can only be changed by starting again.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _resolve(target: str) -> Any:
    """Import ``"module:qualname"`` and return the object it names."""
    module_name, _, qualname = target.partition(":")
    if not module_name or not qualname:
        raise ValueError(f"target {target!r} is not spelled 'module:qualname'")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _save_outputs(path: str, outputs: Any) -> None:
    """Write a call's outputs as NumPy leaves, for the deviation check."""
    import jax
    import numpy as np

    leaves, treedef = jax.tree_util.tree_flatten(outputs)
    arrays: dict[str, Any] = {
        f"leaf_{index}": np.asarray(leaf) for index, leaf in enumerate(leaves)
    }
    arrays["treedef"] = np.array(repr(treedef))
    np.savez(path, **arrays)


def run(spec: dict[str, Any]) -> dict[str, Any]:
    """Time the callable a spec names and return what a driver records.

    Parameters
    ----------
    spec : dict
        The child specification the driver wrote: ``target``, ``factory``,
        ``args``, ``enable_x64``, ``reps``, ``warmup``, ``out``,
        ``save_outputs``, ``label`` and ``round``.

    Returns
    -------
    dict
        Percentiles in milliseconds, every sample, the compile seconds, the
        versions that ran, and the affinity and flags this process actually
        had -- which is the only proof that the driver's pinning and
        environment took effect.
    """
    import jax

    # x64 is a trace-time property of every array built below, so it has to
    # be set before the target module gets a chance to build one.
    jax.config.update("jax_enable_x64", bool(spec["enable_x64"]))

    import jax.numpy as jnp

    target = _resolve(spec["target"])
    if spec["factory"]:
        fn, args = target()
    else:
        fn = target
        with open(spec["args"], "rb") as handle:
            raw = pickle.load(handle)
        args = tuple(jnp.asarray(value) for value in raw)

    # `lower` is what every already-jitted callable has and no plain Python
    # function does; jitting one twice would time the extra dispatch.
    if not hasattr(fn, "lower"):
        fn = jax.jit(fn)

    started = time.perf_counter()
    outputs = jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - started

    for _ in range(int(spec["warmup"])):
        jax.block_until_ready(fn(*args))

    reps = int(spec["reps"])
    samples: list[float] = []
    for _ in range(reps):
        start_ns = time.perf_counter_ns()
        outputs = jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter_ns() - start_ns) / 1e6)

    if spec["save_outputs"]:
        _save_outputs(spec["save_outputs"], outputs)

    ordered = sorted(samples)
    import jaxlib

    return {
        "label": spec["label"],
        "round": spec["round"],
        "median_ms": statistics.median(ordered),
        "p95_ms": _percentile(ordered, 0.95),
        # a p99 of fewer than 100 samples is the maximum wearing another
        # name, and reading it as a tail statistic is the trap
        "p99_ms": _percentile(ordered, 0.99) if reps >= 100 else None,
        "max_ms": ordered[-1] if ordered else None,
        "mean_ms": statistics.fmean(ordered) if ordered else None,
        "samples_ms": samples,
        "compile_s": compile_s,
        "reps": reps,
        "warmup": int(spec["warmup"]),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "affinity": sorted(os.sched_getaffinity(0)),
        "pid": os.getpid(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }


def _percentile(ordered: list[float], fraction: float) -> float | None:
    """Return the nearest-rank percentile of a sorted list."""
    if not ordered:
        return None
    rank = max(0, min(len(ordered) - 1, round(fraction * len(ordered)) - 1))
    return ordered[rank]


def main(argv: list[str]) -> int:
    """Run one child from its spec file.

    Parameters
    ----------
    argv : list of str
        One element: the path of the JSON spec.

    Returns
    -------
    int
        ``0`` when the timings were written, ``1`` when the failure was
        written instead.
    """
    if len(argv) != 1:
        print(
            "usage: python -m jax2exec._tune_child <spec.json>",
            file=sys.stderr,
        )
        return 2
    spec = json.loads(Path(argv[0]).read_text())
    out = Path(spec["out"])
    try:
        result = run(spec)
    # Every exception, deliberately: a failed child is one arm with a
    # reason, not a dead campaign, and the driver can only read a reason it
    # was given.
    except Exception as exc:  # noqa: BLE001
        out.write_text(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        return 1
    out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - the child's entry point
    raise SystemExit(main(sys.argv[1:]))
