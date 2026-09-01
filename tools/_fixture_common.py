"""Shared definitions for the benchmark/correctness fixtures.

A fixture case is stored as a flat little-endian float64 blob: all inputs
concatenated in call order, followed by all outputs in call order.  The
companion manifest describes the sizes so the C++ side can read the blob
without a zip or npz parser.
"""

import json
import typing as tp

import numpy as np

# Call order of `mpc_solver`, from `mpc_export.py` in the `comp` repository.
# The npz fixtures key inputs by name, and alphabetical order is *not* call
# order, so the order has to be written down explicitly.
MPC_INPUT_NAMES: tuple[str, ...] = (
    "personnel_mode",
    "prediction_mode",
    "weight_mode",
    "acc_ref",
    "omega_ref",
    "last_control",
    "prefilt0",
    "filt0",
    "vstate0_irl",
    "vstate0_sim",
    "y_vest_sim_hist",
    "xyz_hist",
    "yaw_hist",
    "quat_hist",
    "terminal_param",
    "iter",
)

# `out_13` is `iter + 1` and is integral, which makes it a free exactness
# check on every call.
MPC_INTEGRAL_OUTPUT = 13


def input_names(npz: tp.Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return input names in call order, for either naming convention."""
    if all(f"in_{name}" in npz for name in MPC_INPUT_NAMES):
        return tuple(f"in_{name}" for name in MPC_INPUT_NAMES)
    # positional fallback, e.g. for the synthetic kernel
    names = sorted(k for k in npz if k.startswith("in_"))
    return tuple(names)


def output_names(npz: tp.Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return output names in call order (`out_00`, `out_01`, ...)."""
    return tuple(sorted(k for k in npz if k.startswith("out_")))


def case_blob(
    npz: tp.Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Flatten one case into (blob, input_sizes, output_sizes)."""
    ins = [np.asarray(npz[k], dtype=np.float64).ravel()
           for k in input_names(npz)]
    outs = [np.asarray(npz[k], dtype=np.float64).ravel()
            for k in output_names(npz)]
    blob = np.concatenate(ins + outs).astype("<f8")
    return blob, [a.size for a in ins], [a.size for a in outs]


def write_manifest(
    path: str,
    name: str,
    input_sizes: list[int],
    output_sizes: list[int],
    cases: list[str],
    integral_output: int | None = None,
) -> None:
    """Write the manifest that the C++ bench reads alongside the blobs."""
    meta = {
        "schema": 1,
        "name": name,
        "dtype": "float64",
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        "cases": cases,
    }
    if integral_output is not None:
        meta["integral_output"] = integral_output
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
