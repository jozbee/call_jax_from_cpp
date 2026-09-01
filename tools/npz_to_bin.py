"""Convert reference-IO npz fixtures into flat binary blobs + a manifest.

The C++ benchmark and correctness checks read raw little-endian float64, so
that the test binaries stay free of any archive-format dependency.

Usage
-----
    python3 tools/npz_to_bin.py \
        --name mpc_solver \
        --out-dir tests/assets/mpc \
        case0.npz case1.npz
"""

import argparse
import os

import numpy as np

from _fixture_common import (
    MPC_INTEGRAL_OUTPUT,
    case_blob,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="+", help="reference-IO npz files")
    parser.add_argument("--name", required=True, help="fixture name")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--integral-output",
        type=int,
        default=None,
        help="index of an output known to be integral (exactness check)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    input_sizes: list[int] | None = None
    output_sizes: list[int] | None = None
    cases: list[str] = []

    for i, npz_path in enumerate(args.npz):
        npz = np.load(npz_path)
        blob, in_sizes, out_sizes = case_blob(npz)

        if input_sizes is None:
            input_sizes, output_sizes = in_sizes, out_sizes
        elif (in_sizes, out_sizes) != (input_sizes, output_sizes):
            raise SystemExit(
                f"{npz_path}: shapes disagree with the first case"
            )

        case_name = f"{args.name}_case{i}.bin"
        blob.tofile(os.path.join(args.out_dir, case_name))
        cases.append(case_name)
        print(f"wrote {case_name}: {blob.size} doubles")

    assert input_sizes is not None and output_sizes is not None
    manifest = os.path.join(args.out_dir, f"{args.name}_cases.json")
    integral = args.integral_output
    if integral is None and args.name == "mpc_solver":
        integral = MPC_INTEGRAL_OUTPUT
    write_manifest(
        manifest,
        args.name,
        input_sizes,
        output_sizes,
        cases,
        integral_output=integral,
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
