"""``python -m jax2exec check <base>``: inspect an artifact set.

Answers, without JAX, the two questions asked when a load fails on a machine
that did not export: what does the sidecar declare, and will the executable
run here.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from ._dtypes import SUPPORTED_DTYPES
from ._isa import host_isa_level, isa_supports
from ._sidecar import (
    SCHEMA_VERSION,
    load_sidecar,
    normalize_arrays,
    sha256_hex,
)

__all__ = ["main"]

_SUFFIXES = (".json", ".binpb", ".mlirbc")


def _strip_suffix(base: str) -> Path:
    """Accept the artifact base with or without one of its extensions."""
    path = Path(base)
    if path.suffix in _SUFFIXES:
        return path.with_suffix("")
    return path


def _fmt_table(
    rows: list[list[str]], headers: list[str], align: str
) -> list[str]:
    """Render a table, one character of ``align`` (``l``/``r``) per column."""
    widths = [
        max(len(str(row[i])) for row in (*rows, headers))
        for i in range(len(headers))
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    for row in rows:
        cells = [
            cell.ljust(width) if how == "l" else cell.rjust(width)
            for cell, width, how in zip(row, widths, align)
        ]
        lines.append("  ".join(cells).rstrip())
    return lines


def _array_rows(entries: list[dict[str, Any]]) -> list[list[str]]:
    """One table row per input or output."""
    return [
        [
            str(entry.get("index", index)),
            str(entry.get("name", f"?{index}")),
            str(entry.get("dtype", "?")),
            str(entry.get("shape", [])),
            str(entry.get("numel", "?")),
            str(entry.get("nbytes", "?")),
        ]
        for index, entry in enumerate(entries)
    ]


def _check_sizes(entries: list[dict[str, Any]], kind: str) -> list[str]:
    """Report entries whose ``numel`` or ``nbytes`` disagree with the shape.

    The loader allocates ``nbytes`` and reads ``numel`` elements out of it,
    so a sidecar that disagrees with itself is a buffer overrun.
    """
    problems = []
    for index, entry in enumerate(entries):
        info = SUPPORTED_DTYPES.get(str(entry.get("dtype")))
        if info is None:
            problems.append(
                f"{kind} {index} ('{entry.get('name')}') has dtype "
                f"{entry.get('dtype')}, which this loader does not support"
            )
            continue
        numel = math.prod(int(dim) for dim in entry.get("shape", []))
        if int(entry.get("numel", -1)) != numel:
            problems.append(
                f"{kind} {index} ('{entry.get('name')}') declares numel "
                f"{entry.get('numel')} but its shape holds {numel}"
            )
        if int(entry.get("nbytes", -1)) != numel * info.itemsize:
            problems.append(
                f"{kind} {index} ('{entry.get('name')}') declares nbytes "
                f"{entry.get('nbytes')} but {numel} x {info.itemsize} is "
                f"{numel * info.itemsize}"
            )
    return problems


def _check_artifacts(
    base: Path, sidecar: dict[str, Any], out: list[str]
) -> list[str]:
    """Verify the digest of every artifact the sidecar names.

    A missing ``.mlirbc`` is not a problem: it only means this artifact set
    cannot fall back to compiling in-process.
    """
    artifacts = sidecar.get("artifacts") or {}
    if not artifacts:  # a v1 sidecar names nothing; the executable is <base>
        artifacts = {"executable": f"{base.name}.binpb"}

    problems = []
    for key, digest_key, required in (
        ("executable", "executable_sha256", True),
        ("mlir", "mlir_sha256", False),
    ):
        file_name = artifacts.get(key)
        if not file_name:
            continue
        path = base.parent / str(file_name)
        if not path.exists():
            out.append(f"  {file_name}: missing")
            if required:
                problems.append(
                    f"{file_name} is named by the sidecar but is not here"
                )
            continue

        data = path.read_bytes()
        recorded = artifacts.get(digest_key)
        actual = sha256_hex(data)
        if not recorded:
            out.append(f"  {file_name}: {len(data)} bytes, no digest recorded")
        elif recorded == actual:
            out.append(f"  {file_name}: {len(data)} bytes, sha256 matches")
        else:
            out.append(f"  {file_name}: {len(data)} bytes, sha256 MISMATCH")
            problems.append(
                f"{file_name} hashes to {actual[:16]}... but the sidecar "
                f"records {str(recorded)[:16]}..."
            )
    return problems


def _check_isa(sidecar: dict[str, Any], out: list[str]) -> list[str]:
    """Compare the exporting host's ISA level with this one's."""
    host_block = (sidecar.get("export") or {}).get("host") or {}
    required = host_block.get("isa_level")
    host = host_isa_level()

    if not required:
        out.append(
            f"  host is {host}; the sidecar records no ISA level, so the "
            ".binpb is still locked to whatever machine exported it"
        )
        return []

    if required == "unknown" or host == "unknown":
        out.append(
            f"  host is {host}, exported on {required}: not comparable, load "
            "the .binpb to find out"
        )
        return []

    verdict = isa_supports(host, required)
    if verdict:
        out.append(f"  host is {host}, exported on {required}: the .binpb runs")
        return []

    weaker = verdict is False
    label = "TOO WEAK" if weaker else "WRONG ARCHITECTURE"
    reason = (
        "a weaker instruction set" if weaker else "a different architecture"
    )
    out.append(f"  host is {host}, exported on {required}: {label}")
    problem = (
        f"this host is {host} and the executable was built for "
        f"{required}, {reason}: the .binpb will not run here. Compile the "
        ".mlirbc instead (the C++ loader does this on its own when "
        "isa_guard is on), or re-export on this machine."
    )
    return [problem]


def _check(base_arg: str) -> int:
    """Run the ``check`` subcommand, returning the process exit code."""
    base = _strip_suffix(base_arg)
    sidecar_path = base.with_name(f"{base.name}.json")
    if not sidecar_path.exists():
        print(f"no sidecar at {sidecar_path}", file=sys.stderr)
        return 2
    try:
        sidecar = load_sidecar(sidecar_path)
    except (OSError, ValueError) as exc:
        print(f"cannot read {sidecar_path}: {exc}", file=sys.stderr)
        return 2

    try:
        schema = int(sidecar.get("schema", 1))
    except (TypeError, ValueError):
        print(f"{sidecar_path} has no usable schema field", file=sys.stderr)
        return 1
    if schema > SCHEMA_VERSION:
        print(
            f"schema {schema} is newer than this package understands "
            f"(supports 1-{SCHEMA_VERSION}); upgrade jax2exec",
            file=sys.stderr,
        )
        return 1

    generator = sidecar.get("generator") or {}
    export_block = sidecar.get("export") or {}
    host_block = export_block.get("host") or {}

    out = [
        f"{sidecar.get('name', base.name)}  ({sidecar_path})",
        (
            f"  schema {schema}, written by "
            f"{generator.get('tool', 'the v1 exporter')} "
            f"{generator.get('version', '')}"
        ).rstrip(),
        (
            f"  jax {sidecar.get('jax_version', '?')} / jaxlib "
            f"{sidecar.get('jaxlib_version', '?')}, platform "
            f"{sidecar.get('platform', '?')}"
        ),
    ]
    if export_block:
        out.append(
            f"  exported {export_block.get('time_utc', '?')} on "
            f"{host_block.get('os', '?')}/{host_block.get('arch', '?')} "
            f"{host_block.get('isa_level', '?')}, python "
            f"{host_block.get('python', '?')}, x64 "
            f"{'on' if export_block.get('x64_enabled') else 'OFF'}"
        )
        if host_block.get("cpu_model"):
            out.append(f"  cpu {host_block['cpu_model']}")
        if export_block.get("xla_flags"):
            out.append(f"  XLA_FLAGS {export_block['xla_flags']}")

    try:
        inputs, outputs = normalize_arrays(sidecar)
        problems = _check_sizes(inputs, "input") + _check_sizes(
            outputs, "output"
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"{sidecar_path} is malformed: {exc}", file=sys.stderr)
        return 1

    headers = ["idx", "name", "dtype", "shape", "numel", "nbytes"]
    for label, entries in (("inputs", inputs), ("outputs", outputs)):
        out.append("")
        out.append(f"{label} ({len(entries)})")
        if entries:
            table = _fmt_table(_array_rows(entries), headers, "rllrrr")
            out.extend("  " + line for line in table)

    out.append("")
    out.append("artifacts")
    problems += _check_artifacts(base, sidecar, out)

    out.append("")
    out.append("instruction set")
    problems += _check_isa(sidecar, out)

    print("\n".join(out))
    if problems:
        print()
        for problem in problems:
            print(f"error: {problem}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m jax2exec``.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 when the artifact set is consistent and will run on this host, 1
        when it is not, 2 when the sidecar could not be read at all.
    """
    parser = argparse.ArgumentParser(
        prog="python -m jax2exec",
        description="Inspect artifacts exported by jax2exec.",
    )
    # `dest` only names the subcommand in the error when none is given.
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check",
        help="describe an artifact set and verify it against this host",
    )
    check.add_argument(
        "base",
        help="artifact base path, with or without an extension "
        "(artifacts/trajopt)",
    )

    return _check(parser.parse_args(argv).base)
