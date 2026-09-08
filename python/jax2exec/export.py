"""Export a JAX function to artifacts a C++ caller can load and run.

One call produces three files in ``directory``: ``<name>.binpb``, the
serialized PJRT executable that the C++ loader deserializes and relinks;
``<name>.mlirbc``, StableHLO bytecode it can compile in-process instead when
the ``.binpb`` will not run here; and ``<name>.json``, the sidecar describing
every input and output, because the PJRT C API cannot be asked what parameters
an executable takes.

# docs: begin export
import jax
import jax.numpy as jnp

from jax2exec import export

jax.config.update("jax_enable_x64", True)  # or everything traces as float32


def rollout(x0, u):
    return x0 @ x0 + u.sum(), u * 0.5


result = export(
    rollout,
    (jax.ShapeDtypeStruct((48,), jnp.float64),
     jax.ShapeDtypeStruct((50, 6), jnp.float64)),
    directory="artifacts",
    name="trajopt",
)
print(result.executable, result.sidecar)
# docs: end export

Nothing reaches the disk until every check has passed.  The exporter this
replaces wrote the executable first and asserted afterwards, so a rejected
function left a stale ``.binpb`` beside a sidecar describing something else --
which the C++ side then loaded, and which failed a long way from the cause.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import jax
import jaxlib

from . import _ifrt
from ._dtypes import SUPPORTED_DTYPES, dtype_name, unsupported_dtype_message
from ._sidecar import (
    SCHEMA_VERSION,
    array_entry,
    atomic_write_bytes,
    build_sidecar,
    sha256_hex,
    write_sidecar,
)

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_JAX",
    "ExportError",
    "ExportResult",
    "default_input_names",
    "default_output_names",
    "export",
    "jax2exec",
]

#: The JAX release this exporter was written against and is tested on.  A
#: different version is a warning, not a refusal: artifacts are validated by
#: the C++ loader on the way in, and a bump usually just works.
SUPPORTED_JAX = "0.11.0"

_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


class ExportError(ValueError):
    """A function, or its arguments, cannot be exported.

    Raised before anything is written, so a failed export never leaves
    artifacts behind.
    """


@dataclasses.dataclass(frozen=True)
class ExportResult:
    """What :func:`export` produced.

    Attributes
    ----------
    executable : Path
        The written ``.binpb``.
    mlir : Path or None
        The written ``.mlirbc``, or None when ``write_mlir`` was false.
    sidecar : Path
        The written ``.json``.
    metadata : dict
        The sidecar contents, so a caller can assert on shapes and names
        without re-reading the file.
    compiled : Any
        The ``jax.stages.Compiled`` the artifacts came from, kept for callers
        that want to run the function in-process for comparison.
    """

    executable: Path
    mlir: Path | None
    sidecar: Path
    metadata: dict
    compiled: Any


def default_input_names(
    fun: Any, count: int, provided: Sequence[str] | None = None
) -> list[str]:
    """Choose a name for each flattened input.

    Parameters
    ----------
    fun : Any
        The function being exported; its signature is consulted.
    count : int
        Number of flattened inputs.
    provided : Sequence of str or None, optional
        Explicit names, which win outright.

    Returns
    -------
    list of str
        ``provided`` when given, else the parameter names of ``fun`` when they
        line up one-to-one with the flattened inputs, else ``arg0``, ``arg1``,
        and so on.  A pytree argument makes the counts disagree, and then
        positional names are the only honest answer.

    Raises
    ------
    ExportError
        If ``provided`` has the wrong length.
    """
    if provided is not None:
        names = [str(n) for n in provided]
        if len(names) != count:
            raise ExportError(
                f"input_names has {len(names)} entries but the function takes "
                f"{count} flattened inputs"
            )
        return names

    try:
        parameters = list(inspect.signature(fun).parameters.values())
    except (TypeError, ValueError):
        parameters = []

    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if len(parameters) == count and all(
        p.kind in positional for p in parameters
    ):
        return [p.name for p in parameters]

    return [f"arg{i}" for i in range(count)]


def _key_name(key: Any) -> str | None:
    """Return the name a pytree key carries, or None when it carries none.

    Dict keys and attribute names are names a caller can look an output up by
    from C++.  Sequence indices and flattened indices are positions, and
    ``out_0`` says that better than ``[0]`` does.
    """
    if isinstance(key, jax.tree_util.DictKey) and isinstance(key.key, str):
        return key.key or None
    if isinstance(key, jax.tree_util.GetAttrKey):
        return key.name or None
    return None


def default_output_names(
    out_info: Any, count: int, provided: Sequence[str] | None = None
) -> list[str]:
    """Name each output, preferring the pytree path when it carries meaning.

    Parameters
    ----------
    out_info : Any
        The output pytree, of ``ShapeDtypeStruct`` or of arrays.
    count : int
        Number of flattened outputs.
    provided : Sequence of str or None, optional
        Explicit names, which win outright.

    Returns
    -------
    list of str
        A dict or namedtuple result names its own leaves better than any
        scheme here could, so those get ``jax.tree_util.keystr`` of the key
        path.  A bare array or a plain tuple carries nothing but position, so
        those get ``out_0``, ``out_1``, and so on.

    Raises
    ------
    ExportError
        If ``provided`` has the wrong length.
    """
    if provided is not None:
        names = [str(n) for n in provided]
        if len(names) != count:
            raise ExportError(
                f"output_names has {len(names)} entries but the function "
                f"returns {count} flattened outputs"
            )
        return names

    names = []
    paths = jax.tree_util.tree_flatten_with_path(out_info)[0]
    for index, (path, _leaf) in enumerate(paths):
        keys = [_key_name(key) for key in path]
        if len(keys) == 1 and keys[0]:
            names.append(keys[0])  # a flat dict or namedtuple result
        elif any(keys):
            names.append(jax.tree_util.keystr(path))
        else:
            names.append(f"out_{index}")  # position is all there is
    return names


def _leaf_spec(leaf: Any, kind: str, index: int) -> tuple[Any, tuple[int, ...]]:
    """Return ``(dtype, shape)`` of an argument leaf, or refuse it."""
    dtype = getattr(leaf, "dtype", None)
    shape = getattr(leaf, "shape", None)
    if dtype is None or shape is None:
        raise ExportError(
            f"{kind} {index} is a {type(leaf).__name__}; pass "
            "jax.ShapeDtypeStruct or an array, not a Python scalar or an "
            "object JAX has to trace as a constant"
        )
    return dtype, tuple(int(d) for d in shape)


def _check_dtypes(
    leaves: Sequence[Any], names: Sequence[str], kind: str
) -> None:
    """Reject any leaf whose element type has no place in the sidecar."""
    for index, leaf in enumerate(leaves):
        dtype, _shape = _leaf_spec(leaf, kind, index)
        if dtype_name(dtype) not in SUPPORTED_DTYPES:
            raise ExportError(
                unsupported_dtype_message(kind, index, names[index], dtype)
            )


def _check_not_empty(
    leaves: Sequence[Any], names: Sequence[str], kind: str
) -> None:
    """Reject zero-element arrays.

    A zero-byte PJRT buffer is untested on this path: the C++ side allocates
    an arena per array and wraps it zero-copy, and nothing establishes what
    ``posix_memalign(64, 0)`` followed by ``BufferFromHostBuffer`` does.
    """
    for index, leaf in enumerate(leaves):
        _dtype, shape = _leaf_spec(leaf, kind, index)
        if 0 in shape:
            raise ExportError(
                f"{kind} {index} ('{names[index]}') has shape "
                f"{list(shape)} and no elements; zero-element arrays are not "
                "supported"
            )


def _check_x64_trap(
    requested: Sequence[Any], traced: Sequence[Any], names: Sequence[str]
) -> None:
    """Catch the silent float64-to-float32 narrowing.

    Without ``jax_enable_x64`` JAX traces a float64 argument as float32 and
    says nothing.  The export then succeeds, the sidecar honestly records
    float32, and a C++ caller writing doubles into a 4-byte-per-element arena
    walks off the end of it.
    """
    if len(requested) != len(traced):  # pragma: no cover - shapes must match
        return
    for index, (want, got) in enumerate(zip(requested, traced)):
        want_name = dtype_name(getattr(want, "dtype", want))
        got_name = dtype_name(getattr(got, "dtype", got))
        if want_name != got_name:
            raise ExportError(
                f"input {index} ('{names[index]}') was requested as "
                f"{want_name} but JAX traced it as {got_name}: call "
                "jax.config.update('jax_enable_x64', True) before exporting"
            )


def _compile_args(lowered: Any) -> dict[str, Any]:
    """Return the lowering's compile arguments, or an empty dict.

    Three facts the public API does not expose live in here: which parameters
    survived, what effects the function has, and how many devices it wants.
    All three decide whether a C++ caller can use the result at all, so they
    are worth reading from a private attribute -- carefully.  A JAX bump that
    moves them costs the diagnostics below, not the export.
    """
    compile_args = getattr(
        getattr(lowered, "_lowering", None), "compile_args", None
    )
    return compile_args if isinstance(compile_args, dict) else {}


def _check_no_effects(lowered: Any) -> None:
    """Refuse a function that needs the host mid-computation.

    An ``io_callback`` or a ``debug_print`` compiles into a call back into the
    Python process, and the C++ runtime has no Python process.  Caught here
    rather than at ``jax.export``, which reports the same thing as an
    unimplemented host_callback serialization several steps later.
    """
    compile_args = _compile_args(lowered)
    effects = tuple(compile_args.get("ordered_effects") or ()) + tuple(
        compile_args.get("unordered_effects") or ()
    )
    if effects or compile_args.get("host_callbacks"):
        raise ExportError(
            "function has effects (io_callback/debug_print); the C++ runtime "
            "cannot service them"
        )


def _check_no_pruned_inputs(lowered: Any, names: Sequence[str]) -> None:
    """Refuse a function XLA has silently narrowed the parameter list of.

    An argument that reaches no output is dropped from the executable, which
    then takes fewer parameters than the sidecar declares.  The C++ loader
    cannot see this -- the PJRT C API has no parameter query -- so it arrives
    as an opaque warm-up failure.  Catch it here, where the cause is still
    visible, and before the compile that would otherwise be wasted.
    """
    kept = _compile_args(lowered).get("kept_var_idx")
    if kept is None:
        return

    dropped = [i for i in range(len(names)) if i not in kept]
    if not dropped:
        return

    listed = ", ".join(f"{i} ('{names[i]}')" for i in dropped)
    subject = (
        f"input {listed} does" if len(dropped) == 1 else f"inputs {listed} do"
    )
    raise ExportError(
        f"{subject} not reach any output, so XLA dropped it from the "
        f"executable: it takes {len(kept)} of the {len(names)} inputs the "
        "sidecar declares. Make an output depend on every input (even through "
        "a multiply by zero), or stop passing it."
    )


def _serialized_executable(
    compiled: Any,
) -> tuple[bytes, Any, str, bytes]:
    """Return ``(pjrt_bytes, loaded_executable, how, raw_blob)``.

    ``pjrt_bytes`` is what gets written; ``raw_blob`` is what jaxlib returned,
    kept so the in-process round-trip check can be given bytes its own client
    understands.

    There is no public, stable API for the bytes of a serialized PJRT
    executable.  This is the route ``jax.experimental.serialize_executable``
    takes internally, which makes it the least fragile one available, but
    ``Compiled.runtime_executable()`` is documented as debugging-only and the
    fallback below reaches into a private attribute outright.  Both are
    load-bearing and both are checked here rather than left to fail later; see
    ``docs/developer/bumping-jax.md`` when a JAX bump lands on this function.
    """
    try:
        loaded = compiled.runtime_executable()
    except NotImplementedError:
        loaded = None

    if loaded is None:
        executable = getattr(compiled, "_executable", None)
        loaded = getattr(executable, "xla_executable", None)
    if loaded is None:
        raise ExportError(
            "Compiled.runtime_executable() returned None and the private "
            "fallback Compiled._executable.xla_executable is not present in "
            f"jax {jax.__version__}; there is no public API for a serialized "
            "PJRT executable, so this path has to be repaired by hand. See "
            "docs/developer/bumping-jax.md."
        )

    client = getattr(loaded, "client", None)
    serialize = getattr(client, "serialize_executable", None)
    if serialize is None:
        raise ExportError(
            f"the loaded executable from jax {jax.__version__} has no "
            "client.serialize_executable; see docs/developer/bumping-jax.md."
        )

    try:
        blob = serialize(loaded)
    except Exception as exc:
        raise ExportError(
            f"serializing the executable failed: {exc}. See "
            "docs/developer/bumping-jax.md."
        ) from exc

    if not blob:
        raise ExportError(
            "the serialized executable is empty; nothing was compiled that a "
            "C++ caller could load"
        )

    # jaxlib wraps the PJRT bytes in an IFRT envelope, which the PJRT C API
    # cannot parse. See jax2exec._ifrt for the format and why unwrapping is
    # safe to attempt.
    pjrt_bytes, how = _ifrt.unwrap(bytes(blob))
    if not _ifrt.looks_like_pjrt_payload(pjrt_bytes):
        raise ExportError(
            f"jax {jax.__version__} returned {len(blob)} bytes that do not "
            "look like a serialized PJRT executable, and no envelope this "
            "version understands. The C++ loader would fall back to compiling "
            "the .mlirbc on every load. See docs/developer/bumping-jax.md."
        )
    return pjrt_bytes, loaded, how, bytes(blob)


def _export_mlir(jit_fun: Any, args: Sequence[Any]) -> tuple[bytes, int | None]:
    """Return ``(stablehlo_bytecode, calling_convention_version)``.

    The bytecode is the answer to the architecture lock: a ``.binpb`` embeds
    machine code for the exporting host, while this can be compiled in-process
    on whatever machine actually runs the function.
    """
    try:
        exported = jax.export.export(jit_fun, platforms=("cpu",))(*args)
    except Exception as exc:
        raise ExportError(
            f"jax.export.export failed: {exc}. Pass write_mlir=False to "
            "export the .binpb alone, giving up the portable fallback."
        ) from exc

    nr_devices = getattr(exported, "nr_devices", None)
    if nr_devices is None:
        raise ExportError(
            f"jax.export in jax {jax.__version__} no longer reports "
            "nr_devices; see docs/developer/bumping-jax.md."
        )
    if nr_devices != 1:
        raise ExportError(
            f"function lowers for {nr_devices} devices; the C++ runtime "
            "creates a single-device CPU client"
        )

    ordered = tuple(getattr(exported, "ordered_effects", ()) or ())
    unordered = tuple(getattr(exported, "unordered_effects", ()) or ())
    if ordered or unordered:
        raise ExportError(
            "function has effects (io_callback/debug_print); the C++ runtime "
            "cannot service them"
        )

    mlir_bytes = getattr(exported, "mlir_module_serialized", None)
    if not mlir_bytes:
        raise ExportError(
            f"jax.export in jax {jax.__version__} produced no "
            "mlir_module_serialized; see docs/developer/bumping-jax.md."
        )

    version = getattr(exported, "calling_convention_version", None)
    return bytes(mlir_bytes), version


def _verify_roundtrip(loaded: Any, blob: bytes) -> None:
    """Deserialize the blob once, here, where a failure is still cheap.

    ``blob`` must be what jaxlib produced, NOT the unwrapped PJRT payload that
    gets written to disk. jaxlib's client is an IFRT client and only
    understands its own envelope, so handing it the payload reports a parse
    failure for bytes that are in fact correct for their real consumer, the
    PJRT C API. What this checks, then, is that the compile produced a sound
    executable; that the written payload is well formed is checked separately
    in :func:`_serialized_executable`, and the only complete proof is a C++
    load, which the test suite does.

    Every failure is a warning: the API is not public, and a signature change
    here must not fail an export whose artifacts are fine.
    """
    try:
        client = loaded.client
        devices = list(client.devices())
        client.deserialize_executable(blob, devices)
    except Exception as exc:  # noqa: BLE001 - any failure here is a warning
        warnings.warn(
            "the serialized executable did not deserialize in-process "
            f"({type(exc).__name__}: {exc}); the artifacts were written "
            "anyway, but load them from C++ before trusting them",
            RuntimeWarning,
            stacklevel=3,
        )


def _prepare_directory(directory: str | Path) -> Path:
    """Return ``directory`` as a Path, creating it when it does not exist."""
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"cannot use directory {path}: {exc}") from exc
    if not path.is_dir():
        raise ExportError(f"cannot use directory {path}: not a directory")
    return path


def export(
    fun: Any,
    args: Sequence[Any],
    directory: str | Path,
    name: str,
    *,
    donate_argnums: Iterable[int] = (),
    input_names: Sequence[str] | None = None,
    write_mlir: bool = True,
    verify: bool = True,
    overwrite: bool = True,
) -> ExportResult:
    """Compile ``fun`` and write the artifacts a C++ caller needs.

    Parameters
    ----------
    fun : callable
        Anything ``jax.jit`` accepts.  It is traced once, for CPU.
    args : Sequence
        Example arguments, positional only.  ``jax.ShapeDtypeStruct`` avoids
        materializing data that is only used for its shape and dtype.
    directory : str or Path
        Where the artifacts go.  Created if missing.
    name : str
        Base name, matching ``[A-Za-z0-9_.-]+``.  The C++ side is handed this
        same path without an extension.
    donate_argnums : Iterable of int, optional
        Passed to ``jax.jit`` and recorded in the sidecar.  Note that the C++
        runtime wraps input arenas once and reuses them, so a donated input is
        described faithfully but not yet exploited.
    input_names : Sequence of str or None, optional
        Names for the flattened inputs.  Defaults to the parameter names of
        ``fun`` when they line up, else ``arg0``, ``arg1``, ...
    write_mlir : bool, optional
        Also write StableHLO bytecode, which the C++ loader compiles in-process
        when the ``.binpb`` was built for a different machine.
    verify : bool, optional
        Deserialize the executable once before writing it.  Failures warn.
    overwrite : bool, optional
        When false, refuse rather than replace existing artifacts.

    Returns
    -------
    ExportResult
        Paths, the sidecar contents, and the compiled function.

    Raises
    ------
    ExportError
        For anything about ``fun`` or ``args`` that a C++ caller could not
        work with: keyword arguments, unsupported dtypes, zero-element arrays,
        effects, multiple devices, or an argument XLA pruned away.
    FileExistsError
        If ``overwrite`` is false and an artifact is already there.

    Notes
    -----
    Nothing is written until every check has passed, and the three files are
    written ``.binpb``, ``.mlirbc``, ``.json`` so that a reader never sees a
    sidecar promising artifacts that are not there yet.
    """
    # 1. Version: warn, do not refuse. Artifacts are validated on the way in.
    if jax.__version__ != SUPPORTED_JAX:
        warnings.warn(
            f"jax2exec is tested against jax {SUPPORTED_JAX} but this is "
            f"{jax.__version__}; a serialized executable is not portable "
            "across a JAX bump, so re-export rather than reusing artifacts",
            RuntimeWarning,
            stacklevel=2,
        )
    x64_enabled = bool(jax.config.jax_enable_x64)

    # 2. Everything cheap that can refuse, before anything expensive runs.
    if not _NAME_RE.fullmatch(name):
        raise ExportError(
            f"name {name!r} must match [A-Za-z0-9_.-]+; it becomes a file "
            "name and a C++ load path"
        )
    if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
        raise ExportError(
            "args must be a sequence of jax.ShapeDtypeStruct or arrays, not "
            f"{type(args).__name__}"
        )
    args = tuple(args)
    for index, leaf in enumerate(jax.tree_util.tree_leaves(args)):
        _leaf_spec(leaf, "input", index)

    out_dir = _prepare_directory(directory)
    executable_path = out_dir / f"{name}.binpb"
    mlir_path = out_dir / f"{name}.mlirbc" if write_mlir else None
    sidecar_path = out_dir / f"{name}.json"
    if not overwrite:
        existing = [
            str(p)
            for p in (executable_path, mlir_path, sidecar_path)
            if p is not None and p.exists()
        ]
        if existing:
            raise FileExistsError(
                f"{', '.join(existing)} already exist(s); pass "
                "overwrite=True to replace"
            )

    # 3. Lower once. Keyword arguments have no place in a positional C++ call.
    donate = tuple(int(i) for i in donate_argnums)
    jit_fun = jax.jit(fun, donate_argnums=donate)
    lowered = jit_fun.lower(*args)
    if lowered.args_info[1]:
        raise ExportError("keyword arguments are not supported")
    _check_no_effects(lowered)

    # 4. Flatten. out_info is a bare ShapeDtypeStruct for a single-array
    #    return, so it goes through the tree utilities like everything else.
    in_info = jax.tree_util.tree_leaves(lowered.args_info[0])
    out_info = jax.tree_util.tree_leaves(lowered.out_info)
    in_names = default_input_names(fun, len(in_info), input_names)
    out_names = default_output_names(lowered.out_info, len(out_info))

    # 5. Refuse what the C++ side could not represent or would misread.
    _check_dtypes(in_info, in_names, "input")
    _check_dtypes(out_info, out_names, "output")
    _check_x64_trap(jax.tree_util.tree_leaves(args), in_info, in_names)
    _check_not_empty(in_info, in_names, "input")
    _check_not_empty(out_info, out_names, "output")
    _check_no_pruned_inputs(lowered, in_names)

    # 6. Compile and serialize.
    compiled = lowered.compile()
    blob, loaded, serialized_how, raw_blob = _serialized_executable(compiled)

    # 7. The portable fallback.
    mlir_bytes: bytes | None = None
    calling_convention_version: int | None = None
    if write_mlir:
        mlir_bytes, calling_convention_version = _export_mlir(jit_fun, args)

    # 8. Describe it.
    donated_flags = [bool(getattr(info, "donated", False)) for info in in_info]
    sidecar = build_sidecar(
        name=name,
        inputs=[
            array_entry(
                index,
                in_names[index],
                info.dtype,
                info.shape,
                donated=donated_flags[index],
            )
            for index, info in enumerate(in_info)
        ],
        outputs=[
            array_entry(index, out_names[index], info.dtype, info.shape)
            for index, info in enumerate(out_info)
        ],
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        x64_enabled=x64_enabled,
        executable=executable_path.name,
        executable_sha256=sha256_hex(blob),
        mlir=None if mlir_bytes is None else mlir_path.name,
        mlir_sha256=None if mlir_bytes is None else sha256_hex(mlir_bytes),
        calling_convention_version=calling_convention_version,
        executable_source=serialized_how,
        donate_argnums=donate,
    )

    # 9. Prove the blob comes back before promising it does.
    if verify:
        _verify_roundtrip(loaded, raw_blob)

    # 10. Executable, bytecode, then sidecar: the sidecar is the manifest, so
    #     it lands last and a reader that has it has everything it names.
    atomic_write_bytes(executable_path, blob)
    if mlir_bytes is not None and mlir_path is not None:
        atomic_write_bytes(mlir_path, mlir_bytes)
    write_sidecar(sidecar_path, sidecar)

    return ExportResult(
        executable=executable_path,
        mlir=mlir_path if mlir_bytes is not None else None,
        sidecar=sidecar_path,
        metadata=sidecar,
        compiled=compiled,
    )


def jax2exec(
    fun: Any, args: Sequence[Any], directory: str | Path, fun_name: str
) -> None:
    """Export ``fun`` the way the first exporter did.

    Parameters
    ----------
    fun : callable
        Anything ``jax.jit`` accepts.
    args : Sequence
        Example arguments.
    directory : str or Path
        Where the artifacts go.
    fun_name : str
        Base name for the artifacts.

    Warns
    -----
    DeprecationWarning
        Always.  This exists so code written against the v1 exporter keeps
        running; it now writes a schema 2 sidecar and a ``.mlirbc`` as well,
        and it no longer restricts inputs to rank-1 float64.

    See Also
    --------
    export : The current entry point, which returns the paths it wrote.
    """
    warnings.warn(
        "jax2exec() is deprecated; call jax2exec.export(fun, args, "
        "directory, name), which returns an ExportResult and writes a "
        "schema 2 sidecar",
        DeprecationWarning,
        stacklevel=2,
    )
    export(fun, args, directory, fun_name)
