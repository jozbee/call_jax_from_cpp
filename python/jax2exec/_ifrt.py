"""Unwrap the IFRT envelope that jaxlib puts around a serialized executable.

Why this exists
---------------
The C++ side loads a serialized executable through the PJRT C API, with
``PJRT_Executable_DeserializeAndLoad``, which expects XLA's
``ExecutableAndOptionsProto``.

jaxlib no longer hands out those bytes directly. Since jaxlib moved its client
onto IFRT, both ``client.serialize_executable(loaded)`` and
``loaded.serialize()`` return the same thing: an IFRT envelope wrapping the
PJRT bytes. Feeding it to the plugin fails with

    PjRtCpuClient::DeserializeExecutable proto deserialization failed

The envelope is a length-delimited header followed by the payload::

    varint(header_length) || header_proto || pjrt_payload

where the header names the serialization format and carries device and
sharding information, and the payload is byte-for-byte the
``ExecutableAndOptionsProto`` the PJRT C API wants. Verified at
jaxlib 0.11.0: stripping the header makes the payload deserialize, and the
resulting executable runs and returns correct results.

This is an internal format, so everything here is defensive: the envelope is
recognised by its own contents rather than assumed, an unrecognised blob is
passed through unchanged (older jaxlib handed back plain PJRT bytes, and a
future one may again), and the C++ loader keeps its ``.mlirbc`` fallback for
the case where a future envelope slips past these checks. A bad unwrap
therefore costs a compile at load time, never a wrong answer.

See ``docs/developer/bumping-jax.md``: confirming this envelope is a step of
every JAX bump.
"""

from __future__ import annotations

# The format tag IFRT writes into the header for a PJRT-backed executable.
_PJRT_IFRT_TAG = b"pjrt_ifrt"

# Protobuf wire-format constants. Only what is needed to read the header.
_WIRE_VARINT = 0
_WIRE_FIXED64 = 1
_WIRE_LENGTH_DELIMITED = 2
_WIRE_FIXED32 = 5

# ExecutableAndOptionsProto.serialized_executable is field 1, length-delimited,
# so a valid payload starts with this tag byte.
_EXECUTABLE_AND_OPTIONS_FIRST_TAG = 0x0A


class EnvelopeError(ValueError):
    """The blob is not an envelope this module knows how to unwrap."""


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read a base-128 varint, returning ``(value, next_offset)``."""
    value = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise EnvelopeError("truncated varint")
        if shift > 63:
            raise EnvelopeError("varint too long to be a length")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _header_contains_tag(header: bytes, tag: bytes) -> bool:
    """Whether any length-delimited field of ``header`` equals ``tag``.

    Walking the fields rather than searching for the bytes anywhere means a
    payload that happens to contain the same characters cannot be mistaken for
    a header.
    """
    offset = 0
    while offset < len(header):
        key, offset = _read_varint(header, offset)
        wire_type = key & 0x07
        if wire_type == _WIRE_VARINT:
            _, offset = _read_varint(header, offset)
        elif wire_type == _WIRE_FIXED64:
            offset += 8
        elif wire_type == _WIRE_LENGTH_DELIMITED:
            length, offset = _read_varint(header, offset)
            if header[offset : offset + length] == tag:
                return True
            offset += length
        elif wire_type == _WIRE_FIXED32:
            offset += 4
        else:
            raise EnvelopeError(f"unsupported wire type {wire_type}")
    return False


def looks_like_pjrt_payload(blob: bytes) -> bool:
    """Whether ``blob`` plausibly is an ``ExecutableAndOptionsProto``."""
    return bool(blob) and blob[0] == _EXECUTABLE_AND_OPTIONS_FIRST_TAG


def unwrap(blob: bytes) -> tuple[bytes, str]:
    """Return ``(pjrt_bytes, how)`` for a serialized executable.

    Parameters
    ----------
    blob :
        Whatever jaxlib returned from ``serialize_executable``.

    Returns
    -------
    pjrt_bytes :
        Bytes suitable for ``PJRT_Executable_DeserializeAndLoad``.
    how :
        ``"ifrt-unwrapped"`` when an envelope was removed, ``"as-is"`` when the
        blob already looked like PJRT bytes.

    Notes
    -----
    Never raises for an unrecognised blob: it is returned unchanged, and the
    C++ loader's ``.mlirbc`` fallback covers the case where it turns out not to
    be loadable.
    """
    if looks_like_pjrt_payload(blob):
        return blob, "as-is"

    try:
        header_length, offset = _read_varint(blob, 0)
        header = blob[offset : offset + header_length]
        if len(header) != header_length:
            raise EnvelopeError("header shorter than its declared length")
        if not _header_contains_tag(header, _PJRT_IFRT_TAG):
            raise EnvelopeError(
                f"header does not name {_PJRT_IFRT_TAG.decode()}"
            )
        payload = blob[offset + header_length :]
        if not looks_like_pjrt_payload(payload):
            raise EnvelopeError(
                "the payload behind the envelope does not start like an "
                "ExecutableAndOptionsProto"
            )
    except EnvelopeError:
        # Unrecognised. Hand it back untouched rather than guessing: a wrong
        # guess would write a file that fails at load, while passing it
        # through at worst reproduces the behaviour of not having this module.
        return blob, "as-is"

    return payload, "ifrt-unwrapped"
