"""Unwrapping the IFRT envelope jaxlib puts around a serialized executable.

``jax2exec._ifrt.unwrap`` removes the header when it recognises one and hands
the blob back untouched when it does not: an unrecognised blob passed through
at worst reproduces not having the module, while a wrong guess writes a file
that fails at load.  Every unrecognised shape below comes back byte-identical
and labelled ``as-is``, and nothing may raise.  The envelopes are synthetic
-- ``tag || length || bytes`` -- because that is the only way to reach the
paths a real jaxlib does not produce.
"""

from __future__ import annotations

import pytest
from jax2exec._ifrt import looks_like_pjrt_payload, unwrap

#: The format tag IFRT writes for a PJRT-backed executable, as field 1 of the
#: header: 0x0a is (field 1, length-delimited), 0x09 is the length.
_PJRT_IFRT_FIELD = b"\x0a\x09pjrt_ifrt"

#: Some other format, same wire shape.  Eleven characters, so the envelope's
#: length prefix cannot be mistaken for a payload's first tag.
_OTHER_FIELD = b"\x0a\x0bnot_an_ifrt"


def envelope(header: bytes, payload: bytes) -> bytes:
    """Wrap ``payload`` the way IFRT does: varint length, header, payload."""
    assert len(header) < 128, "one-byte varint only"
    assert len(header) != 0x0A, (
        "a header of exactly ten bytes makes the length prefix look like an "
        "ExecutableAndOptionsProto tag, which is a different code path"
    )
    return bytes([len(header)]) + header + payload


@pytest.fixture(scope="module")
def pjrt_payload(artifacts):
    """The real serialized executable ``examples/01_basic`` exports, so the
    round trips below are the bytes the C++ loader deserializes."""
    blob = (artifacts / "basic.binpb").read_bytes()
    assert blob, "the exported executable is empty"
    return blob


def test_a_written_executable_is_already_plain(pjrt_payload):
    """``.binpb`` on disk has been through ``unwrap`` once, so it must look
    like an ``ExecutableAndOptionsProto`` to a second pass."""
    assert looks_like_pjrt_payload(pjrt_payload)
    unwrapped, how = unwrap(pjrt_payload)
    assert how == "as-is"
    assert unwrapped == pjrt_payload


def test_unwrap_is_idempotent(pjrt_payload):
    """Unwrapping twice is unwrapping once, not a header eaten off the front
    of a payload."""
    wrapped = envelope(_PJRT_IFRT_FIELD, pjrt_payload)
    once, how_once = unwrap(wrapped)
    assert how_once == "ifrt-unwrapped"
    assert once == pjrt_payload

    twice, how_twice = unwrap(once)
    assert how_twice == "as-is"
    assert twice == once


@pytest.mark.parametrize(
    ("label", "blob"),
    [
        ("empty", b""),
        ("garbage", b"\xff\xfe\xfd\xfc"),
        ("text", b"this is not a protobuf at all"),
        # A varint that never terminates: every byte has the continuation bit.
        ("unterminated varint", b"\x80" * 12),
    ],
)
def test_unrecognised_blobs_come_back_untouched(label, blob):
    """Never raises, never edits; the loader's ``.mlirbc`` fallback covers
    bytes that turn out not to be loadable."""
    unwrapped, how = unwrap(blob)
    assert how == "as-is", label
    assert unwrapped == blob, label


def test_a_truncated_envelope_does_not_raise(pjrt_payload):
    """Truncation is the shape a half-written file has.  Cut inside the
    header the blob passes through; cut inside the payload the remains come
    back.  Either is fine; an exception is not."""
    wrapped = envelope(_PJRT_IFRT_FIELD, pjrt_payload)

    inside_header = wrapped[: len(_PJRT_IFRT_FIELD) - 2]
    unwrapped, how = unwrap(inside_header)
    assert how == "as-is"
    assert unwrapped == inside_header

    inside_payload = wrapped[: len(wrapped) // 2]
    unwrapped, how = unwrap(inside_payload)
    assert how in {"as-is", "ifrt-unwrapped"}
    assert unwrapped == inside_payload[-len(unwrapped) :]


def test_an_envelope_naming_another_format_is_left_alone(pjrt_payload):
    """Only ``pjrt_ifrt`` is unwrapped: stripping another format's header
    would produce a file that deserializes to the wrong thing."""
    wrapped = envelope(_OTHER_FIELD, pjrt_payload)
    assert not looks_like_pjrt_payload(wrapped), (
        "this blob has to reach the header check, not the payload shortcut"
    )
    unwrapped, how = unwrap(wrapped)
    assert how == "as-is"
    assert unwrapped == wrapped


def test_an_envelope_around_something_else_is_left_alone():
    """The header alone is not proof; the payload behind it has to start like
    an ``ExecutableAndOptionsProto`` as well."""
    wrapped = envelope(_PJRT_IFRT_FIELD, b"\x12 not a serialized executable")
    unwrapped, how = unwrap(wrapped)
    assert how == "as-is"
    assert unwrapped == wrapped


def test_the_tag_is_found_by_walking_fields_not_by_searching(pjrt_payload):
    """A payload that merely contains the characters is not a header."""
    coincidence = b"\x0a\x0dxxpjrt_ifrtxx"
    wrapped = envelope(coincidence, pjrt_payload)
    unwrapped, how = unwrap(wrapped)
    assert how == "as-is"
    assert unwrapped == wrapped
