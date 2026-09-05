"""Unwrapping the IFRT envelope jaxlib puts around a serialized executable.

The C++ loader deserializes through ``PJRT_Executable_DeserializeAndLoad``,
which wants XLA's ``ExecutableAndOptionsProto``.  jaxlib no longer hands those
bytes out: since its client moved onto IFRT, ``serialize_executable`` returns
a length-delimited header followed by that payload, and feeding the whole
thing to the plugin fails with ``proto deserialization failed``.

``jax2exec._ifrt.unwrap`` removes the header when it recognises one and hands
the blob back untouched when it does not, because an unrecognised blob passed
through at worst reproduces the behaviour of not having the module at all,
while a wrong guess writes a file that fails at load.  That asymmetry is what
these tests pin down: every unrecognised shape below has to come back
byte-identical and labelled ``as-is``, and nothing may raise.

The envelopes here are synthetic on purpose.  A protobuf header is
``tag || length || bytes`` for a length-delimited field, so one can be built
by hand, and building it by hand is the only way to test the paths a real
jaxlib does not produce -- a truncated header, a foreign format tag.
"""

from __future__ import annotations

import pytest
from jax2exec._ifrt import looks_like_pjrt_payload, unwrap

#: The format tag IFRT writes for a PJRT-backed executable, as field 1 of the
#: header: 0x0a is (field 1, length-delimited), 0x09 is the length.
_PJRT_IFRT_FIELD = b"\x0a\x09pjrt_ifrt"

#: Some other serialization format, same wire shape.  Eleven characters, so
#: the envelope's length prefix cannot be mistaken for a payload's first tag.
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
    """The real serialized executable ``examples/01_basic`` exports.

    Read from disk rather than synthesized: what the exporter writes is what
    the C++ loader deserializes, so the round trips below are the real bytes.
    """
    blob = (artifacts / "basic.binpb").read_bytes()
    assert blob, "the exported executable is empty"
    return blob


def test_a_written_executable_is_already_plain(pjrt_payload):
    """What the exporter writes is the payload, not another envelope.

    ``.binpb`` on disk has been through ``unwrap`` once already, so it must
    look like an ``ExecutableAndOptionsProto`` to the C++ side and to a second
    pass here.
    """
    assert looks_like_pjrt_payload(pjrt_payload)
    unwrapped, how = unwrap(pjrt_payload)
    assert how == "as-is"
    assert unwrapped == pjrt_payload


def test_unwrap_is_idempotent(pjrt_payload):
    """Unwrapping twice is unwrapping once.

    The exporter unwraps and the C++ side does not, but nothing in the format
    stops a caller from asking twice, and the second answer has to be the same
    bytes rather than a header eaten off the front of a payload.
    """
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
    """Never raises, never edits.  The loader's ``.mlirbc`` fallback covers
    the case where the bytes turn out not to be loadable, so a bad unwrap
    costs a compile at load time -- but only if nothing here throws first."""
    unwrapped, how = unwrap(blob)
    assert how == "as-is", label
    assert unwrapped == blob, label


def test_a_truncated_envelope_does_not_raise(pjrt_payload):
    """Truncation is the shape a half-written file has.

    Cut inside the header the declared length can no longer be satisfied and
    the blob is passed through; cut inside the payload the header is still
    intact and the remains of the payload come back.  Either answer is fine.
    What is not fine is an exception out of a module whose whole contract is
    that it never raises for an unrecognised blob.
    """
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
    """Only ``pjrt_ifrt`` is unwrapped.

    A future envelope carrying something other than PJRT bytes must survive
    intact: stripping its header would produce a file that deserializes to the
    wrong thing, where passing it through produces a load error naming it.
    """
    wrapped = envelope(_OTHER_FIELD, pjrt_payload)
    assert not looks_like_pjrt_payload(wrapped), (
        "this blob has to reach the header check, not the payload shortcut"
    )
    unwrapped, how = unwrap(wrapped)
    assert how == "as-is"
    assert unwrapped == wrapped


def test_an_envelope_around_something_else_is_left_alone():
    """A ``pjrt_ifrt`` header over bytes that are not a PJRT payload.

    The header alone is not proof; the payload behind it has to start like an
    ``ExecutableAndOptionsProto`` as well, or the envelope is not one this
    module understands.
    """
    wrapped = envelope(_PJRT_IFRT_FIELD, b"\x12 not a serialized executable")
    unwrapped, how = unwrap(wrapped)
    assert how == "as-is"
    assert unwrapped == wrapped


def test_the_tag_is_found_by_walking_fields_not_by_searching(pjrt_payload):
    """A payload that merely contains the characters is not a header.

    ``_header_contains_tag`` walks the wire format rather than searching the
    bytes, which is what stops a coincidence in the executable from being read
    as a format name.
    """
    coincidence = b"\x0a\x0dxxpjrt_ifrtxx"
    wrapped = envelope(coincidence, pjrt_payload)
    unwrapped, how = unwrap(wrapped)
    assert how == "as-is"
    assert unwrapped == wrapped
