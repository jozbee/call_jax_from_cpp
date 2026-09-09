"""``conftest.parse_kv_lines``, checked against real ``example_01_basic``
output.  Every assertion the integration tests make about a binary goes
through it, and a parser that quietly returned an empty dict would turn
those into tests that cannot fail."""

from __future__ import annotations

from conftest import parse_kv_lines

BASIC_OUTPUT = """\
load_kind=deserialized
synchronous_supported=1
sync_mode=inline
num_inputs=2 num_outputs=2
input[0]: dtype=float64 shape=[4,4] numel=16 nbytes=128
output[1]: dtype=float64 shape=[] numel=1 nbytes=8
x=[-0.0853295001417,0.159887040996,0.194250782611,0.125583251801]
residual_inf_norm=1.526557e-16
debug=0 (checks disabled; see --debug)
"""

DEBUG_OUTPUT = """\
debug_check[out_of_range]: input index 99 is out of range: function 'basic' \
has 2 inputs
debug_check[dtype_mismatch]: input 0 ('A') has dtype float64 but was accessed \
as float32
"""


def test_bare_pairs_land_at_the_top_level():
    parsed = parse_kv_lines(BASIC_OUTPUT)
    assert parsed["load_kind"] == "deserialized"
    assert parsed["synchronous_supported"] == "1"
    assert parsed["sync_mode"] == "inline"
    # Several pairs on one line are several entries.
    assert parsed["num_inputs"] == "2"
    assert parsed["num_outputs"] == "2"


def test_labelled_pairs_land_under_their_label():
    parsed = parse_kv_lines(BASIC_OUTPUT)
    assert parsed["input[0]"] == {
        "dtype": "float64",
        "shape": "[4,4]",
        "numel": "16",
        "nbytes": "128",
    }
    # A scalar prints an empty shape, which must survive as one.
    assert parsed["output[1]"]["shape"] == "[]"
    assert parsed["output[1]"]["numel"] == "1"


def test_values_keep_their_punctuation():
    """A vector or an exponent is one token, not a number to be guessed at."""
    parsed = parse_kv_lines(BASIC_OUTPUT)
    assert parsed["x"].startswith("[-0.0853295001417,")
    assert parsed["residual_inf_norm"] == "1.526557e-16"
    # Trailing prose after a pair is not a pair, and is dropped.
    assert parsed["debug"] == "0"


def test_a_labelled_line_without_pairs_keeps_its_message():
    """The debug checks print a sentence, and the sentence is the value."""
    parsed = parse_kv_lines(DEBUG_OUTPUT)
    assert "is out of range: function" in parsed["debug_check[out_of_range]"]
    assert "but was accessed as" in parsed["debug_check[dtype_mismatch]"]


def test_unparseable_output_is_ignored_not_invented():
    parsed = parse_kv_lines("no pairs here\n\n   \nstill none\n")
    assert parsed == {}
