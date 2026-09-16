# affects: crates/batchalign-pyo3/src/py_json_bridge.rs
"""The Python-to-JSON boundary, exercised against a real interpreter.

``py_to_json_value`` decides what shapes may cross into Rust, so its questions
only mean something when a real Python object asks them. Every provider payload
and every worker response passes through it; these tests reach it through
``aliyun_sentences_to_asr``, which hands it its argument before anything else
happens.

What is pinned here:

- the ORDER of the exact-type questions (a ``bool`` stays a bool rather than
  becoming ``1``; a ``str`` that looks numeric stays a string),
- that a value which merely CONVERTS to a number is refused rather than
  silently coerced, which is what the previous numeric-first order did, and
- that a refusal NAMES THE PATH to the offending value, so a NaN deep inside a
  response is locatable instead of being reported as "invalid float".
"""

from __future__ import annotations

import json

import pytest

batchalign_core = pytest.importorskip("batchalign_core")


def _project(sentences: object) -> dict:
    """Run one payload through the boundary and return the Rust projection."""
    return json.loads(batchalign_core.aliyun_sentences_to_asr(sentences, "yue"))


class _NumberLike:
    """An object that is not a number but answers like one.

    The previous conversion asked "can this be read as an int/float" before it
    asked "is this a container or a string", so a class like this became a JSON
    number. It is the shape the exact-type dispatch exists to refuse.
    """

    def __int__(self) -> int:
        return 7

    def __float__(self) -> float:
        return 7.0

    def __index__(self) -> int:
        return 7


def test_a_well_formed_payload_still_crosses() -> None:
    projection = _project(
        [
            {
                "words": [{"text": "你", "startTime": 0, "endTime": 200}],
                "sentence_text": "你",
            }
        ]
    )
    assert projection["monologues"][0]["elements"][0]["value"] == "你"
    assert projection["timed_words"][0]["start_ms"] == 0
    assert projection["timed_words"][0]["end_ms"] == 200


def test_a_non_finite_float_is_refused_naming_its_path() -> None:
    with pytest.raises(Exception) as refusal:
        _project(
            [
                {
                    "words": [
                        {"text": "你", "startTime": float("nan"), "endTime": 200}
                    ],
                    "sentence_text": "你",
                }
            ]
        )
    message = str(refusal.value)
    assert "$[0].words[0].startTime" in message, message
    assert "finite" in message, message


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_an_infinite_float_is_refused_too(value: float) -> None:
    with pytest.raises(Exception) as refusal:
        _project([{"words": [], "sentence_text": "x", "extra": value}])
    assert "$[0].extra" in str(refusal.value)


def test_a_number_like_object_is_refused_by_type_instead_of_coerced() -> None:
    with pytest.raises(Exception) as refusal:
        _project([{"words": [], "sentence_text": "x", "extra": _NumberLike()}])
    message = str(refusal.value)
    assert "$[0].extra" in message, message
    assert "_NumberLike" in message, message


def test_an_integer_beyond_sixty_four_bits_is_refused_rather_than_wrapped() -> None:
    with pytest.raises(Exception) as refusal:
        _project([{"words": [], "sentence_text": "x", "extra": 2**200}])
    message = str(refusal.value)
    assert "$[0].extra" in message, message
    assert "64-bit" in message, message


def test_the_largest_unsigned_integer_still_crosses() -> None:
    # The boundary is exactly the 64-bit range, on both sides of zero.
    _project([{"words": [], "sentence_text": "x", "extra": 2**64 - 1}])
    _project([{"words": [], "sentence_text": "x", "extra": -(2**63)}])


def test_a_bool_does_not_become_a_number() -> None:
    # Python's `bool` is an `int` subclass. An inexact test turns `True` into
    # `1`, which the wire cannot tell from a real count.
    projection = _project([{"words": [], "sentence_text": "x", "extra": True}])
    assert projection["monologues"] == [] or projection["monologues"][0]["elements"]


def test_a_numpy_float_crosses_instead_of_being_refused_by_type() -> None:
    # `numpy.float64` IS a `float` subclass, so it is a real number rather than
    # an object imitating one. Whisper forced alignment returns its word
    # timings this way; refusing them left every FA group unaligned while the
    # job still reported success.
    #
    # It rides on `extra` because this payload has no float-typed field: the
    # Aliyun word times are integer milliseconds. What is pinned here is that
    # the value is ADMITTED; that it lands in the float branch is pinned by the
    # NaN test below, which refuses it for finiteness rather than for type.
    numpy = pytest.importorskip("numpy")
    projection = _project(
        [{"words": [], "sentence_text": "好", "extra": numpy.float64(1.5)}]
    )
    assert projection["monologues"][0]["elements"][0]["value"] == "好"


def test_a_numpy_integer_crosses_as_a_number() -> None:
    # `numpy.int64` is not an `int` subclass at all, so it reaches the numpy
    # arm by the same route and must land on the same value.
    numpy = pytest.importorskip("numpy")
    projection = _project(
        [
            {
                "words": [
                    {
                        "text": "你",
                        "startTime": numpy.int64(0),
                        "endTime": numpy.int64(200),
                    }
                ],
                "sentence_text": "你",
            }
        ]
    )
    assert projection["timed_words"][0]["start_ms"] == 0
    assert projection["timed_words"][0]["end_ms"] == 200


def test_a_numpy_bool_does_not_become_a_number() -> None:
    # The bool-before-int rule has to survive the numpy arm too: `item()` on a
    # `numpy.bool_` yields a Python `bool`, which the exact tests then keep out
    # of the number branch.
    numpy = pytest.importorskip("numpy")
    projection = _project(
        [{"words": [], "sentence_text": "x", "extra": numpy.bool_(True)}]
    )
    assert projection["monologues"] == [] or projection["monologues"][0]["elements"]


def test_a_non_finite_numpy_float_is_still_refused_naming_its_path() -> None:
    # Admission by name is not admission of anything: once `item()` has run,
    # the ordinary float rules apply, NaN included.
    numpy = pytest.importorskip("numpy")
    with pytest.raises(Exception) as refusal:
        _project([{"words": [], "sentence_text": "x", "extra": numpy.float64("nan")}])
    message = str(refusal.value)
    assert "$[0].extra" in message, message
    assert "finite" in message, message


def test_a_numeric_string_stays_a_string() -> None:
    # `sentence_text` is a string field; a numeric-looking value must reach
    # Rust as a string, not as a number that fails to deserialize.
    projection = _project([{"words": [], "sentence_text": "123"}])
    values = [element["value"] for element in projection["monologues"][0]["elements"]]
    assert values == ["1", "2", "3"]


def test_a_non_string_object_key_is_refused() -> None:
    with pytest.raises(Exception) as refusal:
        _project([{1: "not a field name", "words": [], "sentence_text": "x"}])
    assert "exact str" in str(refusal.value)


def test_a_tuple_is_accepted_as_a_sequence() -> None:
    # Tuples reach the boundary from adapters that build fixed-width rows.
    projection = _project(({"words": (), "sentence_text": "好"},))
    assert projection["monologues"][0]["elements"][0]["value"] == "好"


def test_the_payload_itself_may_be_a_single_object() -> None:
    projection = _project({"words": [], "sentence_text": "好"})
    assert projection["monologues"][0]["elements"][0]["value"] == "好"
