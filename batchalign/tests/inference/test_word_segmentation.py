"""Tests for the worker's word segmentation (``batchalign.inference.word_segmentation``).

The Rust control plane refuses any token list that does not re-spell its run,
so the property that matters here is that the segmenter re-spells exactly; the
expected groupings pin taibun's lexicon for a few everyday phrases, and change
only when the pinned taibun version does.
"""

from __future__ import annotations

import json

import pytest

from batchalign.inference.word_segmentation import segment_batch
from batchalign.worker._types_v2 import WordSegmentationResultPayloadV2, WordSegmenterV2

PHRASES = ["我欲去學校", "你食飽未", "阮阿公佇厝"]
SEGMENTED = [
    ["我", "欲", "去", "學校"],
    ["你", "食飽", "未"],
    ["阮", "阿公", "佇", "厝"],
]


def _segment(runs: list[str]) -> WordSegmentationResultPayloadV2:
    return segment_batch(
        WordSegmenterV2.TAIBUN_TAIWANESE.value, json.dumps({"items": runs})
    )


def test_taibun_groups_taiwanese_words() -> None:
    result = _segment(PHRASES)
    assert result.segmenter.startswith("taibun@")
    assert result.items == SEGMENTED


@pytest.mark.parametrize(
    "run",
    ["臺灣", "馬上馬上", "甜", "伊講伊毋知影", "食飽未"],
)
def test_tokens_respell_their_run(run: str) -> None:
    (tokens,) = _segment([run]).items
    assert "".join(tokens) == run
    assert all(tokens)


def test_one_token_list_per_run_in_order() -> None:
    runs = ["食飽未", "臺灣", "甜"]
    assert ["".join(tokens) for tokens in _segment(runs).items] == runs


def test_an_unknown_segmenter_is_refused() -> None:
    with pytest.raises(ValueError):
        segment_batch("unknown", json.dumps({"items": ["臺灣"]}))
