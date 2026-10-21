"""Word segmentation of Han-character text, for the segmenter Rust names.

Rust decides which segmenter a language uses (``WordSegmenterV2::for_language``)
and sends runs of consecutive one-character words; this module returns each
run's tokens. It never sees timing or CHAT: the Rust control plane merges the
timed characters into words, and refuses any result whose tokens do not
re-spell their run exactly.

Taiwanese uses taibun's dictionary tokeniser. Its code is MIT-licensed; its
lexicon, which the package ships, comes from the Taiwanese-Chinese Online
Dictionary and iTaigi via ChhoeTaigi under CC BY-SA 4.0. BA3 depends on the
package and does not redistribute or modify the lexicon.

Importing taibun loads its lexicon, a few tens of milliseconds, so it is
imported on first use rather than at worker bootstrap.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, assert_never

from pydantic import BaseModel

from batchalign.worker._types import reported_engine_name
from batchalign.worker._types_v2 import (
    WordSegmentationResultPayloadV2,
    WordSegmenterV2,
)

if TYPE_CHECKING:
    from taibun import Tokeniser


class WordSegmentationPreparedBatchV2(BaseModel):
    """Prepared word-segmentation batch frozen by Rust: one Han run per item.

    Mirrors `crates/batchalign/src/worker/text_request_v2.rs::PreparedWordSegmentationBatchV2`.
    """

    items: list[str]


@cache
def _taibun() -> tuple[Tokeniser, str]:
    """The taibun tokeniser and its reported identity, loaded once."""
    try:
        from taibun import Tokeniser

        package_version = version("taibun")
    except (ImportError, PackageNotFoundError) as error:
        # The Rust taxonomy reports NotImplementedError as model_unavailable.
        raise NotImplementedError(f"taibun is not installed: {error}") from error
    return Tokeniser(), reported_engine_name(f"taibun@{package_version}")


def segment_batch(segmenter: str, batch_json: str) -> WordSegmentationResultPayloadV2:
    """Segment every run of a prepared batch.

    Returns the segmenter's identity and one token list per run, in batch
    order, which the Rust executor parses into ``WordSegmentationResultV2``.
    An unknown segmenter or a malformed batch raises ``ValueError``, which the
    executor reports as an invalid payload.
    """
    batch = WordSegmentationPreparedBatchV2.model_validate_json(batch_json)
    match WordSegmenterV2(segmenter):
        case WordSegmenterV2.TAIBUN_TAIWANESE:
            tokeniser, identity = _taibun()
            items = [tokeniser.tokenise(run) for run in batch.items]
        case unreachable:
            assert_never(unreachable)
    return WordSegmentationResultPayloadV2(segmenter=identity, items=items)
