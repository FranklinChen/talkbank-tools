"""Integration test: Stanza control-token leak is stripped and logged
inside ``batch_infer_morphosyntax``.

The pure-function tests in ``tests/inference/test_control_token_filter.py``
prove the detector and the stripper work on fabricated sentence
dicts. The Stanza-only reproducer in ``test_stanza_fi_mwt_sos_leak.py``
proves the leak happens upstream.

What this file adds: a live path through our own inference boundary.
``batch_infer_morphosyntax`` must detect the known Stanza leak, strip
it in place, emit a ``tracing.warning`` so the workaround is visible
in ops monitoring, and still produce a clean UD response. The
workaround follows the project's documented Stanza-defect pattern
(see ``book/src/reference/stanza-limitations.md`` Defect 3).

Any future contributor who removes the strip call or silences the
log will see this test fail.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import pytest

from batchalign.inference._control_token_filter import CONTROL_TOKEN_RE
from batchalign.inference._tokenizer_realign import TokenizerContext
from batchalign.inference.morphosyntax import (
    MorphosyntaxBatchItem,
    batch_infer_morphosyntax,
)
from batchalign.providers import BatchInferRequest

# Stanza is imported lazily inside the fixture so pytest collection
# (including runs that never touch this file's tests) doesn't pay the
# ~1-2s cascade of stanza+torch+numpy+transformers.
if TYPE_CHECKING:
    import stanza
from batchalign.worker._types import InferTask


# Three-word input that triggers the Stanza 1.11.1 Finnish MWT leak.
# Kept identical to the standalone reproducer so a reader diffing the
# two files sees the shared trigger immediately.
_FINNISH_WORDS = ["a", "tollei", "b"]
_FINNISH_LANG = "fin"


@pytest.fixture(scope="module")
def finnish_pipeline_with_ctx() -> tuple[stanza.Pipeline, TokenizerContext]:
    """Finnish Stanza pipeline matching the production configuration
    for MWT-capable non-English languages in ``_stanza_loading.py:133``.
    """
    import stanza
    from stanza import DownloadMethod

    from batchalign.inference._tokenizer_realign import make_tokenizer_postprocessor

    ctx = TokenizerContext()
    pp = make_tokenizer_postprocessor(ctx, alpha2="fi")
    nlp = stanza.Pipeline(
        lang="fi",
        processors="tokenize,pos,lemma,depparse,mwt",
        download_method=DownloadMethod.REUSE_RESOURCES,
        tokenize_no_ssplit=True,
        tokenize_postprocessor=pp,
        verbose=False,
    )
    return nlp, ctx


@pytest.mark.golden
def test_batch_infer_strips_stanza_control_token_leak_and_logs_warning(
    finnish_pipeline_with_ctx: tuple[stanza.Pipeline, TokenizerContext],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The leak must be stripped in place and logged, not silent.

    Why strip rather than raise: Stanza's ``<SOS>`` leak is a known
    upstream defect we catalogue in
    ``book/src/reference/stanza-limitations.md`` under the slug
    ``stanza-fi-mwt-sos-leak``. Failing the whole language group on
    a rare library bug is inconsistent with how we handle other
    Stanza defects (silent rewrite + tracing log + registry entry).
    ``chatter validate`` is the downstream gate if a leak variant
    escapes the stripper.

    Why log loudly: silent-strip hides the workaround. An ops reader
    seeing a warning can confirm Stanza still emits the leak on the
    next upgrade; a silent strip would make the defect un-observable
    until some future Stanza version stops triggering it and we
    wonder why the warning vanished.
    """
    nlp, ctx = finnish_pipeline_with_ctx
    pipelines = {_FINNISH_LANG: nlp}
    contexts = {_FINNISH_LANG: ctx}

    item = MorphosyntaxBatchItem(
        words=list(_FINNISH_WORDS),
        lang=_FINNISH_LANG,
    )
    req = BatchInferRequest(
        task=InferTask.MORPHOSYNTAX,
        lang=_FINNISH_LANG,
        items=[item.model_dump()],
        retokenize=False,
    )

    with caplog.at_level(logging.WARNING, logger="batchalign.worker"):
        response = batch_infer_morphosyntax(
            req=req,
            nlp_pipelines=pipelines,
            contexts=contexts,
            nlp_lock=threading.Lock(),
            free_threaded=False,
        )

    # (1) The response is successful — no per-item error, no exception.
    assert len(response.results) == 1
    item_result = response.results[0]
    assert item_result.error is None, (
        f"batch_infer_morphosyntax must not fail the item on a known "
        f"Stanza defect; got error: {item_result.error!r}"
    )

    # (2) The returned UD is clean — the <SOS> prefix is stripped
    # from both text and lemma fields.
    result = item_result.result
    assert isinstance(result, dict)
    raw_sentences = result.get("raw_sentences")
    assert isinstance(raw_sentences, list) and len(raw_sentences) == 1
    sent = raw_sentences[0]
    assert isinstance(sent, list)

    # Reuse the production regex so the test's "what counts as a leak"
    # vocabulary stays in sync with the stripper's definition. A
    # hand-rolled list here would silently drift if CONTROL_TOKEN_RE
    # ever picks up a new token (e.g., adding ``<RESERVED>`` upstream).
    for tok in sent:
        for field in ("text", "lemma"):
            value = tok.get(field)
            if isinstance(value, str):
                assert CONTROL_TOKEN_RE.search(value) is None, (
                    f"Stanza leak escaped stripper in {field} field: {value!r}"
                )

    # (3) A WARNING was logged naming the stripped leak. Operators
    # monitoring the server log must be able to see that the workaround
    # fired on this batch — that's what lets us re-evaluate the defect
    # on the next Stanza upgrade.
    matching_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "Stripped" in r.getMessage()
        and "control-token" in r.getMessage()
    ]
    assert matching_records, (
        f"expected a tracing.warning naming the stripped Stanza control-token "
        f"leak; captured warnings: {[r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]}"
    )
    msg = matching_records[0].getMessage()
    assert _FINNISH_LANG in msg, f"warning should name the language; got: {msg!r}"
    assert "<SOS>" in msg or "<sos>" in msg.lower(), (
        f"warning should name the original leaked value; got: {msg!r}"
    )
