"""Loading policy of the morphosyntax batch-infer handler.

The handler used to PRELOAD every language mentioned anywhere in a request
before inference began. Against a bounded cache that thrashes: with a capacity
of two and a three-language request, the third preload evicted the first, and
the first language group then reloaded it immediately. These tests pin that
nothing is loaded before it is used.
"""

from __future__ import annotations

from typing import Any

import pytest

from batchalign.worker._pipeline_cache import StanzaPipelineCache
from batchalign.worker._types import BatchInferRequest, InferTask, _state


class _FakePipeline:
    """Stand-in for a loaded pipeline; the handler never calls into one."""

    def __call__(self, text: str) -> Any:  # pragma: no cover - never invoked
        raise AssertionError("this test never runs inference")


@pytest.fixture
def loaded_languages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every language the handler asks the loader for."""
    import batchalign.inference.morphosyntax as morphosyntax
    import batchalign.worker._stanza_loading as stanza_loading

    requested: list[str] = []

    def _fake_load(lang: str) -> None:
        requested.append(lang)

    monkeypatch.setattr(stanza_loading, "load_stanza_models", _fake_load)
    # The handler is a thin shell over this; it is replaced so the test never
    # touches Stanza and so the arguments it is handed can be inspected.
    monkeypatch.setattr(
        morphosyntax,
        "batch_infer_morphosyntax",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(_state, "stanza_pipelines", StanzaPipelineCache())
    return requested


def _three_language_request() -> BatchInferRequest:
    return BatchInferRequest(
        task=InferTask.MORPHOSYNTAX,
        lang="eng",
        items=[
            {"words": ["hello"], "lang": "fra"},
            {"words": ["bonjour"], "lang": "spa"},
        ],
    )


def test_the_handler_preloads_nothing(loaded_languages: list[str]) -> None:
    """RED FIRST (review item 7): a three-language request over a
    capacity-two cache must not load three pipelines before inference."""
    from batchalign.worker._infer_hosts import build_morphosyntax_batch_infer_handler

    handler = build_morphosyntax_batch_infer_handler()
    handler(_three_language_request())

    assert loaded_languages == [], (
        "languages must be loaded per group at the point of use, not upfront: "
        f"{loaded_languages}"
    )


def test_the_handler_hands_the_cache_over_as_one_lookup(
    loaded_languages: list[str],
) -> None:
    """The pipeline and its context reach inference as ONE lookup.

    They used to travel as two mappings (`nlp_pipelines` and `contexts`),
    which is what allowed a read of one to be separated from a read of the
    other by an eviction.
    """
    from batchalign.worker._infer_hosts import build_morphosyntax_batch_infer_handler

    handler = build_morphosyntax_batch_infer_handler()
    call = handler(_three_language_request())

    assert isinstance(call, dict)
    assert call["pipelines"] is _state.stanza_pipelines
    assert "contexts" not in call
    assert "nlp_pipelines" not in call
    # And the on-demand loader is still wired, since that is what makes
    # dropping the preload safe.
    assert call["load_pipeline"] is not None
