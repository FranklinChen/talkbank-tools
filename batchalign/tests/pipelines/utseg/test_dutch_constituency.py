"""Utseg's Stanza pipeline: loaded only with a constituency parser, and once.

Source: an operator's bug report, 2026-03-28 (Dutch utseg crashed on a
missing constituency model).
"""

from types import ModuleType, SimpleNamespace

import pytest

from batchalign.inference._domain_types import StanzaCode
from batchalign.worker._stanza_capabilities import (
    StanzaCapabilityTable,
    StanzaLanguageCapability,
)


def _table(monkeypatch, code: str, stanza_code: str, processors: set[str]) -> None:
    from batchalign.worker import _stanza_capabilities

    monkeypatch.setattr(
        _stanza_capabilities,
        "get_cached_capability_table",
        lambda: StanzaCapabilityTable(
            languages={
                code: StanzaLanguageCapability(
                    stanza_code=StanzaCode(stanza_code),
                    processors=frozenset(processors),
                )
            },
        ),
    )


def _fake_stanza(monkeypatch) -> list[dict[str, object]]:
    """Install a fake ``stanza`` whose pipelines record their construction."""
    built: list[dict[str, object]] = []

    class _Pipeline:
        def __init__(self, **kwargs: object) -> None:
            built.append(kwargs)

    module = ModuleType("stanza")
    module.Pipeline = _Pipeline
    module.DownloadMethod = SimpleNamespace(REUSE_RESOURCES="reuse")
    monkeypatch.setitem(__import__("sys").modules, "stanza", module)
    return built


@pytest.fixture
def fresh_cache(monkeypatch):
    """A fresh pipeline cache, and no download notice from the probe."""
    from batchalign.worker import _stanza_loading
    from batchalign.worker._pipeline_cache import StanzaPipelineCache
    from batchalign.worker._types import _state

    monkeypatch.setattr(_state, "stanza_pipelines", StanzaPipelineCache())
    monkeypatch.setattr(
        _stanza_loading,
        "_emit_stanza_lang_download_event_if_missing",
        lambda lang, alpha2: None,
    )


def test_a_language_without_a_constituency_parser_is_refused_not_loaded(
    monkeypatch, fresh_cache
):
    """RED FIRST: Dutch's catalog entry ships no constituency parser. The
    loader used to build a pipeline without one, and every multi-word item
    then failed with "produced no tree". It now refuses, naming the
    processors, before any pipeline is built."""
    from batchalign.worker._stanza_loading import (
        UnsupportedLanguageError,
        load_utseg_pipeline,
    )

    _table(monkeypatch, "nld", "nl", {"tokenize", "pos", "lemma", "depparse", "mwt"})
    built = _fake_stanza(monkeypatch)
    with pytest.raises(UnsupportedLanguageError) as refusal:
        load_utseg_pipeline("nld")
    assert "constituency" in str(refusal.value)
    assert built == []


def test_the_pipeline_loads_its_requirement_and_no_lemmatizer(monkeypatch, fresh_cache):
    """Utseg loads its own requirement (Rust's, through ``batchalign_core``):
    tokenize, pos and constituency, plus mwt where shipped, and no lemmatizer,
    so an entry with constituency but no lemma (Vietnamese) is served."""
    from batchalign.worker._stanza_loading import load_utseg_pipeline

    _table(monkeypatch, "vie", "vi", {"tokenize", "pos", "depparse", "constituency"})
    built = _fake_stanza(monkeypatch)
    load_utseg_pipeline("vie")
    assert built == [
        {
            "lang": "vi",
            "processors": "tokenize,pos,constituency",
            "tokenize_pretokenized": True,
            "download_method": "reuse",
        }
    ]


def test_the_pipeline_is_built_once_and_reused(monkeypatch, fresh_cache):
    """RED FIRST: the Stanza route built a new pipeline for every batch, a
    multi-hundred-megabyte load each time. It is now built once per worker
    and language, through the same cache and per-key construction slot
    morphotag uses, and every later batch gets the same guarded entry."""
    from batchalign.worker._stanza_loading import load_utseg_pipeline

    _table(monkeypatch, "eng", "en", {"tokenize", "pos", "lemma", "constituency"})
    built = _fake_stanza(monkeypatch)
    first = load_utseg_pipeline("eng")
    second = load_utseg_pipeline("eng")
    assert first is second
    assert len(built) == 1


def test_a_worker_never_prepared_for_a_language_still_serves_the_stanza_route(
    monkeypatch, fresh_cache
):
    """RED FIRST: the Stanza route ran only through a loader that
    ``prepare_utseg_stanza`` installed, and a worker started for ``auto`` or
    ``per-file`` skips that call, so every Stanza request it received failed
    with "No utseg Stanza loader installed". The handler now calls the
    loader itself, which builds the request language's pipeline on first use,
    whatever language the worker was started for."""
    from batchalign.inference.utseg import UtsegInferenceRequest
    from batchalign.worker._infer_hosts import build_utseg_inference_handler
    from batchalign.worker._types_v2 import UtsegSegmenterV2

    _table(monkeypatch, "spa", "es", {"tokenize", "pos", "constituency"})
    built = _fake_stanza(monkeypatch)
    response = build_utseg_inference_handler()(
        UtsegInferenceRequest(
            lang="spa",
            segmenter=UtsegSegmenterV2.STANZA_CONSTITUENCY,
            items=({"words": ["hola", "amigo"], "text": "hola amigo"},),
        )
    )
    assert [kwargs["lang"] for kwargs in built] == ["es"]
    # The fake pipeline is not callable, so the item fails at its own
    # position: the point is that a pipeline was loaded and called at all.
    assert response.results[0].error is not None
    assert "No utseg Stanza loader installed" not in response.results[0].error
