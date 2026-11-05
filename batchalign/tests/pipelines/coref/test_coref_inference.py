"""Tests for the thin Python coreference inference boundary."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from batchalign.inference.coref import (
    ChainRef,
    CorefBatchItem,
    CorefRawAnnotation,
    CorefRawResponse,
    PreparedCoref,
    batch_infer_coref,
    prepare_coref,
)
from batchalign.providers import BatchInferRequest


def _loaded_with(monkeypatch, pipeline_class: type) -> PreparedCoref:
    """Install a fake ``stanza`` and run the real coref preparation against
    it, so the handle a request takes is always one ``prepare_coref`` built."""
    monkeypatch.setitem(
        sys.modules,
        "stanza",
        SimpleNamespace(Pipeline=pipeline_class, __version__="1.99.0"),
    )
    return prepare_coref()


class TestCorefModels:
    """Verify the typed coref wire models remain stable."""

    def test_coref_batch_item_roundtrip(self) -> None:
        item = CorefBatchItem(sentences=[["the", "dog"], ["it", "ran"]])
        assert item.model_dump() == {"sentences": [["the", "dog"], ["it", "ran"]]}

    def test_coref_raw_response_roundtrip(self) -> None:
        response = CorefRawResponse(
            annotations=[
                CorefRawAnnotation(
                    sentence_idx=0,
                    words=[[ChainRef(chain_id=1, is_start=True, is_end=False)], []],
                )
            ]
        )
        data = response.model_dump()
        back = CorefRawResponse.model_validate(data)
        assert back.annotations[0].sentence_idx == 0
        assert back.annotations[0].words[0][0].chain_id == 1


def test_residency_is_scoped_to_the_batch_and_sparse_annotations_come_back(
    monkeypatch,
) -> None:
    """Preparing builds the pipeline once to fetch its files, then releases
    it; each batch builds its own from those files and releases it at the
    end, so no handle keeps the model resident. Only chains are emitted, as
    sparse annotations.
    """

    pipeline_inits: list[dict[str, object]] = []
    seen_texts: list[str] = []

    class _FakeWord:
        def __init__(self, text: str, coref_chains: list[object] | None = None) -> None:
            self.text = text
            self.coref_chains = coref_chains or []

    class _FakeSentence:
        def __init__(self, words: list[_FakeWord]) -> None:
            self.words = words

    class _FakePipeline:
        def __init__(self, **kwargs) -> None:
            pipeline_inits.append(kwargs)

        def __call__(self, text: str):
            seen_texts.append(text)
            if text == "the dog\n\nit ran":
                return SimpleNamespace(
                    sentences=[
                        _FakeSentence(
                            [
                                _FakeWord(
                                    "the",
                                    [
                                        SimpleNamespace(
                                            chain=SimpleNamespace(index=7),
                                            is_start=True,
                                            is_end=True,
                                        )
                                    ],
                                ),
                                _FakeWord("dog"),
                            ]
                        ),
                        _FakeSentence([_FakeWord("it"), _FakeWord("ran")]),
                    ]
                )
            return SimpleNamespace(sentences=[_FakeSentence([_FakeWord("hello")])])

    released: list[int] = []
    _FakePipeline.__del__ = lambda self: released.append(1)  # type: ignore[attr-defined]

    coref = _loaded_with(monkeypatch, _FakePipeline)
    # Built by the preparation, before any request, and already released.
    assert len(pipeline_inits) == 1
    assert len(released) == 1

    response = batch_infer_coref(
        BatchInferRequest(
            task="coref",
            lang="eng",
            items=[
                {"sentences": [["the", "dog"], ["it", "ran"]]},
                {"sentences": [["hello"]]},
            ],
        ),
        coref,
    )
    second = batch_infer_coref(
        BatchInferRequest(task="coref", lang="eng", items=[{"sentences": [["hello"]]}]),
        coref,
    )

    # One build per batch, each released when its batch ended.
    assert len(pipeline_inits) == 3
    assert len(released) == 3
    assert second.results[0].error is None
    assert pipeline_inits[0] == {
        "lang": "en",
        "processors": "tokenize,coref",
        "package": {"coref": "ontonotes-singletons_roberta-large-lora"},
        "tokenize_pretokenized": True,
    }
    assert seen_texts == ["the dog\n\nit ran", "hello", "hello"]
    assert response.results[0].error is None
    assert response.results[0].result == {
        "kind": "resolved",
        "engine": "stanza-1.99.0/ontonotes-singletons_roberta-large-lora",
        "annotations": [
            {
                "sentence_idx": 0,
                "words": [[{"chain_id": 7, "is_start": True, "is_end": True}], []],
            }
        ],
    }
    assert response.results[1].result == {
        "kind": "resolved",
        "annotations": [],
        "engine": "stanza-1.99.0/ontonotes-singletons_roberta-large-lora",
    }


def test_batch_infer_coref_refuses_extra_sentences_from_runtime(monkeypatch) -> None:
    """Extra native sentences are a producer fault, not successful truncation."""

    class _FakeWord:
        def __init__(self, coref_chains: list[object] | None = None) -> None:
            self.coref_chains = coref_chains or []

    class _FakeSentence:
        def __init__(self, words: list[_FakeWord]) -> None:
            self.words = words

    class _FakePipeline:
        def __init__(self, **_kwargs) -> None:
            pass

        def __call__(self, _text: str):
            return SimpleNamespace(
                sentences=[
                    _FakeSentence(
                        [
                            _FakeWord(
                                [
                                    SimpleNamespace(
                                        chain=SimpleNamespace(index=2),
                                        is_start=True,
                                        is_end=True,
                                    )
                                ]
                            )
                        ]
                    ),
                    _FakeSentence(
                        [
                            _FakeWord(
                                [
                                    SimpleNamespace(
                                        chain=SimpleNamespace(index=9),
                                        is_start=True,
                                        is_end=True,
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            )

    coref = _loaded_with(monkeypatch, _FakePipeline)

    response = batch_infer_coref(
        BatchInferRequest(
            task="coref",
            lang="eng",
            items=[{"sentences": [["she"]]}],
        ),
        coref,
    )

    assert response.results[0].result is None
    assert "sentence coverage mismatch" in response.results[0].error


def test_batch_infer_coref_reports_invalid_items_and_empty_documents(
    monkeypatch,
) -> None:
    """Invalid items should fail explicitly, while empty documents stay no-op,
    and a batch with nothing to resolve builds no pipeline."""

    builds: list[int] = []

    class _UnusedPipeline:
        def __init__(self, **_kwargs) -> None:
            builds.append(1)

        def __call__(self, _text: str):
            raise AssertionError("invalid or empty items must never reach the model")

    coref = _loaded_with(monkeypatch, _UnusedPipeline)
    builds.clear()

    response = batch_infer_coref(
        BatchInferRequest(
            task="coref",
            lang="eng",
            items=[{"bad": "shape"}, {"sentences": []}],
        ),
        coref,
    )

    assert response.results[0].error == "Invalid CorefBatchItem"
    # Never sent to the engine, so it names none.
    assert response.results[1].result == {"kind": "no_sentences"}
    assert builds == []


def test_load_refuses_a_model_whose_engine_cannot_be_named(monkeypatch) -> None:
    """With no Stanza version there is no identity to report, so nothing loads:
    refused at the load, before anything downloads."""

    class _UnusedPipeline:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("no pipeline should be built without an identity")

    monkeypatch.setitem(
        sys.modules, "stanza", SimpleNamespace(Pipeline=_UnusedPipeline)
    )

    with pytest.raises(RuntimeError, match="cannot name the Stanza version"):
        prepare_coref()


def test_batch_infer_coref_fails_the_document_on_runtime_failure(
    monkeypatch,
) -> None:
    """A Stanza failure fails that document instead of emptying its chains.

    POLICY CHANGE: this used to assert an empty annotation list, which is the
    same bytes as a document with no chains, so the file was written without
    its ``%xcoref`` tiers and reported as a success.
    """

    class _ExplodingPipeline:
        def __init__(self, **_kwargs) -> None:
            pass

        def __call__(self, _text: str):
            raise RuntimeError("coref runtime exploded")

    coref = _loaded_with(monkeypatch, _ExplodingPipeline)

    response = batch_infer_coref(
        BatchInferRequest(
            task="coref",
            lang="eng",
            items=[{"sentences": [["she"], ["left"]]}],
        ),
        coref,
    )

    assert response.results[0].result is None
    assert response.results[0].error == "Coref failed: coref runtime exploded"
    assert response.results[0].elapsed_s >= 0.0


@pytest.mark.parametrize(
    "native",
    [
        [],
        [["she"]],
        [["she"], []],
        [["she"], ["left", "now"]],
        [["he"], ["left"]],
    ],
)
def test_unannotated_native_shape_faults_cannot_be_resolved(
    monkeypatch, native
) -> None:
    """Coverage is checked even when every native word has no chain."""

    class Pipeline:
        def __init__(self, **_kwargs) -> None:
            pass

        def __call__(self, _text: str):
            return SimpleNamespace(
                sentences=[
                    SimpleNamespace(
                        words=[
                            SimpleNamespace(text=word, coref_chains=[])
                            for word in sentence
                        ]
                    )
                    for sentence in native
                ]
            )

    coref = _loaded_with(monkeypatch, Pipeline)
    response = batch_infer_coref(
        BatchInferRequest(
            task="coref",
            lang="eng",
            items=[{"sentences": [["she"], ["left"]]}],
        ),
        coref,
    )
    assert response.results[0].result is None
    assert response.results[0].error is not None
    assert "mismatch" in response.results[0].error


def test_incomplete_native_item_does_not_poison_next_complete_document(
    monkeypatch,
) -> None:
    """A reusable model can fail one item and still admit the next."""

    class Pipeline:
        def __init__(self, **_kwargs) -> None:
            pass

        def __call__(self, text: str):
            words = (
                []
                if text == "broken"
                else [SimpleNamespace(text="complete", coref_chains=[])]
            )
            return SimpleNamespace(sentences=[SimpleNamespace(words=words)])

    coref = _loaded_with(monkeypatch, Pipeline)
    response = batch_infer_coref(
        BatchInferRequest(
            task="coref",
            lang="eng",
            items=[{"sentences": [["broken"]]}, {"sentences": [["complete"]]}],
        ),
        coref,
    )
    assert response.results[0].result is None
    assert response.results[0].error is not None
    assert response.results[1].error is None
    assert response.results[1].result["kind"] == "resolved"
    assert response.results[1].result["annotations"] == []
