"""Tests for the thin Python utterance-segmentation inference boundary."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from batchalign.inference.utseg import (
    UtsegBatchItem,
    _assemble,
    _leaf_count,
    _parse_tree_indices,
    batch_infer_utseg,
    compute_assignments,
)
from batchalign.models.utterance.evidence import (
    BoundaryAction,
    BoundaryProbability,
    ClassifiedBoundaryEvidence,
    ModelShortCircuit,
    UtteranceBoundaryPrediction,
)
from batchalign.providers import BatchInferRequest


class _SteppingMonotonic:
    """Deterministic ``time.monotonic`` stub advancing one second per call.

    Every measured item calls it exactly twice, once to start and once to
    stop, so each item's own elapsed time is exactly one step regardless of
    how many items the batch holds. That is what makes the timing assertions
    below a proof of per-item attribution rather than a restatement of an
    arbitrary number: under the batch-total stamping this module used until
    2026-09-16, the first item reported the whole batch span and every other
    item reported zero.
    """

    def __init__(self, step: float = 1.0) -> None:
        self._next = 0.0
        self._step = step

    def __call__(self) -> float:
        current = self._next
        self._next += self._step
        return current


class _FakeTree:
    """Small tree double for constituency-helper tests."""

    def __init__(
        self,
        label: str | None = None,
        children: list[_FakeTree] | None = None,
    ) -> None:
        self.label = label
        self.children = children or []

    def is_leaf(self) -> bool:
        return not self.children


def _leaf(label: str = "W") -> _FakeTree:
    return _FakeTree(label=label)


def _install_fake_stanza(
    monkeypatch,
    *,
    pipeline_factory,
    multilingual_factory=None,
) -> None:
    """Install one tiny fake stanza module for utseg tests."""

    module = ModuleType("stanza")
    module.Pipeline = pipeline_factory
    module.MultilingualPipeline = multilingual_factory or pipeline_factory
    module.DownloadMethod = SimpleNamespace(REUSE_RESOURCES="reuse")
    monkeypatch.setitem(__import__("sys").modules, "stanza", module)


class TestUtsegModels:
    """Verify the typed utseg wire models remain stable."""

    def test_utseg_batch_item_roundtrip(self) -> None:
        """Wire-format roundtrip, which no type pins on its own.

        `text` is supplied because the Rust schema declares it required and
        always sends it. This asserted `text == ""` until 2026-08-14, which
        pinned a default that made "the item lost its text" and "the text is
        empty" the same value.

        The item carried a `lang` field too, which Rust never sent and nothing
        read; it was removed the same day along with the conformance check's
        extra-field allowance that existed for it.
        """
        item = UtsegBatchItem(words=["I", "eat", "cookies"], text="I eat cookies")
        assert item.model_dump() == {
            "words": ["I", "eat", "cookies"],
            "text": "I eat cookies",
        }
        assert UtsegBatchItem.model_validate(item.model_dump()) == item


class TestBatchInferUtseg:
    """Verify the thin Python utseg adapter behavior."""

    def test_short_circuits_invalid_and_single_word_items(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            _SteppingMonotonic(),
        )
        calls: list[list[str]] = []

        def build_stanza_config(
            langs: list[str],
        ) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
            calls.append(langs)
            return ["en"], {"en": {"processors": "tokenize,constituency"}}

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[{"words": ["hello"], "text": "hello"}, {"bad": "shape"}],
            ),
            build_stanza_config,
        )

        assert calls == []
        assert response.results[0].result == {"assignments": [0]}
        assert response.results[1].error == "Invalid batch item"
        # The short-circuited item is measured like any other item: it did the
        # work of deciding that one word is one utterance.
        assert response.results[0].elapsed_s == 1.0
        # The rejected item never parsed, so no work is attributable to it.
        # Its zero is the rejection variant speaking, not a measurement.
        assert response.results[1].elapsed_s == 0.0

    def test_builds_single_language_pipeline_and_serializes_trees(
        self, monkeypatch
    ) -> None:
        # The Stanza branch is opt-in via the request-level typed field
        # `allow_stanza_fallback`; the CLI surface is
        # `--utseg-fallback-stanza` on every utseg-invoking subcommand.
        init_kwargs: list[dict[str, Any]] = []
        seen_texts: list[str] = []

        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            _SteppingMonotonic(),
        )

        class _FakePipeline:
            def __init__(self, **kwargs) -> None:
                init_kwargs.append(kwargs)

            def __call__(self, text: str):
                seen_texts.append(text)
                return SimpleNamespace(
                    sentences=[
                        SimpleNamespace(constituency="(S (NP I eat) (VP cookies))"),
                        SimpleNamespace(constituency=None),
                    ]
                )

        _install_fake_stanza(monkeypatch, pipeline_factory=_FakePipeline)

        def build_stanza_config(
            langs: list[str],
        ) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
            assert langs == ["eng"]
            return ["en"], {"en": {"processors": "tokenize,constituency"}}

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="",
                items=[{"words": ["I", "eat", "cookies"], "text": "I eat cookies"}],
                allow_stanza_fallback=True,
            ),
            build_stanza_config,
        )

        assert init_kwargs == [
            {
                "lang": "en",
                "processors": "tokenize,constituency",
                "download_method": "reuse",
            }
        ]
        assert seen_texts == ["I eat cookies"]
        assert response.results[0].result == {"trees": ["(S (NP I eat) (VP cookies))"]}
        # This item's own span, not the batch's. Building the Stanza pipeline
        # happens outside any item's measurement and is charged to no item.
        assert response.results[0].elapsed_s == 1.0

    def test_refuses_when_no_bert_model_and_fallback_not_opted_in(
        self, monkeypatch
    ) -> None:
        """Default behavior: no BERT for language + no opt-in → typed raise.

        Mirrors `whisper_hub.py`'s `WhisperHubModelNotFoundError`
        pattern. Silent substitution is the foot-gun this raise exists
        to prevent.
        """
        from batchalign.inference.utseg import UtsegModelNotFoundError

        with pytest.raises(UtsegModelNotFoundError) as exc_info:
            batch_infer_utseg(
                BatchInferRequest(
                    task="utseg",
                    lang="spa",
                    items=[
                        {"words": ["hola", "como", "estas"], "text": "hola como estas"}
                    ],
                ),
                lambda langs: (_ for _ in ()).throw(
                    AssertionError(f"refusal must skip Stanza load: {langs}")
                ),
                utterance_boundary_model=None,
            )

        message = str(exc_info.value)
        assert "spa" in message
        # The CLI flag is the primary user-facing remediation; the
        # resolver-entry path is the developer fix.
        assert "--utseg-fallback-stanza" in message
        # Env var must NOT appear; it was removed in favor of the
        # typed protocol field + CLI flag.
        assert "BA3_UTSEG_FALLBACK_STANZA" not in message

    def test_emits_loud_fallback_notice_when_opted_in(self, monkeypatch) -> None:
        """Opt-in fallback path: request field set → Stanza loads, notice fires."""
        emit_calls: list[tuple[str, str | None]] = []

        def _capture_emit(requested_lang: str, pack: str | None) -> None:
            emit_calls.append((requested_lang, pack))

        monkeypatch.setattr(
            "batchalign.inference.utseg._emit_stanza_fallback_notice",
            _capture_emit,
        )

        class _FakePipeline:
            def __init__(self, **kwargs) -> None:
                pass

            def __call__(self, text: str):
                return SimpleNamespace(sentences=[])

        _install_fake_stanza(monkeypatch, pipeline_factory=_FakePipeline)

        def build_stanza_config(
            langs: list[str],
        ) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
            assert langs == ["spa"]
            return ["es"], {"es": {"processors": "tokenize,constituency"}}

        batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="spa",
                items=[{"words": ["hola", "como", "estas"], "text": "hola como estas"}],
                allow_stanza_fallback=True,
            ),
            build_stanza_config,
            utterance_boundary_model=None,
        )

        assert emit_calls == [("spa", "es")], (
            "Stanza fallback must announce the substitution; see BUG-032"
        )

    def test_fallback_notice_dedupes_per_language_pack_pair(self, monkeypatch) -> None:
        """A worker processing N batches in the same language must not
        emit N identical fallback warnings, one warn per (lang, pack)
        per process is enough; more is dashboard noise.
        """
        from batchalign.inference import utseg as utseg_module

        monkeypatch.setattr(utseg_module, "_FALLBACK_NOTICE_FIRED", set())

        emit_count: list[int] = [0]

        def _count_emit(stage: str, user_message: str, **_: object) -> None:
            emit_count[0] += 1

        monkeypatch.setattr(
            "batchalign.worker._progress.emit_download_event",
            _count_emit,
        )

        utseg_module._emit_stanza_fallback_notice("spa", "es")
        utseg_module._emit_stanza_fallback_notice("spa", "es")
        utseg_module._emit_stanza_fallback_notice("spa", "es")
        assert emit_count[0] == 1

        utseg_module._emit_stanza_fallback_notice("deu", "de")
        assert emit_count[0] == 2

    def test_builds_multilingual_pipeline_and_handles_runtime_failure(
        self, monkeypatch
    ) -> None:
        # Multilingual Stanza pipeline is also opt-in, no BERT was loaded here.
        init_kwargs: list[dict[str, Any]] = []
        seen_texts: list[str] = []

        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            _SteppingMonotonic(),
        )

        class _FakeMultilingualPipeline:
            def __init__(self, **kwargs) -> None:
                init_kwargs.append(kwargs)

            def __call__(self, text: str):
                seen_texts.append(text)
                if text == "boom now":
                    raise AttributeError("missing constituency")
                return SimpleNamespace(
                    sentences=[SimpleNamespace(constituency="(S good path)")]
                )

        _install_fake_stanza(
            monkeypatch,
            pipeline_factory=_FakeMultilingualPipeline,
            multilingual_factory=_FakeMultilingualPipeline,
        )

        def build_stanza_config(
            langs: list[str],
        ) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
            assert langs == ["eng"]
            return [
                "en",
                "es",
            ], {
                "en": {"processors": "tokenize,constituency"},
                "es": {"processors": "tokenize,constituency"},
            }

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[
                    {"words": ["good", "path"], "text": "good path"},
                    {"words": ["boom", "now"], "text": "boom now"},
                ],
                allow_stanza_fallback=True,
            ),
            build_stanza_config,
        )

        assert init_kwargs == [
            {
                "lang_configs": {
                    "en": {"processors": "tokenize,constituency"},
                    "es": {"processors": "tokenize,constituency"},
                },
                "lang_id_config": {"langid_lang_subset": ["en", "es"]},
                "download_method": "reuse",
            }
        ]
        assert seen_texts == ["good path", "boom now"]
        assert response.results[0].result == {"trees": ["(S good path)"]}
        assert response.results[1].result == {"trees": []}
        # Both items report their own equal spans. This is the assertion that
        # would have caught the old misattribution: it read 4.0 and 0.0.
        assert response.results[0].elapsed_s == 1.0
        assert response.results[1].elapsed_s == 1.0

    def test_returns_empty_trees_when_no_language_pipeline_is_available(
        self, monkeypatch
    ) -> None:
        # Empty-langs path is reachable only when the operator opted in
        # to the Stanza fallback; otherwise the dispatcher refuses earlier.
        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            _SteppingMonotonic(),
        )
        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="",
                items=[{"words": ["still", "works"], "text": "still works"}],
                allow_stanza_fallback=True,
            ),
            lambda langs: ([], {}),
        )

        assert response.results[0].result == {"trees": []}
        # Still this item's own measured span, even though the outcome is empty.
        assert response.results[0].elapsed_s == 1.0

    def test_uses_boundary_model_assignments_when_available(self) -> None:
        class _FakeBoundaryModel:
            def predict_boundary_evidence(
                self, words: list[str]
            ) -> UtteranceBoundaryPrediction:
                assert words == ["On", "television", "Have", "you"]
                low = BoundaryProbability.from_float(0.1)
                high = BoundaryProbability.from_float(0.9)
                return UtteranceBoundaryPrediction(
                    model_id="test-boundary-model",
                    model_revision="0123456789abcdef0123456789abcdef01234567",
                    word_evidence=(
                        ClassifiedBoundaryEvidence(
                            raw_action=BoundaryAction.ORDINARY,
                            applied_action=BoundaryAction.ORDINARY,
                            boundary_probability=low,
                        ),
                        ClassifiedBoundaryEvidence(
                            raw_action=BoundaryAction.PERIOD_BOUNDARY,
                            applied_action=BoundaryAction.PERIOD_BOUNDARY,
                            boundary_probability=high,
                        ),
                        ClassifiedBoundaryEvidence(
                            raw_action=BoundaryAction.ORDINARY,
                            applied_action=BoundaryAction.ORDINARY,
                            boundary_probability=low,
                        ),
                        ClassifiedBoundaryEvidence(
                            raw_action=BoundaryAction.ORDINARY,
                            applied_action=BoundaryAction.ORDINARY,
                            boundary_probability=low,
                        ),
                    ),
                )

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[
                    {
                        "words": ["On", "television", "Have", "you"],
                        "text": "On television Have you",
                    }
                ],
            ),
            lambda langs: (_ for _ in ()).throw(
                AssertionError(f"unexpected Stanza load: {langs}")
            ),
            utterance_boundary_model=_FakeBoundaryModel(),
        )

        assert response.results[0].result == {
            "assignments": [0, 0, 1, 1],
            "boundary_model_evidence": {
                "model_id": "test-boundary-model",
                "model_revision": "0123456789abcdef0123456789abcdef01234567",
                "normalization_revision": "lower-strip-ascii-punctuation-v1",
                "adjacency_policy_revision": "suppress-earlier-adjacent-nonordinary-v1",
                "word_evidence": [
                    {
                        "kind": "classified",
                        "raw_action": "ordinary",
                        "applied_action": "ordinary",
                        "boundary_probability_micros": 100_000,
                    },
                    {
                        "kind": "classified",
                        "raw_action": "period_boundary",
                        "applied_action": "period_boundary",
                        "boundary_probability_micros": 900_000,
                    },
                    {
                        "kind": "classified",
                        "raw_action": "ordinary",
                        "applied_action": "ordinary",
                        "boundary_probability_micros": 100_000,
                    },
                    {
                        "kind": "classified",
                        "raw_action": "ordinary",
                        "applied_action": "ordinary",
                        "boundary_probability_micros": 100_000,
                    },
                ],
            },
        }

    def test_boundary_model_exposes_single_word_short_circuit_evidence(self) -> None:
        class _FakeBoundaryModel:
            def predict_boundary_evidence(
                self, words: list[str]
            ) -> UtteranceBoundaryPrediction:
                assert words == ["hello"]
                return UtteranceBoundaryPrediction(
                    model_id="test-boundary-model",
                    model_revision="0123456789abcdef0123456789abcdef01234567",
                    word_evidence=(ModelShortCircuit(),),
                )

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[{"words": ["hello"], "text": "hello"}],
            ),
            lambda langs: (_ for _ in ()).throw(
                AssertionError(f"unexpected Stanza load: {langs}")
            ),
            utterance_boundary_model=_FakeBoundaryModel(),
        )

        assert response.results[0].result == {
            "assignments": [0],
            "boundary_model_evidence": {
                "model_id": "test-boundary-model",
                "model_revision": "0123456789abcdef0123456789abcdef01234567",
                "normalization_revision": "lower-strip-ascii-punctuation-v1",
                "adjacency_policy_revision": "suppress-earlier-adjacent-nonordinary-v1",
                "word_evidence": [{"kind": "model_short_circuit"}],
            },
        }

    def test_boundary_model_failure_is_not_silently_changed_to_keep(self) -> None:
        class _FailingBoundaryModel:
            def predict_boundary_evidence(
                self, words: list[str]
            ) -> UtteranceBoundaryPrediction:
                raise ValueError(f"deliberate model failure for {words!r}")

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[{"words": ["hello", "there"], "text": "hello there"}],
            ),
            lambda langs: (_ for _ in ()).throw(
                AssertionError(f"unexpected Stanza load: {langs}")
            ),
            utterance_boundary_model=_FailingBoundaryModel(),
        )

        assert response.results[0].result is None
        assert response.results[0].error is not None
        assert "deliberate model failure" in response.results[0].error

    def test_boundary_model_timing_is_attributed_to_each_item(
        self, monkeypatch
    ) -> None:
        """Every item carries its own span, and none carries the batch's.

        This is the provenance-bearing path, so a timing written beside an
        item is a claim about that item. Until 2026-09-16 the whole batch's
        elapsed time was written onto the first item and every other item
        reported zero, so the first item's cost was overstated by all the
        others and every other item's was simply wrong.
        """
        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            _SteppingMonotonic(),
        )

        class _FakeBoundaryModel:
            def predict_boundary_evidence(
                self, words: list[str]
            ) -> UtteranceBoundaryPrediction:
                low = BoundaryProbability.from_float(0.1)
                return UtteranceBoundaryPrediction(
                    model_id="test-boundary-model",
                    model_revision="0123456789abcdef0123456789abcdef01234567",
                    word_evidence=tuple(
                        ClassifiedBoundaryEvidence(
                            raw_action=BoundaryAction.ORDINARY,
                            applied_action=BoundaryAction.ORDINARY,
                            boundary_probability=low,
                        )
                        for _ in words
                    ),
                )

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[
                    {"words": ["one", "two"], "text": "one two"},
                    {"words": ["three", "four"], "text": "three four"},
                    {"words": ["five", "six"], "text": "five six"},
                ],
            ),
            lambda langs: (_ for _ in ()).throw(
                AssertionError(f"unexpected Stanza load: {langs}")
            ),
            utterance_boundary_model=_FakeBoundaryModel(),
        )

        # Three equal spans. The old stamping produced [7.0, 0.0, 0.0] here.
        assert [result.elapsed_s for result in response.results] == [1.0, 1.0, 1.0]

    def test_a_failing_item_is_reported_at_its_own_position(self, monkeypatch) -> None:
        """The failure lands at the index that produced it, with its own time.

        The index travels with the work rather than being restated at the
        write, so a middle item's failure cannot be recorded against the
        first item's evidence.
        """
        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            _SteppingMonotonic(),
        )

        class _SecondItemFails:
            def predict_boundary_evidence(
                self, words: list[str]
            ) -> UtteranceBoundaryPrediction:
                if words == ["bad", "item"]:
                    raise ValueError("deliberate model failure")
                return UtteranceBoundaryPrediction(
                    model_id="test-boundary-model",
                    model_revision="0123456789abcdef0123456789abcdef01234567",
                    word_evidence=tuple(
                        ClassifiedBoundaryEvidence(
                            raw_action=BoundaryAction.ORDINARY,
                            applied_action=BoundaryAction.ORDINARY,
                            boundary_probability=BoundaryProbability.from_float(0.1),
                        )
                        for _ in words
                    ),
                )

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[
                    {"words": ["good", "one"], "text": "good one"},
                    {"words": ["bad", "item"], "text": "bad item"},
                    {"words": ["good", "two"], "text": "good two"},
                ],
            ),
            lambda langs: (_ for _ in ()).throw(
                AssertionError(f"unexpected Stanza load: {langs}")
            ),
            utterance_boundary_model=_SecondItemFails(),
        )

        assert response.results[0].error is None
        assert response.results[1].error is not None
        assert "deliberate model failure" in response.results[1].error
        assert response.results[2].error is None
        # The failing item is measured too: it spent real time failing.
        assert [result.elapsed_s for result in response.results] == [1.0, 1.0, 1.0]

    def test_assemble_refuses_a_batch_with_an_unreached_position(self) -> None:
        """An unreached position raises instead of reporting a plausible result.

        The previous shape pre-filled the result list with an empty-trees
        default and overwrote it by index, so a position that no branch
        reached was indistinguishable from a successful empty segmentation.
        """
        with pytest.raises(RuntimeError) as exc_info:
            _assemble(2, [], [])

        assert "positions [0, 1]" in str(exc_info.value)


class TestUtsegTreeHelpers:
    """Verify the local constituency helper behavior."""

    def test_leaf_count_handles_missing_children(self) -> None:
        # _leaf_count treats a missing-children object as a leaf with
        # weight 0. This is not the silent-failure pattern we banned
        # ``object()`` here is a sentinel-leaf in the test, and
        # downstream length math depends on the 0-count behavior.
        assert _leaf_count(object()) == 0

    def test_parse_tree_indices_raises_on_missing_children(self) -> None:
        # Pinned because the previous implementation swallowed
        # AttributeError and silently returned ``[]``, masking
        # malformed Stanza constituency output as an empty utseg
        # result. The system-wide graceful-failure invariant requires
        # this to raise so the caller can attribute the failure back
        # to the affected file rather than emit empty assignments.
        import pytest

        with pytest.raises(AttributeError):
            _parse_tree_indices(object(), 0)

    def test_leaf_count_recurses_into_nested_subtrees(self) -> None:
        tree = _FakeTree(
            label="ROOT",
            children=[
                _FakeTree(label="NP", children=[_leaf(), _leaf()]),
                _FakeTree(
                    label="VP",
                    children=[_FakeTree(label="V", children=[_leaf()])],
                ),
            ],
        )

        assert _leaf_count(tree) == 3

    def test_parse_tree_indices_extracts_s_ranges_under_coordination(self) -> None:
        tree = _FakeTree(
            label="ROOT",
            children=[
                _FakeTree(label="S", children=[_leaf(), _leaf(), _leaf()]),
                _FakeTree(label="CC", children=[_leaf("and")]),
                _FakeTree(label="S", children=[_leaf(), _leaf(), _leaf()]),
            ],
        )

        assert _leaf_count(tree.children[0]) == 3
        assert _parse_tree_indices(tree, 0) == [[0, 1, 2], [4, 5, 6]]

    def test_compute_assignments_splits_coordinated_ranges(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "batchalign.inference.utseg._parse_tree_indices",
            lambda _subtree, _offset: [[0, 1, 2], [4, 5, 6]],
        )

        def fake_nlp(_text: str):
            return SimpleNamespace(
                sentences=[SimpleNamespace(constituency=_FakeTree(label="ROOT"))]
            )

        assignments = compute_assignments(
            ["I", "eat", "cookies", "and", "he", "likes", "cake"],
            fake_nlp,
        )

        assert assignments == [0, 0, 0, 1, 1, 1, 1]

    def test_compute_assignments_backfills_trailing_unassigned_words(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "batchalign.inference.utseg._parse_tree_indices",
            lambda _subtree, _offset: [[0, 1, 2]],
        )

        def fake_nlp(_text: str):
            return SimpleNamespace(
                sentences=[SimpleNamespace(constituency=_FakeTree(label="ROOT"))]
            )

        assignments = compute_assignments(
            ["the", "dog", "ran", "fast"],
            fake_nlp,
        )

        assert assignments == [0, 0, 0, 0]

    def test_compute_assignments_merges_short_trailing_groups(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "batchalign.inference.utseg._parse_tree_indices",
            lambda _subtree, _offset: [[0, 1, 2], [3, 4]],
        )

        def fake_nlp(_text: str):
            return SimpleNamespace(
                sentences=[SimpleNamespace(constituency=_FakeTree(label="ROOT"))]
            )

        assignments = compute_assignments(
            ["I", "eat", "cookies", "right", "now"],
            fake_nlp,
        )

        assert assignments == [0, 0, 0, 0, 0]

    def test_compute_assignments_merges_all_short_groups_into_one_pending_group(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "batchalign.inference.utseg._parse_tree_indices",
            lambda _subtree, _offset: [[0, 1], [2, 3]],
        )

        def fake_nlp(_text: str):
            return SimpleNamespace(
                sentences=[SimpleNamespace(constituency=_FakeTree(label="ROOT"))]
            )

        assignments = compute_assignments(
            ["we", "all", "go", "home"],
            fake_nlp,
        )

        assert assignments == [0, 0, 0, 0]

    def test_compute_assignments_returns_zeroes_for_single_words_or_singleton_ranges(
        self, monkeypatch
    ) -> None:
        def fake_nlp(_text: str):
            return SimpleNamespace(
                sentences=[SimpleNamespace(constituency=_FakeTree(label="ROOT"))]
            )

        assert compute_assignments(["hello"], fake_nlp) == [0]

        monkeypatch.setattr(
            "batchalign.inference.utseg._parse_tree_indices",
            lambda _subtree, _offset: [[0]],
        )
        assert compute_assignments(["hello", "world"], fake_nlp) == [0, 0]

    def test_compute_assignments_returns_zeroes_when_phrase_mapping_stays_unassigned(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "batchalign.inference.utseg._parse_tree_indices",
            lambda _subtree, _offset: [[0], [10, 11]],
        )

        def fake_nlp(_text: str):
            return SimpleNamespace(
                sentences=[SimpleNamespace(constituency=_FakeTree(label="ROOT"))]
            )

        assert compute_assignments(["hello", "world"], fake_nlp) == [0, 0]
