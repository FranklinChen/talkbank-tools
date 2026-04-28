"""Tests for the thin Python utterance-segmentation inference boundary."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

from batchalign.inference.utseg import (
    UtsegBatchItem,
    _leaf_count,
    _parse_tree_indices,
    batch_infer_utseg,
    compute_assignments,
)
from batchalign.providers import BatchInferRequest


class _FakeTree:
    """Small tree double for constituency-helper tests."""

    def __init__(
        self,
        label: str | None = None,
        children: list["_FakeTree"] | None = None,
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
        item = UtsegBatchItem(words=["I", "eat", "cookies"], lang="eng")
        assert item.model_dump() == {
            "words": ["I", "eat", "cookies"],
            "text": "",
            "lang": "eng",
        }
        assert UtsegBatchItem.model_validate(item.model_dump()) == item


class TestBatchInferUtseg:
    """Verify the thin Python utseg adapter behavior."""

    def test_short_circuits_invalid_and_single_word_items(self) -> None:
        calls: list[list[str]] = []

        def build_stanza_config(langs: list[str]) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
            calls.append(langs)
            return ["en"], {"en": {"processors": "tokenize,constituency"}}

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[{"words": ["hello"]}, {"bad": "shape"}],
            ),
            build_stanza_config,
        )

        assert calls == []
        assert response.results[0].result == {"assignments": [0]}
        assert response.results[0].elapsed_s == 0.0
        assert response.results[1].error == "Invalid batch item"

    def test_builds_single_language_pipeline_and_serializes_trees(self, monkeypatch) -> None:
        init_kwargs: list[dict[str, Any]] = []
        seen_texts: list[str] = []
        monotonic = iter([100.0, 104.0])

        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            lambda: next(monotonic),
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

        def build_stanza_config(langs: list[str]) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
            assert langs == ["eng"]
            return ["en"], {"en": {"processors": "tokenize,constituency"}}

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="",
                items=[{"words": ["I", "eat", "cookies"]}],
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
        assert response.results[0].elapsed_s == 4.0

    def test_builds_multilingual_pipeline_and_handles_runtime_failure(self, monkeypatch) -> None:
        init_kwargs: list[dict[str, Any]] = []
        seen_texts: list[str] = []
        monotonic = iter([5.0, 9.0])

        monkeypatch.setattr(
            "batchalign.inference.utseg.time.monotonic",
            lambda: next(monotonic),
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

        def build_stanza_config(langs: list[str]) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
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
                items=[{"words": ["good", "path"]}, {"words": ["boom", "now"]}],
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
        assert response.results[0].elapsed_s == 4.0
        assert response.results[1].result == {"trees": []}

    def test_returns_empty_trees_when_no_language_pipeline_is_available(self) -> None:
        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="",
                items=[{"words": ["still", "works"]}],
            ),
            lambda langs: ([], {}),
        )

        assert response.results[0].result == {"trees": []}
        assert response.results[0].elapsed_s == 0.0

    def test_uses_boundary_model_assignments_when_available(self) -> None:
        class _FakeBoundaryModel:
            def predict_assignments(self, words: list[str]) -> list[int]:
                assert words == ["On", "television", "Have", "you"]
                return [0, 0, 1, 1]

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[{"words": ["On", "television", "Have", "you"]}],
            ),
            lambda langs: (_ for _ in ()).throw(AssertionError(f"unexpected Stanza load: {langs}")),
            utterance_boundary_model=_FakeBoundaryModel(),
        )

        assert response.results[0].result == {"assignments": [0, 0, 1, 1]}

    def test_boundary_model_short_circuits_single_word_items(self) -> None:
        class _FakeBoundaryModel:
            def predict_assignments(self, words: list[str]) -> list[int]:
                raise AssertionError(f"single-word item should not reach model: {words}")

        response = batch_infer_utseg(
            BatchInferRequest(
                task="utseg",
                lang="eng",
                items=[{"words": ["hello"]}],
            ),
            lambda langs: (_ for _ in ()).throw(AssertionError(f"unexpected Stanza load: {langs}")),
            utterance_boundary_model=_FakeBoundaryModel(),
        )

        assert response.results[0].result == {"assignments": [0]}


class TestUtsegTreeHelpers:
    """Verify the local constituency helper behavior."""

    def test_leaf_count_and_parse_tree_indices_handle_missing_children(self) -> None:
        assert _leaf_count(object()) == 0
        assert _parse_tree_indices(object(), 0) == []

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

    def test_compute_assignments_backfills_trailing_unassigned_words(self, monkeypatch) -> None:
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

    def test_compute_assignments_merges_short_trailing_groups(self, monkeypatch) -> None:
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

    def test_compute_assignments_merges_all_short_groups_into_one_pending_group(self, monkeypatch) -> None:
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

    def test_compute_assignments_returns_zeroes_for_single_words_or_singleton_ranges(self, monkeypatch) -> None:
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

    def test_compute_assignments_returns_zeroes_when_phrase_mapping_stays_unassigned(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "batchalign.inference.utseg._parse_tree_indices",
            lambda _subtree, _offset: [[0], [10, 11]],
        )

        def fake_nlp(_text: str):
            return SimpleNamespace(
                sentences=[SimpleNamespace(constituency=_FakeTree(label="ROOT"))]
            )

        assert compute_assignments(["hello", "world"], fake_nlp) == [0, 0]
