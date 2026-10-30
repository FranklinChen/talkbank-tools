"""Native UD/model-boundary regressions; no CHAT or model download fixtures."""

from copy import deepcopy

import pytest

from batchalign.inference._domain_types import StanzaCode
from batchalign.inference._morphosyntax_graph import AnalyzableSentence, DependencyGraph
from batchalign.inference._tokenizer_realign import TokenizerContext
from batchalign.inference.morphosyntax import RepairedSentence, batch_infer_morphosyntax
from batchalign.providers import BatchInferRequest
from batchalign.worker._pipeline_cache import static_pipelines


class _Document:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self):
        return [deepcopy(self.rows)]


def _word(index, text, head, relation="dep"):
    return {
        "id": index,
        "text": text,
        "lemma": text,
        "upos": "NOUN",
        "head": head,
        "deprel": relation,
    }


def _run(rows, words):
    """Exercise the actual batch-infer producer, replacing only external NLP."""
    context = TokenizerContext(StanzaCode("fr"))

    def nlp(text):
        assert text == " ".join([*words, "."])
        assert context.original_words == [[*words, "."]]
        return _Document(rows)

    request = BatchInferRequest(
        task="morphosyntax",
        lang="fra",
        items=[
            {
                "words": words,
                "terminator": ".",
                "special_forms": [],
                "runs": [len(words)] if words else [],
                "lang": "fra",
            }
        ],
    )
    response = batch_infer_morphosyntax(
        request,
        static_pipelines({"fra": nlp}, {"fra": context}),
    )
    assert context.original_words == []
    return response.results[0]


@pytest.mark.parametrize(
    "retry_kind", ["valid", "missing_head", "wrong_count", "raises"]
)
def test_ambiguous_cue_uses_one_model_owned_lexical_retry(retry_kind):
    context = TokenizerContext(StanzaCode("fr"))
    calls = []
    ambiguous = [_word(1, "alpha", 3), _word(2, "beta", 3), _word(3, ".", 0, "root")]

    def nlp(text):
        calls.append((text, deepcopy(context.original_words)))
        if len(calls) == 1:
            return _Document(ambiguous)
        assert len(calls) == 2, "model retry must be bounded"
        if retry_kind == "raises":
            raise RuntimeError("retry refused")
        if retry_kind == "wrong_count":
            return _Document([_word(1, "alpha", 0, "root")])
        return _Document(
            [
                _word(1, "alpha", 2 if retry_kind == "valid" else 7, "nsubj"),
                _word(2, "beta", 0, "root"),
            ]
        )

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="fra",
            items=[
                {
                    "words": ["alpha", "beta"],
                    "terminator": ".",
                    "special_forms": [],
                    "runs": [2],
                    "lang": "fra",
                }
            ],
        ),
        static_pipelines({"fra": nlp}, {"fra": context}),
    )
    assert calls == [
        ("alpha beta .", [["alpha", "beta", "."]]),
        ("alpha beta", [["alpha", "beta"]]),
    ]
    assert context.original_words == []
    result = response.results[0]
    if retry_kind == "valid":
        assert result.error is None
        assert [row["head"] for row in result.result["raw_sentences"][0]] == [2, 0]
        assert [row["text"] for row in result.result["raw_sentences"][0]] == [
            "alpha",
            "beta",
        ]
    else:
        assert result.result is None
        assert "cue-free" in result.error


@pytest.mark.parametrize(
    "rows,expected_heads,expected_relations",
    [
        (
            [
                _word(1, "alpha", 2),
                _word(2, "beta", 0, "root"),
                _word(3, ".", 2, "punct"),
            ],
            [2, 0],
            ["dep", "root"],
        ),
        (
            [
                _word(1, "alpha", 3, "nsubj"),
                _word(2, "beta", 0, "root"),
                _word(3, ".", 2, "punct"),
            ],
            [2, 0],
            ["nsubj", "root"],
        ),
        (
            [
                _word(1, "alpha", 2, "obj"),
                _word(2, "beta", 3, "dep"),
                _word(3, ".", 0, "root"),
            ],
            [2, 0],
            ["obj", "root"],
        ),
    ],
)
def test_realigned_cue_projection_preserves_closed_dependency_graph(
    rows,
    expected_heads,
    expected_relations,
):
    original = deepcopy(rows)
    result = _run(rows, ["alpha", "beta"])
    assert result.error is None
    projected = result.result["raw_sentences"][0]
    assert [row["id"] for row in projected] == [1, 2]
    assert [row["head"] for row in projected] == expected_heads
    assert [row["deprel"] for row in projected] == expected_relations
    assert rows == original, "projection must not mutate the model's document"
    expected_repairs = (
        []
        if rows[-1]["head"] != 0
        else [
            {
                "kind": "cue_root_promotion",
                "word": "beta",
                "from_relation": "dep",
                "to_relation": "root",
            }
        ]
    )
    assert result.result["repairs"] == expected_repairs


@pytest.mark.parametrize(
    "rows,reason",
    [
        (
            [_word(1, "alpha", 3), _word(2, "beta", 3), _word(3, ".", 0, "root")],
            "ambiguous",
        ),
        (
            [_word(1, "alpha", 3), _word(2, "beta", 0, "root"), _word(3, ".", 1)],
            "cycle",
        ),
        ([_word(1, "alpha", 2), _word(2, "beta", 0, "root")], "boundaries"),
        (
            [_word(1, "alpha", 2), _word(1, "beta", 0, "root"), _word(3, ".", 1)],
            "duplicate",
        ),
        (
            [_word(1, "alpha", 7), _word(2, "beta", 0, "root"), _word(3, ".", 2)],
            "missing",
        ),
        (
            [_word(0, "alpha", 2), _word(2, "beta", 0, "root"), _word(3, ".", 2)],
            "invalid word id",
        ),
        (
            [{"id": (1, 2), "text": "compound"}, _word(1, "part", 0), _word(3, ".", 1)],
            "multiword",
        ),
        ([{"id": (1, 999999999), "text": "compound"}], "multiword"),
        ([], "boundaries"),
    ],
)
def test_invalid_or_ambiguous_model_projection_is_an_item_failure(rows, reason):
    result = _run(rows, ["alpha", "beta"])
    assert result.result is None, "malformed graph must never be tagged analyzed"
    assert reason in result.error.lower()


def test_realigned_multiword_token_projection_preserves_components():
    rows = [
        {"id": (1, 2), "text": "compound"},
        _word(1, "part", 2),
        _word(2, "other", 3),
        _word(3, "beta", 4),
        _word(4, ".", 0, "root"),
    ]
    result = _run(rows, ["compound", "beta"])
    assert result.error is None
    projected = result.result["raw_sentences"][0]
    assert [row["id"] for row in projected] == [[1, 2], 1, 2, 3]
    assert [row["head"] for row in projected] == [0, 2, 3, 0]


def test_owned_sentence_does_not_leak_mutable_wire_rows():
    graph = DependencyGraph(
        RepairedSentence(
            [
                _word(1, "alpha", 2),
                _word(2, ".", 0, "root"),
            ]
        )
    )
    admitted = graph.project_cue(2)
    emitted = admitted.wire_rows()
    emitted[0]["head"] = 99
    assert admitted.wire_rows()[0]["head"] == 0
    with pytest.raises(TypeError):
        AnalyzableSentence()


@pytest.mark.parametrize("bad_shape", [False, True])
def test_projection_failure_does_not_poison_other_items_in_language_group(bad_shape):
    ctx = TokenizerContext(StanzaCode("fr"))
    good = [_word(1, "alpha", 0, "root"), _word(2, ".", 1, "punct")]
    bad = [_word(1, "beta", 7), _word(2, ".", 1, "punct")]
    if bad_shape:
        del bad[0]["text"]

    class BatchDocument:
        def to_dict(self):
            return deepcopy([good, bad])

    result = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="fra",
            items=[
                {
                    "words": [word],
                    "terminator": ".",
                    "special_forms": [],
                    "runs": [1],
                    "lang": "fra",
                }
                for word in ["alpha", "beta"]
            ],
        ),
        static_pipelines({"fra": lambda text: BatchDocument()}, {"fra": ctx}),
    )
    assert result.results[0].result["kind"] == "analyzed"
    assert result.results[1].result is None
    assert ("text" if bad_shape else "missing") in result.results[1].error
    assert all(item.elapsed_s is not None for item in result.results)


def test_free_tokenization_does_not_claim_positional_cue_ownership():
    rows = [_word(1, "alpha", 0, "root"), _word(2, ".", 1, "punct")]
    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="fra",
            retokenize=True,
            items=[
                {
                    "words": ["alpha"],
                    "terminator": ".",
                    "special_forms": [],
                    "runs": [1],
                    "lang": "fra",
                },
            ],
        ),
        static_pipelines({"fra": lambda text: _Document(rows)}),
    )
    assert [row["id"] for row in response.results[0].result["raw_sentences"][0]] == [
        1,
        2,
    ]


def test_pos_transition_cannot_change_admitted_heads_or_word_order():
    from batchalign.inference._morphosyntax_graph import DependencyProjectionError

    admitted = DependencyGraph(
        RepairedSentence(
            [
                _word(1, "alpha", 2),
                _word(2, ".", 0, "root"),
            ]
        )
    ).project_cue(2)

    def malicious_head(rows):
        rows[0]["upos"] = "VERB"
        rows[0]["head"] = 99
        return rows

    tagged = admitted.with_pos(malicious_head).wire_rows()
    assert tagged[0]["upos"] == "VERB"
    assert tagged[0]["head"] == 0
    with pytest.raises(DependencyProjectionError):
        admitted.with_pos(lambda rows: [{**rows[0], "id": 9}])
    with pytest.raises(DependencyProjectionError):
        admitted.with_pos(lambda rows: [])


def test_cantonese_pos_sees_only_projected_lexical_rows(monkeypatch):
    seen = []

    def tag(rows):
        seen.extend(deepcopy(rows))
        return [{**row, "upos": "VERB"} for row in rows]

    monkeypatch.setattr(
        "batchalign.inference.morphosyntax._override_pos_with_pycantonese", tag
    )
    ctx = TokenizerContext(StanzaCode("zh-hant"))
    rows = [_word(1, "alpha", 2), _word(2, ".", 0, "root")]
    result = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="yue",
            items=[
                {
                    "words": ["alpha"],
                    "terminator": ".",
                    "special_forms": [],
                    "runs": [1],
                    "lang": "yue",
                },
            ],
        ),
        static_pipelines({"yue": lambda text: _Document(rows)}, {"yue": ctx}),
    ).results[0]
    assert result.error is None
    assert [(row["text"], row["head"]) for row in seen] == [("alpha", 0)]
    assert result.result["raw_sentences"][0][0]["upos"] == "VERB"


def test_cue_repair_is_excluded_by_position_not_by_surface_text():
    result = _run(
        [
            _word(1, ".", 2, "iob"),
            _word(2, "beta", 0, "root"),
            _word(3, ".", 2, "iob"),
        ],
        [".", "beta"],
    )
    assert result.error is None
    assert result.result["repairs"] == [
        {
            "kind": "relation_alias",
            "word": ".",
            "from_relation": "iob",
            "to_relation": "iobj",
        }
    ], "only the retained row's repair belongs to the output"


def test_empty_nodes_cannot_interrupt_multiword_components():
    from batchalign.inference._morphosyntax_graph import DependencyProjectionError

    with pytest.raises(DependencyProjectionError, match="multiword"):
        DependencyGraph(
            RepairedSentence(
                [
                    {"id": (1, 2), "text": "compound"},
                    _word(1, "part", 2),
                    _word(1.1, "empty", 0),
                    _word(2, "other", 0, "root"),
                ]
            )
        )
