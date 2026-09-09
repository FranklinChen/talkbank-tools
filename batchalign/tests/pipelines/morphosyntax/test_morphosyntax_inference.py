"""Tests for the thin Python morphosyntax inference boundary."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from batchalign.inference.morphosyntax import (
    _is_bogus_lemma,
    batch_infer_morphosyntax,
    validate_ud_words,
)
from batchalign.providers import BatchInferRequest
from batchalign.worker._pipeline_cache import (
    LoadedPipeline,
    StaticPipelines,
    static_pipelines,
)


class _RecordingLock:
    """Minimal context-manager lock that counts acquisitions."""

    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self) -> None:
        self.enter_count += 1

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeDoc:
    """Tiny doc-like object exposing Stanza's ``to_dict()`` seam."""

    def __init__(self, rows: list[list[dict[str, Any]]]) -> None:
        self._rows = rows

    def to_dict(self) -> list[list[dict[str, Any]]]:
        return deepcopy(self._rows)


class _RecordingNlp:
    """Callable test double for one Stanza pipeline."""

    def __init__(
        self,
        ctx: SimpleNamespace | None,
        rows: list[list[dict[str, Any]]],
        *,
        error: Exception | None = None,
    ) -> None:
        self.ctx = ctx
        self.rows = rows
        self.error = error
        self.calls: list[tuple[str, list[list[str]]]] = []

    def __call__(self, text: str) -> _FakeDoc:
        original_words = (
            []
            if self.ctx is None
            else [list(words) for words in self.ctx.original_words]
        )
        self.calls.append((text, original_words))
        if self.error is not None:
            raise self.error
        return _FakeDoc(self.rows)


def _raw_sentence(words: list[str]) -> list[dict[str, Any]]:
    """Build one raw-Stanza-like sentence AS IT LEAVES THE PIPELINE.

    Includes the optional fields as explicit ``None``. Stanza itself omits
    them, but every sentence now passes through ``validate_ud_words``, whose
    ``UdWord.model_dump()`` materializes the full model. Rust's ``UdWord``
    declares ``xpos``/``feats``/``deps``/``misc`` as ``Option``, so explicit
    nulls deserialize identically to absent keys.

    Before 2026-07-28 that validation was never invoked on the production
    path, so responses carried Stanza's raw dict verbatim and this helper
    matched it. Wiring the validator in (which is what stops non-UD deprels
    like ``iob`` reaching ``%gra``) also normalizes the shape.
    """

    rows = []
    for i, word in enumerate(words, start=1):
        rows.append(
            {
                "id": i,
                "text": word,
                "lemma": word.lower(),
                "upos": "NOUN",
                "xpos": None,
                "feats": None,
                "head": 0 if i == 1 else 1,
                "deprel": "root" if i == 1 else "obj",
                "deps": None,
                "misc": None,
            }
        )
    return rows


def test_is_bogus_lemma_flags_punctuation_only_lemmas() -> None:
    """Bogus-lemma detection should ignore surface matches and real empty lemmas."""

    assert _is_bogus_lemma("hello", "...") is True
    assert _is_bogus_lemma("hello", "hello") is False
    assert _is_bogus_lemma("hello", "") is False
    assert _is_bogus_lemma("?!", "?!") is False


def test_validate_ud_words_falls_back_from_bogus_punctuation_lemma() -> None:
    """Punctuation-only lemmas for lexical words should fall back to the surface form."""

    sentences = [
        [
            {
                "id": 1,
                "text": "bonjour",
                "lemma": "...",
                "upos": "INTJ",
                "head": 0,
                "deprel": "root",
            }
        ]
    ]

    validate_ud_words(sentences)

    assert sentences[0][0]["lemma"] == "bonjour"
    assert sentences[0][0]["deprel"] == "root"


def test_validate_ud_words_coerces_tuple_ids_inside_sentence_rows() -> None:
    """Tuple IDs from Stanza should be normalized before row validation."""

    sentences = [[{"id": (2, 3), "text": "au"}]]

    validate_ud_words(sentences)

    assert sentences[0][0]["id"] == [2, 3]
    assert sentences[0][0]["lemma"] == ""


def test_batch_infer_morphosyntax_groups_by_language_and_uses_lock(monkeypatch) -> None:
    """Batch morphosyntax should group by language, set tokenizer context, and lock."""

    monotonic = iter([10.0, 13.5])
    monkeypatch.setattr(
        "batchalign.inference.morphosyntax.time.monotonic",
        lambda: next(monotonic),
    )

    lock = _RecordingLock()
    eng_ctx = SimpleNamespace(original_words=[])
    fra_ctx = SimpleNamespace(original_words=[])
    # Stanza is given the terminator, so it returns a word for it; these
    # fakes therefore return N+1 words, and the assertions below expect the
    # terminator to have been dropped again on the way out.
    eng_nlp = _RecordingNlp(
        eng_ctx,
        [
            _raw_sentence(["hello", "world", "."]),
            _raw_sentence(["goodbye", "moon", "."]),
        ],
    )
    fra_nlp = _RecordingNlp(fra_ctx, [_raw_sentence(["salut", "."])])

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="eng",
            items=[
                {
                    "words": ["hello", "world"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "eng",
                },
                {
                    "words": ["goodbye", "moon"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "eng",
                },
                {
                    "words": ["salut"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "fra",
                },
                {"words": [], "terminator": ".", "special_forms": [], "lang": "eng"},
                {"bad": "shape"},
            ],
        ),
        static_pipelines(
            {"eng": eng_nlp, "fra": fra_nlp}, {"eng": eng_ctx, "fra": fra_ctx}
        ),
        lock,
        free_threaded=False,
    )

    assert lock.enter_count == 2
    assert eng_nlp.calls == [
        (
            "hello world .\n\ngoodbye moon .",
            [["hello", "world", "."], ["goodbye", "moon", "."]],
        )
    ]
    assert fra_nlp.calls == [("salut .", [["salut", "."]])]
    assert eng_ctx.original_words == []
    assert fra_ctx.original_words == []
    assert response.results[0].result == {
        "raw_sentences": [_raw_sentence(["hello", "world"])]
    }
    assert response.results[0].elapsed_s == 3.5
    assert response.results[1].result == {
        "raw_sentences": [_raw_sentence(["goodbye", "moon"])]
    }
    assert response.results[2].result == {"raw_sentences": [_raw_sentence(["salut"])]}
    # Each of the three above asserts the CHAT words only: the terminator the
    # fakes returned is gone, which is the contract.
    assert response.results[3].result == {"sentences": []}
    assert response.results[4].error == "Invalid batch item"


def test_batch_infer_morphosyntax_uses_fallback_context_and_resets_after_failure(
    monkeypatch,
) -> None:
    """If a lang-specific context is absent, the request-lang context should be used and reset.

    POLICY CHANGE, recorded here because this test used to assert the loss:
    its final assertion was ``result == {"sentences": []}``, i.e. that a
    pipeline which RAISED still returned a well-formed empty analysis. That is
    the silent-empty-tier defect, written down as an expectation. The reset of
    the borrowed context is what this test is actually for, and it still holds;
    the outcome for the item is now an error.
    """

    monotonic = iter([1.0, 2.0])
    monkeypatch.setattr(
        "batchalign.inference.morphosyntax.time.monotonic",
        lambda: next(monotonic),
    )

    lock = _RecordingLock()
    fallback_ctx = SimpleNamespace(original_words=[])
    fra_nlp = _RecordingNlp(
        fallback_ctx,
        [],
        error=RuntimeError("stanza exploded"),
    )
    # The request-language pipeline is resident and carries the context that
    # `fra` borrows. A context with no pipeline behind it is no longer a state
    # the fixture can express, and it was never one the real cache could
    # reach: `LoadedPipeline` holds the two together.
    eng_nlp = _RecordingNlp(fallback_ctx, [])

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="eng",
            items=[
                {
                    "words": ["salut", "toi"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "fra",
                }
            ],
        ),
        static_pipelines({"fra": fra_nlp, "eng": eng_nlp}, {"eng": fallback_ctx}),
        lock,
        free_threaded=True,
    )

    assert lock.enter_count == 0
    assert fra_nlp.calls == [("salut toi .", [["salut", "toi", "."]])]
    assert fallback_ctx.original_words == []
    assert response.results[0].result is None
    assert (
        response.results[0].error
        == "Stanza pipeline raised for language fra: stanza exploded"
    )
    assert response.results[0].elapsed_s == 1.0


def test_batch_infer_morphosyntax_reports_a_failed_language_group_as_an_error(
    monkeypatch,
) -> None:
    """A language group whose pipeline RAISES must return an error per item.

    The fact that Stanza failed has to travel in the response, because the
    Rust consumer branches on it: `morphosyntax/worker.rs` turns a per-item
    `error` into a failed file, and its own comment says why, "so the
    affected file fails rather than silently producing empty %mor tiers".
    An empty result carries no such fact, so the file used to be WRITTEN with
    empty `%mor` and `%gra` and nothing anywhere said the analysis never ran.

    Items in a language whose pipeline SUCCEEDED are unaffected: the failure
    is scoped to its own group.
    """

    monotonic = iter([1.0, 2.0])
    monkeypatch.setattr(
        "batchalign.inference.morphosyntax.time.monotonic",
        lambda: next(monotonic),
    )

    eng_ctx = SimpleNamespace(original_words=[])
    fra_ctx = SimpleNamespace(original_words=[])
    # The fake returns the terminator too, as Stanza does; it is stripped on
    # the way out, which is why the expectation below is the CHAT words alone.
    eng_nlp = _RecordingNlp(eng_ctx, [_raw_sentence(["hello", "world", "."])])
    fra_nlp = _RecordingNlp(fra_ctx, [], error=RuntimeError("stanza exploded"))

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="eng",
            items=[
                {
                    "words": ["salut", "toi"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "fra",
                },
                {
                    "words": ["hello", "world"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "eng",
                },
                {
                    "words": ["bonjour"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "fra",
                },
            ],
        ),
        static_pipelines(
            {"eng": eng_nlp, "fra": fra_nlp}, {"eng": eng_ctx, "fra": fra_ctx}
        ),
        _RecordingLock(),
        free_threaded=True,
    )

    for index in (0, 2):
        failed = response.results[index]
        assert failed.error is not None, f"item {index} carries no error"
        assert "fra" in failed.error
        assert "stanza exploded" in failed.error
        # Never an empty analysis: an empty `%mor` is indistinguishable from a
        # real one for an utterance Stanza legitimately had nothing to say
        # about, which is exactly how this defect stayed invisible.
        assert failed.result is None

    survivor = response.results[1]
    assert survivor.error is None
    assert survivor.result == {"raw_sentences": [_raw_sentence(["hello", "world"])]}


def test_batch_infer_morphosyntax_reloads_a_language_that_is_not_resident(
    monkeypatch,
) -> None:
    """A missing pipeline asks the loader once, and a good reload is no failure.

    This is what makes a BOUNDED pipeline cache safe. The cache holds two
    pipelines, so a batch spanning three languages will find one evicted
    between the host loading it and this function reaching its group. Without
    this seam that group would be reported as unanalysable and the whole file
    would fail, which is a worse outcome than the memory ceiling buys.
    """

    monotonic = iter([40.0, 41.0])
    monkeypatch.setattr(
        "batchalign.inference.morphosyntax.time.monotonic",
        lambda: next(monotonic),
    )

    fra_ctx = SimpleNamespace(original_words=[])
    fra_nlp = _RecordingNlp(fra_ctx, [_raw_sentence(["salut", "."])])
    # Empty: stands for a cache that has evicted `fra` since the host loaded it.
    entries: dict[str, LoadedPipeline] = {}
    pipelines = StaticPipelines(entries=entries)
    loaded: list[str] = []

    def _load(lang: str) -> None:
        loaded.append(lang)
        # The loader installs the pipeline AND its context together, which is
        # the only way an entry ever comes into existence.
        entries[lang] = LoadedPipeline(nlp=fra_nlp, context=fra_ctx)

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="fra",
            items=[
                {
                    "words": ["salut"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "fra",
                }
            ],
        ),
        pipelines,
        _RecordingLock(),
        free_threaded=True,
        load_pipeline=_load,
    )

    assert loaded == ["fra"], "the loader must be asked exactly once"
    assert response.results[0].error is None
    assert response.results[0].result == {"raw_sentences": [_raw_sentence(["salut"])]}


def test_batch_infer_morphosyntax_reports_why_a_reload_failed(monkeypatch) -> None:
    """A loader that raises hands its reason to the failure, not only to the log.

    The bare absence of a pipeline and "Stanza has no model for this language"
    are different operator leads, and the second one is only knowable here.
    """

    monotonic = iter([50.0, 51.0])
    monkeypatch.setattr(
        "batchalign.inference.morphosyntax.time.monotonic",
        lambda: next(monotonic),
    )

    def _load(lang: str) -> None:
        raise RuntimeError(f"no Stanza model for {lang}")

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="zzz",
            items=[
                {
                    "words": ["kuku"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "zzz",
                }
            ],
        ),
        static_pipelines({}, {}),
        _RecordingLock(),
        free_threaded=True,
        load_pipeline=_load,
    )

    assert response.results[0].result is None
    assert response.results[0].error == (
        "no Stanza pipeline loaded for language zzz: loading it failed: "
        "no Stanza model for zzz"
    )


def test_batch_infer_morphosyntax_reports_missing_pipelines_and_mismatches(
    monkeypatch,
) -> None:
    """A missing pipeline and sentence-count drift both fail their group.

    POLICY CHANGE, and this test used to be named "leaves_defaults..." and to
    assert exactly the defect: three items whose analysis never happened came
    back as well-formed empty results, so the Rust side injected empty %mor
    and %gra and wrote the file. Both situations mean nothing is KNOWN about
    the utterances, which is not the same as knowing they have no morphology.

    They stay two distinct error kinds because the operator action differs:
    a missing pipeline is an installation or routing problem, a count mismatch
    is a tokenization drift to investigate.
    """

    monotonic = iter([20.0, 21.0])
    monkeypatch.setattr(
        "batchalign.inference.morphosyntax.time.monotonic",
        lambda: next(monotonic),
    )

    lock = _RecordingLock()
    eng_ctx = SimpleNamespace(original_words=[])
    mismatch_nlp = _RecordingNlp(
        eng_ctx,
        [_raw_sentence(["only", "one"])],
    )

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="eng",
            items=[
                {
                    "words": ["no", "pipeline"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "spa",
                },
                {
                    "words": ["hello", "world"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "eng",
                },
                {
                    "words": ["goodbye", "moon"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "eng",
                },
            ],
        ),
        static_pipelines({"eng": mismatch_nlp}, {"eng": eng_ctx}),
        lock,
        free_threaded=False,
    )

    assert mismatch_nlp.calls == [
        (
            "hello world .\n\ngoodbye moon .",
            [["hello", "world", "."], ["goodbye", "moon", "."]],
        )
    ]
    assert response.results[0].result is None
    assert (
        response.results[0].error
        == "no Stanza pipeline loaded for language spa: this worker has no "
        "loaded pipeline for the language"
    )
    for index in (1, 2):
        assert response.results[index].result is None
        assert (
            response.results[index].error
            == "Stanza sentence count mismatch for language eng: expected 2 "
            "sentences, got 1"
        )


def test_batch_infer_morphosyntax_returns_early_when_no_nonempty_items(
    monkeypatch,
) -> None:
    """All-invalid or empty items should hit the no-work early return."""

    monotonic = iter([30.0, 31.0])
    monkeypatch.setattr(
        "batchalign.inference.morphosyntax.time.monotonic",
        lambda: next(monotonic),
    )

    response = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="eng",
            items=[
                {"words": [], "terminator": ".", "special_forms": [], "lang": "eng"},
                {"bad": "shape"},
            ],
        ),
        static_pipelines({}, {}),
        _RecordingLock(),
        free_threaded=False,
    )

    assert response.results[0].result == {"sentences": []}
    assert response.results[0].elapsed_s == 0.0
    assert response.results[1].error == "Invalid batch item"


def test_batch_infer_morphosyntax_normalizes_deprels_on_the_production_path() -> None:
    """The PRODUCTION path must validate Stanza's output, not just the tests.

    `validate_ud_words` and its `<PAD>` sanitizer existed and were unit-tested
    for months, yet `PAD` and `IOB` both reached the published corpora. The
    reason: nothing on the live path ever called them. `batch_infer_morphosyntax`
    took `doc.to_dict()` straight into the response, so every validator in this
    module was dead code the moment real data flowed.

    This test drives the real entrypoint with a Stanza double that emits
    `iob` (which Stanza's Italian model genuinely produces for clitics, and
    which is not a UD relation) and asserts the response carries a valid UD
    relation. A unit test on `UdWord` cannot catch the wiring gap; only a test
    at this seam can.
    """
    lock = _RecordingLock()
    ctx = SimpleNamespace(original_words=[])
    nlp = _RecordingNlp(
        ctx,
        [
            [
                {
                    "id": 1,
                    "text": "attenzi",
                    "lemma": "attenzare",
                    "upos": "VERB",
                    "head": 0,
                    "deprel": "root",
                },
                {
                    "id": 2,
                    "text": "ne",
                    "lemma": "ne",
                    "upos": "PRON",
                    "head": 1,
                    "deprel": "iob",
                },
                {
                    "id": 3,
                    "text": ".",
                    "lemma": ".",
                    "upos": "PUNCT",
                    "head": 1,
                    "deprel": "punct",
                },
            ]
        ],
    )

    resp = batch_infer_morphosyntax(
        BatchInferRequest(
            task="morphosyntax",
            lang="ita",
            items=[
                {
                    "words": ["attenzione"],
                    "terminator": ".",
                    "special_forms": [],
                    "lang": "ita",
                }
            ],
        ),
        static_pipelines({"ita": nlp}, {"ita": ctx}),
        lock,
        free_threaded=False,
    )

    sentence = resp.results[0].result["raw_sentences"][0]
    deprels = [w["deprel"] for w in sentence]
    assert "iob" not in deprels, (
        f"non-UD deprel reached the response: {deprels!r}, "
        "the production path is not validating Stanza output"
    )
    assert deprels[1] == "iobj", f"expected iob normalized to iobj, got {deprels!r}"


# ---------------------------------------------------------------------------
# The terminator is a parsing cue for Stanza, never data
# ---------------------------------------------------------------------------


def _run_eng(
    items: list[dict[str, Any]], sentences: list[list[dict[str, Any]]]
) -> tuple[Any, _RecordingNlp]:
    """Run one English batch through the boundary and hand back both sides."""
    ctx = SimpleNamespace(original_words=[])
    nlp = _RecordingNlp(ctx, sentences)
    response = batch_infer_morphosyntax(
        BatchInferRequest(task="morphosyntax", lang="eng", items=items),
        static_pipelines({"eng": nlp}, {"eng": ctx}),
        _RecordingLock(),
        free_threaded=False,
    )
    return response, nlp


def test_batch_item_without_a_terminator_is_invalid() -> None:
    """A missing terminator is a caller bug, not a period.

    Rust always sends one, typed, and serializes it to its CHAT surface form
    at the IPC boundary. A `"."` default here is a sentinel that is also a
    legal value: it turns "the caller forgot" into "the utterance ended in a
    period", which is indistinguishable downstream and changes what Stanza is
    told about the sentence.
    """
    response, nlp = _run_eng([{"words": ["hello"]}], [_raw_sentence(["hello"])])

    assert response.results[0].error == "Invalid batch item"
    assert nlp.calls == [], "an invalid item must not reach Stanza"


def test_words_ending_in_the_terminator_are_rejected_not_silently_deduped() -> None:
    """The terminator is not a main-tier word, so `words` must not contain it.

    Silently dropping the duplicate hides a caller that has confused the two,
    and the two are exactly what this boundary exists to keep apart. Signal
    the unexpected pathway instead.
    """
    response, nlp = _run_eng(
        [
            {
                "words": ["hello", "."],
                "terminator": ".",
                "special_forms": [],
                "lang": "eng",
            }
        ],
        [_raw_sentence(["hello"])],
    )

    assert response.results[0].error == "Invalid batch item"
    assert nlp.calls == [], "a contract violation must not reach Stanza"


def test_terminator_word_is_stripped_by_position_not_by_recognition() -> None:
    """The cue Stanza was given must not come back as a `%mor` item.

    It is removed BY CONSTRUCTION: we appended exactly one boundary, so its UD
    word is the last one, and no inspection of what Stanza made of it is
    needed or wanted. This fixture returns the terminator tagged NOUN rather
    than PUNCT precisely so that a recognition-based filter would miss it and
    this test would fail.
    """
    response, nlp = _run_eng(
        [{"words": ["hello"], "terminator": ".", "special_forms": [], "lang": "eng"}],
        [_raw_sentence(["hello", "."])],
    )

    assert nlp.calls == [("hello .", [["hello", "."]])], (
        "Stanza must still SEE the terminator: it is the evidence the model "
        "uses for sentence-final analysis."
    )
    returned = response.results[0].result["raw_sentences"][0]
    assert [w["text"] for w in returned] == ["hello"], (
        "the terminator must not survive into the payload Rust maps to %mor"
    )
