"""Tests for translation worker bootstrap engine selection.

Mirrors the ASR engine-resolution tests in
``batchalign/tests/pipelines/asr/test_asr_model_loading.py``. The translation
loader previously discarded its ``engine_overrides`` parameter outright,
hard-coding Google as the backend even when the Rust control plane passed
``{"translate": "seamless"}``. These tests pin the resolver behavior so the
discard cannot recur silently.
"""

from __future__ import annotations

import pytest

from batchalign.inference._domain_types import TranslationBackend
from batchalign.worker._model_loading.translation import resolve_translate_engine


class TestResolveTranslateEngine:
    """Engine selection must stay deterministic, typed, and loud on bad input."""

    def test_seamless_override_wins(self) -> None:
        assert (
            resolve_translate_engine({"translate": "seamless"})
            is TranslationBackend.SEAMLESS
        )

    def test_google_override_wins(self) -> None:
        assert (
            resolve_translate_engine({"translate": "google"})
            is TranslationBackend.GOOGLE
        )

    def test_nllb_override_wins(self) -> None:
        assert (
            resolve_translate_engine({"translate": "nllb"})
            is TranslationBackend.NLLB
        )

    def test_tencent_override_wins(self) -> None:
        # Tencent Cloud TMT (Text Translation). Cloud-API engine that
        # outperforms NLLB on short Mandarin greetings (verified
        # 2026-05-26 against the same fixture set). Cantonese (`yue`)
        # is NOT supported by Tencent — those routes still go through
        # NLLB. See ``docs/investigations/2026-05-23-translate-engine-comparison.md``.
        assert (
            resolve_translate_engine({"translate": "tencent"})
            is TranslationBackend.TENCENT
        )

    def test_default_without_overrides_is_google(self) -> None:
        assert resolve_translate_engine(None) is TranslationBackend.GOOGLE

    def test_empty_dict_falls_through_to_default(self) -> None:
        assert resolve_translate_engine({}) is TranslationBackend.GOOGLE

    def test_unrelated_override_keys_are_ignored(self) -> None:
        # Only the ``translate`` key matters here; other engine keys
        # belong to other resolvers.
        assert (
            resolve_translate_engine({"asr": "seamless"})
            is TranslationBackend.GOOGLE
        )

    def test_unknown_engine_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown translate engine 'gogle'"):
            resolve_translate_engine({"translate": "gogle"})

    def test_unknown_engine_error_mentions_supported_options(self) -> None:
        with pytest.raises(ValueError, match="google, seamless, nllb, tencent"):
            resolve_translate_engine({"translate": "whisper"})


# ---------------------------------------------------------------------------
# Behavioral tests: Seamless backend actually translates
# ---------------------------------------------------------------------------
#
# The unit tests above prove ``resolve_translate_engine`` picks the
# Seamless backend when asked. They do NOT prove the Seamless backend,
# once selected, actually produces English translations. The tests
# below probe the full runtime path: model download (first run only),
# load, inference per source language, edge cases, state-restoration
# semantics. Before these tests, the only evidence that Seamless
# "works" was that the Python import resolves and the function exists —
# which proves nothing about runtime behavior.
#
# All marked ``integration`` and ``slow`` because the FIRST run
# downloads ~1.2 GB from HuggingFace (``facebook/hf-seamless-m4t-medium``)
# and the model load takes 30–60 s on CPU. Subsequent runs hit the
# cache and complete in seconds. The module-scoped fixture
# ``seamless_translate_fn`` amortizes the load across every test in
# this suite so the whole pass takes one model-load, not N.
#
# Excluded from the default fast-test selection (per pytest.ini
# ``addopts = ... -m "not slow and not golden and not integration"``).
# Run explicitly with:
#
#     uv run pytest -m integration \
#         batchalign/tests/pipelines/translate/test_translation_model_loading.py


def _load_translate_engine_fixture(
    engine_wire_name: str, expected_backend: TranslationBackend
):
    """Load a translate engine once for a module-scoped fixture.

    Restores prior worker state on teardown so the loaded model
    doesn't leak into other tests sharing the same pytest worker
    process (some test runners reuse worker processes).
    """
    from batchalign.worker._model_loading.translation import load_translation_engine
    from batchalign.worker._types import (
        InferTask,
        WorkerBootstrapRuntime,
        _state,
    )

    saved_backend = _state.translate_backend
    saved_fn = _state.translate_fn
    load_translation_engine(
        WorkerBootstrapRuntime(
            task=InferTask.TRANSLATE,
            lang="spa",
            num_speakers=1,
            engine_overrides={"translate": engine_wire_name},
        )
    )
    assert _state.translate_backend is expected_backend, (
        f"fixture failed to select {expected_backend.name}; "
        f"got {_state.translate_backend!r}"
    )
    assert _state.translate_fn is not None
    try:
        yield _state.translate_fn
    finally:
        _state.translate_backend = saved_backend
        _state.translate_fn = saved_fn


@pytest.fixture(scope="module")
def seamless_translate_fn():
    yield from _load_translate_engine_fixture("seamless", TranslationBackend.SEAMLESS)


@pytest.mark.integration
@pytest.mark.slow
class TestSeamlessTranslatesPerLanguage:
    """One test per source language Seamless claims to support that we
    care about. Each asserts the output is non-empty English with a
    plausible word for the input. The exact phrasing is NOT pinned —
    SeamlessM4T is non-deterministic across model revisions; the test
    detects "the path works AT ALL" not "the path produces a specific
    string."
    """

    def test_spanish_to_english(self, seamless_translate_fn) -> None:
        result = seamless_translate_fn("Hola mundo", "spa")
        assert isinstance(result, str)
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "world")), (
            f"Seamless spa→eng didn't produce expected words: {result!r}"
        )

    def test_french_to_english(self, seamless_translate_fn) -> None:
        result = seamless_translate_fn("Bonjour le monde", "fra")
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "world", "good", "day")), (
            f"Seamless fra→eng didn't produce expected words: {result!r}"
        )

    def test_german_to_english(self, seamless_translate_fn) -> None:
        result = seamless_translate_fn("Guten Tag", "deu")
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "good", "day", "morning")), (
            f"Seamless deu→eng didn't produce expected words: {result!r}"
        )

    def test_mandarin_to_english(self, seamless_translate_fn) -> None:
        # 你好世界 — Hello world. High-value because mandarin is the
        # motivating case (ECNU, mainland-China hosts where Google
        # Translate is GFW-blocked).
        result = seamless_translate_fn("你好世界", "cmn")
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "world")), (
            f"Seamless cmn→eng didn't produce expected words: {result!r}"
        )

    def test_cantonese_to_english(self, seamless_translate_fn) -> None:
        # Cantonese (yue) — HK research relevance. Seamless's
        # multilingual coverage of yue is the open question this
        # test exists to answer empirically rather than by reading
        # the model card.
        result = seamless_translate_fn("你好", "yue")
        # Looser assertion — if Seamless can't handle yue, the result
        # might be a romanization or a passthrough. Document what
        # actually happens.
        assert isinstance(result, str)
        assert result.strip(), "Seamless yue→eng returned empty output"

    def test_japanese_to_english(self, seamless_translate_fn) -> None:
        result = seamless_translate_fn("こんにちは", "jpn")
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "good")), (
            f"Seamless jpn→eng didn't produce expected words: {result!r}"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestSeamlessEdgeCases:
    """Edge cases the production translate pipeline can plausibly
    feed Seamless. Each is a probe for "does the path crash or
    return garbage on this input?" — we don't pin the exact output.
    """

    def test_empty_string_does_not_crash(self, seamless_translate_fn) -> None:
        # Production code's batched-text-infer handler short-circuits
        # empty utterances upstream of the worker, but a defensive
        # check here documents what happens if one slips through.
        result = seamless_translate_fn("", "spa")
        assert isinstance(result, str), f"expected str on empty input, got {type(result).__name__}"

    def test_single_word_translates(self, seamless_translate_fn) -> None:
        result = seamless_translate_fn("perro", "spa")
        # "perro" → "dog"
        assert isinstance(result, str)
        assert result.strip(), "single-word input produced empty translation"

    def test_punctuation_only_does_not_crash(self, seamless_translate_fn) -> None:
        result = seamless_translate_fn(".", "spa")
        assert isinstance(result, str)

    def test_multi_sentence_input(self, seamless_translate_fn) -> None:
        # Real CHAT utterances are typically one sentence each, but
        # nothing prevents a worker from getting a longer string.
        result = seamless_translate_fn(
            "Hola mundo. ¿Cómo estás? Me llamo Juan.", "spa"
        )
        assert isinstance(result, str)
        assert result.strip()
        # Should contain SOMETHING that maps to one of the source
        # sentences' meanings — sanity check, not output-pinning.
        lower = result.lower()
        assert any(
            w in lower for w in ("hello", "world", "how", "my", "name", "juan")
        ), f"multi-sentence translation lost meaning: {result!r}"


@pytest.mark.integration
@pytest.mark.slow
class TestSeamlessRuntimeInvariants:
    """Behavior the production worker relies on: idempotent inference
    (calling twice on the same input returns the same shape of
    result), state population (loader actually sets ``_state``),
    and backend-switching (loading Google after Seamless works, and
    vice versa, so a daemon that handles both never poisons state).
    """

    def test_repeated_inference_returns_stable_shape(
        self, seamless_translate_fn
    ) -> None:
        # Not pinning output (Seamless can have small variance even
        # on identical input), but the type and non-emptiness must
        # be stable across invocations.
        first = seamless_translate_fn("Hola mundo", "spa")
        second = seamless_translate_fn("Hola mundo", "spa")
        assert isinstance(first, str) and isinstance(second, str)
        assert first.strip() and second.strip()

    def test_backend_loader_populates_state(self) -> None:
        # The loader's contract: after a successful load, both
        # ``_state.translate_backend`` and ``_state.translate_fn``
        # are populated. The batch-infer handler in
        # ``_infer_hosts.py`` asserts both are non-None before
        # accepting work, so the loader must satisfy that.
        from batchalign.worker._model_loading.translation import load_translation_engine
        from batchalign.worker._types import (
            InferTask,
            WorkerBootstrapRuntime,
            _state,
        )

        saved_backend = _state.translate_backend
        saved_fn = _state.translate_fn
        try:
            load_translation_engine(
                WorkerBootstrapRuntime(
                    task=InferTask.TRANSLATE,
                    lang="spa",
                    num_speakers=1,
                    engine_overrides={"translate": "seamless"},
                )
            )
            assert _state.translate_backend is TranslationBackend.SEAMLESS
            assert _state.translate_fn is not None
            assert callable(_state.translate_fn)
        finally:
            _state.translate_backend = saved_backend
            _state.translate_fn = saved_fn

    def test_can_switch_from_google_to_seamless_in_same_process(self) -> None:
        # A daemon serving both engine pools (one worker on Google,
        # one on Seamless) must be able to load each independently
        # without the Google load polluting Seamless state or vice
        # versa. Test the transition explicitly.
        from batchalign.worker._model_loading.translation import load_translation_engine
        from batchalign.worker._types import (
            InferTask,
            WorkerBootstrapRuntime,
            _state,
        )

        saved_backend = _state.translate_backend
        saved_fn = _state.translate_fn
        try:
            load_translation_engine(
                WorkerBootstrapRuntime(
                    task=InferTask.TRANSLATE,
                    lang="spa",
                    num_speakers=1,
                    engine_overrides={"translate": "google"},
                )
            )
            assert _state.translate_backend is TranslationBackend.GOOGLE
            google_fn = _state.translate_fn

            load_translation_engine(
                WorkerBootstrapRuntime(
                    task=InferTask.TRANSLATE,
                    lang="spa",
                    num_speakers=1,
                    engine_overrides={"translate": "seamless"},
                )
            )
            assert _state.translate_backend is TranslationBackend.SEAMLESS
            assert _state.translate_fn is not google_fn, (
                "Seamless load must replace ``_state.translate_fn`` — "
                "leaving the Google fn in place would silently route "
                "Seamless-pool requests to Google."
            )
        finally:
            _state.translate_backend = saved_backend
            _state.translate_fn = saved_fn


# ---------------------------------------------------------------------------
# Behavioral tests: NLLB backend actually translates
# ---------------------------------------------------------------------------
#
# Mirror of the Seamless suite above. ``integration`` + ``slow`` because
# the FIRST run downloads ~5 GB from HuggingFace.


@pytest.fixture(scope="module")
def nllb_translate_fn():
    yield from _load_translate_engine_fixture("nllb", TranslationBackend.NLLB)


@pytest.mark.integration
@pytest.mark.slow
class TestNllbTranslatesPerLanguage:
    """Per-source-language behavioral tests. Each asserts non-empty
    English with at least one plausible word — exact output is NOT
    pinned (NLLB varies across model revisions; the empirical
    fixture-output data is captured in the investigation doc).
    """

    def test_spanish_to_english(self, nllb_translate_fn) -> None:
        result = nllb_translate_fn("Hola mundo", "spa")
        assert isinstance(result, str)
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "hey", "world")), (
            f"NLLB spa→eng didn't produce expected words: {result!r}"
        )

    def test_french_to_english(self, nllb_translate_fn) -> None:
        result = nllb_translate_fn("Bonjour le monde", "fra")
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "world", "good", "day")), (
            f"NLLB fra→eng didn't produce expected words: {result!r}"
        )

    def test_german_to_english(self, nllb_translate_fn) -> None:
        result = nllb_translate_fn("Guten Tag", "deu")
        lower = result.lower()
        assert any(w in lower for w in ("hello", "hi", "good", "day", "morning", "afternoon")), (
            f"NLLB deu→eng didn't produce expected words: {result!r}"
        )

    def test_mandarin_long_form_to_english(self, nllb_translate_fn) -> None:
        # Long-form input — NLLB handles real sentences; short greetings
        # are a documented weakness handled in TestNllbRuntimeInvariants.
        result = nllb_translate_fn(
            "今天天气很好，我想去公园散步。", "cmn"
        )
        lower = result.lower()
        assert any(
            w in lower for w in ("weather", "today", "park", "walk", "nice", "good")
        ), f"NLLB cmn→eng (long-form) didn't produce expected words: {result!r}"

    def test_japanese_long_form_to_english(self, nllb_translate_fn) -> None:
        result = nllb_translate_fn("今日はとても良い天気ですね。", "jpn")
        lower = result.lower()
        assert any(
            w in lower for w in ("weather", "today", "day", "nice", "good")
        ), f"NLLB jpn→eng (long-form) didn't produce expected words: {result!r}"


@pytest.mark.integration
@pytest.mark.slow
class TestNllbRuntimeInvariants:
    """Loader-state contract + backend-switching contract."""

    def test_backend_loader_populates_state(self) -> None:
        from batchalign.worker._model_loading.translation import load_translation_engine
        from batchalign.worker._types import (
            InferTask,
            WorkerBootstrapRuntime,
            _state,
        )

        saved_backend = _state.translate_backend
        saved_fn = _state.translate_fn
        try:
            load_translation_engine(
                WorkerBootstrapRuntime(
                    task=InferTask.TRANSLATE,
                    lang="spa",
                    num_speakers=1,
                    engine_overrides={"translate": "nllb"},
                )
            )
            assert _state.translate_backend is TranslationBackend.NLLB
            assert _state.translate_fn is not None
            assert callable(_state.translate_fn)
        finally:
            _state.translate_backend = saved_backend
            _state.translate_fn = saved_fn

    def test_unsupported_language_raises_clear_error(
        self, nllb_translate_fn
    ) -> None:
        # The FLORES-200 mapping is a closed set — an unmapped source
        # language must raise rather than silently produce wrong-language
        # output. "xyz" is not a valid ISO-639-3 code anywhere.
        with pytest.raises(ValueError, match="FLORES-200 mapping"):
            nllb_translate_fn("hola", "xyz")
