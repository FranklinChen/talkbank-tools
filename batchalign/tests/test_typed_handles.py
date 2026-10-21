"""Tests for typed model handles, behavioral contracts only.

Verifies non-trivial logic in WhisperASRHandle (one gen_kwargs per decoding,
callable forwarding) and WhisperFAHandle (instance-local monkey-patching).
Field-storage tests removed: dataclass fields storing what was passed in
is tested by Python itself, not us.
"""

from __future__ import annotations

from batchalign.inference.audio import bind_whisper_token_timestamp_extractor
from batchalign.inference.types import (
    CheckpointPinnedLanguage,
    WhisperASRHandle,
)
from batchalign.worker._types_v2 import DetectWhisperLanguageV2, ForceWhisperLanguageV2


class TestWhisperASRHandle:
    """WhisperASRHandle replaces monkey-patched _ba_* attributes.

    How the handle is told its language is fixed at construction (one
    ``decoding`` value), so ``gen_kwargs`` takes no language: a request cannot
    re-force a fine-tune, and Cantonese's detect rule is the control plane's
    decision, tested on the Rust side.
    """

    def test_callable_forwards_to_pipe(self) -> None:
        calls: list[tuple[str, dict[str, int | dict[str, str]]]] = []

        def fake_pipe(
            audio: str, **kwargs: int | dict[str, str]
        ) -> dict[str, list[dict[str, str | tuple[float, float]]]]:
            calls.append((audio, kwargs))
            return {"chunks": [{"text": "hello", "timestamp": (0.0, 1.0)}]}

        handle = WhisperASRHandle(
            pipe=fake_pipe,
            config="cfg",
            decoding=DetectWhisperLanguageV2(),
            sample_rate=16000,
        )
        result = handle(
            "audio_data", batch_size=1, generate_kwargs={"task": "transcribe"}
        )
        assert len(calls) == 1
        assert result["chunks"][0]["text"] == "hello"  # type: ignore[index]

    def test_a_forced_language_is_named_by_its_whisper_token(self) -> None:
        handle = WhisperASRHandle(
            pipe=None,
            config="my_config",
            decoding=ForceWhisperLanguageV2(token="el"),
            sample_rate=16000,
        )
        assert handle.gen_kwargs() == {
            "repetition_penalty": 1.001,
            "generation_config": "my_config",
            "task": "transcribe",
            "language": "el",
        }

    def test_detection_omits_language_and_task(self) -> None:
        """Whisper detects the language itself, with the stock tuning knobs."""
        handle = WhisperASRHandle(
            pipe=None,
            config="my_config",
            decoding=DetectWhisperLanguageV2(),
            sample_rate=16000,
        )
        assert handle.gen_kwargs() == {
            "repetition_penalty": 1.001,
            "generation_config": "my_config",
        }

    def test_a_fine_tune_gets_only_the_runaway_cap(self) -> None:
        """Fine-tunes come with their own ``generation_config`` baked into
        the checkpoint: language hints, suppress-token sets, beam and
        temperature settings tuned during training. The pipeline uses that
        by default.

        ``load_whisper_asr`` starts from ``GenerationConfig.from_pretrained``
        and applies ``no_repeat_ngram_size=4`` and ``use_cache=True`` on top.
        Passing that as ``generate_kwargs={"generation_config": ...}``
        **overrides** the checkpoint's own config, including settings the
        fine-tune author relied on. Empirically (2026-04-22,
        thennal/whisper-medium-ml on a Malayalam sample) the override
        produced cross-script gibberish with ``"letter"`` substrings and
        stray digits, while empty ``generate_kwargs`` produced clean
        62-word folk-tale output.

        So a fine-tune passes exactly the runaway-generation cap (Whisper's
        ``max_target_positions=448`` minus 3 special start tokens minus 1
        margin). Knobs for fine-tunes belong in a new decoding variant, NOT
        in overrides here.
        """
        handle = WhisperASRHandle(
            pipe=None,
            config="ft_config",
            decoding=CheckpointPinnedLanguage(),
            sample_rate=16000,
        )
        assert handle.gen_kwargs() == {"max_new_tokens": 444}


class TestWhisperFAHandle:
    """WhisperFAHandle replaces (model, processor, sample_rate) tuple."""

    def test_timestamp_extractor_binding_stays_instance_local(self) -> None:
        """The Whisper workaround should patch one model instance, not a class."""

        class _FakeModel:
            """Small test double for an instance-bound Whisper override."""

            marker = "fake-model"

        model = _FakeModel()
        bind_whisper_token_timestamp_extractor(model)  # type: ignore[arg-type]

        bound = model._extract_token_timestamps  # type: ignore[attr-defined]
        assert bound.__self__ is model
