"""Qwen3-ASR helpers for the built-in Cantonese engine.

Wraps the ``qwen-asr`` PyPI package. Mirrors the
``_funaudio_common.FunAudioRecognizer`` shape so the worker
bootstrap can swap engines uniformly. The actual model load is
deferred until first inference call — keeps worker startup time
bounded when no Qwen jobs are dispatched to the worker.

Engine selection rationale: Qwen3-ASR is an open-weight
Cantonese-capable ASR model from Alibaba. External evaluations on
per-utterance Cantonese child speech report competitive CER
relative to the major cloud APIs and to Cantonese-finetuned Whisper
variants; it is wired here as one engine option among several.

Device default is ``cpu``. Apple Silicon hosts have no CUDA, and
empirical testing found that MPS-backed inference produces degraded
output on the 1.7B model (the upstream ``transformers`` MPS path
isn't fully supported for Qwen3 attention ops yet). Hosts with CUDA
can opt in via ``engine_overrides["qwen_device"]="cuda"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ._asr_types import AsrElement, AsrGenerationPayload, AsrMonologue, TimedWord
from batchalign.inference._domain_types import LanguageCode

L = logging.getLogger("batchalign.hk.qwen")


# ISO-639-3 → English language label expected by Qwen3-ASR's
# ``language=`` parameter. Pinned in code rather than via pycountry
# because pycountry returns ``"Yue Chinese"`` for ``yue``, which Qwen
# does not accept (silent fall-through to auto-detect). The fix is
# explicit per-code mapping with a fail-loud default.
_QWEN_LANG_LABELS: dict[LanguageCode, str] = {
    "yue": "Cantonese",
    "zho": "Chinese",
    "cmn": "Chinese",
    "eng": "English",
}


# Canonical forced-alignment companion model from the Qwen3-ASR family.
# qwen-asr's ``Qwen3ASRModel.transcribe(..., return_time_stamps=True)``
# raises ValueError when ``forced_aligner`` was not supplied at
# ``from_pretrained``. We always want word-level timestamps (the
# downstream BA3 FA pipeline depends on them for ``%wor`` tier
# injection), so the aligner is always wired. The model ID matches
# Qwen's own README example for paired ASR + forced-alignment usage.
_QWEN_FORCED_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


def _resolve_qwen_language(lang: LanguageCode) -> str:
    label = _QWEN_LANG_LABELS.get(lang)
    if label is None:
        raise ValueError(
            f"Qwen3-ASR has no language label mapped for ISO-639-3 "
            f"{lang!r}; add it to _QWEN_LANG_LABELS in "
            f"_qwen_common.py if the model supports the language."
        )
    return label


@dataclass
class QwenAsrSegment:
    """One Qwen3-ASR utterance with optional word-level timestamps.

    ``text`` is the full utterance string. ``word_timestamps`` is a
    list of ``(start_s, end_s, word_text)`` tuples — empty when the
    model didn't return timing for this segment (returns can be
    text-only for short utterances).
    """

    text: str
    word_timestamps: list[tuple[float, float, str]] = field(
        default_factory=list
    )

    @classmethod
    def from_raw(cls, raw: Any) -> QwenAsrSegment:
        """Parse one qwen-asr result object (duck-typed: has ``.text``
        and optionally ``.time_stamps`` with an ``.items`` list of
        word entries carrying ``.start_time``, ``.end_time``, ``.text``)."""
        text = str(getattr(raw, "text", "") or "")
        word_ts: list[tuple[float, float, str]] = []
        ts_container = getattr(raw, "time_stamps", None)
        if ts_container is not None:
            items = getattr(ts_container, "items", None) or []
            for item in items:
                start = float(getattr(item, "start_time", 0.0))
                end = float(getattr(item, "end_time", 0.0))
                word = str(getattr(item, "text", "") or "")
                if word:
                    word_ts.append((start, end, word))
        return cls(text=text, word_timestamps=word_ts)


class QwenRecognizer:
    """Wrapper around Qwen3-ASR model invocation.

    The transcribe API returns a typed ``AsrGenerationPayload`` +
    timed-words list, matching the contract the BA3 worker protocol
    expects. Projection from per-utterance results to speaker-tagged
    monologues is done locally in Python (single-speaker for now;
    Qwen3-ASR doesn't emit speaker diarization).

    For multi-speaker audio, the downstream BA3 pipeline applies
    diarization separately — Qwen3-ASR's role here is the
    speech-to-character transcription only. This matches how
    BA3 currently uses other diarization-free ASR engines.
    """

    def __init__(
        self,
        lang: LanguageCode = "yue",
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cpu",
    ) -> None:
        self.lang = lang
        self.model_id = model_id
        self.device = device
        self._model: Any | None = None
        # Resolved at construction so unsupported-language errors
        # surface at worker startup, not on first inference.
        self._qwen_language = _resolve_qwen_language(lang)

    def warm(self) -> None:
        """Force the lazy model load. Call at worker bootstrap so the
        first inference request sees a warm cache, not a ~3.4 GB
        download + load delay."""
        self._get_model()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import torch  # type: ignore[import-not-found]
            from qwen_asr import Qwen3ASRModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Qwen3-ASR engine dependency 'qwen-asr' (and its "
                "transitive 'torch' dep) is missing from this "
                "environment. Reinstall batchalign3 or install "
                "qwen-asr explicitly."
            ) from exc

        if self.device == "cuda":
            dtype = torch.bfloat16
        elif self.device == "mps":
            # User opted into MPS via override; warn rather than raise
            # because the user is sovereign on device selection.
            L.warning(
                "Qwen3-ASR MPS device requested; empirical testing "
                "2026-05-26 found degraded output on 1.7B (~78%% CER "
                "vs ~57%% CPU). Verify current qwen-asr/transformers "
                "MPS support before relying on MPS in production."
            )
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            max_inference_batch_size=32,
            # max_new_tokens caps per-chunk generation. The qwen-asr
            # package handles internal long-audio chunking, but each
            # chunk's text is bounded by this. 4096 is generous for
            # the ~5-60 second chunks the package's VAD produces.
            max_new_tokens=4096,
            # qwen-asr requires the forced aligner at init when the
            # caller asks for return_time_stamps=True later. The
            # aligner loads on the same device + dtype as the ASR
            # model so cross-device tensor moves don't surprise the
            # downstream batch loop.
            forced_aligner=_QWEN_FORCED_ALIGNER_MODEL_ID,
            forced_aligner_kwargs={
                "dtype": dtype,
                "device_map": self.device,
            },
        )
        L.info(
            "Qwen3-ASR loaded: model=%s, lang=%s (%s), device=%s, dtype=%s",
            self.model_id,
            self.lang,
            self._qwen_language,
            self.device,
            dtype,
        )
        return self._model

    def _run_model(self, source_path: str) -> list[QwenAsrSegment]:
        model = self._get_model()
        results = model.transcribe(
            audio=source_path,
            # Per Qwen3-ASR README, ``language=None`` enables
            # auto-detection. We pass the explicit label since the
            # caller has already established the session language via
            # ``@Languages`` — auto-detect on a known-language input
            # risks the model mis-classifying short or low-energy
            # segments.
            language=self._qwen_language,
            return_time_stamps=True,
        )
        if not isinstance(results, list):
            # Defensive: the API can return a single result for
            # single-input calls. Wrap so downstream code sees a
            # uniform list.
            results = [results]
        return [QwenAsrSegment.from_raw(r) for r in results]

    def transcribe(
        self, source_path: str
    ) -> tuple[AsrGenerationPayload, list[TimedWord]]:
        """Run Qwen3-ASR on the audio file and return the shared
        ``(monologues_payload, timed_words)`` tuple.

        Unlike the FunASR path, this projection is done entirely in
        Python (no Rust ``batchalign_core`` call). The Qwen3-ASR
        output structure is simple enough that a per-engine Rust
        projection isn't warranted yet; if the multi-speaker /
        diarization story changes in a future Qwen release we can
        consolidate into a shared Rust helper alongside the FunASR
        and Tencent projections.
        """
        segments = self._run_model(source_path)

        elements: list[AsrElement] = []
        timed_words: list[TimedWord] = []
        for seg in segments:
            if seg.word_timestamps:
                for start_s, end_s, word in seg.word_timestamps:
                    # Pre-filter empty/whitespace tokens so downstream
                    # never sees them — saves allocation and matches
                    # the FunASR + Tencent shape.
                    if not word.strip():
                        continue
                    elements.append(
                        AsrElement(
                            type="text",
                            ts=start_s,
                            end_ts=end_s,
                            value=word,
                        )
                    )
                    timed_words.append(
                        TimedWord(
                            word=word,
                            start_ms=int(start_s * 1000),
                            end_ms=int(end_s * 1000),
                        )
                    )
            elif seg.text.strip():
                # Fall back to the whole-segment text when the model
                # didn't return per-word timestamps. CJK output is
                # tokenized per-character downstream; we leave that
                # to BA3's standard Cantonese tokenizer rather than
                # splitting here (avoid double-tokenization).
                elements.append(
                    AsrElement(
                        type="text",
                        ts=None,
                        end_ts=None,
                        value=seg.text,
                    )
                )

        monologues: list[AsrMonologue] = [
            AsrMonologue(speaker=0, elements=elements)
        ]
        payload: AsrGenerationPayload = AsrGenerationPayload(monologues=monologues)
        return payload, timed_words
