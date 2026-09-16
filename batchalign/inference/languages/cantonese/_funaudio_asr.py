"""FunASR ASR inference provider for built-in HK/Cantonese engines.

New-style provider: a ``load`` function (called once at worker startup)
and an ``infer`` function (called per batch_infer request).
"""

from __future__ import annotations

import logging
import time

from pydantic import ValidationError

from batchalign.inference._domain_types import LanguageCode
from batchalign.inference.asr import AsrBatchItem, MonologueAsrResponse
from batchalign.worker._types import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
)

from ._asr_types import admit_provider_monologues
from ._common import EngineOverrides
from ._funaudio_common import FunAudioRecognizer

L = logging.getLogger("batchalign.hk.funaudio_asr")


# ---------------------------------------------------------------------------
# Module-level state (populated by load_funaudio_asr)
# ---------------------------------------------------------------------------

_recognizer: FunAudioRecognizer | None = None

# The error returned when inference is attempted before the model is loaded.
#
# A constant because the test for that path asserted the same sentence as a
# second literal, and the 2026-07-29 dash sweep then resolved the one em-dash
# they shared differently on each side: a colon here, a comma there. Both edits
# obeyed the rule (punctuation is chosen per occurrence), and the pair still
# broke, because nothing recorded that these two occurrences were one sentence.
# Duplicated knowledge with no owner behaves exactly like this.
NOT_LOADED_ERROR = "FunAudio ASR not loaded: call load_funaudio_asr first"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_funaudio_asr(
    lang: LanguageCode,
    engine_overrides: EngineOverrides | None,
    *,
    model: str | None = None,
    model_path: str | None = None,
    model_revision: str | None = None,
    vad_model: str | None = None,
    vad_model_path: str | None = None,
    vad_revision: str | None = None,
    punc_model: str | None = None,
    punc_revision: str | None = None,
) -> None:
    """Initialize the FunAudio ASR recognizer.

    Called once at worker startup.  Stores the recognizer in module-level
    state so that ``infer_funaudio_asr`` can use it for every request.

    Parameters
    ----------
    lang : str
        ISO 639-3 language code (e.g. ``"yue"``).
    engine_overrides : EngineOverrides or None
        Optional dict of overrides applied on top of the defaults
        (``model="FunAudioLLM/SenseVoiceSmall"``, ``device="cpu"``).
        Two keys are recognized:

        * ``"funaudio_model"``: Hugging Face model id of the FunASR
          checkpoint to load (e.g. a Paraformer variant). The
          downstream :class:`FunAudioRecognizer` branches on whether
          the chosen name contains ``"paraformer"``.
        * ``"funaudio_device"``: torch device string passed through
          to the recognizer (e.g. ``"cuda"`` or ``"mps"``).

        Other keys in the dict are ignored.
    """
    global _recognizer

    selected = model or "FunAudioLLM/SenseVoiceSmall"
    device = "cpu"
    if engine_overrides:
        if model is None and "funaudio_model" in engine_overrides:
            selected = str(engine_overrides["funaudio_model"])
        if "funaudio_device" in engine_overrides:
            device = str(engine_overrides["funaudio_device"])

    _recognizer = FunAudioRecognizer(
        lang=lang,
        model=selected,
        device=device,
        model_path=model_path,
        model_revision=model_revision,
        vad_model=vad_model,
        vad_model_path=vad_model_path,
        vad_revision=vad_revision,
        punc_model=punc_model,
        punc_revision=punc_revision,
    )
    L.info(
        "FunAudio ASR recognizer loaded: lang=%s, model=%s, device=%s",
        lang,
        model_path or selected,
        device,
    )


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------


def infer_funaudio_asr(req: BatchInferRequest) -> BatchInferResponse:
    """Process a batch of ASR inference items using FunASR.

    Each item should be an :class:`AsrBatchItem`.  For each item the
    recognizer transcribes the audio and returns the provider-shaped
    monologue payload directly. Rust owns the shared normalization layer.

    Parameters
    ----------
    req : BatchInferRequest
        Batch of ``AsrBatchItem`` payloads.

    Returns
    -------
    BatchInferResponse
        One :class:`InferResponse` per item, each containing a tagged
        monologue payload on success.
    """
    if _recognizer is None:
        return BatchInferResponse(
            results=[
                InferResponse(error=NOT_LOADED_ERROR, elapsed_s=0.0) for _ in req.items
            ]
        )

    t0 = time.monotonic()
    n = len(req.items)
    results: list[InferResponse] = []

    for item_idx, raw_item in enumerate(req.items):
        try:
            item = AsrBatchItem.model_validate(raw_item)
        except ValidationError:
            results.append(InferResponse(error="Invalid AsrBatchItem", elapsed_s=0.0))
            continue

        try:
            response = _transcribe_to_monologues(item)
            results.append(InferResponse(result=response.model_dump(), elapsed_s=0.0))
        except Exception as e:
            L.warning("FunAudio ASR failed for item %d: %s", item_idx, e, exc_info=True)
            results.append(InferResponse(error=str(e), elapsed_s=0.0))

    elapsed = time.monotonic() - t0
    if results:
        first = results[0]
        results[0] = InferResponse(
            result=first.result, error=first.error, elapsed_s=elapsed
        )

    L.info("batch_infer funaudio_asr: %d items, %.3fs", n, elapsed)
    return BatchInferResponse(results=results)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _transcribe_to_monologues(item: AsrBatchItem) -> MonologueAsrResponse:
    """Run FunAudio recognizer and return raw speaker monologues."""
    if _recognizer is None:
        raise RuntimeError("FunAudio recognizer not initialized")

    payload, _timed_words = _recognizer.transcribe(item.audio_path)
    return admit_provider_monologues(payload, item.lang)


def infer_funaudio_asr_v2(item: AsrBatchItem) -> MonologueAsrResponse:
    """Run one typed FunAudio ASR request for worker protocol V2.

    The live V2 worker path calls the transport directly so Python
    does not reassemble a batch envelope for one request.
    """

    return _transcribe_to_monologues(item)
