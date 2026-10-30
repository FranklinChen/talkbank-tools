"""The one runner-error taxonomy at the Rust worker boundary.

Every executor calls its Python runner through ``LoadedRunner::call1`` in
``batchalign-pyo3/src/worker_execute.rs``, which classifies a raised
exception by its CLASS into a protocol code. These tests drive that
classification through two real executors (a text task and forced
alignment), because the text executor used to keep its own copy of the
classifier and forced alignment stringified every exception into a runtime
failure.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import numpy as np
import pytest
import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import BaseModel, ValidationError

from batchalign.inference._model_access_errors import ModelAccessDeniedError
from batchalign.inference._provider_errors import ProviderQuotaExhaustedError
from batchalign.inference.qwen_forced_alignment import (
    PairedWindowAlignment,
    WindowAlignedWord,
)
from batchalign.inference.qwen_timing import (
    AudioWindow,
    RecordingChunkSpeech,
    RecordingStart,
    SpokenWindowWords,
    WindowAlignment,
)
from batchalign.inference.types import Wave2VecWordAlignment
from batchalign.inference.utseg import UtsegModelNotFoundError
from batchalign.worker._execute_v2 import WorkerExecutionHostV2, execute_request_v2
from batchalign.worker._fa_v2 import (
    ForcedAlignmentExecutionHostV2,
    execute_forced_alignment_request_v2,
)
from batchalign.worker._language_errors import UnsupportedLanguageError
from batchalign.worker._text_v2 import TextExecutionHostV2
from batchalign.worker._types import BatchInferResponse
from batchalign.worker._types_v2 import (
    ExecuteErrorV2,
    ExecuteRequestV2,
    FaBackendV2,
    FaTextModeV2,
    ForcedAlignmentRequestV2,
    InferenceTaskV2,
    MorphosyntaxRequestV2,
    PreparedAudioEncodingV2,
    PreparedAudioRefV2,
    PreparedTextEncodingV2,
    PreparedTextRefV2,
    ProtocolErrorCodeV2,
)


class _Strict(BaseModel):
    """A model whose validation failure stands in for a malformed batch."""

    count: int


def _pydantic_validation_error() -> Exception:
    try:
        _Strict.model_validate({"count": "not a number"})
    except ValidationError as error:
        return error
    raise AssertionError("the model accepted a malformed value")


def _download_failed_in_transit() -> Exception:
    try:
        try:
            raise httpx2.ConnectError("connection refused")
        except httpx2.ConnectError as cause:
            raise LocalEntryNotFoundError("no cached copy") from cause
    except LocalEntryNotFoundError as raised:
        return raised


def _quota_error() -> Exception:
    return ProviderQuotaExhaustedError("the provider's resource pack is exhausted")


def _bad_qwen_timing() -> Exception:
    try:
        WindowAlignment(
            PairedWindowAlignment.of(
                ["hello", "world"],
                [
                    WindowAlignedWord("hello", 0.7, 0.8),
                    WindowAlignedWord("world", 0.5, 0.6),
                ],
            ),
            AudioWindow(1.0),
        )
    except Exception as error:
        return error
    raise AssertionError("backwards model timing was admitted")


def _bad_recording_placement() -> Exception:
    aligned = WindowAlignment(
        PairedWindowAlignment.of(["hello"], [WindowAlignedWord("hello", 0.1, 0.4)]),
        AudioWindow(1.0),
    )
    speech = SpokenWindowWords(aligned)
    prefix = RecordingStart().append(RecordingChunkSpeech("hello", 180.0, speech))
    try:
        prefix.append(RecordingChunkSpeech("hello", 179.0, speech))
    except Exception as error:
        return error
    raise AssertionError("backwards recording placement was admitted")


# (what the runner raises, the protocol code the boundary must report).
# Most specific class first in the classifier: UnsupportedLanguageError
# subclasses ValueError and ModelAccessDeniedError subclasses RuntimeError,
# yet each keeps its own code.
_CASES = [
    pytest.param(
        _bad_recording_placement,
        ProtocolErrorCodeV2.MODEL_OUTPUT_INVALID,
        id="recording-placement",
    ),
    pytest.param(
        _bad_qwen_timing,
        ProtocolErrorCodeV2.MODEL_OUTPUT_INVALID,
        id="generated-qwen-timing",
    ),
    pytest.param(
        lambda: UnsupportedLanguageError("no Stanza pipeline for 'que'"),
        ProtocolErrorCodeV2.UNSUPPORTED_LANGUAGE,
        id="unsupported-language",
    ),
    pytest.param(
        lambda: ModelAccessDeniedError(
            "https://huggingface.co/owner/model/resolve/main/config.json", "gated"
        ),
        ProtocolErrorCodeV2.MODEL_ACCESS_DENIED,
        id="model-access-denied",
    ),
    pytest.param(
        _quota_error,
        ProtocolErrorCodeV2.PROVIDER_QUOTA_EXHAUSTED,
        id="provider-quota",
    ),
    # RED FIRST: a request naming the boundary model on a worker that has
    # none loaded subclassed RuntimeError, so it was a local model's runtime
    # failure (not retried, and reported as the model failing on the input).
    # It is the worker's model-not-loaded fact, model_unavailable.
    pytest.param(
        lambda: UtsegModelNotFoundError("no boundary model loaded"),
        ProtocolErrorCodeV2.MODEL_UNAVAILABLE,
        id="utseg-model-not-loaded",
    ),
    pytest.param(
        lambda: NotImplementedError("model not loaded"),
        ProtocolErrorCodeV2.MODEL_UNAVAILABLE,
        id="not-implemented",
    ),
    pytest.param(
        _pydantic_validation_error,
        ProtocolErrorCodeV2.INVALID_PAYLOAD,
        id="pydantic-validation",
    ),
    pytest.param(
        lambda: ValueError("window too short for its transcript"),
        ProtocolErrorCodeV2.RUNTIME_FAILURE,
        id="plain-value-error",
    ),
    # RED FIRST: a device that ran out of memory was a local model's runtime
    # failure, which the server does not retry. torch raises its own class
    # for it, and that class is the device's shortfall, retried after a
    # backoff.
    pytest.param(
        lambda: torch.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        ),
        ProtocolErrorCodeV2.DEVICE_OUT_OF_MEMORY,
        id="torch-out-of-memory",
    ),
    # RED FIRST: a Hub download that could not connect (what hf_hub_download
    # raises: no cached copy, raised from the connection failure) was a
    # runtime failure, never retried.
    pytest.param(
        _download_failed_in_transit,
        ProtocolErrorCodeV2.MODEL_DOWNLOAD_FAILED,
        id="download-failed-in-transit",
    ),
    # The words are not the class: a plain RuntimeError saying "out of
    # memory" (which is also what torch raises for an MPS shortfall, with no
    # class of its own) stays a runtime failure.
    pytest.param(
        lambda: RuntimeError("CUDA out of memory"),
        ProtocolErrorCodeV2.RUNTIME_FAILURE,
        id="runtime-error-text-is-not-a-class",
    ),
]


def _morphosyntax_request(tmp_path: Path) -> ExecuteRequestV2:
    payload_path = tmp_path / "morphosyntax-batch.json"
    item = {
        "words": ["hi"],
        "terminator": ".",
        "special_forms": [[None, None]],
        "runs": [1],
        "lang": "eng",
    }
    payload_path.write_text(json.dumps({"items": [item], "mwt": {}}), encoding="utf-8")
    return ExecuteRequestV2(
        request_id="req-runner-error-morphosyntax",
        task=InferenceTaskV2.MORPHOSYNTAX,
        payload=MorphosyntaxRequestV2(
            lang="eng",
            payload_ref_id="text-ref-runner-error",
            item_count=1,
        ),
        attachments=[
            PreparedTextRefV2(
                id="text-ref-runner-error",
                path=str(payload_path),
                encoding=PreparedTextEncodingV2.UTF8_JSON,
                byte_offset=0,
                byte_len=payload_path.stat().st_size,
            )
        ],
    )


def _fa_request(tmp_path: Path) -> ExecuteRequestV2:
    payload_path = tmp_path / "fa-payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "words": ["hello", "world"],
                "word_ids": ["u0:w0", "u0:w1"],
                "word_utterance_indices": [0, 0],
                "word_utterance_word_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )
    audio_path = tmp_path / "fa-audio.pcm"
    audio_path.write_bytes(np.asarray([0.1, 0.2, 0.3, 0.4], dtype="<f4").tobytes())
    return ExecuteRequestV2(
        request_id="req-runner-error-fa",
        task=InferenceTaskV2.FORCED_ALIGNMENT,
        payload=ForcedAlignmentRequestV2(
            backend=FaBackendV2.WAVE2VEC,
            payload_ref_id="fa-payload-ref",
            audio_ref_id="fa-audio-ref",
            text_mode=FaTextModeV2.SPACE_JOINED,
        ),
        attachments=[
            PreparedTextRefV2(
                id="fa-payload-ref",
                path=str(payload_path),
                encoding=PreparedTextEncodingV2.UTF8_JSON,
                byte_offset=0,
                byte_len=payload_path.stat().st_size,
            ),
            PreparedAudioRefV2(
                id="fa-audio-ref",
                path=str(audio_path),
                encoding=PreparedAudioEncodingV2.PCM_F32LE,
                channels=1,
                sample_rate_hz=16000,
                frame_count=4,
                byte_offset=0,
                byte_len=audio_path.stat().st_size,
            ),
        ],
    )


@pytest.mark.parametrize(("make_error", "expected"), _CASES)
def test_a_text_runner_exception_keeps_its_class_code(
    tmp_path: Path,
    make_error: Callable[[], Exception],
    expected: ProtocolErrorCodeV2,
) -> None:
    def _raise(*_args: object) -> BatchInferResponse:
        raise make_error()

    response = execute_request_v2(
        request=_morphosyntax_request(tmp_path),
        host=WorkerExecutionHostV2(
            text=TextExecutionHostV2(morphosyntax_runner=_raise)
        ),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is expected
    assert response.result is None


@pytest.mark.parametrize(("make_error", "expected"), _CASES)
def test_a_forced_alignment_runner_exception_keeps_its_class_code(
    tmp_path: Path,
    make_error: Callable[[], Exception],
    expected: ProtocolErrorCodeV2,
) -> None:
    def _raise(*_args: object) -> list[Wave2VecWordAlignment]:
        raise make_error()

    response = execute_forced_alignment_request_v2(
        _fa_request(tmp_path),
        ForcedAlignmentExecutionHostV2(wave2vec_runner=_raise),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is expected
    assert response.result is None
