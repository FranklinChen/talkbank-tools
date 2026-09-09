"""Thin worker-protocol V2 forced-alignment wrapper.

Rust now owns the prepared-artifact reads, request validation, backend dispatch,
and typed V2 response shaping for the worker FA boundary. Python stays only at
the model-host callback edge.

**See also:** `../../INTERFACE_MAP.md` section "3. Forced Alignment V2" for:
- Rust FFI function: `crates/batchalign-pyo3/src/worker_fa_exec.rs::execute_forced_alignment_request_v2()`
- Full Rust/Python responsibility split and input/output contracts.

Concurrency note
----------------
The GPU worker serves V2 requests via ``_serve_stdio_concurrent(max_threads=4)``,
which dispatches up to 4 requests simultaneously through a ``ThreadPoolExecutor``.
PyTorch releases the GIL during compute, so threads truly run in parallel on the
CPU side.  However, the torchaudio MMS_FA / ``forced_align`` kernel is **not
thread-safe** for concurrent CPU execution from multiple threads, concurrent
calls cause SIGSEGV/SIGABRT in LibTorch, crashing the entire worker process and
failing every pending request in the job.

``_fa_inference_lock`` serializes all calls to ``execute_forced_alignment_request_v2``
so that only one FA inference runs at a time.  The lock is module-level (one per
worker process), which is the correct granularity, each GPU worker process is a
single Python interpreter serving one language's models.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from batchalign.worker._types_v2 import (
    ExecuteRequestV2,
    ExecuteResponseV2,
    ForcedAlignmentRequestV2,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from batchalign.inference.languages.cantonese._cantonese_fa import CantoneseFaHost
    from batchalign.inference.qwen_forced_alignment import (
        IndexedWordTimingRow,
        QwenFaHost,
    )
    from batchalign.inference.types import (
        Wave2VecFAHandle,
        Wave2VecFAResult,
        WhisperFAHandle,
    )

# Serialize all FA inference calls within a single worker process.
#
# The torchaudio MMS_FA forced_align kernel is not thread-safe for concurrent
# CPU execution. _serve_stdio_concurrent runs a ThreadPoolExecutor(max_threads=4)
# and PyTorch releases the GIL during compute, so without this lock up to 4
# threads can race inside the native C++ kernel simultaneously, causing
# SIGSEGV/SIGABRT. One lock per process is the right granularity because each
# worker process owns exactly one set of loaded models for one language.
_fa_inference_lock = threading.Lock()


class PreparedFaPayloadV2(BaseModel):
    """Prepared FA payload written by the Rust-side V2 request builder."""

    words: list[str]
    word_ids: list[str]
    word_utterance_indices: list[int]
    word_utterance_word_indices: list[int]


@dataclass(frozen=True, slots=True)
class ForcedAlignmentExecutionHostV2:
    """Injected FA execution hooks for the live V2 path."""

    whisper_runner: Callable[[np.ndarray, str], list[tuple[str, float]]] | None = None
    wave2vec_runner: Callable[[np.ndarray, list[str]], Wave2VecFAResult] | None = None
    canto_runner: (
        Callable[
            [np.ndarray, PreparedFaPayloadV2, ForcedAlignmentRequestV2],
            Wave2VecFAResult,
        ]
        | None
    ) = None
    # Same call shape as `wave2vec_runner`: the Qwen3 host is handed the word
    # list and rebuilds its own transcript, so nothing about the request beyond
    # the words reaches it. Its rows differ from wave2vec's in one way that
    # matters: `interval_ms` may be `None`, because the Qwen3 aligner can time
    # some of a group's words and not others, and says which.
    qwen3_runner: (
        Callable[[np.ndarray, list[str]], list[IndexedWordTimingRow]] | None
    ) = None


def build_default_fa_execution_host_v2(
    *,
    whisper_model: WhisperFAHandle | None,
    wave2vec_model: Wave2VecFAHandle | None,
    canto_host: CantoneseFaHost | None,
    qwen_host: QwenFaHost | None,
) -> ForcedAlignmentExecutionHostV2:
    """Build the live V2 FA host from already loaded model handles.

    `canto_host` and `qwen_host` are passed in rather than read from module
    globals so the seam stays explicit and a test can supply a fake.
    """

    import torch

    from batchalign.inference.fa import infer_wave2vec_fa, infer_whisper_fa

    def _as_tensor(audio: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(audio, dtype=np.float32))

    whisper_runner = None
    if whisper_model is not None:

        def _run_whisper(audio: np.ndarray, text: str) -> list[tuple[str, float]]:
            # `text` arrives shaped: Rust applied the request's `FaTextModeV2`,
            # including the character-spacing that `--pauses` selects. This
            # used to take a `pauses` flag and reshape the text a second time
            # here, which split one decision across two languages.
            return infer_whisper_fa(whisper_model, _as_tensor(audio), text)

        whisper_runner = _run_whisper

    wave2vec_runner = None
    if wave2vec_model is not None:

        def _run_wave2vec(
            audio: np.ndarray,
            words: list[str],
        ) -> Wave2VecFAResult:
            return infer_wave2vec_fa(wave2vec_model, _as_tensor(audio), words)

        wave2vec_runner = _run_wave2vec

    canto_runner = None
    if canto_host is not None:

        def _run_canto(
            audio: np.ndarray,
            payload: PreparedFaPayloadV2,
            _request: ForcedAlignmentRequestV2,
        ) -> Wave2VecFAResult:
            return canto_host.align_words(_as_tensor(audio), payload.words)

        canto_runner = _run_canto

    qwen3_runner = None
    if qwen_host is not None:

        def _run_qwen3(
            audio: np.ndarray, words: list[str]
        ) -> list[IndexedWordTimingRow]:
            # The host owns the whole verb (segment, align, fold onto our
            # words), so nothing is reassembled here. Times come back
            # WINDOW-relative on purpose: Rust adds the group offset.
            #
            # `to_indexed_timings` is the one lossy step, and it is called
            # here and nowhere else: the fold knows WHY a word is untimed and
            # the worker protocol's row has no slot for it, so the reason
            # stops at this line rather than at an unmarked boundary.
            return qwen_host.align_words(audio, words).to_indexed_timings()

        qwen3_runner = _run_qwen3

    return ForcedAlignmentExecutionHostV2(
        whisper_runner=whisper_runner,
        wave2vec_runner=wave2vec_runner,
        canto_runner=canto_runner,
        qwen3_runner=qwen3_runner,
    )


def _wrap_canto_runner(
    runner: Callable[
        [np.ndarray, PreparedFaPayloadV2, ForcedAlignmentRequestV2], object
    ]
    | None,
) -> Callable[[np.ndarray, str, str], object] | None:
    """Adapt the typed Cantonese host hook to the Rust bridge shape."""

    if runner is None:
        return None

    def _run(audio: np.ndarray, payload_json: str, request_json: str) -> object:
        return runner(
            audio,
            PreparedFaPayloadV2.model_validate_json(payload_json),
            ForcedAlignmentRequestV2.model_validate_json(request_json),
        )

    return _run


def execute_forced_alignment_request_v2(
    request: ExecuteRequestV2,
    host: ForcedAlignmentExecutionHostV2,
) -> ExecuteResponseV2:
    """Execute one live V2 forced-alignment request through the Rust bridge.

    The ``_fa_inference_lock`` is held for the duration of this call to
    prevent concurrent FA inference within the same worker process.  See the
    module-level docstring for the full rationale.
    """
    import batchalign_core

    with _fa_inference_lock:
        return ExecuteResponseV2.model_validate_json(
            batchalign_core.execute_forced_alignment_request_v2(
                request,
                host.whisper_runner,
                host.wave2vec_runner,
                _wrap_canto_runner(host.canto_runner),
                host.qwen3_runner,
            )
        )
