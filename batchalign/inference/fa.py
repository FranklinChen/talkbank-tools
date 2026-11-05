"""Forced alignment inference: audio+text -> timings.

Pure inference, no CHAT, no caching, no pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, model_validator

from batchalign.inference._domain_types import (
    AudioPath,
    ConfidenceScore,
    SampleRate,
    TimestampMs,
)
from batchalign.inference.types import Wave2VecWordAlignment, WhisperModelIdentity

if TYPE_CHECKING:
    from transformers import WhisperProcessor

    from batchalign.inference.types import (
        Wave2VecFAHandle,
        Wave2VecFAResult,
        WhisperFAHandle,
    )
    from batchalign.inference.whisper_attention import WhisperAttentionCapture
    from batchalign.worker._model_loading.pinned_hub import TransformersSnapshot


L = logging.getLogger("batchalign.worker")

MMS_FA_ALGORITHM_ID = "wave2vec-fa-mms-v2-label-complete"


# ---------------------------------------------------------------------------
# Pydantic models (FA payload/response)
# ---------------------------------------------------------------------------


class FaInferItem(BaseModel):
    """A single FA inference request item."""

    words: list[str]
    word_ids: list[str]
    word_utterance_indices: list[int]
    word_utterance_word_indices: list[int]
    audio_path: AudioPath
    audio_start_ms: TimestampMs
    audio_end_ms: TimestampMs

    @model_validator(mode="after")
    def validate_parallel_arrays(self) -> FaInferItem:
        expected = len(self.words)
        if len(self.word_ids) != expected:
            raise ValueError(
                f"word_ids length mismatch: expected {expected}, got {len(self.word_ids)}"
            )
        if len(self.word_utterance_indices) != expected:
            raise ValueError(
                f"word_utterance_indices length mismatch: expected {expected}, "
                f"got {len(self.word_utterance_indices)}"
            )
        if len(self.word_utterance_word_indices) != expected:
            raise ValueError(
                f"word_utterance_word_indices length mismatch: expected {expected}, "
                f"got {len(self.word_utterance_word_indices)}"
            )
        return self


class FaIndexedTiming(BaseModel):
    """A word-level timing result."""

    start_ms: TimestampMs
    end_ms: TimestampMs
    confidence: ConfidenceScore | None = None


class Wave2VecIndexedResponse(BaseModel):
    """Wave2Vec FA output: indexed timings aligned to input words."""

    indexed_timings: list[FaIndexedTiming | None]


# ---------------------------------------------------------------------------
# Whisper FA load/infer
# ---------------------------------------------------------------------------

# Namespace the callback's acoustic-window algorithm, not only its checkpoint.
# Old padded-grid token evidence must not enter this implementation's cache.
WHISPER_FA_ENGINE_ID = "whisper-fa-large-v2-open-begin-v3"


WHISPER_FA_MODEL_ID = "openai/whisper-large-v2"
"""The hub model the Whisper FA engine loads."""


def load_whisper_fa(
    snapshot: TransformersSnapshot,
    target_sample_rate: SampleRate = 16000,
    *,
    device_policy=None,
) -> WhisperFAHandle:
    """Load a Whisper FA model from ``snapshot``. Returns a typed handle
    whose identity is the snapshot's model and commit.

    The snapshot is the identity: it used to be found by a separate
    ``config.json`` download before a load by name, two hub calls that could
    disagree, and a local directory loaded with no identity at all.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from batchalign.device import resolve_inference_device
    from batchalign.inference.audio import bind_whisper_token_timestamp_extractor
    from batchalign.inference.types import WhisperFAHandle

    device = resolve_inference_device(device_policy)

    if device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        snapshot.path,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    )
    processor = WhisperProcessor.from_pretrained(snapshot.path)
    bind_whisper_token_timestamp_extractor(whisper_model)
    whisper_model.to(device)
    whisper_model.eval()

    return WhisperFAHandle(
        model=whisper_model,
        processor=processor,
        sample_rate=target_sample_rate,
        model_identity=WhisperModelIdentity(snapshot.model_id, snapshot.commit),
    )


class _WhisperAlignmentWindow:
    """Label-bound attention over lexical rows and the actual waveform.

    Conditioning rows cannot constrain a lexical alignment. The terminal row
    remains an acoustic end sentinel; prefix labels retain explicit zero times
    in the worker response. This owner pairs row selection with output labels.
    """

    def __init__(
        self,
        audio: torch.Tensor,
        sample_rate: SampleRate,
        hop_length: int,
        weights: torch.Tensor,
        tokens: torch.Tensor,
        prefix_tokens: list[int],
        eos_token_id: int,
    ) -> None:
        if audio.ndim != 1 or sample_rate <= 0 or hop_length <= 0:
            raise ValueError(
                "Whisper FA requires mono audio and positive frame geometry"
            )
        if tokens.ndim != 1 or weights.ndim != 3 or weights.shape[-2] != tokens.numel():
            raise ValueError("Whisper FA attention rows do not match decoder labels")
        self._tokens = tuple(int(token) for token in tokens)
        self._prefix_length = len(prefix_tokens)
        if (
            not prefix_tokens
            or self._tokens[: self._prefix_length] != tuple(prefix_tokens)
            or len(self._tokens) <= self._prefix_length + 1
            or self._tokens[-1] != eos_token_id
        ):
            raise ValueError(
                "Whisper FA requires the tokenizer prefix, lexical labels and terminal label"
            )
        # Whisper's encoder downsamples mel frames by two, as in the shared
        # token-timestamp extractor. The feature extractor's hop is the model's
        # geometry, not a hard-coded recording duration or fabricated endpoint.
        frames = audio.numel() // hop_length // 2
        if frames == 0:
            raise ValueError("Whisper FA audio contains no complete encoder frame")
        if frames > weights.shape[-1]:
            raise ValueError("Whisper FA audio exceeds the supplied encoder attention")
        self._weights = weights[:, self._prefix_length :, :frames]
        self._time_precision = 2 * hop_length / sample_rate

    def normalized(self) -> torch.Tensor:
        std, mean = torch.std_mean(self._weights, dim=-2, keepdim=True, unbiased=False)
        return (self._weights - mean) / std

    def timed_tokens(
        self, processor: WhisperProcessor, rows: np.ndarray, indices: np.ndarray
    ) -> list[tuple[str, float]]:
        if (
            indices.ndim != 1
            or not np.array_equal(
                rows, np.arange(len(self._tokens) - self._prefix_length)
            )
            or len(indices) != len(rows)
        ):
            raise ValueError(
                "Whisper FA DTW did not cover every admitted decoder label"
            )
        if (
            np.any(indices < 0)
            or np.any(indices >= self._weights.shape[-1])
            or np.any(np.diff(indices) < 0)
        ):
            raise ValueError("Whisper FA DTW escaped its admitted acoustic window")
        times = np.concatenate(
            (np.zeros(self._prefix_length), indices * self._time_precision)
        )
        return [
            (processor.decode(token), float(time))
            for token, time in zip(self._tokens, times, strict=True)
        ]


def infer_whisper_fa(
    handle: WhisperFAHandle,
    audio_chunk: torch.Tensor,
    text: str,
    *,
    capture: WhisperAttentionCapture | None = None,
) -> list[tuple[str, float]]:
    """Run Whisper forced alignment. Returns [(token_text, timestamp_sec), ...]."""
    import torch
    from transformers.models.whisper.generation_whisper import (
        _median_filter as median_filter,
    )

    from batchalign_core import whisper_lexical_frame_indices

    device = next(handle.model.parameters()).device

    # `text` arrives already shaped. Whether it is space-joined or spelled one
    # character per token is `FaTextModeV2`, applied in Rust by `join_fa_words`
    # before this is called. This used to reshape it a second time behind a
    # `pauses` flag, which is the duplication that fold removed.
    features = handle.processor(
        audio=audio_chunk,
        text=text,
        sampling_rate=handle.sample_rate,
        return_tensors="pt",
    )
    tokens = features["labels"][0]

    with torch.inference_mode():
        output = handle.model(**features.to(device), output_attentions=True)

    cross_attentions = torch.cat(output.cross_attentions).cpu()
    weights = torch.stack(
        [
            cross_attentions[layer][head]
            for layer, head in handle.model.generation_config.alignment_heads
        ]
    )

    window = _WhisperAlignmentWindow(
        audio_chunk,
        handle.sample_rate,
        handle.processor.feature_extractor.hop_length,
        weights,
        tokens,
        handle.processor.tokenizer.prefix_tokens,
        handle.processor.tokenizer.eos_token_id,
    )
    weights = median_filter(
        window.normalized(), handle.model.config.median_filter_width
    )
    matrix = weights.mean(dim=0)

    indices = np.asarray(
        whisper_lexical_frame_indices(-matrix.float().numpy()), dtype=np.int64
    )
    result = window.timed_tokens(handle.processor, np.arange(matrix.shape[0]), indices)
    if capture is not None:
        identity = handle.model_identity
        capture.record(
            token_ids=tokens.tolist(),
            token_texts=[handle.processor.decode(token) for token in tokens],
            prefix_length=window._prefix_length,
            heads=[
                (head[0], head[1])
                for head in handle.model.generation_config.alignment_heads
            ],
            raw_attention=window._weights.float().numpy(),
            costs=-matrix.float().numpy(),
            row_entries=indices,
            hop_length=handle.processor.feature_extractor.hop_length,
            median_width=handle.model.config.median_filter_width,
            model_id=identity.model_id,
            model_revision=identity.revision,
        )
    return result


# ---------------------------------------------------------------------------
# Wave2Vec FA load/infer (replaces Wave2VecFAModel class)
# ---------------------------------------------------------------------------


def load_wave2vec_fa(
    target_sample_rate: SampleRate = 16000,
    *,
    device_policy=None,
) -> Wave2VecFAHandle:
    """Load a Wave2Vec FA model. Returns a typed handle."""
    from torchaudio.pipelines import MMS_FA

    from batchalign.device import resolve_inference_device
    from batchalign.inference.fa_ctc_resources import MmsAlphabet
    from batchalign.inference.types import Wave2VecFAHandle

    device = resolve_inference_device(device_policy)
    model = mms_fa_model(purpose="forced alignment").to(device)
    return Wave2VecFAHandle(
        model=model,
        sample_rate=target_sample_rate,
        alphabet=MmsAlphabet.admit(MMS_FA.get_dict()),
    )


def mms_fa_model(*, purpose: str) -> torch.nn.Module:
    """The torchaudio MMS_FA model with its star token, its first download
    watched.

    Always with the star (``with_star=True``, torchaudio's default): forced
    alignment maps characters its dictionary lacks to the ``*`` wildcard
    (``infer_wave2vec_fa``), and placement confirmation lets it absorb
    unspoken material, so neither can use the model without that output.

    ``MMS_FA.get_model()`` downloads to torch hub's cache on first use
    (~1.2 GB). Under ``download_liveness`` that is announced only when bytes
    arrive (a cached bundle says nothing) and its growth reported, which keeps
    it inside the worker's startup stall window. Shared by forced alignment
    (``load_wave2vec_fa``) and placement confirmation (``load_fa_confirm``),
    so the bundle is watched and named once.
    """
    import torchaudio

    from batchalign.worker._download_liveness import TorchHubBundle, download_liveness

    with download_liveness(
        TorchHubBundle("mms_fa"),
        what=f"the Wave2Vec MMS_FA bundle for {purpose} (~1.2 GB)",
    ):
        return torchaudio.pipelines.MMS_FA.get_model(with_star=True)


def infer_wave2vec_fa(
    handle: Wave2VecFAHandle,
    audio_chunk: torch.Tensor,
    words: list[str],
) -> Wave2VecFAResult:
    """Run Wave2Vec FA, retaining indexed intervals and CTC path scores.

    The score is the duration-weighted mean of the merged token scores for one
    word, matching TorchAudio's multilingual forced-alignment tutorial. It is
    model evidence, not a calibrated probability that the boundary is right.
    """
    import torch
    import torchaudio.functional as AF

    from batchalign.inference.fa_ctc_resources import MmsTargetSequence

    targets = MmsTargetSequence.from_words(words, handle.alphabet)
    device = next(handle.model.parameters()).device

    audio = audio_chunk.to(device)
    with torch.inference_mode():
        emission, _ = handle.model(audio.unsqueeze(0))
        emission = emission.cpu().detach()

    transcript = torch.tensor(targets.ids, dtype=torch.int64)

    path, scores = AF.forced_align(emission, transcript.unsqueeze(0))
    alignments, scores = path[0], scores[0]
    scores = scores.exp()
    merged_path = AF.merge_tokens(alignments, scores)

    aligned = targets.bind_spans(merged_path)
    ratio = audio.size(0) / emission.size(1)
    result: Wave2VecFAResult = [
        Wave2VecWordAlignment(
            word=word.source,
            interval_ms=(
                int(((word.spans[0].start * ratio) / handle.sample_rate) * 1000),
                int(((word.spans[-1].end * ratio) / handle.sample_rate) * 1000),
            ),
            model_score=(
                sum(span.score * (span.end - span.start) for span in word.spans)
                / sum(span.end - span.start for span in word.spans)
            ),
        )
        for word in aligned.words
    ]
    return result
