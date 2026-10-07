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
from batchalign.inference.types import Wave2VecWordAlignment

if TYPE_CHECKING:
    from transformers import WhisperProcessor

    from batchalign.inference.types import (
        Wave2VecFAHandle,
        Wave2VecFAResult,
        WhisperFAHandle,
    )


L = logging.getLogger("batchalign.worker")


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
WHISPER_FA_ENGINE_ID = "whisper-fa-large-v2-lexical-window-v2"


def load_whisper_fa(
    model: str = "openai/whisper-large-v2",
    target_sample_rate: SampleRate = 16000,
    *,
    device_policy=None,
) -> WhisperFAHandle:
    """Load a Whisper FA model. Returns a typed handle."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from batchalign.device import resolve_inference_device
    from batchalign.inference.audio import bind_whisper_token_timestamp_extractor
    from batchalign.inference.types import WhisperFAHandle
    from batchalign.worker._progress import (
        HF_ARTIFACTS_WHISPER,
        emit_hf_download_if_missing,
    )

    device = resolve_inference_device(device_policy)

    if device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Multi-GB Whisper FA model download, make the wait visible. Probe
    # the full Whisper artifact set so a partial-cache state (e.g.,
    # tokenizer.json evicted while weights remain) still triggers a
    # download notification.
    emit_hf_download_if_missing(
        model, kind="forced alignment", artifacts=HF_ARTIFACTS_WHISPER
    )

    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model, attn_implementation="eager", torch_dtype=torch_dtype
    )
    bind_whisper_token_timestamp_extractor(whisper_model)
    whisper_model.to(device)
    whisper_model.eval()
    processor = WhisperProcessor.from_pretrained(model)

    return WhisperFAHandle(
        model=whisper_model,
        processor=processor,
        sample_rate=target_sample_rate,
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
) -> list[tuple[str, float]]:
    """Run Whisper forced alignment. Returns [(token_text, timestamp_sec), ...]."""
    import torch
    from transformers.models.whisper.generation_whisper import (
        _dynamic_time_warping as dtw,
    )
    from transformers.models.whisper.generation_whisper import (
        _median_filter as median_filter,
    )

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

    text_idx, time_idx = dtw(-matrix)
    jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
    return window.timed_tokens(handle.processor, text_idx[jumps], time_idx[jumps])


# ---------------------------------------------------------------------------
# Wave2Vec FA load/infer (replaces Wave2VecFAModel class)
# ---------------------------------------------------------------------------


def load_wave2vec_fa(
    target_sample_rate: SampleRate = 16000,
    *,
    device_policy=None,
) -> Wave2VecFAHandle:
    """Load a Wave2Vec FA model. Returns a typed handle."""
    import torchaudio

    from batchalign.device import resolve_inference_device
    from batchalign.inference.types import Wave2VecFAHandle
    from batchalign.worker._progress import emit_download_event

    bundle = torchaudio.pipelines.MMS_FA
    device = resolve_inference_device(device_policy)

    # ``MMS_FA.get_model()`` downloads to torchaudio's hub cache on first use
    # (~1.2 GB). torchaudio prints its own progress to stderr; surface a
    # parallel event on the BA3 protocol channel so every UI sees the wait.
    # Best-effort cache check: torchaudio doesn't expose a clean API for
    # this, so we always emit. False positives (cached, but we still notify)
    # are a much smaller UX cost than silent multi-minute waits.
    emit_download_event(
        stage="downloading_torchaudio_mms_fa",
        user_message=(
            "Downloading Wave2Vec MMS_FA bundle for forced alignment "
            "(one-time, ~1.2 GB; future runs will use the local cache)…"
        ),
    )
    model = bundle.get_model()
    model = model.to(device)
    return Wave2VecFAHandle(model=model, sample_rate=target_sample_rate)


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
    from torchaudio.pipelines import MMS_FA as bundle

    def _build_target_tokens(
        source_words: list[str],
        dictionary: dict[str, int],
    ) -> tuple[torch.Tensor, list[int]]:
        # MMS_FA uses CTC blank index 0 internally, and at least '-' maps there in
        # the live dictionary. Strip blank-mapped chars at the engine boundary
        # instead of changing the shared word model. If a word would become empty,
        # fall back to the wildcard token so word-slot accounting still works.
        wildcard = dictionary["*"]
        blank_index = 0
        transcript_tokens: list[int] = []
        word_lengths: list[int] = []

        for word in source_words:
            word_tokens: list[int] = []
            for char in word.lower():
                token = dictionary.get(char, wildcard)
                if token == blank_index:
                    continue
                word_tokens.append(token)
            if not word_tokens:
                word_tokens = [wildcard]
            transcript_tokens.extend(word_tokens)
            word_lengths.append(len(word_tokens))

        return torch.tensor(transcript_tokens, dtype=torch.int64), word_lengths

    device = next(handle.model.parameters()).device

    audio = audio_chunk.to(device)
    emission, _ = handle.model(audio.unsqueeze(0))
    emission = emission.cpu().detach()

    dictionary = bundle.get_dict()
    transcript, word_lengths = _build_target_tokens(words, dictionary)

    path, scores = AF.forced_align(emission, transcript.unsqueeze(0))
    alignments, scores = path[0], scores[0]
    scores = scores.exp()
    merged_path = AF.merge_tokens(alignments, scores)

    def unflatten(
        list_: list[torch.Tensor], lengths: list[int]
    ) -> list[list[torch.Tensor]]:
        i = 0
        ret = []
        for length in lengths:
            ret.append(list_[i : i + length])
            i += length
        return ret

    word_spans = unflatten(merged_path, word_lengths)
    ratio = audio.size(0) / emission.size(1)
    result: Wave2VecFAResult = [
        Wave2VecWordAlignment(
            word=word,
            interval_ms=(
                int(((spans[0].start * ratio) / handle.sample_rate) * 1000),
                int(((spans[-1].end * ratio) / handle.sample_rate) * 1000),
            ),
            model_score=(
                sum(span.score * (span.end - span.start) for span in spans)
                / sum(span.end - span.start for span in spans)
            ),
        )
        # `strict` because `_build_target_tokens` appends exactly one entry to
        # `word_lengths` per word (empty words fall back to the wildcard token),
        # and `unflatten` returns one span-group per length. That 1:1 is held by
        # two functions agreeing, not by a type, so enforce it here rather than
        # dropping trailing words if either side ever changes.
        for word, spans in zip(words, word_spans, strict=True)
    ]
    return result
