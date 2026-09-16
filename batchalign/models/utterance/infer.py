"""Runtime inference for BA2-style utterance boundary models."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from transformers import AutoTokenizer, BertForTokenClassification

from batchalign.inference._domain_types import LanguageCode
from batchalign.models.utterance.dataset import BOUNDARIES
from batchalign.models.utterance.evidence import (
    BoundaryAction,
    BoundaryAdjacencyPolicy,
    BoundaryProbability,
    ClassifiedBoundaryEvidence,
    ModelShortCircuit,
    NormalizationOmission,
    NormalizedBoundaryEvidence,
    UtteranceBoundaryPrediction,
    apply_boundary_adjacency_policy,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_STRIP_PUNCT_RE = re.compile(r"[.?!,]")

# Sliding-window inference parameters for inputs that exceed the model's
# `max_position_embeddings`. The window covers the inner content; the
# model call adds [CLS] and [SEP], so the actual sequence length passed
# to the model is _WINDOW_INNER_TOKENS + 2, well under the standard 512
# position-embedding ceiling. The 96-token overlap (~20%) matches the
# long-document classification convention; logits are averaged across
# overlapping windows before argmax so cross-window context survives.
_WINDOW_INNER_TOKENS = 480
_WINDOW_OVERLAP_TOKENS = 96


# Cantonese sentence-final particles used to pre-chunk yue inputs before
# BERT classification. Each particle reliably marks an utterance
# boundary in spoken Cantonese, so chunking at particles gives the model
# linguistically-grounded inputs and keeps the BERT context window from
# straddling natural sentence breaks. Multi-character particles come
# first in the iteration so longest-match wins; standalone 㗎 only
# matches when 㗎喇 does not.
_YUE_FINAL_PARTICLES: tuple[tuple[str, ...], ...] = (
    ("㗎", "喇"),  # multi-char first so longest-match wins
    ("呀",),
    ("啦",),
    ("喎",),
    ("嘞",),
    ("囉",),
    ("㗎",),
    ("啊",),
    ("嗯",),
)


def _split_yue_at_particles(words: list[str]) -> list[tuple[int, int]]:
    """Split a Cantonese word list at sentence-final particles.

    Returns a list of half-open `(start, end)` index ranges. Each range
    ends *after* a particle (the particle is included in its chunk),
    except possibly the trailing range when the input does not end on
    a particle. Returns `[]` for empty input.

    The chunking is purely positional; it does not consult
    morphology, only literal-string matches against the particle list.
    """
    n = len(words)
    if n == 0:
        return []
    chunks: list[tuple[int, int]] = []
    chunk_start = 0
    i = 0
    while i < n:
        matched_len = 0
        for particle in _YUE_FINAL_PARTICLES:
            plen = len(particle)
            if i + plen <= n and tuple(words[i : i + plen]) == particle:
                matched_len = plen
                break  # longest-match-first iteration
        if matched_len > 0:
            chunks.append((chunk_start, i + matched_len))
            chunk_start = i + matched_len
            i += matched_len
        else:
            i += 1
    if chunk_start < n:
        chunks.append((chunk_start, n))
    return chunks


def _normalize_utterance_word_mapping(
    words: Sequence[str],
) -> tuple[list[str], list[int]]:
    """Normalize ASR words and keep the original-index mapping."""
    normalized: list[str] = []
    original_indices: list[int] = []
    for original_index, word in enumerate(words):
        lowered = word.lower().strip()
        if not lowered:
            continue
        cleaned = _STRIP_PUNCT_RE.sub("", lowered)
        if cleaned:
            normalized.append(cleaned)
            original_indices.append(original_index)
    return normalized, original_indices


def normalize_utterance_words(words: Sequence[str]) -> list[str]:
    """Normalize ASR words for BA2-style utterance model inference."""
    normalized, _ = _normalize_utterance_word_mapping(words)
    return normalized


class BertUtteranceModel:
    """Typed BA2-style utterance boundary classifier.

    The model predicts one action per input word. BA3 consumes only the
    utterance-boundary decisions as typed group assignments; punctuation
    reconstruction remains a Rust concern.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str,
        model_revision: str,
        lang: LanguageCode | None = None,
    ) -> None:
        # The id is what provenance records; the path is only where the bytes
        # happen to live on this machine and must never be reported, or a
        # stamp would carry a local directory layout.
        self.model_id = model_id
        # The revision the plan pinned and the snapshot resolver verified on
        # disk. It is NOT read back from ``config._commit_hash``: that is a
        # library internal which can be absent, and its absence is exactly why
        # a boundary model's revision used to be optional.
        self.model_revision = model_revision
        # `lang` is optional for backward compatibility with any caller
        # that constructs a model without one. yue-specific particle
        # pre-chunking only fires when `lang == "yue"`.
        self.lang: LanguageCode | None = lang

        # Loaded from the resolved local snapshot, never by name: a load by
        # name is the floating load this pin exists to remove.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path).to(DEVICE)
        self.model.eval()

    def predict_actions(self, words: Sequence[str]) -> list[int]:
        """Predict BA2-style token actions for one pretokenized word sequence.

        Uses sliding-window inference when the tokenized input exceeds
        `max_position_embeddings`. Short inputs take a single-pass path;
        long inputs get classified across multiple overlapping windows
        with logit averaging at overlap positions before argmax.
        """
        return [
            action.label_index
            for action in self.predict_boundary_evidence(words).applied_actions
        ]

    def predict_boundary_evidence(
        self, words: Sequence[str]
    ) -> UtteranceBoundaryPrediction:
        """Return a complete typed prediction parallel to original words."""

        normalized_words, original_indices = _normalize_utterance_word_mapping(words)
        if len(normalized_words) <= 1:
            normalized_index_set = set(original_indices)
            return UtteranceBoundaryPrediction(
                model_id=self.model_id,
                model_revision=self.model_revision,
                word_evidence=tuple(
                    ModelShortCircuit()
                    if original_index in normalized_index_set
                    else NormalizationOmission()
                    for original_index in range(len(words))
                ),
            )

        normalized_evidence = self._predict_normalized_evidence(normalized_words)

        # Existing post-processing drops the earlier of two adjacent non-zero
        # labels. Evidence retains both the raw decision and applied result so
        # this legacy policy can be evaluated independently of the model.
        applied_actions = apply_boundary_adjacency_policy(
            normalized_evidence,
            BoundaryAdjacencyPolicy.SUPPRESS_EARLIER_ADJACENT_NONORDINARY_V1,
        )

        normalized_by_original_index = {
            original_index: ClassifiedBoundaryEvidence(
                raw_action=normalized_evidence[normalized_index].raw_action,
                applied_action=applied_actions[normalized_index],
                boundary_probability=normalized_evidence[
                    normalized_index
                ].boundary_probability,
            )
            for normalized_index, original_index in enumerate(original_indices)
        }
        return UtteranceBoundaryPrediction(
            model_id=self.model_id,
            model_revision=self.model_revision,
            word_evidence=tuple(
                normalized_by_original_index.get(
                    original_index, NormalizationOmission()
                )
                for original_index in range(len(words))
            ),
        )

    def _predict_normalized_evidence(
        self, normalized_words: list[str]
    ) -> list[NormalizedBoundaryEvidence]:
        """Return typed raw evidence for every normalized input word."""

        if self.lang == "yue":
            ranges = _split_yue_at_particles(normalized_words)
            if len(ranges) > 1:
                # Multi-chunk path: classify each chunk independently,
                # concatenate the per-word action lists. Particle
                # boundaries are NOT explicitly forced as splits, we
                # trust the model to predict them, which it should
                # because particles are reliable boundary markers in
                # the training distribution. Forcing chunk-boundary
                # splits is a separate refinement we can layer on
                # later if empirical evidence justifies it.
                evidence: list[NormalizedBoundaryEvidence] = []
                for start, end in ranges:
                    chunk_words = normalized_words[start:end]
                    evidence.extend(self._classify_chunk_evidence(chunk_words))
                return evidence
            # Single chunk (no particles found, or all chunked into
            # one): fall through to the standard path.
        return self._classify_chunk_evidence(normalized_words)

    def _classify_chunk_evidence(
        self, normalized_words: list[str]
    ) -> list[NormalizedBoundaryEvidence]:
        """Classify one normalized chunk and retain boundary probabilities."""

        # Tokenize without special tokens so we can manage [CLS]/[SEP]
        # per window. word_ids(0) returns one entry per WordPiece token.
        full_encoding = self.tokenizer(
            [normalized_words],
            return_tensors="pt",
            is_split_into_words=True,
            add_special_tokens=False,
        )
        full_input_ids = full_encoding.input_ids[0]  # 1D tensor of token ids
        full_word_ids = full_encoding.word_ids(0)
        n_tokens = full_input_ids.shape[0]
        n_words = len(normalized_words)
        max_pos = int(self.model.config.max_position_embeddings)
        n_classes = int(self.model.config.num_labels)

        # `_WINDOW_INNER_TOKENS` is the inner content; the model call adds
        # [CLS] and [SEP], so the actual sequence length is
        # `inner + 2`. Cap inner at `max_pos - 2` to stay safe even if
        # someone configures a model with a smaller position-embedding
        # table.
        inner_window = min(_WINDOW_INNER_TOKENS, max_pos - 2)
        if inner_window <= _WINDOW_OVERLAP_TOKENS:
            # Pathological config: fall back to non-overlapping chunks
            # (still correct, just no overlap reconciliation).
            stride = inner_window
        else:
            stride = inner_window - _WINDOW_OVERLAP_TOKENS

        # Accumulators over the full token sequence. Logits are summed
        # across overlapping windows; counts track how many windows
        # contributed to each position so we can average at the end.
        accumulated = torch.zeros((n_tokens, n_classes), dtype=torch.float32)
        counts = torch.zeros(n_tokens, dtype=torch.float32)

        # Build [CLS] / [SEP] singleton tensors once; reused per window.
        cls_tensor = torch.tensor(
            [self.tokenizer.cls_token_id], dtype=full_input_ids.dtype
        )
        sep_tensor = torch.tensor(
            [self.tokenizer.sep_token_id], dtype=full_input_ids.dtype
        )

        # Iterate windows. For inputs ≤ inner_window this loop runs once
        # and the result equals the single-pass behavior of the prior
        # implementation modulo numerical-precision noise in the
        # accumulate-then-divide vs direct-argmax paths (counts is 1 at
        # every position so the division is a no-op).
        start = 0
        while True:
            end = min(start + inner_window, n_tokens)
            window_inner_ids = full_input_ids[start:end]
            n_inner = int(window_inner_ids.shape[0])

            # Build [CLS] inner [SEP] for this window's model call.
            window_input_ids = (
                torch.cat([cls_tensor, window_inner_ids, sep_tensor])
                .unsqueeze(0)
                .to(DEVICE)
            )
            attention_mask = torch.ones_like(window_input_ids)

            window_output = self.model(
                input_ids=window_input_ids,
                attention_mask=attention_mask,
            )
            # Shape: (1, n_inner + 2, n_classes). Strip [CLS] (index 0)
            # and [SEP] (last index) to align with the inner positions.
            window_logits = (
                window_output.logits[0].detach().to("cpu", dtype=torch.float32)
            )
            inner_logits = window_logits[1 : n_inner + 1]

            accumulated[start:end] += inner_logits
            counts[start:end] += 1.0

            if end >= n_tokens:
                break
            start += stride

        # Average and argmax at the WordPiece-token level.
        averaged = accumulated / counts.unsqueeze(-1)
        token_actions = torch.argmax(
            averaged, dim=1
        ).tolist()  # list[int], len == n_tokens

        # Map per-token actions back to per-word actions, taking the
        # action of the first WordPiece token of each word. Matches the
        # prior implementation's per-word reduction.
        raw_actions = [BoundaryAction.ORDINARY] * n_words
        boundary_probabilities = [BoundaryProbability.from_float(0.0)] * n_words
        previous_word_idx: int | None = None
        for token_idx, word_idx in enumerate(full_word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            previous_word_idx = word_idx
            raw_actions[word_idx] = BoundaryAction.from_label_index(
                token_actions[token_idx]
            )
            probabilities = torch.softmax(averaged[token_idx], dim=0)
            boundary_probability = float(probabilities[BOUNDARIES].sum().item())
            boundary_probabilities[word_idx] = BoundaryProbability.from_float(
                boundary_probability
            )

        return [
            NormalizedBoundaryEvidence(
                raw_action=raw_action,
                boundary_probability=boundary_probability,
            )
            for raw_action, boundary_probability in zip(
                raw_actions, boundary_probabilities, strict=True
            )
        ]

    def predict_assignments(self, words: Sequence[str]) -> list[int]:
        """Predict typed utterance-group assignments for one word sequence."""
        return list(self.predict_boundary_evidence(words).assignments)


__all__ = [
    "BertUtteranceModel",
    "BoundaryAction",
    "ClassifiedBoundaryEvidence",
    "ModelShortCircuit",
    "NormalizationOmission",
    "UtteranceBoundaryPrediction",
    "normalize_utterance_words",
]
