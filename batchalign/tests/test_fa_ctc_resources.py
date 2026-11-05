"""CTC admission is expressed in actual labels, repeats, frames and bytes."""

from __future__ import annotations

import pytest
import torch

from batchalign.inference.fa_ctc_resources import (
    AdmittedCtcShape,
    CtcFrameCapacityError,
    MmsAlphabet,
    MmsTargetSequence,
)

ALPHABET = MmsAlphabet.admit({"-": 0, "a": 1, "b": 2, "*": 3})


def test_targets_own_the_same_word_partition_used_for_native_alignment() -> None:
    dictionary = {"-": 0, "a": 1, "b": 2, "*": 3}
    alphabet = MmsAlphabet.admit(dictionary)
    dictionary["a"] = 9
    targets = MmsTargetSequence.from_words(["aa", "b", "-"], alphabet)
    assert targets.ids == (1, 1, 2, 3)
    assert targets.word_lengths == (2, 1, 1)
    assert targets.minimum_frames == 5


def test_actual_ctc_frame_constraint_has_a_distinguishing_boundary() -> None:
    targets = MmsTargetSequence.from_words(["aa", "b"], ALPHABET)
    with pytest.raises(CtcFrameCapacityError) as failure:
        AdmittedCtcShape.admit(targets, frames=3, classes=4, scalar_bytes=4)
    assert failure.value.available_frames == 3
    assert failure.value.required_frames == 4
    admitted = AdmittedCtcShape.admit(targets, frames=4, classes=4, scalar_bytes=4)
    assert admitted.targets is targets


def test_cpu_native_workspace_comes_from_its_allocation_shapes() -> None:
    targets = MmsTargetSequence.from_words(["ab"], ALPHABET)
    admitted = AdmittedCtcShape.admit(targets, frames=20, classes=4, scalar_bytes=4)
    # Native CPU source: two alpha rows of S=2L+1 scalars; T*S int8
    # backpointers; an int64 path per frame. Sizes are types, not time limits.
    states = 2 * len(targets.ids) + 1
    expected = (
        2 * states * 4 + 20 * states * torch.int8.itemsize + 20 * torch.int64.itemsize
    )
    assert admitted.native_workspace_bytes == expected
    larger = AdmittedCtcShape.admit(targets, frames=40, classes=4, scalar_bytes=4)
    assert larger.native_workspace_bytes > admitted.native_workspace_bytes


@pytest.mark.parametrize("words", [[], [""]])
def test_missing_requested_words_are_rejected(words: list[str]) -> None:
    with pytest.raises(ValueError):
        MmsTargetSequence.from_words(words, ALPHABET)


def test_a_label_outside_the_actual_emission_vocabulary_is_rejected() -> None:
    targets = MmsTargetSequence.from_words(
        ["a"], MmsAlphabet.admit({"-": 0, "a": 9, "*": 1})
    )
    with pytest.raises(ValueError, match="vocabulary"):
        AdmittedCtcShape.admit(targets, frames=20, classes=4, scalar_bytes=4)


@pytest.mark.parametrize(
    "dictionary", [{}, {"*": 0}, {"*": -1}, {"*": 1, "a": -1}, {"*": True}]
)
def test_alphabet_rejects_malformed_native_vocabulary(
    dictionary: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        MmsAlphabet.admit(dictionary)
