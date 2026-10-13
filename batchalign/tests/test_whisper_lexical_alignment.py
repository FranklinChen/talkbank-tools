"""Real PyO3 numeric boundary, independent of model downloads and test doubles."""

import numpy as np
import pytest

from batchalign_core import whisper_lexical_frame_indices


def test_open_begin_preserves_every_label_and_can_skip_audio_prefix():
    costs = np.asarray([[9, 9, -4, 9], [9, 9, 9, -4]], dtype=np.float32)
    assert whisper_lexical_frame_indices(costs) == [2, 3]
    # The bridge must handle non-contiguous arrays in row-major logical order.
    assert whisper_lexical_frame_indices(np.asfortranarray(costs)) == [2, 3]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_costs_are_refused(value):
    with pytest.raises(ValueError, match="non-finite"):
        whisper_lexical_frame_indices(np.asarray([[value]], dtype=np.float32))


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_empty_costs_are_refused(shape):
    with pytest.raises(ValueError, match="empty"):
        whisper_lexical_frame_indices(np.zeros(shape, dtype=np.float32))


def test_repeated_rows_are_not_deduplicated_and_ties_remain_bounded():
    assert whisper_lexical_frame_indices(np.zeros((4, 2), dtype=np.float32)) == [
        0,
        0,
        0,
        1,
    ]
