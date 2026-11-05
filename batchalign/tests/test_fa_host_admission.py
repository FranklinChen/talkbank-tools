"""Production FA hosts cannot be assembled from independent callback facts."""

from __future__ import annotations

import pytest

from batchalign.inference.fa_virtual_resources import UnavailableFaPreparation
from batchalign.worker._loads import LoadedWhisperFa, LoadedWordFa
from batchalign.worker._types_v2 import FaBackendV2


def test_word_host_requires_its_model_specific_loading_transition() -> None:
    with pytest.raises(TypeError):
        LoadedWordFa(  # type: ignore[call-arg]
            backend=FaBackendV2.WAVE2VEC,
            runner=lambda _audio, _words: [],
            prepare=UnavailableFaPreparation("a different backend's preparation"),
            engine="unrelated-model",
        )


def test_whisper_host_requires_its_model_specific_loading_transition() -> None:
    with pytest.raises(TypeError):
        LoadedWhisperFa(  # type: ignore[call-arg]
            runner=lambda _audio, _text, _capture: [],
            prepare=UnavailableFaPreparation("a different backend's preparation"),
            engine="unrelated-model",
        )
