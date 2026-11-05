"""Speaker-model loads for ``ensure_task``.

The speaker task loads on first use: the pool sends ``ensure_task("speaker")``
before a worker's first speaker request, under a loading stall window that
download growth re-arms. The overrides name the one speaker model the plan
selects (``GpuModelPlan``'s speaker slot, Rust side), which
``task_load`` parses into a :class:`~batchalign.worker._types.TaskLoad`; the
model is not worker identity, so one GPU worker serves every speaker model and
each model is its own load. Each function here builds exactly one model and
returns the host whose runner closes over it. pyannoteAI is hosted and builds
nothing; its load returns the host that calls the API.
"""

from __future__ import annotations

import logging
from functools import partial

from batchalign.worker._loads import LoadedDiarization, LoadedSpeakerEmbedding
from batchalign.worker._types_v2 import SpeakerBackendV2, SpeakerEmbeddingBackendV2

L = logging.getLogger("batchalign.worker")


def load_local_pyannote_task() -> LoadedDiarization:
    """Build the local pyannote diarization pipeline and its host."""
    from batchalign.inference.pyannote_local import load_local_pyannote_pipeline
    from batchalign.inference.speaker import infer_local_pyannote_prepared_audio

    host = LoadedDiarization(
        backend=SpeakerBackendV2.PYANNOTE,
        runner=partial(
            infer_local_pyannote_prepared_audio,
            pipeline=load_local_pyannote_pipeline(),
        ),
    )
    L.info("Loaded the local pyannote diarization pipeline")
    return host


def hosted_speaker_task() -> LoadedDiarization:
    """The pyannoteAI host: a hosted API, so nothing is loaded."""
    from batchalign.inference.speaker import infer_pyannote_ai_prepared_audio

    L.info("No speaker model to load: the plan's engine is hosted")
    return LoadedDiarization(
        backend=SpeakerBackendV2.PYANNOTE_AI,
        runner=infer_pyannote_ai_prepared_audio,
    )


def load_speaker_embedding_task() -> LoadedSpeakerEmbedding:
    """Build the speaker embedding model and its host."""
    from batchalign.inference.speaker_embedding import load_speaker_embedding_model
    from batchalign.worker._speaker_embedding_v2 import span_runner

    model = load_speaker_embedding_model()
    L.info("Loaded the speaker embedding model at %s", model.revision())
    return LoadedSpeakerEmbedding(
        backend=SpeakerEmbeddingBackendV2.PYANNOTE,
        runner=span_runner(model),
    )
