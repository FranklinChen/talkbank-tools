"""FunASR helpers for built-in HK/Cantonese engines."""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from typing import Any, assert_never

from batchalign.inference._domain_types import LanguageCode
from batchalign.worker._model_errors import ModelOutputInvalidError
from batchalign.worker._model_loading.modelscope_hub import ModelScopeSnapshot
from batchalign.worker._model_loading.pinned_hub import RepositorySnapshot

from ._asr_types import AsrGenerationPayload, AsrMonologue, TimedWord

L = logging.getLogger("batchalign.hk.funaudio")


@dataclass(frozen=True)
class FunAsrSegment:
    """Parsed output from one FunASR result.

    FunASR returns raw dicts: this type captures the fields we use and
    validates at the boundary so downstream code never touches raw dicts.

    ``timestamp`` holds one ``[start_ms, end_ms]`` entry per recognized unit,
    and the units live in a checkpoint-specific list that Rust pairs it with
    (``funasr_projection`` in batchalign-pyo3): SenseVoice's ``words``, or
    Paraformer's pre-punctuation ``raw_text`` (requested with
    ``return_raw_text=True``). Paraformer's ``text`` is punctuated and
    space-free, so it cannot be split back into units.
    """

    text: str
    timestamp: list[list[int | float]] = field(default_factory=list)
    words: list[str] | None = None
    raw_text: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> FunAsrSegment:
        """Parse a raw FunASR segment dict into a typed object."""
        text = raw.get("text")
        if not isinstance(text, str):
            raise ModelOutputInvalidError("FunASR segment has no string text field")
        timestamps = raw.get("timestamp")
        if timestamps is None:
            timestamps = []
        elif not isinstance(timestamps, list):
            raise ModelOutputInvalidError("FunASR timestamp must be a list or absent")
        words = raw.get("words")
        raw_text = raw.get("raw_text")
        if words is not None and (
            not isinstance(words, list)
            or any(not isinstance(word, str) for word in words)
        ):
            raise ModelOutputInvalidError(
                "FunASR words must be a list of strings or absent"
            )
        if raw_text is not None and not isinstance(raw_text, str):
            raise ModelOutputInvalidError("FunASR raw_text must be a string or absent")
        return cls(
            text=text,
            timestamp=timestamps,
            words=words,
            raw_text=raw_text,
        )


@dataclass(frozen=True, slots=True)
class SenseVoiceSnapshots:
    """SenseVoice and its voice-activity model, materialized on local disk.

    Each member is a ``RepositorySnapshot``: every file of a Hugging Face
    repository, resolved at its pinned commit by ``resolve_repository``
    (FunASR reads whatever its ``config.yaml`` names), not a path anyone
    could name, and not a Transformers snapshot, which would lack them.
    """

    asr: RepositorySnapshot
    vad: RepositorySnapshot


@dataclass(frozen=True, slots=True)
class ParaformerSnapshots:
    """Paraformer with its voice-activity and punctuation models, on local disk.

    Each member is a ``ModelScopeSnapshot``, admitted only complete (no
    partial download, checkpoint present) by
    ``_model_loading.modelscope_hub.resolve_modelscope_snapshot``. A bare path
    used to be accepted here, and a directory a stalled download had left
    without its ``model.pt`` loaded with untrained weights.
    """

    asr: ModelScopeSnapshot
    vad: ModelScopeSnapshot
    punc: ModelScopeSnapshot


# The two FunASR compositions BA3 runs. Which one a recognizer holds decides
# how it is constructed and how it is asked to transcribe, by an exhaustive
# match rather than by looking for "paraformer" in a model name.
FunAsrSnapshots = SenseVoiceSnapshots | ParaformerSnapshots


class FunAudioRecognizer:
    """A FunASR model loaded in memory, ready to transcribe.

    The only way to make one in production is ``load``, from snapshots already
    on local disk, so a recognizer always holds a loaded model: there is no
    unloaded state and no first-use path. That is the point. The model used to
    be built lazily, inside the first transcription request, and when the
    snapshot was not cached FunASR downloaded it right there, under a time
    budget meant for decoding the audio. On 2026-11-05 a 14-minute Paraformer
    download timed that request out and a retry started the download over.
    Downloads now happen before the worker reports ready (see
    ``_download_liveness``), and this type is why a request cannot reach one.
    """

    def __init__(
        self,
        *,
        lang: LanguageCode,
        device: str,
        snapshots: FunAsrSnapshots,
        label: str,
        model: Any,
    ) -> None:
        """Wrap an already-constructed FunASR model. Use ``load``.

        Direct construction exists for tests, which supply a fake model.
        """
        self.lang = lang
        self.device = device
        self.snapshots = snapshots
        self.label = label
        self._model = model

    @classmethod
    def load(
        cls,
        snapshots: FunAsrSnapshots,
        *,
        lang: LanguageCode,
        device: str,
        label: str,
    ) -> FunAudioRecognizer:
        """Construct the FunASR model from local snapshots, eagerly.

        ``label`` names the checkpoint for logs and errors (its model id).
        Loading from disk still takes seconds, so it is announced as a
        ``progress_v2`` pair for the time-transparency rule; it downloads
        nothing, and ``disable_update`` keeps FunASR from checking the network
        for a newer release of itself.
        """
        try:
            with redirect_stdout(io.StringIO()):
                from funasr import AutoModel
        except Exception as exc:
            raise ImportError(
                "FunAudio engine dependency 'funasr' is missing from this "
                "environment. Reinstall batchalign3 or install funasr."
            ) from exc

        from batchalign.worker._progress import Phase, emit_phase_event
        from batchalign.worker._types_v2 import PhaseEdgeV2

        emit_phase_event(
            Phase.LOADING_FUNAUDIO_ASR,
            PhaseEdgeV2.BEGIN,
            f"Loading FunASR model {label} ({device})…",
        )
        # FunASR prints a version banner to stdout, which carries the worker
        # protocol; it is swallowed here and around every generate call.
        with redirect_stdout(io.StringIO()):
            match snapshots:
                case SenseVoiceSnapshots(asr=asr, vad=vad):
                    model = AutoModel(
                        model=asr.path,
                        output_timestamps=True,
                        vad_model=vad.path,
                        vad_kwargs={"max_single_segment_time": 30000},
                        device=device,
                        hub="hf",
                        cache={},
                        language=lang,
                        use_itn=True,
                        batch_size_s=60,
                        output_timestamp=True,
                        ban_emo_unk=False,
                        merge_vad=True,
                        merge_length_s=15,
                        disable_update=True,
                    )
                case ParaformerSnapshots(asr=asr, vad=vad, punc=punc):
                    model = AutoModel(
                        model=str(asr.path),
                        vad_model=str(vad.path),
                        punc_model=str(punc.path),
                        device=device,
                        disable_update=True,
                    )
                case _ as unreachable:
                    assert_never(unreachable)
        emit_phase_event(
            Phase.LOADING_FUNAUDIO_ASR,
            PhaseEdgeV2.COMPLETE,
            f"FunASR model loaded ({label}).",
        )
        return cls(
            lang=lang, device=device, snapshots=snapshots, label=label, model=model
        )

    def _run_model(self, source_path: str) -> list[FunAsrSegment]:
        """Invoke FunASR and parse output into typed segments."""
        model = self._model
        with redirect_stdout(io.StringIO()):
            match self.snapshots:
                case ParaformerSnapshots():
                    # `raw_text` is the only Paraformer field whose tokens are
                    # parallel to `timestamp`; see `FunAsrSegment`.
                    output = model.generate(
                        input=source_path, output_timestamp=True, return_raw_text=True
                    )
                case SenseVoiceSnapshots():
                    # No model names here: the voice-activity model is the one
                    # loaded from its snapshot. The `vad_model="fsmn-vad"` and
                    # `spk_model="cam++"` this call used to pass were inert
                    # (identical output with and without them, measured on
                    # 2026-11-05), and a hub name reaching FunASR at request
                    # time is the route a download could take.
                    output = model.generate(
                        input=source_path,
                        cache={},
                        language=self.lang,
                        output_timestamps=True,
                        vad_kwargs={"max_single_segment_time": 60000},
                        ban_emo_unk=False,
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=True,
                        merge_length_s=15,
                        output_timestamp=True,
                    )
                case _ as unreachable:
                    assert_never(unreachable)

        raw_list: list[dict[str, Any]]
        if isinstance(output, dict):
            raw_list = [output]
        elif isinstance(output, list):
            raw_list = output
        else:
            raise ModelOutputInvalidError(
                f"FunASR returned {type(output).__name__}; expected a dict or list of dicts"
            )

        segments = []
        for index, raw in enumerate(raw_list):
            if not isinstance(raw, dict):
                raise ModelOutputInvalidError(
                    f"FunASR model {self.label}, segment {index + 1}: "
                    f"expected a dict, got {type(raw).__name__}"
                )
            try:
                segments.append(FunAsrSegment.from_raw(raw))
            except ModelOutputInvalidError as error:
                raise ModelOutputInvalidError(
                    f"FunASR model {self.label}, segment {index + 1}: {error}"
                ) from error
        return segments

    def transcribe(
        self, source_path: str
    ) -> tuple[AsrGenerationPayload, list[TimedWord]]:
        """Return `(monologues_payload, timed_words)` for the source audio.

        The Python side only owns model invocation and shallow parsing. Rust
        owns token cleanup, Cantonese tokenization, timestamp pairing, and the
        projection into the shared ASR worker payload shape.
        """
        import batchalign_core

        segments = self._run_model(source_path)
        projection = json.loads(
            batchalign_core.funaudio_segments_to_asr(
                [asdict(segment) for segment in segments],
            )
        )
        monologues: list[AsrMonologue] = projection["monologues"]
        timed_words: list[TimedWord] = projection["timed_words"]
        return {"monologues": monologues}, timed_words
