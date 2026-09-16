"""FunASR helpers for built-in HK/Cantonese engines."""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from typing import Any

from batchalign.inference._domain_types import LanguageCode

from ._asr_types import AsrGenerationPayload, AsrMonologue, TimedWord

L = logging.getLogger("batchalign.hk.funaudio")


@dataclass
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
        text = str(raw.get("text", ""))
        timestamps = raw.get("timestamp")
        if not isinstance(timestamps, list):
            timestamps = []
        words = raw.get("words")
        raw_text = raw.get("raw_text")
        return cls(
            text=text,
            timestamp=timestamps,
            words=[str(word) for word in words] if isinstance(words, list) else None,
            raw_text=raw_text if isinstance(raw_text, str) else None,
        )


class FunAudioRecognizer:
    """Wrapper around FunASR model invocation plus Rust-owned projection."""

    def __init__(
        self,
        lang: LanguageCode = "yue",
        model: str = "FunAudioLLM/SenseVoiceSmall",
        device: str = "cpu",
        *,
        model_path: str | None = None,
        model_revision: str | None = None,
        vad_model: str | None = None,
        vad_model_path: str | None = None,
        vad_revision: str | None = None,
        punc_model: str | None = None,
        punc_revision: str | None = None,
    ) -> None:
        """Store language/model configuration and defer model loading until first use.

        The worker fills the keyword arguments from the pinned plan, and the
        two branches need different things because the two hubs behave
        differently. The Hugging Face branch gets LOCAL PATHS, because FunASR's
        HF download drops the revision it is handed and would otherwise fetch
        whatever the hub currently serves. The ModelScope branch gets ids and
        tags, which it does honour. All ``None`` is the direct-caller path.
        """
        self.lang = lang
        self.model_name = model
        self.device = device
        self.model_path = model_path
        self.model_revision = model_revision
        self.vad_model = vad_model
        self.vad_model_path = vad_model_path
        self.vad_revision = vad_revision
        self.punc_model = punc_model
        self.punc_revision = punc_revision
        self._model: Any | None = None

    def _get_model(self) -> Any:
        """Return a cached FunASR model instance, creating it on first call.

        On cache miss (first call per worker lifetime), emits a pair of
        ``progress_v2`` download events bracketing the
        ``AutoModel(...)`` construction so the daemon log / dashboard /
        TUI show "Loading FunASR model… (first-run download may take
        several minutes)" instead of dead air. Subsequent calls hit the
        ``self._model is not None`` short-circuit above and emit
        nothing: a future call cannot mislead the user into thinking
        another download is happening.

        Per CLAUDE.md §11 time-transparency rule; mirrors the eager
        ``warm()`` pattern in ``_qwen_common.QwenRecognizer`` but
        without the eager-call-at-bootstrap shape because FunASR's
        worker bootstrap is deliberately lazy. Late binding of
        ``emit_download_event`` (import inside the function) keeps the
        recognizer importable in test contexts that don't bring up the
        full worker stack.
        """
        if self._model is not None:
            return self._model

        try:
            with redirect_stdout(io.StringIO()):
                from funasr import AutoModel
        except Exception as exc:
            raise ImportError(
                "FunAudio engine dependency 'funasr' is missing from this "
                "environment. Reinstall batchalign3 or install funasr."
            ) from exc

        from batchalign.worker._progress import emit_download_event

        emit_download_event(
            stage="downloading_funaudio_asr",
            user_message=(
                f"Loading FunASR model {self.model_name} ({self.device}); "
                f"first-run HuggingFace download may take several minutes…"
            ),
        )

        with redirect_stdout(io.StringIO()):
            if "paraformer" not in self.model_name:
                # SenseVoice, on Hugging Face. Both the checkpoint and its
                # voice-activity model are passed as resolved local paths, so
                # FunASR loads exactly what the plan pinned rather than
                # re-resolving a name against the live hub.
                self._model = AutoModel(
                    model=self.model_path or self.model_name,
                    output_timestamps=True,
                    vad_model=self.vad_model_path or self.vad_model or "fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=self.device,
                    hub="hf",
                    cache={},
                    language=self.lang,
                    use_itn=True,
                    batch_size_s=60,
                    output_timestamp=True,
                    ban_emo_unk=False,
                    merge_vad=True,
                    merge_length_s=15,
                )
            else:
                # Paraformer, on ModelScope, which DOES honour the revisions
                # it is given. The auxiliary models and their tags come from
                # the plan; they used to be five literals here, which is how a
                # model could move without anything recording that it had.
                # ``model_revision`` alone may be absent, for a checkpoint the
                # manifest does not pin.
                if not (
                    self.vad_model
                    and self.vad_revision
                    and self.punc_model
                    and self.punc_revision
                ):
                    raise ValueError(
                        "the Paraformer composition needs its voice-activity and "
                        "punctuation models with their revisions; the worker "
                        "supplies them from the pinned plan"
                    )
                # The checkpoint's own revision is passed only when the plan
                # pinned one. It used to fall back to the branch name
                # ``master``, which names a moving head rather than a release:
                # nothing had asked for that branch, and the registry records
                # this member's revision as not exposed, so the substitution
                # reached no provenance and a checkpoint could move between two
                # runs that report the same identity. An unpinned checkpoint is
                # a real state (a user may name a Paraformer this build does
                # not pin, which the manifest types as floating), so the honest
                # load asks for no revision instead of inventing one.
                paraformer_kwargs: dict[str, Any] = {
                    "model": self.model_path or self.model_name,
                    "vad_model": self.vad_model,
                    "vad_model_revision": self.vad_revision,
                    "punc_model": self.punc_model,
                    "punc_model_revision": self.punc_revision,
                }
                if self.model_revision is not None:
                    paraformer_kwargs["model_revision"] = self.model_revision
                self._model = AutoModel(**paraformer_kwargs)

        emit_download_event(
            stage="downloading_funaudio_asr_complete",
            user_message=f"FunASR model loaded ({self.model_name}).",
        )
        return self._model

    def _run_model(self, source_path: str) -> list[FunAsrSegment]:
        """Invoke FunASR and parse output into typed segments."""
        model = self._get_model()
        with redirect_stdout(io.StringIO()):
            if "paraformer" in self.model_name:
                # `raw_text` is the only Paraformer field whose tokens are
                # parallel to `timestamp`; see `FunAsrSegment`.
                output = model.generate(
                    input=source_path, output_timestamp=True, return_raw_text=True
                )
            else:
                output = model.generate(
                    input=source_path,
                    cache={},
                    language=self.lang,
                    output_timestamps=True,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 60000},
                    ban_emo_unk=False,
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                    output_timestamp=True,
                    spk_model="cam++",
                )

        raw_list: list[dict[str, Any]]
        if isinstance(output, dict):
            raw_list = [output]
        elif isinstance(output, list):
            raw_list = [item for item in output if isinstance(item, dict)]
        else:
            raw_list = []

        return [FunAsrSegment.from_raw(raw) for raw in raw_list]

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
