"""Structural Protocol types for third-party objects used across inference.

All protocols are defined here to avoid runtime imports of heavy libraries
(stanza, torch, torchaudio).  They are only used for static type checking.

Usage::

    from __future__ import annotations
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from batchalign.inference.types import StanzaDoc, StanzaNLP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    assert_never,
    overload,
    runtime_checkable,
)

import numpy as np

from batchalign.inference._domain_types import SampleRate
from batchalign.worker._types_v2 import DetectWhisperLanguageV2, ForceWhisperLanguageV2

if TYPE_CHECKING:
    import torch
    from transformers import (
        GenerationConfig,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.pipelines import AutomaticSpeechRecognitionPipeline

    from batchalign.worker._types_v2 import AsrModelIdentityV2


# ---------------------------------------------------------------------------
# Stanza Protocols
# ---------------------------------------------------------------------------


class StanzaWord(Protocol):
    """Structural type for ``stanza.models.common.doc.Word``."""

    @property
    def text(self) -> str: ...

    @property
    def upos(self) -> str: ...

    @property
    def lemma(self) -> str: ...

    @property
    def feats(self) -> str | None: ...

    @property
    def deprel(self) -> str: ...

    @property
    def head(self) -> int: ...

    @property
    def id(self) -> int | tuple[int, ...]: ...


class StanzaToken(Protocol):
    """Structural type for ``stanza.models.common.doc.Token``."""

    @property
    def text(self) -> str: ...

    @property
    def id(self) -> tuple[int, ...]: ...

    @property
    def words(self) -> list[StanzaWord]: ...


class ConstituencyTree(Protocol):
    """Structural type for a constituency parse tree node."""

    @property
    def children(self) -> list[ConstituencyTree]: ...

    @property
    def label(self) -> str | None: ...

    def is_leaf(self) -> bool: ...


class StanzaSentence(Protocol):
    """Structural type for ``stanza.models.common.doc.Sentence``."""

    @property
    def tokens(self) -> list[StanzaToken]: ...

    @property
    def words(self) -> list[StanzaWord]: ...

    @property
    def constituency(self) -> ConstituencyTree: ...


class StanzaDoc(Protocol):
    """Structural type for ``stanza.models.common.doc.Document``."""

    @property
    def sentences(self) -> list[StanzaSentence]: ...

    def to_dict(
        self,
    ) -> list[list[dict[str, str | int | float | list[int] | tuple[int, ...] | None]]]:
        """Serialize the document to a nested list of word-level dicts."""
        ...


class StanzaNLP(Protocol):
    """Structural type for a Stanza Pipeline.

    Both call forms are declared because both are used. A Pipeline accepts a
    string, and it also accepts a LIST of pre-built Documents, which is how the
    Italian MWT probe runs a whole batch in one call. Only the string form was
    declared, so the batch caller had to type its pipeline as `object`, which is
    not callable, and the resulting error stayed invisible behind the
    `batchalign.inference.*` mypy carve-out.
    """

    @overload
    def __call__(self, text: str) -> StanzaDoc: ...

    @overload
    def __call__(self, text: list[Any]) -> list[StanzaDoc]: ...

    def __call__(self, text: str | list[Any]) -> StanzaDoc | list[StanzaDoc]: ...


# ---------------------------------------------------------------------------
# Audio file Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioFile(Protocol):
    """Structural type for ``ASRAudioFile``."""

    @property
    def file(self) -> str: ...

    @property
    def tensor(self) -> torch.Tensor: ...

    @property
    def rate(self) -> int: ...

    def file_identity(self) -> str: ...

    def chunk(self, begin_ms: int, end_ms: int) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# FA model return type aliases
# ---------------------------------------------------------------------------

WhisperFAResult = list[tuple[str, float]]
"""Whisper FA output: list of (token_text, timestamp_seconds)."""


class Wave2VecWordAlignment(NamedTuple):
    """One indexed word interval and its optional model alignment score."""

    word: str
    interval_ms: tuple[int, int]
    model_score: float | None


Wave2VecFAResult = list[Wave2VecWordAlignment]
"""Wave2Vec2 FA output, preserving one slot and score per input word."""


# ---------------------------------------------------------------------------
# Typed model handles (replace monkey-patched object / tuple hacks)
# ---------------------------------------------------------------------------


"""Generation kwargs for Whisper ASR: repetition_penalty, config, task, language, max_new_tokens."""
GenerateKwargs: TypeAlias = "dict[str, str | int | float | GenerationConfig]"


@dataclass(frozen=True)
class CheckpointPinnedLanguage:
    """A fine-tune decodes with the language and task its own generation
    config pins.

    HuggingFace Whisper fine-tunes bake their language hint and suppress-token
    set in during training; forcing a language or task on top produces
    cross-script gibberish. Used by the ``whisper_hub`` engine; see
    ``batchalign/inference/whisper_hub.py``.
    """


WhisperDecoding: TypeAlias = (
    DetectWhisperLanguageV2 | ForceWhisperLanguageV2 | CheckpointPinnedLanguage
)
"""How one loaded Whisper is told its language: the control plane's decision
for a stock checkpoint, or the checkpoint's own for a fine-tune."""


class WhisperASRHandle:
    """Typed wrapper for a HuggingFace ASR pipeline with metadata.

    Replaces the monkey-patching pattern where config, lang, and sample_rate
    were stashed as ``_ba_*`` attributes on the pipeline object.
    """

    def __init__(
        self,
        pipe: AutomaticSpeechRecognitionPipeline,
        config: GenerationConfig,
        decoding: WhisperDecoding,
        sample_rate: SampleRate,
    ) -> None:
        self._pipe = pipe
        self.config = config
        # Fixed for the handle's life: a worker is spawned for one language,
        # and the decision was made once, by the control plane or the
        # checkpoint, before the model loaded.
        self.decoding = decoding
        self.sample_rate = sample_rate

        # Which checkpoint this handle actually loaded, set by the worker
        # layer that resolved it (``batchalign/worker/_model_loading/asr.py``).
        # It travels on every ASR result this handle produces, so a hub
        # repository that moved is refused by name at the bridge instead of
        # silently changing transcripts. ``None`` only between construction
        # and that assignment.
        self.model_identity: AsrModelIdentityV2 | None = None

    def __call__(
        self,
        audio: np.ndarray | str,  # mono waveform or provider-native input path
        *,
        batch_size: int = 1,
        generate_kwargs: GenerateKwargs | None = None,
        observation_mode: Literal["windows", "legacy"] = "windows",
    ) -> dict[str, Any]:
        return self._pipe(  # type: ignore[no-any-return]
            audio,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs or {},
            observation_mode=observation_mode,
        )

    def gen_kwargs(self) -> GenerateKwargs:
        """Build generation kwargs from the handle's decoding decision.

        - A fine-tune (``CheckpointPinnedLanguage``) gets only
          ``max_new_tokens=444``. That cap is a hard upper bound on
          per-chunk generation, not a decoding override: Whisper's
          ``max_target_positions=448`` includes the 3 special start
          tokens, so 444 is one below the legal max, a no-op under
          successful operation, a terminator for the intermittent
          runaway-generation case where the decoder fails to predict
          end-of-utterance and would otherwise spin indefinitely.

        - ``Detect`` (stock Whisper) omits ``language`` and ``task`` so
          Whisper detects, keeping the stock-model tuning knobs.

        - ``Force`` (stock Whisper) names the language by Whisper's own
          token and transcribes.
        """
        match self.decoding:
            case CheckpointPinnedLanguage():
                return {"max_new_tokens": 444}
            case DetectWhisperLanguageV2():
                return {"repetition_penalty": 1.001, "generation_config": self.config}
            case ForceWhisperLanguageV2(token=token):
                return {
                    "repetition_penalty": 1.001,
                    "generation_config": self.config,
                    "task": "transcribe",
                    "language": token,
                }
            case _:
                assert_never(self.decoding)


@dataclass(frozen=True)
class WhisperModelIdentity:
    """Resolved hub commit passed to both model and processor loaders."""

    model_id: str
    revision: str

    def __post_init__(self) -> None:
        if (
            not self.model_id
            or len(self.revision) != 40
            or any(c not in "0123456789abcdef" for c in self.revision)
        ):
            raise ValueError("Whisper identity requires a resolved hub commit")


class WhisperFAHandle:
    """Typed wrapper for Whisper forced alignment model bundle.

    Replaces the ``(model, processor, sample_rate)`` tuple with named fields.
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        processor: WhisperProcessor,
        sample_rate: SampleRate,
        *,
        model_identity: WhisperModelIdentity | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.sample_rate = sample_rate
        self.model_identity = model_identity


class Wave2VecFAHandle:
    """Typed wrapper for Wave2Vec forced alignment model bundle.

    Replaces the ``(model, sample_rate)`` tuple with named fields.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sample_rate: SampleRate,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
