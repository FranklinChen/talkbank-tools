"""Domain-specific type aliases for inference and worker modules.

These are ``TypeAlias``: zero-cost documentation that makes signatures
self-describing without the wrapping friction of ``NewType``.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias, TypeVar

AudioPath: TypeAlias = str
"""Filesystem path to an audio file."""

SampleRate: TypeAlias = int
"""Audio sample rate in Hz (e.g. 16000)."""

TimestampMs: TypeAlias = int
"""Time offset in milliseconds."""

TimestampSeconds: TypeAlias = float
"""Time offset in seconds."""

LanguageCode: TypeAlias = str
"""ISO-639-3 language code (e.g. 'eng', 'spa', 'zho')."""

LanguageCode2: TypeAlias = str
"""ISO-639-1 two-letter language code (e.g. 'en', 'es', 'zh')."""

SpeakerId: TypeAlias = str
"""Speaker label (e.g. 'SPEAKER_0', 'SPEAKER_1')."""

NumSpeakers: TypeAlias = int
"""Expected number of speakers in audio."""

ConfidenceScore: TypeAlias = float
"""Confidence score in range [0.0, 1.0]."""

CommandName: TypeAlias = str
"""Batchalign command name (e.g. 'morphotag', 'align', 'transcribe')."""

RevAiJobId: TypeAlias = str
"""A Rev.AI server-side job identifier returned after audio submission.

Obtained during preflight batch upload and passed to polling calls so
individual file tasks can retrieve results without re-uploading audio.
"""

RevAiApiKey: TypeAlias = str
"""Raw Rev.AI API credential loaded from the environment.

Never logged or included in error messages. Only used at the worker/SDK
boundary where the Rev.AI client is constructed.
"""

TencentSecretId: TypeAlias = str
"""Tencent Cloud CAM SecretId (AKID-prefixed identifier).

Loaded from ``~/.batchalign.ini`` `[asr]` `engine.tencent.id`. The
CAM user must hold both ASR product permissions and (when used by
the translate backend) ``tmt:TextTranslate``. Never logged or
echoed.
"""

TencentSecretKey: TypeAlias = str
"""Tencent Cloud CAM SecretKey paired with ``TencentSecretId``.

Loaded from ``~/.batchalign.ini`` `[asr]` `engine.tencent.key`.
Treated as a secret: never logged or echoed.
"""

TencentRegion: TypeAlias = str
"""Tencent Cloud region identifier (e.g. ``"ap-guangzhou"``,
``"ap-shanghai"``). Loaded from ``~/.batchalign.ini`` `[asr]`
`engine.tencent.region`. Affects API endpoint latency but not
product authorization.
"""

TcpPort: TypeAlias = int
"""TCP port number in the range 1-65535."""


_Choice = TypeVar("_Choice", bound=Enum)


def parse_choice(kind: type[_Choice], choice: str, label: str) -> _Choice:
    """The member of ``kind`` whose value is ``choice``.

    An unknown value raises with the supported list rather than falling back:
    a typo in a per-host invocation must not silently select another engine.
    One helper for every engine-override key, so the message has one shape.
    """
    try:
        return kind(choice)
    except ValueError as exc:
        supported = ", ".join(str(member.value) for member in kind)
        raise ValueError(
            f"unknown {label} {choice!r}; expected one of: {supported}"
        ) from exc


class WhisperCpuPrecision(Enum):
    """The dtype stock Whisper loads with on a CPU device.

    CUDA always loads float16 and this enum does not apply there. On CPU the
    default is float32, the precision every measured number in the book was
    taken at. ``FLOAT16`` is the experiment knob for the memory saving Houjun's
    Batchalign takes on Apple Silicon (PyTorch 2.5 and later run half-precision
    CPU kernels); it is selected per job through the ``whisper_cpu_dtype``
    engine-override extra and enters the worker key, so a float16 worker never
    serves a float32 job. Adopting it as a default needs the WER comparison the
    action-items register asks for.
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"

    @classmethod
    def from_overrides(
        cls, engine_overrides: dict[str, str] | None
    ) -> WhisperCpuPrecision:
        """The precision a job selected, float32 when it selected none."""
        choice = (engine_overrides or {}).get("whisper_cpu_dtype")
        if choice is None:
            return cls.FLOAT32
        return parse_choice(cls, choice, "whisper_cpu_dtype")


class TranslationBackend(str, Enum):
    """Which translation engine is active.

    Wire values are the lowercase tokens the Rust control plane sends in
    ``engine_overrides["translate"]`` at worker start and in
    ``TranslateRequestV2.engine`` on every request. Adding a variant requires
    a mirror update in ``crates/batchalign/src/types/engines.rs``
    (``TranslateEngineName``) and ``crates/batchalign-types``
    (``TranslateBackendV2``); the IPC conformance test catches drift. The
    dispatcher in ``batchalign/worker/_model_loading/translation.py`` decodes
    the string into this enum via ``TranslationBackend(value)``.
    """

    GOOGLE = "google"
    SEAMLESS = "seamless"
    NLLB = "nllb"
    TENCENT = "tencent"
    ALIYUN = "aliyun"
