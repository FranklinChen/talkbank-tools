"""Domain-specific type aliases for inference and worker modules.

These are ``TypeAlias`` — zero-cost documentation that makes signatures
self-describing without the wrapping friction of ``NewType``.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

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
Treated as a secret — never logged or echoed.
"""

TencentRegion: TypeAlias = str
"""Tencent Cloud region identifier (e.g. ``"ap-guangzhou"``,
``"ap-shanghai"``). Loaded from ``~/.batchalign.ini`` `[asr]`
`engine.tencent.region`. Affects API endpoint latency but not
product authorization.
"""

TcpPort: TypeAlias = int
"""TCP port number in the range 1–65535."""


class TranslationBackend(Enum):
    """Which translation engine is active."""

    GOOGLE = "google"
    SEAMLESS = "seamless"
    NLLB = "nllb"
    TENCENT = "tencent"
