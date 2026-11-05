"""Aliyun NLS ASR provider for built-in HK/Cantonese engines.

The ASR load validates the credentials and builds an :class:`AliyunAsr`, which
the worker's load registry records; a request transcribes through that object,
which also caches the account's NLS token. Nothing is held at module level.

Aliyun NLS is **Cantonese-only** (``lang="yue"``).
"""

from __future__ import annotations

import configparser
import json
import logging
import pathlib
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from batchalign.inference._domain_types import AudioPath, LanguageCode
from batchalign.inference.asr import (
    AsrBatchItem,
    MonologueAsrResponse,
)

from ._common import EngineOverrides, read_asr_config

L = logging.getLogger("batchalign.hk.aliyun")


# ---------------------------------------------------------------------------
# Aliyun NLS response models (parse at the websocket boundary)
# ---------------------------------------------------------------------------


class AliyunWord(BaseModel):
    """One word from an Aliyun NLS SentenceEnd payload.

    The two times are OPTIONAL, and their absence crosses into Rust as absence.
    They used to default to ``0``, which is a legal time: a word Aliyun sent
    without timing therefore claimed to have been spoken in the first
    millisecond of the recording, and nothing downstream could tell that from a
    measurement. Rust's provider admission now turns a missing bound into an
    untimed word with a named cause (see
    ``crates/batchalign-pyo3/src/cantonese_asr_bridge/provider_admission.rs``).
    """

    text: str
    startTime: int | None = None
    endTime: int | None = None


class _AliyunSentencePayload(BaseModel):
    """The ``payload`` object inside a SentenceEnd websocket message."""

    result: str = ""
    words: list[AliyunWord] = []


class _AliyunSentenceMessage(BaseModel):
    """Top-level Aliyun NLS SentenceEnd websocket message."""

    payload: _AliyunSentencePayload = _AliyunSentencePayload()


@dataclass
class AliyunSentenceResult:
    """Parsed result from one Aliyun SentenceEnd callback."""

    words: list[AliyunWord]
    sentence_text: str


# An Aliyun NLS token is valid for 24 hours; refresh after 23 (an hour's margin).
_TOKEN_TTL_S: float = 23 * 3600


# ---------------------------------------------------------------------------
# _AliyunRunner: websocket streaming transcription
# ---------------------------------------------------------------------------


class _AliyunRunner:
    """Streaming runner for Aliyun NLS websocket transcription."""

    def __init__(self, token: str, appkey: str, wav_path: str) -> None:
        self._token = token
        self._appkey = appkey
        self._wav_path = wav_path
        self._sample_rate = 16000
        self._audio_data = b""
        self._results: list[AliyunSentenceResult] = []

    def _load_file(self) -> None:
        with wave.open(self._wav_path, "rb") as wav_file:
            self._audio_data = wav_file.readframes(wav_file.getnframes())
            self._sample_rate = wav_file.getframerate()

    def _on_sentence_begin(self, _message: str, *_args: Any) -> None:
        return

    def _on_sentence_end(self, message: str, *_args: Any) -> None:
        parsed = _AliyunSentenceMessage.model_validate_json(message)
        self._results.append(
            AliyunSentenceResult(
                words=parsed.payload.words,
                sentence_text=parsed.payload.result.strip(),
            )
        )

    def _on_start(self, _message: str, *_args: Any) -> None:
        return

    def _on_error(self, message: str, *_args: Any) -> None:
        raise RuntimeError(f"Aliyun ASR error: {message}")

    def _on_close(self, *_args: Any) -> None:
        return

    def _on_result_changed(self, _message: str, *_args: Any) -> None:
        return

    def _on_completed(self, _message: str, *_args: Any) -> None:
        return

    def start(self) -> list[AliyunSentenceResult]:
        """Run streaming transcription, return parsed sentence results."""
        try:
            import nls
        except Exception as exc:
            raise ImportError(
                "Aliyun engine dependencies are missing from this "
                "environment. Reinstall batchalign3 or install the Aliyun "
                "SDK packages."
            ) from exc

        self._load_file()

        transcriber = nls.NlsSpeechTranscriber(
            url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
            token=self._token,
            appkey=self._appkey,
            on_sentence_begin=self._on_sentence_begin,
            on_sentence_end=self._on_sentence_end,
            on_start=self._on_start,
            on_result_changed=self._on_result_changed,
            on_completed=self._on_completed,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        transcriber.start(
            aformat="pcm",
            enable_intermediate_result=True,
            enable_punctuation_prediction=False,
            enable_inverse_text_normalization=False,
            sample_rate=self._sample_rate,
            ex={"enable_words": True},
        )

        chunk_size = 640
        for offset in range(0, len(self._audio_data), chunk_size):
            transcriber.send_audio(self._audio_data[offset : offset + chunk_size])
            time.sleep(0.01)

        transcriber.ctrl(ex={"source": "batchalign3"})
        transcriber.stop()
        return self._results


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _NlsToken:
    """One Aliyun NLS token and the monotonic time it was issued."""

    token: str
    issued_at: float

    def fresh(self, now: float) -> bool:
        """Whether the token is still inside its refresh window at ``now``."""
        return now - self.issued_at < _TOKEN_TTL_S


def _request_token(ak_id: str, ak_secret: str) -> str:
    """Ask Aliyun for a new NLS token with the account's access key."""
    try:
        from aliyunsdkcore.client import AcsClient
        from aliyunsdkcore.request import CommonRequest
    except Exception as exc:
        raise ImportError(
            "Aliyun engine dependencies are missing from this "
            "environment. Reinstall batchalign3 or install the Aliyun SDK "
            "packages."
        ) from exc

    client = AcsClient(ak_id, ak_secret, "ap-southeast-1")
    request = CommonRequest()
    request.set_method("POST")
    request.set_domain("nlsmeta.ap-southeast-1.aliyuncs.com")
    request.set_version("2019-07-17")
    request.set_action_name("CreateToken")

    response = client.do_action_with_exception(request)
    payload = json.loads(response)

    token = payload.get("Token", {}).get("Id")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Aliyun token request did not return Token.Id")
    L.info("Aliyun NLS token refreshed")
    return token


def _ensure_wav(
    source_path: str,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    """Return a WAV path, converting if needed.  Caller must clean up temp_dir."""
    path = pathlib.Path(source_path)
    if path.suffix.lower() == ".wav":
        return source_path, None

    from batchalign.inference.audio import load_audio, save_audio

    audio, sample_rate = load_audio(source_path)
    if audio.dim() == 2 and audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    temp_dir = tempfile.TemporaryDirectory(prefix="batchalign-hk-aliyun-")
    wav_path = pathlib.Path(temp_dir.name) / f"{path.stem}.wav"
    save_audio(wav_path, audio, sample_rate, bits_per_sample=16)
    return str(wav_path), temp_dir


# ---------------------------------------------------------------------------
# Provider load function
# ---------------------------------------------------------------------------


class AliyunAsr:
    """The Aliyun account one ASR load admitted, and its cached NLS token.

    Not frozen: the token is refreshed in place once its window passes. Two
    requests racing on an expired token each fetch one, and the later write
    wins; both tokens are valid, so nothing is lost.
    """

    __slots__ = ("_ak_id", "_ak_secret", "_appkey", "_token")

    def __init__(self, *, ak_id: str, ak_secret: str, appkey: str) -> None:
        self._ak_id = ak_id
        self._ak_secret = ak_secret
        self._appkey = appkey
        self._token: _NlsToken | None = None

    def _current_token(self) -> str:
        """A token inside its refresh window, fetched when there is none."""
        now = time.monotonic()
        match self._token:
            case _NlsToken() as cached if cached.fresh(now):
                return cached.token
            case _:
                token = _request_token(self._ak_id, self._ak_secret)
                self._token = _NlsToken(token=token, issued_at=now)
                return token

    def transcribe(self, item: AsrBatchItem) -> MonologueAsrResponse:
        """Run Aliyun websocket ASR on one item's audio and return monologues."""
        return self._transcribe_path(item.audio_path)

    def _transcribe_path(self, audio_path: AudioPath) -> MonologueAsrResponse:
        token = self._current_token()
        wav_path, temp_dir = _ensure_wav(audio_path)
        try:
            runner = _AliyunRunner(token=token, appkey=self._appkey, wav_path=wav_path)
            results = runner.start()
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
        return _project_results(results)


def load_aliyun_asr(
    lang: LanguageCode,
    engine_overrides: EngineOverrides | None,
    *,
    config: configparser.ConfigParser | None = None,
) -> AliyunAsr:
    """Validate Aliyun credentials and build the provider one ASR load records.

    Aliyun NLS is Cantonese-only.  Raises ``ValueError`` if ``lang != "yue"``.
    Raises ``ConfigError`` if credentials are missing from ``~/.batchalign.ini``.
    """
    if lang != "yue":
        raise ValueError(
            "Aliyun ASR currently supports lang='yue' (Cantonese) only, "
            f"got lang={lang!r}."
        )

    creds = read_asr_config(
        (
            "engine.aliyun.ak_id",
            "engine.aliyun.ak_secret",
            "engine.aliyun.ak_appkey",
        ),
        engine="Aliyun",
        config=config,
    )

    appkey = creds["engine.aliyun.ak_appkey"]
    L.info("Aliyun ASR credentials loaded (appkey=%s...)", appkey[:4])
    return AliyunAsr(
        ak_id=creds["engine.aliyun.ak_id"],
        ak_secret=creds["engine.aliyun.ak_secret"],
        appkey=appkey,
    )


# ---------------------------------------------------------------------------
# Provider infer function
# ---------------------------------------------------------------------------


def _project_results(results: list[AliyunSentenceResult]) -> MonologueAsrResponse:
    """Delegate Aliyun sentence projection to the shared Rust helper.

    The Python adapter keeps only websocket transport and shallow payload
    parsing. Rust owns the sentence-only fallback tokenization plus the shared
    monologue/timed-word projection rule used across HK ASR providers.
    """
    import batchalign_core

    projection = json.loads(
        batchalign_core.aliyun_sentences_to_asr(
            [
                {
                    "words": [word.model_dump() for word in result.words],
                    "sentence_text": result.sentence_text,
                }
                for result in results
            ],
            "yue",
        )
    )
    return MonologueAsrResponse.model_validate(
        {
            "kind": "monologues",
            "lang": "yue",
            "monologues": projection["monologues"],
        }
    )


__all__ = ["AliyunAsr", "load_aliyun_asr"]
