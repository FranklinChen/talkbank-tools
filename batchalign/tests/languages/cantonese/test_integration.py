"""Integration tests for HK engine providers.

These tests exercise each real provider (the object its load returns) against
audio fixtures.
Each test auto-skips if its required dependencies or credentials are unavailable.

Fixtures (in tests/languages/cantonese/fixtures/):
  - 05b_clip.mp3  -8-second Cantonese audio clip from 05b.cha (4.5s-12.5s)
  - 05b_clip.wav, same clip, 16 kHz mono WAV (required by Aliyun)
  - 05b_clip.cha: 4-utterance CHAT file matching the clip

Run:
  uv run pytest batchalign/tests/languages/cantonese/test_integration.py -v
  uv run pytest -m integration -v
  uv run pytest -m "not integration" -v
"""

from __future__ import annotations

import configparser
import pathlib
import re

import pytest

from batchalign.inference.asr import AsrBatchItem, MonologueAsrResponse

from .conftest import CLIP_MP3, CLIP_WAV

# ---------------------------------------------------------------------------
# Dependency probes
# ---------------------------------------------------------------------------


def _has_funasr() -> bool:
    try:
        import funasr  # noqa: F401

        return True
    except ImportError:
        return False


def _has_pycantonese() -> bool:
    try:
        import pycantonese  # noqa: F401

        return True
    except ImportError:
        return False


def _has_opencc() -> bool:
    try:
        import opencc  # noqa: F401

        return True
    except ImportError:
        return False


def _has_tencent_sdk() -> bool:
    try:
        from qcloud_cos import CosS3Client  # noqa: F401
        from tencentcloud.asr.v20190614.asr_client import AsrClient  # noqa: F401

        return True
    except ImportError:
        return False


def _has_aliyun_sdk() -> bool:
    try:
        import nls  # noqa: F401
        from aliyunsdkcore.client import AcsClient  # noqa: F401

        return True
    except ImportError:
        return False


def _has_tencent_credentials() -> bool:
    cfg = configparser.ConfigParser()
    cfg.read(str(pathlib.Path.home() / ".batchalign.ini"))
    keys = (
        "engine.tencent.id",
        "engine.tencent.key",
        "engine.tencent.region",
        "engine.tencent.bucket",
    )
    return all(cfg.has_option("asr", k) and cfg.get("asr", k).strip() for k in keys)


def _has_aliyun_credentials() -> bool:
    cfg = configparser.ConfigParser()
    cfg.read(str(pathlib.Path.home() / ".batchalign.ini"))
    keys = (
        "engine.aliyun.ak_id",
        "engine.aliyun.ak_secret",
        "engine.aliyun.ak_appkey",
    )
    return all(cfg.has_option("asr", k) and cfg.get("asr", k).strip() for k in keys)


def _require_fixtures() -> bool:
    return CLIP_MP3.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _make_asr_request(audio_path: str, lang: str = "yue") -> AsrBatchItem:
    """One provider ASR item."""
    from batchalign.worker._types_v2 import NotRequestedDiarizationV2

    # These engines separate no speakers, so the request says that, rather than
    # asking for a count of one: that count is the contradiction submission
    # refuses, and it has no spelling on this boundary any more.
    return AsrBatchItem(
        audio_path=audio_path, lang=lang, diarization=NotRequestedDiarizationV2()
    )


def _extract_tokens(response: MonologueAsrResponse) -> list[dict[str, object]]:
    """Extract flat token dicts from the raw monologue provider payload.

    HK provider tests still assert text/timing/speaker behavior, but the
    production worker contract returns tagged monologue payloads instead of a
    flattened shared token schema. This helper mirrors the Rust-side
    normalization only as far as these integration assertions need.
    """
    tokens: list[dict[str, object]] = []
    for monologue in response.monologues:
        speaker = str(monologue.speaker)
        for element in monologue.elements:
            if element.type != "text":
                continue
            if not element.value.strip():
                continue
            tokens.append(
                {
                    "text": element.value,
                    "start_s": element.ts,
                    "end_s": element.end_ts,
                    "speaker": speaker,
                    "confidence": element.confidence,
                }
            )
    return tokens


# ---------------------------------------------------------------------------
# FunASR / SenseVoice
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_funasr(), reason="funasr not installed")
@pytest.mark.skipif(not _has_opencc(), reason="opencc not installed")
@pytest.mark.skipif(not _require_fixtures(), reason="fixtures not found")
class TestFunAudioASR:
    @pytest.fixture(autouse=True)
    def _load_engine(self) -> None:
        from batchalign.inference.languages.cantonese._funaudio_asr import (
            load_funaudio_asr,
        )
        from batchalign.inference.languages.cantonese._funaudio_common import (
            SenseVoiceSnapshots,
        )
        from batchalign.worker._model_loading.pinned_hub import (
            resolve_repository,
            unpinned,
        )

        # The worker materializes both members before loading; so does this.
        asr = resolve_repository(unpinned("FunAudioLLM/SenseVoiceSmall"), kind="ASR")
        vad = resolve_repository(
            unpinned("funasr/fsmn-vad"), kind="voice activity detection"
        )
        provider = load_funaudio_asr(
            "yue",
            None,
            snapshots=SenseVoiceSnapshots(asr=asr, vad=vad),
            label="FunAudioLLM/SenseVoiceSmall",
        )
        self._infer = provider.transcribe

    def test_produces_tokens(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        assert len(tokens) > 0

    def test_tokens_have_text(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        for t in tokens:
            assert str(t["text"]).strip(), f"Empty token text: {t}"

    def test_output_contains_cantonese(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        all_text = "".join(str(t["text"]) for t in tokens)
        assert _CJK_RE.search(all_text), f"No CJK characters in output: {all_text}"

    def test_tokens_have_timestamps(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        timed = [t for t in tokens if t.get("start_s") is not None]
        assert len(timed) > 0

    def test_normalization_applied(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        all_text = "".join(str(t["text"]) for t in tokens)
        for bad_char in ("系", "呀", "噶"):
            if bad_char in all_text:
                pytest.fail(
                    f"Unnormalized character '{bad_char}' found in output. "
                    f"Full text: {all_text}"
                )


# ---------------------------------------------------------------------------
# Tencent Cloud ASR
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_tencent_sdk(), reason="Tencent SDK not installed")
@pytest.mark.skipif(
    not _has_tencent_credentials(), reason="Tencent credentials not found"
)
@pytest.mark.skipif(not _has_opencc(), reason="opencc not installed")
@pytest.mark.skipif(not _require_fixtures(), reason="fixtures not found")
class TestTencentASR:
    @pytest.fixture(autouse=True)
    def _load_engine(self) -> None:
        from batchalign.inference.languages.cantonese._tencent_asr import (
            load_tencent_asr,
        )

        provider = load_tencent_asr("yue", engine_model_type="16k_zh_en_2.0")
        self._infer = provider.transcribe

    def test_produces_tokens(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        assert len(tokens) > 0

    def test_output_contains_cantonese(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        all_text = "".join(str(t["text"]) for t in tokens)
        assert _CJK_RE.search(all_text), f"No CJK in output: {all_text}"

    def test_tokens_have_timestamps(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        timed = [t for t in tokens if t.get("start_s") is not None]
        assert len(timed) > 0

    def test_normalization_applied(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        all_text = "".join(str(t["text"]) for t in tokens)
        assert "系" not in all_text or "係" in all_text, (
            f"Unnormalized '系' in output: {all_text}"
        )

    def test_has_speaker_info(self) -> None:
        req = _make_asr_request(str(CLIP_MP3))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        with_speaker = [t for t in tokens if t.get("speaker") is not None]
        assert len(with_speaker) > 0


# ---------------------------------------------------------------------------
# Aliyun NLS ASR
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_aliyun_sdk(), reason="Aliyun SDK not installed")
@pytest.mark.skipif(
    not _has_aliyun_credentials(), reason="Aliyun credentials not found"
)
@pytest.mark.skipif(not _has_opencc(), reason="opencc not installed")
@pytest.mark.skipif(not _require_fixtures(), reason="fixtures not found")
class TestAliyunASR:
    @pytest.fixture(autouse=True)
    def _load_engine(self) -> None:
        from batchalign.inference.languages.cantonese._aliyun_asr import (
            load_aliyun_asr,
        )

        provider = load_aliyun_asr("yue", None)
        self._infer = provider.transcribe

    def test_produces_tokens(self) -> None:
        req = _make_asr_request(str(CLIP_WAV))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        assert len(tokens) > 0

    def test_output_contains_cantonese(self) -> None:
        req = _make_asr_request(str(CLIP_WAV))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        all_text = "".join(str(t["text"]) for t in tokens)
        assert _CJK_RE.search(all_text), f"No CJK in output: {all_text}"

    def test_tokens_have_timestamps_or_fallback(self) -> None:
        req = _make_asr_request(str(CLIP_WAV))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        assert len(tokens) > 0

    def test_normalization_applied(self) -> None:
        req = _make_asr_request(str(CLIP_WAV))
        resp = self._infer(req)
        tokens = _extract_tokens(resp)
        all_text = "".join(str(t["text"]) for t in tokens)
        assert "系" not in all_text or "係" in all_text

    def test_rejects_non_cantonese(self) -> None:
        from batchalign.inference.languages.cantonese._aliyun_asr import load_aliyun_asr

        with pytest.raises(ValueError, match="yue"):
            load_aliyun_asr("eng", None)


# ---------------------------------------------------------------------------
# Cantonese FA (jyutping + Wave2Vec)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_pycantonese(), reason="pycantonese not installed")
@pytest.mark.skipif(not _require_fixtures(), reason="fixtures not found")
class TestCantoneseFAProvider:
    @pytest.fixture(autouse=True)
    def _load_engine(self) -> None:
        from batchalign.inference.languages.cantonese._cantonese_fa import (
            load_cantonese_fa,
        )

        self._host = load_cantonese_fa("yue", None)

    def _window(self, start_ms: int, end_ms: int) -> object:
        from batchalign.inference.audio import load_audio_file

        return load_audio_file(str(CLIP_MP3)).chunk(start_ms, end_ms)

    def test_empty_words_returns_empty_timings(self) -> None:
        assert self._host.align_words(self._window(0, 1000), []) == []

    def test_cantonese_words_produce_timings(self) -> None:
        rows = self._host.align_words(self._window(350, 1375), ["咁", "搞", "笑", "嘅"])

        assert len(rows) == 4
        assert all(end >= start for _word, (start, end), _score in rows)


# Cantonese normalization has no Python surface to integrate against any more:
# it runs in the Rust server, once per monologue, through
# `AlignedNormalization`. Its behaviour is tested where it lives, in
# `crates/batchalign-transform/src/asr_postprocess/cantonese.rs`.
