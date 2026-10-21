"""Tests for ``batchalign.inference.whisper_hub``: HF fine-tune loader.

These tests cover the *dispatch* and *error* paths of the whisper_hub
loader without actually downloading a 1.5 GB model. The underlying HF
pipeline construction is exercised by the ML-golden suite separately.
"""

from __future__ import annotations

import pytest

from batchalign.device import DevicePolicy


class TestResolveWhisperHubModelId:
    """``resolve_whisper_hub_model_id`` picks model_id from lang or override."""

    def test_explicit_override_wins_over_language_default(self) -> None:
        from batchalign.inference.whisper_hub import resolve_whisper_hub_model_id

        resolved = resolve_whisper_hub_model_id(
            lang="mal",
            engine_overrides={"asr": "whisper_hub", "model_id": "other/mal-alt"},
        )
        assert resolved == "other/mal-alt"

    def test_language_default_is_used_when_no_override(self) -> None:
        from batchalign.inference.whisper_hub import resolve_whisper_hub_model_id

        resolved = resolve_whisper_hub_model_id(
            lang="mal",
            engine_overrides={"asr": "whisper_hub"},
        )
        assert resolved == "thennal/whisper-medium-ml"

    def test_no_default_and_no_override_raises_typed_error(self) -> None:
        from batchalign.inference.whisper_hub import (
            WhisperHubModelNotFoundError,
            resolve_whisper_hub_model_id,
        )

        # Language without a seeded default entry and no explicit model_id
        # must raise a typed error naming the language and pointing at the
        # escape hatch. No silent fallback to a random model.
        with pytest.raises(WhisperHubModelNotFoundError) as excinfo:
            resolve_whisper_hub_model_id(
                lang="xyz",
                engine_overrides={"asr": "whisper_hub"},
            )
        msg = str(excinfo.value)
        assert "xyz" in msg
        assert "model_id" in msg.lower()


class TestWhisperHubLoaderDispatch:
    """``load_whisper_hub_asr`` delegates through resolve_whisper_hub_model_id."""

    def test_loader_resolves_model_id_from_lang(self, monkeypatch) -> None:
        captured: dict[str, object] = {}
        sentinel = object()

        def fake_load_whisper_asr(
            *, model, base, decoding, device_policy, timestamp_mode
        ):
            captured["model"] = model
            captured["base"] = base
            captured["decoding"] = decoding
            captured["device_policy"] = device_policy
            captured["timestamp_mode"] = timestamp_mode
            return sentinel

        monkeypatch.setattr(
            "batchalign.inference.asr.load_whisper_asr",
            fake_load_whisper_asr,
        )

        from batchalign.inference._domain_types import WhisperTimestampMode
        from batchalign.inference.types import CheckpointPinnedLanguage
        from batchalign.inference.whisper_hub import load_whisper_hub_asr

        handle = load_whisper_hub_asr(
            lang="mal",
            engine_overrides={"asr": "whisper_hub"},
            device_policy=DevicePolicy(force_cpu=True),
        )

        assert handle is sentinel
        assert captured["model"] == "thennal/whisper-medium-ml"
        assert captured["base"] == "thennal/whisper-medium-ml"
        # The fine-tune decodes with its own pinned language and task: the
        # handle cannot be told otherwise, by the request or anyone else.
        assert captured["decoding"] == CheckpointPinnedLanguage()
        assert captured["device_policy"] == DevicePolicy(force_cpu=True)
        assert captured["timestamp_mode"] is WhisperTimestampMode.SEGMENT
