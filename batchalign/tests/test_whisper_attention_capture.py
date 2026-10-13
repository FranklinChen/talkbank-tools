"""Real worker/PyO3 diagnostic path and lossless, no-clobber SQLite capture."""

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from batchalign.inference.types import WhisperModelIdentity
from batchalign.inference.whisper_attention import WhisperAttentionCapture
from batchalign.tests.pipelines.fa.test_fa_inference import (
    _FakeWhisperModel,
    _FakeWhisperProcessor,
)
from batchalign.tests.test_worker_fa_v2 import _make_request
from batchalign.worker._fa_v2 import (
    build_default_fa_execution_host_v2,
    execute_forced_alignment_request_v2,
)
from batchalign.worker._types_v2 import (
    ExecuteErrorV2,
    ExecuteSuccessV2,
    FaBackendV2,
    FaTextModeV2,
    WhisperAttentionCaptureRequestV2,
)


def test_actual_worker_capture_preserves_timings_and_binds_audio(tmp_path):
    model = _FakeWhisperModel()
    handle = SimpleNamespace(
        model=model,
        processor=_FakeWhisperProcessor(),
        sample_rate=16000,
        model_identity=WhisperModelIdentity("example/whisper", "a" * 40),
    )
    host = build_default_fa_execution_host_v2(
        whisper_model=handle,
        wave2vec_model=None,
        canto_host=None,
        qwen_host=None,
    )
    request = _make_request(
        tmp_path, backend=FaBackendV2.WHISPER, text_mode=FaTextModeV2.SPACE_JOINED
    )
    audio = np.zeros(1280, dtype="<f4")
    audio.tofile(tmp_path / "audio.pcm")
    request.attachments[1].frame_count = 1280
    request.attachments[1].byte_len = 5120
    baseline = execute_forced_alignment_request_v2(request, host)
    assert isinstance(baseline.outcome, ExecuteSuccessV2)
    assert baseline.result.attention_capture is None
    destination = tmp_path / "attention.sqlite"
    request.payload.attention_capture = WhisperAttentionCaptureRequestV2(
        destination=str(destination),
        semantic_key="a" * 64,
    )
    captured = execute_forced_alignment_request_v2(request, host)
    assert isinstance(captured.outcome, ExecuteSuccessV2)
    assert captured.result.tokens == baseline.result.tokens
    receipt = captured.result.attention_capture
    assert receipt.request_id == request.request_id
    assert receipt.capture == request.payload.attention_capture
    assert len(receipt.blake3) == 64
    with sqlite3.connect(destination) as db:
        assert db.execute(
            "SELECT request_id,semantic_key,audio_sha256 FROM capture"
        ).fetchone() == (
            request.request_id,
            "a" * 64,
            hashlib.sha256(audio.tobytes()).hexdigest(),
        )
        assert db.execute("SELECT count(*) FROM costs").fetchone() == (12,)
        assert db.execute("SELECT count(*) FROM raw_attention").fetchone() == (12,)
        raw = np.asarray(
            [
                r[0]
                for r in db.execute(
                    "SELECT value FROM raw_attention ORDER BY head_ordinal,label_row,frame"
                )
            ]
        ).reshape(3, 4)
        np.testing.assert_array_equal(
            raw, model().cross_attentions[0][0, 0, 2:].numpy()
        )
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    before = destination.read_bytes()
    refused = execute_forced_alignment_request_v2(request, host)
    assert isinstance(refused.outcome, ExecuteErrorV2)
    assert destination.read_bytes() == before


@pytest.mark.parametrize(
    "failure", ["shape", "nonfinite", "out_of_window", "no_model_identity"]
)
def test_capture_refuses_invalid_observation_before_creating_file(tmp_path, failure):
    destination = tmp_path / "attention.sqlite"
    capture = WhisperAttentionCapture(
        json.dumps(
            {
                "request_id": "request",
                "capture": {"destination": str(destination), "semantic_key": "a" * 64},
            }
        ),
        np.zeros(4, dtype=np.float32),
        "word",
        16000,
    )
    raw = np.zeros((1, 2, 2), dtype=np.float32)
    if failure == "shape":
        raw = raw[:, :1]
    if failure == "nonfinite":
        raw[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="admitted labels/audio"):
        capture.record(
            token_ids=[1, 2, 3],
            token_texts=["<|prefix|>", "word", "<|end|>"],
            prefix_length=1,
            heads=[(0, 0)],
            raw_attention=raw,
            costs=np.zeros((2, 2), dtype=np.float32),
            row_entries=np.asarray([0, 2 if failure == "out_of_window" else 1]),
            hop_length=1,
            median_width=1,
            model_id="example/model",
            model_revision="" if failure == "no_model_identity" else "a" * 40,
        )
    assert not destination.exists()


def test_worker_refuses_capture_for_another_backend(tmp_path):
    request = _make_request(
        tmp_path, backend=FaBackendV2.WAVE2VEC, text_mode=FaTextModeV2.SPACE_JOINED
    )
    request.payload.attention_capture = WhisperAttentionCaptureRequestV2(
        destination=str(tmp_path / "attention.sqlite"),
        semantic_key="a" * 64,
    )
    host = build_default_fa_execution_host_v2(
        whisper_model=None,
        wave2vec_model=None,
        canto_host=None,
        qwen_host=None,
    )
    response = execute_forced_alignment_request_v2(request, host)
    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code.value == "invalid_payload"


def test_identity_resolves_snapshot_before_load_without_config_private_fields(
    monkeypatch, tmp_path
):
    from batchalign.inference.fa import _resolve_whisper_identity

    config = tmp_path / "snapshots" / ("b" * 40) / "config.json"
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda model, filename: str(config)
    )
    assert _resolve_whisper_identity("example/whisper") == WhisperModelIdentity(
        "example/whisper", "b" * 40
    )
    assert _resolve_whisper_identity(str(tmp_path)) is None
    with pytest.raises(ValueError, match="resolved hub commit"):
        WhisperModelIdentity("example/whisper", "main")
