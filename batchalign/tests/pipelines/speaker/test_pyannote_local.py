"""Contracts for the immutable local-Pyannote model graph."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from batchalign.inference._model_access_errors import ModelAccessDeniedError
from batchalign.inference.pyannote_local import (
    PinnedHuggingFaceArtifact,
    _download_pinned_artifact,
    _get_pyannote_pipeline,
    _load_pipeline,
    load_local_pyannote_model_graph,
    resolve_huggingface_hub_token,
)


@pytest.mark.parametrize(
    "value",
    [
        {"repo_id": "owner/model", "revision": "main", "filename": "config.yaml"},
        {
            "repo_id": "owner / model",
            "revision": "a" * 40,
            "filename": "config.yaml",
        },
        {
            "repo_id": "owner/model",
            "revision": "a" * 40,
            "filename": "../outside.bin",
        },
    ],
)
def test_pinned_artifact_refuses_moving_or_escaping_inputs(value: object) -> None:
    """A parsed artifact is proof of an immutable in-repository object."""

    with pytest.raises(ValueError):
        assert isinstance(value, dict)
        PinnedHuggingFaceArtifact.parse({**value, "sha256": "b" * 64}, field="test")


@pytest.mark.parametrize("tampered", [None, "pipeline", "embedding", "segmentation"])
def test_get_pyannote_pipeline_loads_only_pinned_artifacts(
    monkeypatch, tmp_path: Path, tampered: str | None
) -> None:
    """The runtime must consume the same immutable graph named by the cache."""

    graph = load_local_pyannote_model_graph()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  name: pyannote.audio.pipelines.SpeakerDiarization\n"
        "  params:\n"
        "    segmentation: moving-segmentation-head\n"
        "    embedding: moving-embedding-head\n"
        "params: {}\n"
    )
    embedding_path = tmp_path / "speaker-embedding.onnx"
    embedding_path.write_bytes(b"fixture")
    segmentation_path = tmp_path / "pytorch_model.bin"
    segmentation_path.write_bytes(b"segmentation fixture")
    graph = replace(
        graph,
        pipeline=replace(
            graph.pipeline, sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()
        ),
        embedding=replace(
            graph.embedding,
            sha256=hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
        ),
        segmentation=replace(
            graph.segmentation,
            sha256=hashlib.sha256(segmentation_path.read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.load_local_pyannote_model_graph",
        lambda: graph,
    )
    downloads: list[tuple[str, str, str, str | None]] = []

    def fake_download(
        *, repo_id: str, filename: str, revision: str, token: str | None
    ) -> str:
        downloads.append((repo_id, filename, revision, token))
        if repo_id == graph.pipeline.repo_id:
            return str(config_path)
        if repo_id == graph.embedding.repo_id:
            return str(embedding_path)
        if repo_id == graph.segmentation.repo_id:
            return str(segmentation_path)
        raise AssertionError(f"unexpected eager download: {repo_id}")

    loaded: list[dict[str, object]] = []
    load_tokens: list[str | None] = []

    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.hf_hub_download", fake_download
    )
    monkeypatch.setattr(
        "batchalign.inference.pyannote_local._load_pipeline",
        lambda config, *, token: (
            loaded.append(config) or load_tokens.append(token) or {"pipeline": config}
        ),
    )
    monkeypatch.setattr(
        "batchalign.inference.pyannote_local._emit_model_download_if_missing",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.resolve_huggingface_hub_token",
        lambda: "stub-token",
    )
    monkeypatch.setattr("batchalign.inference.pyannote_local._PYANNOTE_PIPELINE", None)

    if tampered is not None:
        sources = {
            "pipeline": config_path,
            "embedding": embedding_path,
            "segmentation": segmentation_path,
        }
        sources[tampered].write_bytes(b"changed before admission")
        with pytest.raises(ValueError, match="integrity mismatch"):
            _get_pyannote_pipeline()
        assert loaded == []
        return
    first = _get_pyannote_pipeline()
    second = _get_pyannote_pipeline()

    assert first is second
    assert downloads == [
        (
            graph.pipeline.repo_id,
            graph.pipeline.filename,
            graph.pipeline.revision,
            "stub-token",
        ),
        (
            graph.embedding.repo_id,
            graph.embedding.filename,
            graph.embedding.revision,
            "stub-token",
        ),
        (
            graph.segmentation.repo_id,
            graph.segmentation.filename,
            graph.segmentation.revision,
            "stub-token",
        ),
    ]
    assert load_tokens == ["stub-token"]
    pipeline = loaded[0]["pipeline"]
    assert isinstance(pipeline, dict)
    params = pipeline["params"]
    assert isinstance(params, dict)
    segmentation = params["segmentation"]
    assert isinstance(segmentation, dict)
    snapshot = Path(segmentation["checkpoint"])
    assert snapshot != segmentation_path
    segmentation_path.write_bytes(b"changed after admission")
    assert snapshot.read_bytes() == b"segmentation fixture"
    assert Path(params["embedding"]).read_bytes() == b"fixture"
    # Exercise the actual upstream dispatch seam: an anonymous local ONNX
    # path falls through to the pickle loader despite passing our mocks.
    from pyannote.audio.pipelines import speaker_verification

    selected = object()
    monkeypatch.setattr(
        speaker_verification,
        "ONNXWeSpeakerPretrainedSpeakerEmbedding",
        lambda *args, **kwargs: selected,
    )
    assert (
        speaker_verification.PretrainedSpeakerEmbedding(params["embedding"]) is selected
    )


def test_resolve_huggingface_hub_token_prefers_the_batchalign_ini(
    tmp_path: Path, monkeypatch
) -> None:
    """The operator's own `.batchalign.ini [auth] hf_token` wins first."""

    config_path = tmp_path / ".batchalign.ini"
    config_path.write_text("[auth]\nhf_token = ini-token\n")

    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.get_token", lambda: "hub-fallback-token"
    )

    assert resolve_huggingface_hub_token(config_path=config_path) == "ini-token"


def test_resolve_huggingface_hub_token_falls_back_to_huggingface_hub(
    tmp_path: Path, monkeypatch
) -> None:
    """With no ini token, fall back to huggingface_hub's own resolution."""

    config_path = tmp_path / ".batchalign.ini"
    config_path.write_text("[diarize]\nengine.pyannote.key = unrelated\n")

    monkeypatch.setattr(
        "batchalign.inference.pyannote_local.get_token", lambda: "hub-fallback-token"
    )

    assert (
        resolve_huggingface_hub_token(config_path=config_path) == "hub-fallback-token"
    )


def test_resolve_huggingface_hub_token_is_none_when_neither_source_has_one(
    tmp_path: Path, monkeypatch
) -> None:
    """No ini token and no logged-in Hub session must resolve to `None`, not
    an empty string a caller might mistake for a real (invalid) token."""

    config_path = tmp_path / "missing.batchalign.ini"

    monkeypatch.setattr("batchalign.inference.pyannote_local.get_token", lambda: None)

    assert resolve_huggingface_hub_token(config_path=config_path) is None


def test_load_pipeline_reclassifies_a_gated_repo_error() -> None:
    """A gated-repository load failure must surface as the typed access error.

    This is the exact failure shape from the 2026-09-02 report: pyannote's
    ``Pipeline.from_pretrained`` unconditionally fetches a PLDA artifact whose
    default points at a gated Hub repository, and the resulting
    ``GatedRepoError`` used to propagate as an undifferentiated crash that the
    server reported as a generic pipeline bug rather than a configuration
    condition on the operator's machine.
    """

    from huggingface_hub.errors import GatedRepoError
    from requests import Response

    response = Response()
    response.status_code = 403
    response.url = (
        "https://huggingface.co/pyannote/speaker-diarization-community-1/"
        "resolve/main/plda/xvec_transform.npz"
    )
    gated = GatedRepoError("403 Client Error: gated repo", response=response)

    class _RaisingPipeline:
        @staticmethod
        def from_pretrained(_config: object, *, token: str | None) -> object:
            raise gated

    with pytest.raises(ModelAccessDeniedError) as excinfo:
        with _swap_pyannote_pipeline_class(_RaisingPipeline):
            _load_pipeline({}, token=None)

    assert excinfo.value.repo_id == "pyannote/speaker-diarization-community-1"
    assert excinfo.value.__cause__ is gated


def test_download_pinned_artifact_reclassifies_a_gated_repo_error() -> None:
    """The same reclassification applies to a pinned-artifact download."""

    from huggingface_hub.errors import GatedRepoError
    from requests import Response

    response = Response()
    response.status_code = 403
    response.url = "https://huggingface.co/talkbank/dia-fork/resolve/main/config.yaml"
    gated = GatedRepoError("403 Client Error: gated repo", response=response)

    def fake_download(
        *, repo_id: str, filename: str, revision: str, token: str | None
    ) -> str:
        raise gated

    artifact = PinnedHuggingFaceArtifact(
        repo_id="talkbank/dia-fork",
        revision="a" * 40,
        filename="config.yaml",
        sha256="b" * 64,
    )

    with pytest.raises(ModelAccessDeniedError) as excinfo:
        with _swap_hf_hub_download(fake_download):
            _download_pinned_artifact(artifact, token=None)

    assert excinfo.value.repo_id == "talkbank/dia-fork"
    assert excinfo.value.__cause__ is gated


def test_load_pipeline_does_not_reclassify_an_unrelated_failure() -> None:
    """A non-Hub-access exception must propagate unchanged, not be swallowed."""

    class _RaisingPipeline:
        @staticmethod
        def from_pretrained(_config: object, *, token: str | None) -> object:
            raise ValueError("some other pyannote construction failure")

    with pytest.raises(ValueError, match="some other pyannote construction failure"):
        with _swap_pyannote_pipeline_class(_RaisingPipeline):
            _load_pipeline({}, token=None)


class _swap_pyannote_pipeline_class:
    """Context manager swapping ``pyannote.audio.Pipeline`` for one call."""

    def __init__(self, replacement: object) -> None:
        self._replacement = replacement
        self._original: object | None = None

    def __enter__(self) -> None:
        import pyannote.audio

        self._original = pyannote.audio.Pipeline
        pyannote.audio.Pipeline = self._replacement  # type: ignore[misc]

    def __exit__(self, *_exc_info: object) -> None:
        import pyannote.audio

        pyannote.audio.Pipeline = self._original  # type: ignore[misc]


class _swap_hf_hub_download:
    """Context manager swapping the module-level ``hf_hub_download`` for one call."""

    def __init__(self, replacement: object) -> None:
        self._replacement = replacement
        self._original: object | None = None

    def __enter__(self) -> None:
        import batchalign.inference.pyannote_local as module

        self._original = module.hf_hub_download
        module.hf_hub_download = self._replacement  # type: ignore[assignment]

    def __exit__(self, *_exc_info: object) -> None:
        import batchalign.inference.pyannote_local as module

        module.hf_hub_download = self._original  # type: ignore[assignment]
