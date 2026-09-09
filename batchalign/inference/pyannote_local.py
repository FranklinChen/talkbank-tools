"""Immutable local-Pyannote model graph and lazy worker-local loader."""

from __future__ import annotations

import configparser
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import get_token, hf_hub_download
from omegaconf import OmegaConf

_MODEL_MANIFEST_RESOURCE = "local_pyannote_model.json"
_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")
_HUB_REPOSITORY_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
_PYANNOTE_PIPELINE: _LoadedPipeline | None = None


@dataclass(frozen=True)
class _LoadedPipeline:
    """Keep verified private model files alive for lazy backend reads."""

    pipeline: object
    files: tempfile.TemporaryDirectory[str]


@dataclass(frozen=True)
class PinnedHuggingFaceArtifact:
    """One Hub artifact whose mutable repository name has an exact commit."""

    repo_id: str
    revision: str
    filename: str
    sha256: str

    @classmethod
    def parse(cls, value: object, *, field: str) -> PinnedHuggingFaceArtifact:
        if not isinstance(value, dict) or set(value) != {
            "repo_id",
            "revision",
            "filename",
            "sha256",
        }:
            raise ValueError(
                f"local Pyannote {field} must name exactly one pinned artifact"
            )
        repo_id = value["repo_id"]
        revision = value["revision"]
        filename = value["filename"]
        sha256 = value["sha256"]
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"local Pyannote {field} must name a SHA-256 digest")
        if (
            not isinstance(repo_id, str)
            or _HUB_REPOSITORY_ID.fullmatch(repo_id) is None
        ):
            raise ValueError(f"local Pyannote {field} has an invalid Hub repository id")
        if (
            not isinstance(revision, str)
            or _COMMIT_REVISION.fullmatch(revision) is None
        ):
            raise ValueError(
                f"local Pyannote {field} must use a 40-hex commit revision"
            )
        if not isinstance(filename, str):
            raise ValueError(f"local Pyannote {field} filename must be a string")
        relative = PurePosixPath(filename)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(
                f"local Pyannote {field} filename must stay within its repository"
            )
        return cls(repo_id=repo_id, revision=revision, filename=filename, sha256=sha256)


@dataclass(frozen=True)
class LocalPyannoteModelGraph:
    """Complete immutable dependency graph for the released local backend."""

    pipeline: PinnedHuggingFaceArtifact
    segmentation: PinnedHuggingFaceArtifact
    embedding: PinnedHuggingFaceArtifact


@lru_cache(maxsize=1)
def load_local_pyannote_model_graph() -> LocalPyannoteModelGraph:
    """Validate the packaged graph before any model download is authorized."""

    raw = json.loads(
        resources.files("batchalign.inference")
        .joinpath(_MODEL_MANIFEST_RESOURCE)
        .read_text(encoding="utf-8")
    )
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "pipeline",
        "segmentation",
        "embedding",
    }:
        raise ValueError("local Pyannote manifest has an unsupported shape")
    if raw["schema_version"] != 1:
        raise ValueError("local Pyannote manifest has an unsupported schema version")
    return LocalPyannoteModelGraph(
        pipeline=PinnedHuggingFaceArtifact.parse(raw["pipeline"], field="pipeline"),
        segmentation=PinnedHuggingFaceArtifact.parse(
            raw["segmentation"], field="segmentation"
        ),
        embedding=PinnedHuggingFaceArtifact.parse(raw["embedding"], field="embedding"),
    )


def resolve_huggingface_hub_token(*, config_path: Path | None = None) -> str | None:
    """Resolve a Hugging Face Hub token, preferring the operator's own config.

    Checked in order:

    1. ``~/.batchalign.ini`` ``[auth] hf_token`` (or ``config_path`` when
       given) — the same file batchalign's other provider keys already live
       in (see ``[diarize] engine.pyannote.key`` in ``pyannote_ai.py``).
    2. ``huggingface_hub``'s own resolution (the ``HF_TOKEN`` environment
       variable, then the token cached by ``hf auth login``).

    Returns ``None`` when neither source has a token, never an empty string:
    a caller must be able to tell "no token" from "a token, which happens to
    be empty", and the pinned artifacts download anonymously when no token is
    supplied, which is correct for every artifact that is not gated.
    """

    path = config_path or Path.home() / ".batchalign.ini"
    parser = configparser.ConfigParser()
    try:
        loaded = parser.read(path)
    except (configparser.Error, OSError):
        loaded = []
    if loaded and parser.has_section("auth"):
        value = parser.get("auth", "hf_token", fallback="").strip()
        if value:
            return value

    return get_token()


def _load_pipeline(config: dict[str, Any], *, token: str | None) -> object | None:
    """Keep the heavyweight Pyannote import behind the lazy model boundary."""

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
    except ImportError as exc:
        raise ImportError(
            "Speaker diarization requires pyannote.audio, which is not installed.\n"
            "Reinstall the standard batchalign3 package and confirm "
            "'import pyannote.audio' works in the worker Python runtime."
        ) from exc

    try:
        return PyannotePipeline.from_pretrained(config, token=token)
    except Exception as error:
        raise _reclassified_access_error(error) from error


def _emit_model_download_if_missing(artifact: PinnedHuggingFaceArtifact) -> None:
    """Import progress reporting lazily to avoid the worker protocol cycle."""

    from batchalign.worker._progress import emit_hf_download_if_missing

    emit_hf_download_if_missing(
        artifact.repo_id,
        kind="speaker diarization",
        artifacts=(artifact.filename,),
        revision=artifact.revision,
    )


def _download_pinned_artifact(
    artifact: PinnedHuggingFaceArtifact, *, token: str | None
) -> str:
    """``hf_hub_download`` one pinned artifact, reclassifying a Hub access failure."""

    try:
        return hf_hub_download(
            repo_id=artifact.repo_id,
            filename=artifact.filename,
            revision=artifact.revision,
            token=token,
        )
    except Exception as error:
        raise _reclassified_access_error(error) from error


def _reclassified_access_error(error: Exception) -> Exception:
    """Return the exception that should actually be raised for ``error``.

    A Hub access/credential failure (gated repository, missing/invalid
    token, no cached copy while offline) becomes the typed
    :class:`ModelAccessDeniedError`; every other exception is returned
    unchanged so the caller's ``raise ... from error`` re-raises it verbatim.
    """

    from batchalign.inference._model_access_errors import (
        classify_huggingface_access_error,
    )

    access_error = classify_huggingface_access_error(error)
    return access_error if access_error is not None else error


@dataclass(frozen=True)
class _VerifiedArtifact:
    """Private snapshot admitted by the packaged artifact's complete digest."""

    path: Path

    @classmethod
    def admit(
        cls, artifact: PinnedHuggingFaceArtifact, directory: Path, *, token: str | None
    ) -> _VerifiedArtifact:
        source = Path(_download_pinned_artifact(artifact, token=token))
        # Pyannote's embedding factory dispatches on the repository name in
        # a local path (for example, "wespeaker" selects its ONNX backend).
        # Keep that validated component as well as preventing graph-node
        # filename collisions inside our private snapshot.
        target = directory / artifact.repo_id / artifact.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        # The enclosing private directory owns cleanup on every refusal.
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            while block := incoming.read(1024 * 1024):
                digest.update(block)
                outgoing.write(block)
        if digest.hexdigest() != artifact.sha256:
            raise ValueError(
                f"model artifact integrity mismatch: {artifact.repo_id}/{artifact.filename}"
            )
        return cls(target)


def _pinned_pipeline_config(
    graph: LocalPyannoteModelGraph, *, token: str | None, directory: Path
) -> dict[str, Any]:
    """Materialize a config whose transitive model references cannot move."""

    pipeline_config_path = _VerifiedArtifact.admit(
        graph.pipeline, directory, token=token
    ).path
    embedding_path = _VerifiedArtifact.admit(
        graph.embedding, directory, token=token
    ).path
    segmentation_path = _VerifiedArtifact.admit(
        graph.segmentation, directory, token=token
    ).path
    loaded = OmegaConf.to_container(
        OmegaConf.load(Path(pipeline_config_path)), resolve=True
    )
    if not isinstance(loaded, dict):
        raise ValueError("pinned local Pyannote config is not a mapping")
    pipeline = loaded.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("pinned local Pyannote config has no pipeline mapping")
    params = pipeline.get("params")
    if not isinstance(params, dict):
        raise ValueError("pinned local Pyannote config has no parameter mapping")

    params["segmentation"] = {
        "checkpoint": str(segmentation_path),
    }
    params["embedding"] = str(embedding_path)
    return loaded


def _get_pyannote_pipeline() -> object:
    """Return one lazily loaded, immutable model graph per worker process."""

    global _PYANNOTE_PIPELINE

    if _PYANNOTE_PIPELINE is None:
        graph = load_local_pyannote_model_graph()
        token = resolve_huggingface_hub_token()
        for artifact in (graph.pipeline, graph.segmentation, graph.embedding):
            _emit_model_download_if_missing(artifact)
        files = tempfile.TemporaryDirectory(prefix="batchalign-model-")
        try:
            pipeline = _load_pipeline(
                _pinned_pipeline_config(graph, token=token, directory=Path(files.name)),
                token=token,
            )
            if pipeline is None:
                raise RuntimeError(
                    "pinned local Pyannote model graph could not be loaded"
                )
        except BaseException:
            files.cleanup()
            raise
        _PYANNOTE_PIPELINE = _LoadedPipeline(pipeline, files)
    return _PYANNOTE_PIPELINE.pipeline
