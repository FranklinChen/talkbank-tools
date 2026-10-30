"""Report which files BA3 would download for each hub model it loads.

For every model a loader resolves through ``pinned_hub.resolve_pinned_snapshot``,
ask the hub for the repository's files and sizes (metadata only, nothing is
downloaded), apply the same selection the resolver applies for the loader's
``HubFiles`` family, and print what would be fetched against the whole
repository. A Transformers selection that would carry no weights is reported
as an error, because that loader could not load.

Run after adding a hub model or changing a family:

    uv run --no-sync python -m scripts.check_hub_file_selection
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass
from typing import assert_never

from huggingface_hub import HfApi

from batchalign.worker._model_loading import pinned_hub
from batchalign.worker._model_loading.pinned_hub import HubFiles


@dataclass(frozen=True)
class Load:
    """One hub model a loader resolves, and the family it declares."""

    model_id: str
    files: HubFiles


LOADS = (
    Load("openai/whisper-large-v3", HubFiles.TRANSFORMERS),
    Load("Qwen/Qwen3-ASR-1.7B-hf", HubFiles.TRANSFORMERS),
    Load("Qwen/Qwen3-ForcedAligner-0.6B-hf", HubFiles.TRANSFORMERS),
    Load("talkbank/CHATUtterance-en", HubFiles.TRANSFORMERS),
    Load("talkbank/CHATUtterance-zh_CN", HubFiles.TRANSFORMERS),
    Load("PolyU-AngelChanLab/Cantonese-Utterance-Segmentation", HubFiles.TRANSFORMERS),
    Load("FunAudioLLM/SenseVoiceSmall", HubFiles.WHOLE_REPOSITORY),
    Load("funasr/fsmn-vad", HubFiles.WHOLE_REPOSITORY),
)


def matches(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether a repository file matches any pattern, as ``allow_patterns`` does."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def selection(names: list[str], files: HubFiles) -> list[str]:
    """The files the resolver fetches for ``files`` from a repository of ``names``."""
    match files:
        case HubFiles.WHOLE_REPOSITORY:
            return names
        case HubFiles.TRANSFORMERS:
            support = [n for n in names if matches(n, pinned_hub._TRANSFORMERS_SUPPORT)]
            safetensors = [
                n for n in names if matches(n, pinned_hub._SAFETENSORS_WEIGHTS)
            ]
            if safetensors:
                return support + safetensors
            return support + [
                n for n in names if matches(n, pinned_hub._PYTORCH_WEIGHTS)
            ]
        case _ as unreachable:
            assert_never(unreachable)


def main() -> int:
    api = HfApi()
    failed = False
    for load in LOADS:
        info = api.model_info(load.model_id, files_metadata=True)
        sizes = {
            sibling.rfilename: sibling.size or 0 for sibling in info.siblings or []
        }
        chosen = selection(sorted(sizes), load.files)
        weights = [
            name for name in chosen if name.endswith((".safetensors", ".bin", ".pt"))
        ]
        total = sum(sizes.values()) / 1e9
        fetched = sum(sizes[name] for name in chosen) / 1e9
        print(
            f"{load.model_id} [{load.files.value}]: {fetched:.2f} GB of {total:.2f} GB"
        )
        print(f"  weights: {weights}")
        if not weights:
            print("  ERROR: no weights selected; the loader could not load")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
