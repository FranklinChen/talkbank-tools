"""Report which files BA3 would download for each hub model it loads.

For every model a loader resolves through ``pinned_hub.resolve_transformers``
or ``resolve_repository``, ask the hub for the repository's files and sizes
(metadata only, nothing is downloaded), apply the same selection the resolver
applies (``pinned_hub.default_weights`` for the weight layout), and print what
would be fetched against the whole repository. A Transformers selection that would carry no weights is reported
as an error, because that loader could not load.

Run after adding a hub model or changing a family:

    uv run --no-sync python -m scripts.check_hub_file_selection
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

from huggingface_hub import HfApi

from batchalign.inference.fa import WHISPER_FA_MODEL_ID
from batchalign.inference.machine_ear import CALIBRATED_MODEL_ID
from batchalign.inference.qwen_forced_alignment import QWEN_FORCED_ALIGNER_MODEL_ID
from batchalign.worker._model_loading import pinned_hub
from batchalign.worker._model_loading.asr import DEFAULT_QWEN_ASR_ID, STOCK_WHISPER_ID
from batchalign.worker._model_loading.translation import (
    NLLB_MODEL_ID,
    SEAMLESS_MODEL_ID,
)


class Family(Enum):
    """Which resolve edge a loader goes through."""

    TRANSFORMERS = "transformers"
    """``resolve_transformers``: support files and one weight layout."""

    REPOSITORY = "repository"
    """``resolve_repository``: every file (FunASR)."""


@dataclass(frozen=True)
class Load:
    """One hub model a loader resolves, and the edge it resolves through."""

    model_id: str
    family: Family


LOADS = (
    Load(STOCK_WHISPER_ID, Family.TRANSFORMERS),
    Load(WHISPER_FA_MODEL_ID, Family.TRANSFORMERS),
    Load(DEFAULT_QWEN_ASR_ID, Family.TRANSFORMERS),
    Load(QWEN_FORCED_ALIGNER_MODEL_ID, Family.TRANSFORMERS),
    # Utterance-boundary models: pinned by the Rust manifest, not named in Python.
    Load("talkbank/CHATUtterance-en", Family.TRANSFORMERS),
    Load("talkbank/CHATUtterance-zh_CN", Family.TRANSFORMERS),
    Load("PolyU-AngelChanLab/Cantonese-Utterance-Segmentation", Family.TRANSFORMERS),
    Load(SEAMLESS_MODEL_ID, Family.TRANSFORMERS),
    Load(NLLB_MODEL_ID, Family.TRANSFORMERS),
    Load(CALIBRATED_MODEL_ID, Family.TRANSFORMERS),
    # FunASR SenseVoice members: pinned by the Rust manifest.
    Load("FunAudioLLM/SenseVoiceSmall", Family.REPOSITORY),
    Load("funasr/fsmn-vad", Family.REPOSITORY),
)


def matches(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether a repository file matches any pattern, as ``allow_patterns`` does."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def selection(names: list[str], family: Family) -> list[str] | None:
    """The files the resolver fetches from a repository of ``names``, or
    ``None`` when a Transformers repository has no layout a loader can read.

    The weight layout comes from ``pinned_hub.default_weights``, the rule the
    resolver itself applies, so this report cannot drift from it.
    """
    match family:
        case Family.REPOSITORY:
            return names
        case Family.TRANSFORMERS:
            layout = pinned_hub.default_weights(names)
            if layout is None:
                return None
            return [n for n in names if matches(n, layout.allow_patterns)]
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
        chosen = selection(sorted(sizes), load.family)
        total = sum(sizes.values()) / 1e9
        if chosen is None:
            print(f"{load.model_id} [{load.family.value}]: of {total:.2f} GB")
            print("  ERROR: no weight layout a loader can read")
            failed = True
            continue
        weights = [
            name for name in chosen if name.endswith((".safetensors", ".bin", ".pt"))
        ]
        fetched = sum(sizes[name] for name in chosen) / 1e9
        print(
            f"{load.model_id} [{load.family.value}]: {fetched:.2f} GB of {total:.2f} GB"
        )
        print(f"  weights: {weights}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
