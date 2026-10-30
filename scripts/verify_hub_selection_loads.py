"""Prove each hub model loads from only the files its family selects.

``check_hub_file_selection`` shows what WOULD download; this downloads it.
Each model is resolved by the real ``resolve_pinned_snapshot`` into an EMPTY
Hugging Face cache, so nothing but the selected files exists, then loaded
with the same ``from_pretrained`` calls BA3's loaders make. A file the
selection missed fails the load here instead of in a user's job.

The cache must be empty, so run with a fresh ``HF_HUB_CACHE`` (read when
``huggingface_hub`` is imported, so it has to be set in the environment):

    HF_HUB_CACHE=$(mktemp -d) uv run --no-sync python -m scripts.verify_hub_selection_loads

It downloads several GB (Whisper alone is 3 GB); delete the cache after.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from batchalign.worker._model_loading.pinned_hub import (
    HubFiles,
    resolve_pinned_snapshot,
)


def weights_in(path: str) -> list[str]:
    """Weight files present in a resolved snapshot."""
    return sorted(
        entry.name
        for entry in Path(path).iterdir()
        if entry.suffix in (".safetensors", ".bin", ".pt", ".msgpack", ".h5")
    )


def load_whisper(path: str) -> None:
    from transformers import (
        GenerationConfig,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    GenerationConfig.from_pretrained(path)
    WhisperTokenizer.from_pretrained(path)
    WhisperProcessor.from_pretrained(path)
    WhisperForConditionalGeneration.from_pretrained(path)


def load_bert(path: str) -> None:
    from transformers import AutoTokenizer, BertForTokenClassification

    AutoTokenizer.from_pretrained(path)
    BertForTokenClassification.from_pretrained(path)


def load_qwen_aligner(path: str) -> None:
    from transformers import AutoProcessor, Qwen3ASRForTokenClassification

    AutoProcessor.from_pretrained(path)  # type: ignore[no-untyped-call]
    Qwen3ASRForTokenClassification.from_pretrained(path)


CHECKS = (
    ("openai/whisper-large-v3", load_whisper),
    ("talkbank/CHATUtterance-en", load_bert),
    ("Qwen/Qwen3-ForcedAligner-0.6B-hf", load_qwen_aligner),
)


def main() -> int:
    cache = os.environ.get("HF_HUB_CACHE", "")
    if not cache or any(Path(cache).iterdir()):
        print("HF_HUB_CACHE must name an empty directory", file=sys.stderr)
        return 2
    failed = False
    for model_id, load in CHECKS:
        resolved = resolve_pinned_snapshot(model_id, None, HubFiles.TRANSFORMERS)
        print(f"{model_id} @ {resolved.commit}: weights {weights_in(resolved.path)}")
        try:
            load(resolved.path)
        # Every load failure is the finding, whatever its type.
        except Exception as error:
            print(f"  LOAD FAILED: {error}")
            failed = True
        else:
            print("  loaded")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
