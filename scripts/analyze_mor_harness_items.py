#!/usr/bin/env python3
"""Analyze `mor_render_harness collect` items with the real morphosyntax worker.

The Stanza half of the deploy-free mapping harness
(`crates/batchalign-transform/examples/mor_render_harness.rs`, whose module
docs show the whole loop). Each item goes through `batch_infer_morphosyntax`,
the worker's own entry point, grouped by language exactly as the server sends
a batch, and each outcome is saved as one JSON line:

    {"file": ..., "line_idx": ..., "analysis": {"kind": "analyzed", "raw_sentences": [...]}}

with `"kind": "no_words"` or `{"kind": "failed", "error": ...}` for the other
two outcomes. Run it once per Stanza version; render as often as needed.

    uv run --no-sync python scripts/analyze_mor_harness_items.py items.jsonl analyses.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any


def _analysis_of(error: str | None, result: object) -> dict[str, Any]:
    """One worker outcome as the harness saves it; an unexpected shape stops the run."""
    if error is not None:
        return {"kind": "failed", "error": error}
    if not isinstance(result, dict):
        raise SystemExit(f"worker result is not an object: {result!r}")
    match result.get("kind"):
        case "analyzed":
            return {"kind": "analyzed", "raw_sentences": result["raw_sentences"]}
        case "no_words":
            return {"kind": "no_words"}
        case other:
            raise SystemExit(f"unexpected worker result kind: {other!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "items", type=Path, help="JSONL written by `mor_render_harness collect`"
    )
    parser.add_argument("analyses", type=Path, help="JSONL to write")
    args = parser.parse_args(argv)

    from batchalign.inference.morphosyntax import batch_infer_morphosyntax
    from batchalign.worker._stanza_loading import load_stanza_models
    from batchalign.worker._types import BatchInferRequest, InferTask, _state

    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with args.items.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_lang[record["item"]["lang"]].append(record)

    written = 0
    with args.analyses.open("w", encoding="utf-8") as out:
        for lang in sorted(by_lang):
            records = by_lang[lang]
            load_stanza_models(lang)
            if _state.stanza_pipelines is None:
                raise SystemExit(f"no Stanza pipelines loaded for {lang}")
            request = BatchInferRequest(
                task=InferTask.MORPHOSYNTAX,
                items=[r["item"] for r in records],
                lang=lang,
                retokenize=False,
                mwt={},
            )
            response = batch_infer_morphosyntax(
                request,
                pipelines=_state.stanza_pipelines,
                nlp_lock=threading.Lock(),
                free_threaded=False,
            )
            for record, result in zip(records, response.results, strict=True):
                analysis = _analysis_of(result.error, result.result)
                json.dump(
                    {
                        "file": record["file"],
                        "line_idx": record["line_idx"],
                        "analysis": analysis,
                    },
                    out,
                    ensure_ascii=False,
                )
                out.write("\n")
                written += 1
            print(f"{lang}: {len(records)} items", file=sys.stderr)
    print(f"wrote {written} analyses to {args.analyses}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
