#!/usr/bin/env python3
"""Compare BA3 utterance adjacency policies on retained ASR evidence.

The command performs no ASR request and makes no paid service call.  It loads
one local Hugging Face boundary model, captures raw evidence for every source
monologue in the retained ``*_asr_response.json`` files, and replays both the
current and candidate adjacency policies over that same evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batchalign.models.utterance.infer import BertUtteranceModel
from batchalign.models.utterance.policy_probe import (
    ProbeLanguage,
    ProbeProgress,
    compare_retained_asr_files,
    write_policy_report,
)
from batchalign.worker._model_loading.pinned_hub import resolve_pinned_snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing retained *_asr_response.json artifacts",
    )
    parser.add_argument("output", type=Path, help="Atomic JSON report destination")
    parser.add_argument(
        "--model-id",
        default="talkbank/CHATUtterance-en",
        help="Hugging Face utterance model ID",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="BA3 model language code (default: eng)",
    )
    return parser.parse_args()


def _retained_inputs(input_dir: Path) -> tuple[Path, ...]:
    if not input_dir.is_dir():
        raise SystemExit(f"retained ASR input directory does not exist: {input_dir}")
    inputs = tuple(
        sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.name.endswith("_asr_response.json")
            ),
            key=lambda path: path.name,
        )
    )
    if not inputs:
        raise SystemExit(f"no *_asr_response.json inputs found in {input_dir}")
    return inputs


def _show_progress(progress: ProbeProgress) -> None:
    summary = progress.latest_input
    print(
        f"[{progress.completed_files}/{progress.total_files}] {summary.name}: "
        f"{summary.monologue_count} monologues, {summary.word_count} words, "
        f"{summary.restored_boundary_count} restored boundaries",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    inputs = _retained_inputs(args.input_dir)
    print(
        f"Loading local utterance model {args.model_id} for {len(inputs)} inputs...",
        file=sys.stderr,
        flush=True,
    )
    # The probe is an offline research tool and pins nothing itself, so it
    # resolves the hub default and reports the commit that actually landed.
    # It still goes through the one snapshot resolver, so the model it loads
    # names a real revision rather than none.
    resolved = resolve_pinned_snapshot(args.model_id, None)
    print(f"Resolved {args.model_id} to {resolved.commit}", file=sys.stderr)
    model = BertUtteranceModel(
        model_id=args.model_id,
        model_path=resolved.path,
        model_revision=resolved.commit,
        lang=args.lang,
    )
    print("Model loaded; capturing and replaying boundary evidence...", file=sys.stderr)
    report = compare_retained_asr_files(
        inputs,
        model,
        expected_language=ProbeLanguage.admit(args.lang),
        progress_observer=_show_progress,
    )
    write_policy_report(report, args.output)
    print(
        f"Published {report.assignment_changing_difference_count} "
        f"assignment-changing cases to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
