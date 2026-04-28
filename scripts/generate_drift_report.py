#!/usr/bin/env python
"""Generate a drift-probe report from the Phase B test-history DB.

Phase G of the test-cost revamp. Invoked by
``scripts/run-drift-probes.sh`` after a probe run, or manually by a
developer wanting to inspect the current drift surface.

Usage::

    python scripts/generate_drift_report.py \\
        --db ~/.batchalign3/drift-probe-history.sqlite \\
        --output-dir ~/workspace/docs/investigations \\
        --window-days 1

Behavior:

* Reads the Phase B history schema (``test_runs`` table).
* Queries the most recent run per probe within the window.
* Writes ``docs/investigations/drift-<date>-stanza.md``.
* Prints the output path on stdout.
* **Exit code is 0 even when drift is present** — probes are
  monitors, not tests. A calling CI workflow decides whether to
  open an issue based on the report's ``**Status:**`` line.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from batchalign.tests._drift_report import (  # noqa: E402
    build_drift_report,
    format_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the probe-history SQLite DB.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the dated report into.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=1,
        help="How far back to look for probe runs (default: 1 day).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    since_ts = int(time.time()) - args.window_days * 24 * 3600
    report = build_drift_report(db_path, since_ts=since_ts)

    run_date = time.strftime("%Y-%m-%d")
    report_path = out_dir / f"drift-{run_date}-stanza.md"
    report_path.write_text(format_markdown(report, run_date=run_date))
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
