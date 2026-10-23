"""Write the installed Stanza catalog's shape as a test fixture.

The language census (``batchalign/tests/test_stanza_language_census.py``)
builds the worker's capability table from a real catalog without a network
fetch or a user cache, so it needs the catalog committed. Only the shape the
builder reads is kept: each language entry's processor names (as empty
objects) and each alias, which is a few kilobytes instead of the catalog's
megabyte of model URLs and checksums.

Usage, after a Stanza upgrade (the fixture names the version it came from):

    uv run python scripts/snapshot_stanza_catalog.py

Then rerun the census and update its expectations deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

PROCESSORS = ("tokenize", "mwt", "pos", "lemma", "depparse", "constituency", "coref")
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "batchalign"
    / "tests"
    / "fixtures"
    / "stanza_catalog_shape.json"
)


def main() -> None:
    import stanza
    import stanza.resources.common as resources

    catalog = resources.load_resources_json()
    shape: dict[str, object] = {}
    for code, entry in sorted(catalog.items()):
        if not isinstance(entry, dict):
            continue
        alias = entry.get("alias")
        if isinstance(alias, str):
            shape[code] = {"alias": alias}
        elif "tokenize" in entry:
            shape[code] = {name: {} for name in PROCESSORS if name in entry}
    FIXTURE.write_text(
        json.dumps(
            {"stanza_version": stanza.__version__, "resources": shape},
            indent=1,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {FIXTURE} from Stanza {stanza.__version__}")


if __name__ == "__main__":
    main()
