#!/usr/bin/env python3
"""Generate the Rust hardcoded Stanza language table from resources.json.

Outputs the STANZA_SUPPORTED_ISO3 constant for request.rs and the
SUPPORTED_STANZA_CODES set for stanza_languages.rs. Run this after
upgrading Stanza to keep the fallback tables in sync.

Usage:
    uv run scripts/generate_stanza_language_table.py
"""

from batchalign.worker._stanza_capabilities import build_stanza_capability_table


def main() -> None:
    table = build_stanza_capability_table()

    codes = sorted(table.iso3_to_alpha2.keys())

    print(f"// Generated from Stanza {table.stanza_version} resources.json")
    print(f"// {len(codes)} languages")
    print(f"// Run: uv run scripts/generate_stanza_language_table.py")
    print()
    print("const STANZA_SUPPORTED_ISO3: &[&str] = &[")
    for i in range(0, len(codes), 10):
        chunk = codes[i : i + 10]
        line = ", ".join(f'"{c}"' for c in chunk)
        print(f"    {line},")
    print("];")

    print()
    print("// For stanza_languages.rs:")
    print("static SUPPORTED_STANZA_CODES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {")
    print("    [")
    for i in range(0, len(codes), 10):
        chunk = codes[i : i + 10]
        line = ", ".join(f'"{c}"' for c in chunk)
        print(f"        {line},")
    print("    ]")
    print("    .into_iter()")
    print("    .collect()")
    print("});")

    # Also show per-language processor availability
    print()
    print("// Processor availability:")
    for code in codes:
        caps = table.languages[code]
        procs = []
        if caps.has_tokenize:
            procs.append("tok")
        if caps.has_pos:
            procs.append("pos")
        if caps.has_lemma:
            procs.append("lem")
        if caps.has_depparse:
            procs.append("dep")
        if caps.has_mwt:
            procs.append("mwt")
        if caps.has_constituency:
            procs.append("con")
        print(f"//   {code} ({caps.alpha2}): {' '.join(procs)}")


if __name__ == "__main__":
    main()
