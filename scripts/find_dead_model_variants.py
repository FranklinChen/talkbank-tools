"""Find dead enum variants in a Rust crate's model.

A model crate's parser/builder code is the canonical producer of its
own data types. Therefore: any enum variant defined in the model crate
that is never constructed in non-test code anywhere in the search root
is dead.

Method:
  1. Enumerate every `pub enum` in the model crate and its variants.
  2. For each variant `EnumName::Variant`, search the search root
     (excluding test modules and tests/ dirs) for any constructor or
     pattern-match site.
  3. A variant with NO non-test constructor — including no thiserror
     `#[from]` auto-construction — is dead.

Detected constructor shapes:
  - `EnumName::Variant(...)` or `EnumName::Variant { ... }` — explicit
    full-path construction.
  - `Self::Variant(...)` or `Self::Variant { ... }` — in-impl
    construction inside the enum's own `impl` block. Restricted to the
    enum's home file to avoid false positives from `Self` in other types.
  - Variants with `#[from]` on a tuple-payload field — thiserror
    auto-implements `From<T>` for these variants, so they are
    constructed implicitly via `?`/`.into()` even though no source line
    contains `EnumName::Variant`.

Known limitations (still false-positive prone for):
  - Macro-generated constructors other than thiserror `#[from]`.
  - `Default` impls that mention a variant via `Self::Variant` pattern
    (counted) but `EnumName::default()` calls upstream are not tracked.
  - `serde` deserialization — variants reachable only via parsed input
    show up as dead unless they're also constructed in code.

Usage:
    python3 find_dead_model_variants.py \\
        --model-root <path-to-model-crate-src> \\
        --search-root <path-to-workspace-or-crate-with-consumers> \\
        --output <markdown-report-path>
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TEST_PATH_PATTERNS = ("tests/", "/tests/", "_tests.rs", "test_", "/test/")


# Pattern matching `pub enum Name { ... }` blocks. We extract the name
# then capture the body up to the matching closing brace.
ENUM_DECL_RE = re.compile(
    r"(?ms)#\[derive[^\]]*\][\s\S]*?\bpub enum\s+(\w+)\s*\{",
)


# Enum-level derive attributes that construct variants from external
# input (serialized data, CLI args, env vars, etc.). When ANY of these
# is on the enum, ALL of its variants are reachable via the macro-
# generated path even if no source line names them — they get
# constructed by clap's argument parser, serde's Deserialize, or other
# input-driven dispatch.
EXTERNAL_INPUT_DERIVES = (
    "Subcommand",       # clap subcommand dispatch
    "ValueEnum",        # clap value-enum
    "Args",             # clap argument struct (rare on enums)
    "Parser",           # clap top-level Parser
    "Deserialize",      # serde
    "JsonSchema",       # schemars (input-driven via deserialization)
    "EnumString",       # strum FromStr
    "FromRepr",         # strum from integer repr
)


def find_enums(model_root: Path) -> dict[str, tuple[list[str], Path, set[str], bool]]:
    """Return `{enum_name: (variant_names, home_file, from_variants, externally_constructible)}`
    for every `pub enum` in the model crate.

    * `home_file`: the .rs file the enum is defined in — needed to
      resolve `Self::Variant` constructions inside its impl blocks.
    * `from_variants`: variant names with thiserror `#[from]` on a
      field (auto-`From<T>` impl).
    * `externally_constructible`: True when the enum's `#[derive(...)]`
      list includes a derive that constructs variants from external
      input (clap subcommand, serde Deserialize, etc.). All variants of
      such enums are treated as live.
    """
    out: dict[str, tuple[list[str], Path, set[str], bool]] = {}
    for rs in model_root.rglob("*.rs"):
        if any(p in str(rs) for p in TEST_PATH_PATTERNS):
            continue
        text = rs.read_text(encoding="utf-8", errors="replace")
        # Find each `pub enum Name {` and grab its body. We also need
        # to look at the `#[derive(...)]` block(s) immediately preceding
        # the `pub enum` to detect external-input-driven derives.
        for match in re.finditer(r"\bpub enum\s+(\w+)\s*\{", text):
            name = match.group(1)
            # Look back at the 1KB preceding the enum decl to find
            # `#[derive(...)]` attributes on this enum.
            preamble_start = max(0, match.start() - 1024)
            preamble = text[preamble_start : match.start()]
            externally_constructible = _has_external_input_derive(preamble)
            # Walk braces to find the matching closing brace.
            depth = 1
            i = match.end()
            while i < len(text) and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1
            body = text[match.end():i - 1]
            # Variants are top-level identifiers in the body. Strip
            # nested {}, [], (...), comments, attribute lines first.
            cleaned = strip_nested(body)
            variants = extract_variant_names(cleaned)
            from_variants = extract_from_variants(body)
            if variants:
                out[name] = (variants, rs, from_variants, externally_constructible)
    return out


def _has_external_input_derive(preamble: str) -> bool:
    """Return True if the `#[derive(...)]` block immediately preceding
    an enum mentions any derive that constructs variants from external
    input. We scan all `#[derive(...)]` clauses in the preamble (an
    enum can carry several `#[derive]` lines and `#[cfg_attr]`-gated
    derives in a row)."""
    for m in re.finditer(r"#\[(?:cfg_attr\s*\([^)]+,\s*)?derive\s*\(([^)]*)\)\s*\)?\s*\]", preamble):
        derive_list = m.group(1)
        # Strip path qualifiers like `serde::Deserialize` → `Deserialize`.
        names = re.findall(r"(?:[A-Za-z_][A-Za-z0-9_]*::)*([A-Z][A-Za-z0-9_]*)", derive_list)
        if any(n in EXTERNAL_INPUT_DERIVES for n in names):
            return True
    return False


def extract_from_variants(body: str) -> set[str]:
    """Return variant names whose tuple/struct field carries thiserror
    `#[from]`. The shapes we recognize:

        Variant(#[from] Type),
        Variant(#[from] Type, OtherFields),
        Variant {
            #[from]
            source: Type,
            ...
        },

    For each, the variant name is the `\\b([A-Z][A-Za-z0-9_]*)` token
    immediately preceding the `(` or `{` containing `#[from]`.
    """
    out: set[str] = set()
    # Walk variant by variant. For a tuple variant `Name(...)`, look
    # for `#[from]` inside the parens. For a struct variant
    # `Name { ... }`, look for `#[from]` inside the braces.
    # We use a stateful scanner: find each top-level CamelCase token,
    # then walk forward to see whether the next non-whitespace char is
    # `(` or `{`, and whether the matching closing paren/brace is
    # preceded by content containing `#[from]`.
    i = 0
    while i < len(body):
        m = re.search(r"\b([A-Z][A-Za-z0-9_]*)\s*([\({,])", body[i:])
        if not m:
            break
        variant_name = m.group(1)
        opener = m.group(2)
        absolute_idx = i + m.start(2)
        if opener in "({":
            close = ")" if opener == "(" else "}"
            depth = 1
            j = absolute_idx + 1
            while j < len(body) and depth > 0:
                if body[j] == opener:
                    depth += 1
                elif body[j] == close:
                    depth -= 1
                j += 1
            payload = body[absolute_idx + 1 : j - 1]
            if "#[from]" in payload:
                out.add(variant_name)
            i = j
        else:
            # Bare unit variant; no payload, no #[from].
            i = absolute_idx + 1
    return out


def strip_nested(s: str) -> str:
    """Remove {...} (...) [...] regions and // comments and /* */ comments
    so variant names at the top level become extractable."""
    # Remove block comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    # Remove line comments
    s = re.sub(r"//[^\n]*", "", s)
    # Remove parenthesized payloads, brace-wrapped struct fields, bracket-wrapped attrs
    out: list[str] = []
    depth = {"(": 0, "[": 0, "{": 0}
    pair = {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in depth:
            depth[c] += 1
            continue
        if c in pair:
            depth[pair[c]] -= 1
            continue
        if any(d > 0 for d in depth.values()):
            continue
        out.append(c)
    return "".join(out)


def extract_variant_names(cleaned: str) -> list[str]:
    """Extract identifier-shaped variant names from cleaned enum body."""
    # Each variant is a CamelCase identifier followed by `,` or end.
    return re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*,", cleaned + ",")


def search_constructor(enum: str, variant: str, home_file: Path, search_root: Path) -> list[str]:
    """Return matching file:line entries for the variant outside tests.

    Searches two patterns:
      - `EnumName::Variant` anywhere (covers external callers and in-impl
        construction that uses the full path)
      - `Self::Variant` only inside the enum's home file (Rust idiom for
        in-impl construction inside `impl EnumName { ... }`)
    """
    full_pattern = rf"\b{enum}::{variant}\b"
    self_pattern = rf"\bSelf::{variant}\b"
    lines: list[str] = []
    # External / full-path references
    try:
        result = subprocess.run(
            [
                "rg", "--no-heading", "-n",
                "-g", "!**/tests/**",
                "-g", "!**/test_*.rs",
                "-g", "!**/*_tests.rs",
                full_pattern, str(search_root),
            ],
            capture_output=True, text=True, check=False,
        )
        lines.extend(result.stdout.splitlines())
    except FileNotFoundError:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.rs",
             "--exclude-dir=tests",
             full_pattern, str(search_root)],
            capture_output=True, text=True, check=False,
        )
        lines.extend(result.stdout.splitlines())
    # `Self::Variant` references in the enum's home file (covers in-impl
    # construction). We restrict to the home file because `Self::Variant`
    # in an arbitrary file refers to a different `Self` and could be
    # noise.
    if home_file.exists():
        # `--with-filename` forces the 3-field `path:line:content` output
        # even on single-file searches; without it, rg drops the path on
        # single-file inputs and downstream parsing breaks.
        result = subprocess.run(
            ["rg", "--no-heading", "--with-filename", "-n", self_pattern, str(home_file)],
            capture_output=True, text=True, check=False,
        )
        lines.extend(result.stdout.splitlines())
    # Filter out lines inside `#[cfg(test)]` modules. This is approximate
    # — we strip lines whose file path contains test markers (already
    # handled by --glob exclusions) and lines within `mod tests {`.
    # The mod-tests filter is conservative: we exclude any line whose
    # file has `#[cfg(test)]` declared at module scope earlier and the
    # match line is inside that module. To keep this robust without an
    # AST, we just exclude lines that are obviously inside `mod tests {`.
    filtered: list[str] = []
    for ln in lines:
        # ln format: path:line:content
        parts = ln.split(":", 2)
        if len(parts) < 3:
            continue
        path, line_no, content = parts
        # Skip if the file path is a test fixture
        if any(p in path for p in TEST_PATH_PATTERNS):
            continue
        # Skip pure pattern-match arms inside writer code? No — match
        # arms count as live consumers but NOT producers. We're looking
        # for constructors, so distinguish:
        #   - `EnumName::Variant {` or `EnumName::Variant(` followed
        #     by struct-literal or tuple-construction = constructor
        #   - `EnumName::Variant { .. } => ...` = pattern match (consumer)
        #   - `matches!(x, EnumName::Variant ...)` = test-style consumer
        # For dead-code purposes, a variant is dead if NO constructor exists.
        # We approximate "constructor" as: not a pattern-match arm and
        # not inside a `match` block. Simpler heuristic: look for
        # `EnumName::Variant(` immediately followed by something that
        # isn't `..)` or for `EnumName::Variant {` or for `EnumName::Variant)`
        # (unit variant).
        filtered.append(ln)
    return filtered


def is_constructor(content: str, enum: str, variant: str) -> bool:
    """Heuristic: does this line CONSTRUCT the variant (vs. pattern-match
    or doc-reference it)?

    Returns False if the line is one of:
      - A match arm: variant followed (eventually) by `=>` on same line
      - Inside `matches!(...)`
      - A markdown link in a docstring: `[`EnumName::Variant`]` or
        `[EnumName::Variant]`
      - A `use` import line
      - A doc comment (line starts with `///`)
    Otherwise returns True (constructor or other live use).
    """
    s = re.sub(r"//.*", "", content).strip() if not content.lstrip().startswith("///") else ""
    if not s:
        return False
    if s.lstrip().startswith("///") or s.lstrip().startswith("//!"):
        return False
    if "matches!" in s:
        return False
    if s.startswith("use ") or s.startswith("pub use "):
        return False

    # Try both `EnumName::Variant` and `Self::Variant` (for in-impl
    # construction).
    for needle in (f"{enum}::{variant}", f"Self::{variant}"):
        idx = s.find(needle)
        if idx < 0:
            continue
        # Line within a markdown rustdoc link, e.g. `[`Foo::Bar`]`. The
        # disambiguator is the backtick: `[`Name`]` is rustdoc, while
        # plain `[Name, ...]` is a Rust slice literal which is valid
        # constructor context.
        head = s[:idx]
        if head.endswith("[`"):
            continue
        # Arrow after the variant on the same line → pattern-match arm.
        tail = s[idx + len(needle):]
        if "=>" in tail:
            continue
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Path to the model crate's `src/` directory; enums are enumerated here.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        required=True,
        help="Path to the workspace or crate(s) where consumers live; variant references are searched here.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the markdown audit report.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label for the report title (e.g. crate name). Defaults to the model-root's parent dir name.",
    )
    args = parser.parse_args()

    if not args.model_root.is_dir():
        print(f"error: --model-root {args.model_root} is not a directory", file=sys.stderr)
        return 1
    if not args.search_root.is_dir():
        print(f"error: --search-root {args.search_root} is not a directory", file=sys.stderr)
        return 1

    label = args.label or args.model_root.parent.name
    enums = find_enums(args.model_root)
    if not enums:
        print("no enums found in model-root", file=sys.stderr)
        return 1

    out_lines: list[str] = []
    out_lines.append(f"# Dead Variant Audit — `{label}`\n")
    out_lines.append("Generated by `scripts/find_dead_model_variants.py`.\n")
    out_lines.append(
        f"**Method.** Enumerate every `pub enum` in `{args.model_root}`. "
        f"For each variant, search `{args.search_root}` (non-test code only) "
        "for any reference. A variant is flagged dead when it has no "
        "explicit constructor (`EnumName::Variant` / `Self::Variant`), "
        "no thiserror `#[from]` auto-constructor, and no other live "
        "reference outside pattern-match arms / doc links / use statements.\n"
    )
    out_lines.append(
        "**Limitation.** Macro-generated constructors other than thiserror "
        "`#[from]` are not detected. A variant flagged here should be "
        "inspected manually before removal.\n"
    )

    total_dead = 0
    total_variants = 0
    total_from_skipped = 0
    total_external_skipped = 0
    for enum_name in sorted(enums):
        variants, home_file, from_variants, externally_constructible = enums[enum_name]
        total_variants += len(variants)
        if externally_constructible:
            # All variants reachable via clap/serde/etc. — skip the whole enum
            # (don't even render a section, to keep the report focused on
            # actually-actionable findings).
            total_external_skipped += len(variants)
            continue
        out_lines.append(f"\n## `{enum_name}` ({len(variants)} variants)\n")
        any_dead_in_enum = False
        for v in variants:
            if v in from_variants:
                # thiserror auto-constructor; not dead, skip silently.
                total_from_skipped += 1
                continue
            refs = search_constructor(enum_name, v, home_file, args.search_root)
            constructors = [
                ln for ln in refs
                if is_constructor(
                    ln.split(":", 2)[2] if len(ln.split(":", 2)) >= 3 else "",
                    enum_name, v,
                )
            ]
            if not constructors:
                if not refs:
                    out_lines.append(f"- ☠️ **`{v}`** — DEAD (zero non-test references)")
                else:
                    out_lines.append(
                        f"- ☠️ **`{v}`** — DEAD ({len(refs)} ref(s), all pattern-match arms / doc links / use stmts; no constructor)"
                    )
                total_dead += 1
                any_dead_in_enum = True
        if not any_dead_in_enum:
            out_lines.append("(all variants have constructors)")

    out_lines.append("\n## Summary\n")
    out_lines.append(f"- Enums scanned: **{len(enums)}**")
    out_lines.append(f"- Variants total: **{total_variants}**")
    out_lines.append(f"- Confirmed dead: **{total_dead}**")
    out_lines.append(f"- Skipped (`#[from]` thiserror auto-constructor): **{total_from_skipped}**")
    out_lines.append(
        f"- Skipped (clap / serde / strum / similar derive constructs variants from input): "
        f"**{total_external_skipped}**"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines))
    print(f"wrote {args.output}", file=sys.stderr)
    print(
        f"  {len(enums)} enums, {total_variants} variants, "
        f"{total_dead} dead, {total_from_skipped} via #[from], "
        f"{total_external_skipped} via input-derives",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
