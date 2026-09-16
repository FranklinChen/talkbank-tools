# Number Expansion

**Status:** Current
**Last updated:** 2026-09-15 09:35 EDT

ASR engines emit digit-bearing tokens (`"3"`, `"$5"`, `"1950s"`,
`"3rd"`, `"80%"`) that the CHAT format does not allow on the main
tier for most languages (the validator rejects them as **E220**).
Number expansion rewrites those tokens to language-appropriate word
forms before they reach validation.

> **For developers:** the architecture and per-language
> coverage matrix live at
> [Architecture → Number Expansion](../architecture/number-expansion.md).
> That page is the single source of truth and is kept in lock-step
> with the implementation.

## What expansion does to your output

| Input token | Output (language) |
|------------|--------------------|
| `"3"` (eng) | `"three"` |
| `"3"` (mal, Malayalam) | `"മൂന്ന്"` |
| `"3"` (zho / cmn) | `"三"` |
| `"3"` (jpn / yue) | `"三"` (traditional script) |
| `"3rd"` (eng) | `"third"` |
| `"21st"` (eng) | `"twenty-first"` |
| `"1950s"` (eng) | `"nineteen fifties"` |
| `"54ª"` (por) | `"quinquagésima quarta"` |
| `"1.º"` (por) | `"primeiro"` |
| `"54.ºs"` (por) | `"quinquagésimos quartos"` |
| `"$12"` (any) | `"twelve dollars"` |
| `"€50"` (any) | `"fifty euros"` |
| `"80%"` (eng) | `"eighty percent"` |
| `"21-22"` (eng) | `"twenty-one twenty-two"` |
| `"3"` (cym / vie / nan / min / hak) | `"3"` (validator allows digits inline) |

Expansion is fully deterministic, no ML model, no audio context.
A bug-for-bug repeat of the same ASR output produces the same
expanded text.

## Coverage

The number-expansion table at
`crates/batchalign-transform/data/num2lang.json` covers the long tail
of European, Indic, East Asian, and Semitic languages, the active
list is the JSON file itself; treat it as the canonical source. Most
entries are codegenned from the Python `num2words` library at build
time; a handful are hand-curated where `num2words` is missing the
language or has known defects (Malayalam `mal`, Greek `ell`, Basque
`eus`, Croatian `hrv`).

CJK languages route through the dedicated `num2chinese` converter
(Mandarin → simplified, Cantonese / Japanese → traditional).

Languages whose CHAT validator already accepts inline digits need no
expansion to pass validation. Those with a table (Welsh `cym`,
Vietnamese `vie`, Thai `tha`) are still expanded; those without one
(Min Nan `nan`, Minangkabau `min`, Hakka `hak`) keep the digit, which
the validator accepts.

Languages outside this set hit the validator as E220. To add one,
see the [Adding Language Support](../developer/adding-language-support.md)
checklist's number-expansion section.

## English-specific extras

Beyond cardinals (every covered language), English also has:

- **Ordinals**: `"3rd"` → `"third"`, `"21st"` → `"twenty-first"`,
  `"1234th"` → `"one thousand two hundred and thirty-fourth"`.
- **Decades**: `"1950s"` → `"nineteen fifties"`, `"80s"` → `"eighties"`.
- **Years** (when surrounded by year-form context): handled by the
  ordinal/year/decade composer.

These English-only modes are deterministic Rust composition rules
cross-validated against `num2words` output for every value in the
covered range.

## Portuguese ordinals

Portuguese ordinals written as printed abbreviations expand with the
gender and number the abbreviation marks, for ranks 1 through 1000:
`54ª` → `quinquagésima quarta`, `54º` → `quinquagésimo quarto`,
`3.ª` → `terceira`, `54.ºs` → `quinquagésimos quartos`. The period in
`54.ª` belongs to the abbreviation and does not end the utterance; a
period after it still does. Outside that range (`0ª`, `1001.º`) the
token is left as written and validation reports E220.

## Other ordinal conventions

Other ordinal or decade conventions (Spanish `"3º"`, German `"3."`,
French `"1950s"`) currently pass the digit through; no observed
production traffic has needed them. File a request if your corpus
contains them; the implementation pattern is the same as for English
and Portuguese.

## Currency, percent, and dash ranges

These are **language-agnostic** symbol patterns, expanded by Rust
regardless of target language:

- `$ € £ ¥ ₹ ₩ ₽` prefix or suffix → cardinal expansion of the
  digit portion + English currency word ("dollars", "euros",
  "pounds", …). Rationale: morphosyntax can re-tag in-language
  later; CHAT just needs *some* non-digit word.
- `%` suffix → cardinal + per-language percent word
  (English "percent", Spanish "por ciento", etc.); falls back to
  "percent" for unlisted languages.
- `5-7` or `5-6` → split into `"five seven"` / `"five six"` (em-dashes
  normalize to hyphens; both parts must be pure digits).

## When expansion fails

If a token genuinely cannot be expanded (a language with no table, an
ordinal or decade convention without an expander, a Portuguese ordinal
above 1000, a number the table cannot decompose), the original token
passes through. Validation later emits **E220** with the file and line
number. That is the design: silent fallthrough surfaces as a real
validator error rather than a wrong-but-plausible word.
