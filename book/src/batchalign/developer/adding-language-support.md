# Adding Support for a New Language

**Status:** Current
**Last updated:** 2026-05-19 21:19 EDT

This page is the checklist to run through when someone says "let's add
language X." Skipping any of these checks produces silent quality bugs
that surface later as user complaints (Whisper hallucinating, validator
rejecting digits, retokenize segfaulting, morphotag injecting wrong
counts).

A class of E220 bug happens when a language is declared "supported in
transcribe-only mode" without verifying that the number-expansion
backend covers it. Whisper emits digits, the validator rejects them,
the postprocess pipeline silently passes them through. This page
exists to prevent that class of mistake.

## Pre-flight: capability matrix

For every new language, fill in this table **before writing any code**.
The answers determine which integrations are wired and which docs
need a "not available for X" line.

| Capability | How to check | Affects |
|------------|--------------|---------|
| **ISO 639-3 code** | `pycountry`, `talkbank-types::LanguageCode3` | Everything downstream |
| **Stanza pipeline?** | `python -c "import stanza; print('XXX' in stanza.resources.common.load_resources_json())"` AND check the entry has a `packages` key (not just charlm stubs) | morphotag, utseg, retokenize gating |
| **`num2words` backend?** (build-time only) | `python -c "import num2words; print('XX' in num2words.CONVERTER_CLASSES)"` (use ISO 639-1 2-char code). The Rust `NUM2LANG` table at `crates/batchalign-transform/data/num2lang.json` is the codegenned output of an offline `num2words` sweep; runtime uses Rust only. (No in-tree codegen script today: see [Number Expansion](../architecture/number-expansion.md) for the regeneration protocol.) | Number expansion (E220 risk) |
| **Rev.AI quality?** | Submit a sample to Rev.AI; check for hallucinations, script confusion, repetition. Document result in `book/src/batchalign/reference/revai-language-quality-strategy.md` | Default ASR engine choice |
| **Stock Whisper quality?** | Same: run a representative sample, evaluate | Fallback ASR engine choice |
| **HuggingFace fine-tune available?** | Search HF Hub for `whisper-*-{lang}` checkpoints | `whisper_hub` engine routing in `crates/batchalign/src/model_manifest.rs::WHISPER_HUB_DEFAULTS` |
| **CHAT digit-validator allows digits?** | `rg "{lang}" talkbank-tools/../chatter/crates/talkbank-model/src/validation/word/language/digits.rs` | Whether E220 fires on Whisper digit emissions |
| **PyCantonese / language-specific tools?** | Per-language: relevant for CJK, possibly others | Special-case wiring |

## The five integration points

When the matrix is filled in, work through these in order:

### 1. Stanza wiring

If Stanza ships a real pipeline (the `packages` key is populated, not
just `backward_charlm`/`forward_charlm` stubs):

- Add or verify the language in `batchalign/worker/_stanza_capabilities.py`
 , this is the runtime authority, NOT a hardcoded table.
- Confirm MWT, POS, lemma, depparse, constituency availability via the
  capability table.
- If MWT is present, the Stanza-induced retokenize path
  (`crates/batchalign-transform/src/retokenize.rs` and
  `crates/batchalign-transform/src/retokenize/{rebuild,parse_helpers}.rs`)
  automatically applies.
- Per-language analysis quirks (clitics, compounds, elision) may need a
  `crates/batchalign-transform/src/morphosyntax/lang_<code>.rs` module:
  see Italian (`lang_it.rs`) and French (`lang_fr.rs`) as references.

If Stanza ships only stubs (no `packages`): the language is
**transcribe-only**. Document this on the language's reference page.
Morphotag, utseg, and Stanza-driven retokenize all skip silently
through `with_morphosyntax=false` / `with_utseg=false` plan flags
(`crates/batchalign/src/pipeline/transcribe.rs`).

### 2. Number expansion

> **Authoritative reference:** the
> [Number Expansion architecture page](../architecture/number-expansion.md)
> is the single source of truth for how this works. The summary
> below is a checklist; the page is the deeper explanation, the
> per-language coverage matrix, and the maintenance protocol you
> follow when adding a language. Keep that page updated in the
> same patch as any code change.

The `stage_asr_postprocess` stage runs **for every language**, gated
only by `always_enabled` in
`crates/batchalign/src/pipeline/transcribe.rs`. The expansion
pipeline is Rust-only (no Python IPC) and is NOT Stanza-gated.

What determines whether digits get spelled out:

- **CJK (`zho`/`cmn`/`jpn`/`yue`)**: handled in Rust by `num2chinese`
  in `crates/batchalign-transform/src/asr_postprocess/num2chinese.rs`.
- **English ordinals/years/decades**: handled by
  `crates/batchalign-transform/src/asr_postprocess/ordinal_year_eng.rs`
  via deterministic composition rules.
- **All other cases**: per-language `NUM2LANG` table at
  `crates/batchalign-transform/data/num2lang.json`. The table is the
  offline-codegenned output of a `num2words` sweep; runtime is
  Rust-only.

**Regenerating the table.** The historical codegen script
(`scripts/codegen_num2lang.py`) is no longer in the tree; the
table is committed as a generated artifact. The maintenance protocol
lives in [Number Expansion](../architecture/number-expansion.md),
follow that page when adding or refreshing a language entry. When
`num2words.CONVERTER_CLASSES` does not cover a language (e.g.
Malayalam, Hindi, Tamil, most non-Telugu/Kannada/Bengali Indic
languages), either:

1. Add a hand-curated overlay (digits 0-9 and the common compounds
   you need) following the procedure in the number-expansion page.
2. Add the language to the digit-allowed list via
   `language_allows_numbers` in
   `../chatter/crates/talkbank-model/src/validation/context.rs:34` (consulted by
   the validator through `mixed_language_allows_numbers` in
   `../chatter/crates/talkbank-model/src/validation/word/language/helpers.rs:57`,
   which gates `digits.rs`). Lossy but unblocks transcribe runs.

Pick option 1 unless the user community explicitly accepts digits in
the transcript.

### 3. ASR engine selection

Order of preference, picking the first that produces usable output on
a representative sample:

1. **Stock Whisper** (`--asr-engine whisper`): fast, broad coverage,
   no per-language config. Good baseline.
2. **HuggingFace fine-tune via `whisper_hub`**: when stock Whisper or
   Rev.AI underperform on extended recordings. Configure model
   resolution in
   `crates/batchalign/src/model_manifest.rs::WHISPER_HUB_DEFAULTS`, naming the
   hub commit to pin it to; that is what a planned job resolves from.
3. **Rev.AI** (`--asr-engine rev`): only if it produces clean output
   for this language. Many languages return garbage from Rev.AI; see
   `book/src/batchalign/reference/revai-language-quality-strategy.md` for the
   canonical Malayalam-failure case study.
4. **Specialty engines** (Tencent, Aliyun, FunASR for Cantonese): only
   when domain quality demands it.

Document the choice and the evidence behind it on the language's
reference page. Do NOT silently change engine defaults, every change
needs a rationale in the docs.

### 4. CHAT validator allowlist

Several validators have per-language carve-outs. Check at least:

- `digits.rs` (E220): which languages may have Arabic digits
- Other validators in `talkbank-tools/../chatter/crates/talkbank-model/src/validation/word/language/`

If the language is missing from a relevant allowlist AND the upstream
ASR / transcription convention produces output that triggers the
validator, decide whether to (a) widen the validator, (b) add a
post-processing normalization, or (c) document the constraint and
expect transcribers to manually fix it. Option (b) is preferred when
the input is deterministic (e.g., digits → spelled words).

### 5. Reference documentation

Every language with non-trivial special treatment gets a page under
`book/src/batchalign/reference/languages/<lang>.md`. Even a transcribe-only
language deserves a page so future contributors know where to look.

The page must include:

- ASR engine choice + rationale
- Stanza availability (cite the capability table check)
- Per-stage table (text norm, number expansion, retokenize,
  morphotag, utseg, FA), each with the actual current behavior
  not the *intended* behavior
- Open issues section if any known bugs apply to this language
- Operational notes (chunk size, model parameters, etc.)

Add the language to `book/src/batchalign/reference/languages/overview.md`
index so it shows up in the SUMMARY.

## Verification

After wiring a language, run end-to-end on a small fixture before
declaring "supported":

1. ASR: short audio (< 60s), confirm transcribed text matches
   expected script.
2. Number expansion: feed an utterance containing a spoken number
   ("I have three books"), confirm output is spelled, not digits.
3. CHAT validation: run `chatter validate` (or the equivalent
   pipeline gate) on the output; confirm no E220 / E1xx errors that
   are language-coverage gaps rather than real transcript problems.
4. Morphotag (if Stanza-supported): confirm `%mor` and `%gra` tiers
   inject without count-mismatch errors.
5. FA (if attempted): confirm word-level timings appear and
   `%wor` tier is generated.

Any failure on steps 1-3 means the language is **not yet ready** for
user-visible support, adjust integration before merging.

## Related documentation

- `book/src/batchalign/reference/languages/overview.md`: language index
- `book/src/batchalign/reference/revai-language-quality-strategy.md`: when to
  switch away from Rev.AI
- `book/src/batchalign/reference/whisper-hub-asr.md`: HuggingFace fine-tune
  routing
- `crates/batchalign/CLAUDE.md`: batchalign crate map
- [Number Expansion](../architecture/number-expansion.md), protocol
  for refreshing `crates/batchalign-transform/data/num2lang.json` and
  the hand-curated overlay (the historical
  `scripts/codegen_num2lang.py` script is no longer in-tree)
- `../chatter/crates/talkbank-model/src/validation/word/language/`
 , language-aware validators, including E220 digits
