# BA2 Morphotag Architecture — Retrospective

**Status:** Reference
**Last updated:** 2026-04-14 09:54 EDT

## Correction notice (2026-04-13 22:53 EDT)

An earlier version of this document asserted that BA2-jan9 was
"accidentally-correct by Stanza-version coincidence" and that the
`(text, True)` MWT hint was a "dead channel" in Stanza 1.11.1. **Both
claims are wrong.** Empirical testing against Stanza 1.10.1 (BA2-jan9's
version) and Stanza 1.11.1 (BA3's current version) shows:

* Stanza's tokenizer natively emits `(text, True)` tuples for English
  contractions in both versions — **the hint is Stanza's own output**,
  not something the BA2 postprocessor manufactures.
* Stanza's MWT processor honors those hints in both versions and
  produces Range tokens correctly.
* The Stanza upgrade is not implicated in the 2026-04-13 regression.

The actual BA3 regression mechanism is a local bug in
`batchalign/inference/_tokenizer_realign.py:_realign_sentence`: the
`_conform(tok)` call extracts token text but **discards Stanza's own
MWT hint tuples** before passing to the Rust char-DP aligner. When
`align_tokens` returns `Plain` tokens for a 1:1 mapping, the hint is
gone and Stanza's MWT processor never produces a Range. See
`reference/retokenize-analysis.md` "Preserve-mode MWT regression,
2026-04-13" for the full trace.

Sections below marked **[RETRACTED]** were incorrect in the initial
version; they are preserved in place, clearly marked, because the
reasoning they contained is useful for future readers who might reach
the same false conclusion. Other sections (the narrower architectural
critiques) remain valid.

## What this document is

A detailed architectural retrospective of the morphotag pipeline in
batchalign2 (BA2), pinned to the `84ad500b` commit ("BA2-jan9", 2026-01-09) —
the canonical baseline anchor referenced throughout the BA2→BA3 migration.
The goal is to examine whether BA2 was on sound architectural ground, or
whether its working behavior in production was a fortunate consequence of
implementation details that do not generalize. The analysis informs long-term
BA3 design decisions and highlights patterns to keep, replace, or retire.

Scope: the `batchalign/pipelines/morphosyntax/ud.py` module of BA2-jan9,
specifically the Stanza pipeline construction, the tokenize-postprocessor
flow, and the Stanza→%mor mapping. Not in scope: ASR, forced alignment,
coref, translation.

## TL;DR

BA2-jan9's morphotag pipeline has **two architecturally sound decisions**
(iterating Stanza's Token layer for MWT detection, and using the
documented `(text, True)` MWT hint convention that Stanza natively
honors) and **several narrower compromises** (a regex band-aid for a
known POS-tagger failure on `'s` before nominal gerunds, monolithic
per-word Python assembly with no typed intermediate model, boolean
mode flags, and double-duty in the postprocessor). The compromises are
real but orthogonal to MWT correctness. BA2-jan9 produces correct
tilde-joined output for English contractions in both Stanza 1.10.1 and
1.11.1 — the pipeline is not sensitive to the Stanza minor-version bump.

BA3's current regression is a localized bug in Python that strips
Stanza's own MWT hint tuples before the Rust aligner sees them, not an
architectural incompatibility. See the correction notice above.

## Overview of BA2-jan9's morphotag pipeline

Given a CHAT utterance, BA2-jan9 performs:

1. **Pre-Stanza cleanup** (`ud.py:806-872`). Strips CHAT annotations,
   shortening parens, special-form markers like `@wp`, etc. Inserts
   a placeholder token `xbxxx` for special forms so they can be
   re-identified after Stanza runs.
2. **Pipeline construction** (`ud.py:736-796`). Configures a Stanza
   Pipeline per language (or a `MultilingualPipeline` for code-switched
   documents). English enables the `gum` MWT model; most other
   MWT-capable languages use `default`. Crucially, `tokenize_no_ssplit=True`
   is set, but `tokenize_pretokenized=True` is NOT set for English —
   Stanza's neural tokenizer runs freely on the surface text. MWT_exclusion
   languages (CJK, several others) skip the MWT processor entirely.
3. **Tokenize postprocessor** (`ud.py:743, 1017, 1028`). Wired in via
   `config["tokenize_postprocessor"] = lambda x: adlist_processor(tokenizer_postprocessor(x))`,
   but ONLY when `retokenize=False`. When `retokenize=True`, the
   postprocessor is skipped entirely, leaving Stanza's natural
   tokenization untouched.
4. **Token-level postprocessing** (`ud.py:610-700`, `tokenizer_processor`).
   Uses a char-level DP alignment (`align(targets, refs)`) to fold
   Stanza's tokens back to the space-separated CHAT words of the
   input sentence. After folding, applies language-specific MWT
   hints via `(text, True)` / `(text, False)` tuples: English
   contractions containing `'` (with an `o'`-prefix exception) are
   marked `(text, True)`; Italian `lei` merge and several French
   clitic rules are hand-coded.
5. **Stanza inference** (`ud.py:884`). Runs the pipeline on the
   cleaned input. Produces a `Document` with `Sentence.tokens` and
   `Sentence.words` populated. MWTs appear as tokens whose `id` is a
   tuple of length > 1 (Range over their component words).
6. **Per-utterance parse** (`ud.py:parse_sentence`, called at `ud.py:894`).
   Does two passes over the sentence:
   - **Pass 1** (`ud.py:389-439`) iterates `sentence.tokens` and
     collects (a) `mwts` — list of Range-token ids — and
     (b) `auxiliaries` — list of positions where adjacent words must
     be tilde-joined because of language-specific clitic rules
     (French `-ce/-là`, Italian `l'`, etc.).
   - **Pass 2** (`ud.py:445-492`) iterates `sentence.words` and
     produces a per-word MOR string via a POS-specific handler
     (`handle(word, lang)`), building a flat `mor` list indexed by
     word position.
7. **Post-hoc MOR rewiring** (`ud.py:507-552`). After per-word MORs
   are computed:
   - Clitics and auxiliaries are folded with `~` joins into their
     host word's MOR at the list level (`mor_clone`).
   - MWTs are processed by slicing `mor_clone[mwt_start-1:mwt_end]`
     for each Range and tilde-joining the components
     (`ud.py:545`: `mwt_str = "~".join([...])`).
8. **Regex band-aid** (`ud.py:895`). After `parse_sentence` returns,
   a single regex substitution rewrites `~part|s verb|X-Ger-S` →
   `~aux|is verb|X-Part-Pres-S`. This attempts to repair a specific
   English POS-tagger failure: when `'s` is mis-tagged as a
   particle before a gerund-form verb, it is forcibly rewritten as
   an auxiliary.
9. **Ending punctuation, delimiter, string join** (`ud.py:554-565`).
   Assembles the final `%mor` string by joining the `mor_clone` list
   with spaces and appending the utterance terminator.

## What BA2-jan9 got right

### 1. Token-layer iteration for MWT detection

```python
for indx, token in enumerate(sentence.tokens):
    ...
    if len(token.id) > 1:
        mwts.append(token.id)
```

This recognizes that Stanza's **Token** layer is the correct unit to map
to CHAT words. An MWT appears as a single Token with a Range `id`
covering multiple Words. BA2 detects this by checking `len(token.id) > 1`,
which is the idiomatic UD approach.

This is the most architecturally sound piece of the pipeline. It means
BA2 *in principle* treats `"stool's"` as one CHAT word corresponding to
one Stanza Token, with two internal Words for tilde-joining. The BA3
Rust counterpart (`map_ud_sentence` in `nlp/mapping/mod.rs`) encodes the
same idea with a typed `UdId::Range` variant.

### 2. Free Stanza tokenization for non-CJK languages

By not setting `tokenize_pretokenized=True` for MWT-capable languages,
BA2-jan9 lets Stanza's tokenizer+MWT pipeline run naturally. This is the
state where Stanza's MWT processor is most confident and produces
Range-expanded output. If the postprocessor's subsequent folding
preserves the Token layer, the downstream code sees the correct
structure.

Compare with BA3, which also does not pretokenize English but adds the
`original_words` override that empirically destroys Range tokens. BA2 is
a step closer to "trust Stanza" than BA3's current state.

### 3. CJK-specific exclusion from MWT and compound handling

Chinese, Japanese, Korean, and several other languages are listed in
`mwt_exclusion` (`ud.py:760`) and treated with a different pipeline
shape. This is a legitimate special case — CHAT has gold word
segmentation for CJK that Stanza cannot recover from the surface. BA2
correctly does not try to unify CJK with the MWT-capable languages.

### 4. Language-specific MWT patches centralized in one place

Italian, French, Portuguese, and Dutch quirks live together in
`tokenizer_processor` (`ud.py:660-700`). Whatever one thinks of the
quirks, they are discoverable by grep rather than scattered. BA3 ported
these directly into a `mwt_overrides.rs` module — same organizing
principle.

## What BA2-jan9 got wrong

### 1. [RETRACTED — see correction notice above] Reliance on the `(text, True)` hint

**This critique was wrong.** The original claim was that BA2 relied on
an undocumented Stanza behavior (the MWT processor re-expanding
hinted tokens) that might work in Stanza 1.10.1 but not in 1.11.1,
making BA2 "accidentally correct by Stanza-version coincidence."

Empirical verification contradicts this:

* Stanza's `tokenize_postprocessor` API DOES document the `(text, bool)`
  tuple convention, and the behavior is stable across Stanza 1.10.1 and
  1.11.1 (tested 2026-04-13 with identical input sentences).
* Stanza's tokenizer natively emits `(text, True)` tuples for English
  contractions — BA2's postprocessor largely preserves what Stanza
  already sets rather than inventing it.
* Running BA2's full postprocessor logic against both Stanza versions
  produces Range tokens for all four Copula-contraction sentences used
  as fixtures in `test_stanza_mwt_copula_observations.py`.

The `(text, True)` channel is a documented Stanza API and a legitimate
architectural seam. BA2 using it is not a compromise.

The original "time bomb" concern does apply to a related but different
issue: BA3's `_realign_sentence` discards Stanza's own hint tuples when
flattening for the Rust aligner. That is a real bug, but it's a
BA3-side Python defect, not a BA2-side architectural flaw.

### 2. The `~part|s` → `~aux|is` regex band-aid

The line at `ud.py:895`:

```python
mor = re.sub(r"~part\|s verb\|(\w+)-Ger-S", r"~aux|is verb|\1-Part-Pres-S", mor)
```

rewrites a specific mis-tagging: when Stanza tags contracted `'s` as
PART (particle / possessive) before a gerund verb, coerce it to AUX
(`be` auxiliary). This is a real problem in Stanza — the `'s`
disambiguation between copula and possessive is context-dependent and
Stanza occasionally gets it wrong — but the regex catches only one
specific downstream pattern (the gerund being tagged as a verb). When
the same mis-tagging is followed by a gerund tagged as a nominal
gerund (`noun|X-Ger`), the regex does not fire, and the bad analysis
leaks through to the final `%mor`.

This is a **symptomatic fix, not a root-cause fix**. The principled
fix would be either (a) a better POS-disambiguation for `'s` that
reasons about the syntactic context of the whole utterance, or (b) an
honest acceptance that Stanza's tagging is what it is, with no
rewriting. BA2 chose the middle ground — a narrow string-surgery patch
— which masks the symptom but does not report the failure, and leaves
users to discover the exceptions case-by-case.

Any long-term BA3 architecture that replicates this regex replicates
the compromise. The regex is not architecturally defensible; it is a
survival tactic.

### 3. Monolithic Python assembly with no typed intermediate model

The `parse_sentence` function (`ud.py:~345-580`) is ~230 lines of
imperative Python building two parallel lists (`mor` and `gra`), then
mutating them in place with clitic joins, auxiliary joins, and MWT
joins. There is no typed intermediate structure — everything is
strings in lists, indexed by integer word position, with repeated
`-1` arithmetic because Stanza's ids are 1-indexed but Python lists
are 0-indexed. Bugs in this area would be silent string-splicing
errors rather than type violations.

This is the largest single block of domain logic in BA2 and the hardest
to audit. BA3 correctly moves this to Rust with typed `Mor`,
`GrammaticalRelation`, and `UdWord` structures. That is the right
direction; the BA2 approach is a clear anti-pattern.

### 4. Retokenize vs non-retokenize: same pipeline, different postprocessor — no clear mode type

BA2-jan9 gates the postprocessor with a bare Python bool (`retokenize`).
In `morphoanalyze`, `retokenize` controls whether the postprocessor is
installed (`ud.py:753-756`); in `parse_sentence`, no such flag is
visible — retokenize behavior is expressed only by the absence of the
postprocessor upstream. There is no named type, no enum, no mode
object — just a flag that propagates through kwargs.

This boolean-blindness is the pattern the BA3 CLAUDE.md explicitly
bans. It makes BA2's retokenize behavior invisible to future readers
and fragile across refactors. BA3 currently inherits this shape at the
Python/Rust boundary (`req.retokenize: bool`), and will need to
replace it with a proper enum when building a unified architecture.

### 5. Coupling of sentence alignment with MWT hinting

The `tokenizer_processor` function does two unrelated jobs in one
pass: (a) fold Stanza's tokens back to CHAT word count via char-DP
alignment, and (b) apply language-specific MWT hints. Job (a) is about
recovering from spurious splits (`ice-cream` → `ice + - + cream`). Job
(b) is about steering Stanza's MWT processor for contractions. These
are different concerns, but they share state (the `res` list of tokens
being built) and run interleaved with language-specific branches. A
change to one job risks perturbing the other.

BA3 split these into a Rust `align_tokens` (compound merge) and a
conceptually-separate `is_contraction` predicate, which is an
improvement, but the two are still wired together into a single
postprocessor return value. The architectural cleanup — separate the
two concerns onto different layers — has not been completed in either
codebase.

### 6. Post-hoc `mor_clone` index manipulation is fragile

The final MOR assembly (`ud.py:507-552`) mutates `mor_clone` in place,
nulling out positions that have been folded into a prior entry
(`mor_clone[j-1] = None`, then `" ".join(filter(lambda x: x, mor_clone))`).
This is fine for small sentences but makes the code non-obvious: a
reader has to simulate the index manipulation mentally to understand
what appears where in the final string.

A cleaner architecture expresses the final output as a list of typed
MOR items (one per CHAT word, where each item may carry clitics) and
serializes once at the end. BA3's Rust `Mor::with_post_clitic()` is
exactly this model; BA2's `mor_clone` is not.

### 7. No type-level invariant that "N CHAT words → N %mor items"

The CHAT format's fundamental contract is one `%mor` item per
main-tier word. BA2 realizes this only via careful list arithmetic:
the `mor` list starts with `len(sentence.words)` entries, MWTs collapse
several entries into one via string joining, clitics null out
positions, and the final count happens to match if everything lines
up. A single off-by-one in any of the four merging passes (clitic,
auxiliary, MWT, compound) would silently break the invariant.

The right architecture expresses this invariant as a type: a vector
indexed by CHAT word position where each entry is a `Mor` that may
carry arbitrary internal tilde-joined clitics. BA3's Rust side
approaches this but the Python side of BA3 and all of BA2 do not.

## Is BA2-jan9 salvageable as an architectural reference?

**The MWT handling is sound. The residual compromises are orthogonal
and should be addressed separately.**

Sound and reusable:

* **Token-layer iteration** for MWT detection (`len(token.id) > 1`).
  This is the bedrock observation that a CHAT word = Stanza Token, not
  Stanza Word. BA3's typed `UdId::Range` enum encodes the same idea.
* **Free Stanza tokenization** for non-CJK languages (no
  `tokenize_pretokenized=True`). Lets Stanza's MWT model do its job.
* **`(text, True)` MWT hints via the documented postprocessor API.**
  Stanza's tokenizer natively emits these; the postprocessor's role is
  to preserve or augment them as needed. This is a stable contract,
  not a "time bomb." (Originally flagged as compromised — retracted;
  see correction notice.)

Compromised but orthogonal to MWT correctness:

* The `~part|s` regex band-aid (`ud.py:895`) papers over a specific
  Stanza POS mis-tagging. Narrow and non-principled — should not be
  ported, but its absence does not affect basic MWT behavior.
* The monolithic ~230-line Python `parse_sentence` with string-list
  assembly should not be ported; BA3's Rust typed approach is strictly
  better. Again, orthogonal to MWT correctness.
* Boolean `retokenize` flag should be an enum. A hygiene fix, not a
  correctness fix.
* Two-jobs-one-seam in the postprocessor (compound merging and
  language-specific MWT patches sharing a return value). Legitimately
  separable; BA3's current `align_tokens` + `apply_mwt_patches` split
  is already an improvement.

## Recommendations for BA3's long-term morphotag architecture

Informed by this retrospective (post-correction):

1. **Treat Stanza's Token layer as the CHAT-word-equivalent.** This is
   already correct on BA3's Rust side via `UdId::Range`. Preserve this
   framing in any future changes.

2. **Preserve Stanza's own `(text, True)` MWT hints through the
   postprocessor.** The current BA3 `_realign_sentence` discards them
   via `_conform()` — this is the direct cause of the 2026-04-13
   Preserve-mode regression. Fix by making `align_tokens` (or its
   wrapper) tuple-aware, or by overlaying Stanza's original tuples
   onto the aligned output for positions that were not merged.

3. **Do not port BA2's `~part|s` regex.** Accept Stanza's
   disambiguation as authoritative, or invest in principled
   disambiguation; do not pattern-match over serialized `%mor` output.

4. **Replace boolean mode flags with explicit enums.** `retokenize: bool`
   becomes a `TokenizationMode` enum with explicit variants (Preserve,
   Retokenize, CjkRetokenize, L2Secondary). Rust side already has a
   partial form; finish the job.

5. **Type the "one MOR per CHAT word" invariant.** The per-utterance
   output type is a `Vec<MorForChatWord>` where each entry already
   carries its tilde-joined clitic structure. No post-hoc list
   rewiring. BA3's Rust `Mor::with_post_clitic()` already realizes
   this shape.

6. **Separate compound-merging from MWT-expansion at the type level.**
   These are different alignment jobs; give them distinct function
   contracts even if they share a wrapper. BA3's split of
   `align_tokens` vs `apply_mwt_patches` is in this direction.

7. **Consider whether alignment should move to Rust entirely (the "B"
   option).** This is a separate architectural improvement, not a
   prerequisite for fixing the current regression. Arguments in favor:
   matches the stated "Python = pure ML call" boundary; symmetric
   Preserve/Retokenize paths; fewer layers to reason about. Arguments
   against: the current Python-side postprocessor works once the hint
   bug is fixed; moving to Rust is net-new alignment code that must
   handle every CHAT-vs-Stanza shape the existing code handles.

8. **Write tests that pin Stanza behavior, not that hope for it.**
   Every assumption about Stanza's runtime (what it produces for
   specific inputs, which processors run in what order, which hints
   are honored) should have a permanent test that asserts the
   assumption. The `test_stanza_mwt_copula_observations.py` file is
   the pattern.

## Operational contrasts: BA2 vs BA3 (as of 2026-04-14)

This retrospective has focused on morphotag correctness. Two operational
concerns are worth pinning alongside the architectural critique, because
they are where BA3's investment is most visible to operators and to the
successor team.

### Per-language batch progress

BA2 had **no per-language progress tracking whatsoever**. The pipeline
was a monolithic per-file Python loop — there was no `BatchInferProgress`
equivalent, no cross-file batching, and no way for an operator to watch
a long run and see "eng done, spa 80%, zho stalled." Fleet jobs either
completed or hung with no visibility into which language's Stanza model
was doing work.

BA3 has a typed `BatchInferProgress` model (`runner/util/batch_progress.rs`)
keyed by language, published at 2-second intervals from a drain loop
(`runner/dispatch/infer_batched.rs`), visible in the REST API, the React
dashboard, and the CLI/TUI. As of 2026-04-14, per-language-group reporting
is also **correct** — a per-language tagger
(`morphosyntax/worker.rs::infer_batch`) rewrites each progress event's
`stage` field to the language code before the drain loop keys on it, so
multiple real languages no longer collapse onto a single
`"stanza_processing"` entry. See
[Observability — Per-language-group batch progress](../architecture/observability.md#per-language-group-batch-progress)
for the full mechanism.

### Stall detection

BA2's pool dispatch had **no stall detection at all**. A stuck Stanza
worker, a write-blocked pipe, or a runaway CPU loop would just hang the
job indefinitely; the only operator tool was to kill the process and
restart.

BA3 has:

* **Heartbeat-gap warnings at 120 seconds** in the drain loop, naming
  the stalled language groups. Load-bearing prerequisite: the
  per-language tagger above, without which `incomplete_groups()` would
  always return `[]` and stall detection would be blind.
* **A worker-pool checkout refactor landed 2026-04-14** (see
  `crates/batchalign-app/src/worker/pool/dispatch.rs::checkout`) that
  eliminates a spin-loop previously observed to burn an entire tokio
  worker at 100% CPU for five hours (production, 2026-04-14 chunk-5
  hang). The fast path now falls through to the slow path (spawn or
  async-wait) with `tokio::task::yield_now().await` inserted at both
  re-release sites, so co-tenant tasks — including the health check
  that would otherwise refill the `idle` VecDeque — can be scheduled.
  Regression test: `degenerate_iterations_must_yield_between_retries`
  in the same file.

BA2 carried none of this machinery; operators could not distinguish
"stuck" from "slow."

## Related documents

- `reference/retokenize-analysis.md` — retokenize-mode behavior and
  the 2026-04-04 fix that established part of the BA3 Preserve-mode
  regression.
- `reference/mwt-handling.md` — current BA3 MWT contract.
- `reference/stanza-limitations.md` — Stanza defect registry (Defect 1:
  copula `'s`; Defect 2: MWT hint tuple preservation).
- `reference/morphosyntax.md` — BA3 morphosyntax overview.
- `architecture/observability.md` — per-file and batch-level progress,
  heartbeat-gap detection.
