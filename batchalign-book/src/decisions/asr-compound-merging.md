# ASR Compound Merging — Provenance, Problem, and Proposal

**Status:** Proposal — deferred (current behavior preserved at BA2 parity)
**Last updated:** 2026-04-20 19:00 EDT

## Context

Stage 1 of the ASR post-processing pipeline
(`crates/batchalign-chat-ops/src/asr_postprocess/compounds.rs`) runs
`merge_compounds()`, which takes an adjacent pair of ASR-emitted tokens and
concatenates them without a space when the pair appears in
`data/compounds.json` — a 3,584-entry list of `[word1, word2]` pairs. Example:
ASR emits `"air plane"`, `merge_compounds` returns `"airplane"`.

Reviewers have observed that the same merger consistently turns
verb+particle sequences such as `come back` into `comeback` and `put down`
into `putdown`, forcing manual splitting during transcript review. This
document audits the pipeline stage: where the list came from, what
problem it actually solves, what serious ASR systems and the linguistic
literature do, what CHAT convention prescribes — and what we should do.

## Empirical Confirmation of Current Behavior

The BA2 ancestor pipeline (`merge_on_wordlist` in
`batchalign/pipelines/asr/utils.py` of an archived BA2 checkout) produces
`comeback` and `putdown` from the bare two-token inputs `come back` and
`put down`. The BA3 implementation uses a different algorithm shape (a
left-to-right index versus BA2's two-element buffer) but is bit-for-bit
equivalent on every test case examined — including all reported
verb+particle inputs. This is longstanding inherited behavior, not a BA3
regression. The behavior is wrong by both the CHAT manual's convention
and the empirical contents of the reference CHAT corpora.

## 1. Provenance of `compounds.json`

### 1.1 The one commit

The BA2 file `batchalign/utils/compounds.py` was created and populated in a
single commit on 2025-07-12 with the message `"[ci skip] compounds!"`. No
design note, no source citation, no CI test, no follow-up edit. The BA3
port (`data/compounds.json`) is a direct JSON copy of that Python literal.

### 1.2 Structural fingerprint

- **3,660 raw entries, 3,584 unique.** 76 duplicate slots indicate the
  list was assembled from multiple concatenated sub-lists, not
  de-duplicated.
- **Ordering.** Roughly alphabetical by first word: starts `('air','bag')`,
  the "main block" runs through `('zoo','keeper')` around index 2193, and
  an appended tail block resets to `('hard','back')`/`('heavy','weight')`
  etc. The tail block is organized by semantic category (adjectives,
  participles, particle verbs, plus a small ad-hoc injection at indices
  89–94 of `('american','style')`, `('non','verbal')`, `('t','shirts')`,
  `('post','war')` — exactly the strings hardcoded into
  WER-normalization tests in the same commit).
- **Machine-split artifacts.** `('balloons','man')` ("balloonsman" is not
  a word; the real compound is `balloonman`), `('grille','work')`
  ("grillework" is not a word; the real compound is `grillwork`),
  `('ever','lasting')` (morphologically `ever+last+ing`, not
  `ever+lasting`). These are signatures of a naive **substring-splitter
  heuristic** applied to an upstream single-word compound list — the
  splitter walked each word and accepted any prefix+suffix that both
  happened to be dictionary words, with no morphological knowledge.
- **Closest public match.** `dariusk/corpora` (MIT, 2016,
  [compounds.json](https://github.com/dariusk/corpora/blob/master/data/words/compounds.json))
  ships a ~2,675-entry `[compoundWord, firstWord, secondWord]` list in
  the same alphabetical shape, ending at the same terminal entry
  `('zoo','keeper')`. Intersection with the batchalign list:
  **1,465 / 3,584 pairs (41%).** 1,210 dariusk entries are *absent* from
  batchalign, so dariusk is not the parent. It is the closest known
  ancestor in shape and ordering, but not the source.
- **No public list of ~3,584 entries matches.** Moby Words II is far
  larger and in a different shape; NLTK/WordNet ship no compound-pair
  resource of this size; no Kaggle/HuggingFace dataset matches.

### 1.3 Conclusion on provenance

The list is **a bespoke, incrementally-grown empirical
WER-normalization table**, probably seeded by a Wiktionary (or similar)
closed-compound scrape fed through a substring-splitter, then extended
by appending pairs encountered as ASR/gold mismatches during evaluation.
It has no linguistic design criterion. It is not documented, not
versioned as a dataset, has no inclusion rule, and its generator is not
preserved.

## 2. What Problem the List Actually Tries to Solve

The motivating problem is real: ASR engines (Whisper, Rev.AI, Google
STT, etc.) sometimes emit a lexicalized closed compound as two
space-separated tokens — `airplane` → `air plane`, `bathroom` → `bath
room`. This happens because BPE-level subword decoding produces word
boundaries driven by the language model's training-data frequencies, and
both spellings appear in web text. Low-frequency or domain-specific
compounds split more often than high-frequency ones.

The list tries to undo these ASR tokenization artifacts. On unambiguous
closed compounds it works well: `air plane` → `airplane` is a high-value
fix that the reviewer would otherwise do by hand.

The design **fails on any pair where both a closed and an open reading
are lexicalized** — the noun `comeback` versus the verb+particle
`come back`; the noun `takeoff` versus the verb+particle `take off`; the
noun `setup` versus the verb+particle `set up`. A context-free wordlist
has no way to choose, and the list's designer chose "merge
unconditionally." In spoken conversation (the CHAT domain), the
verb+particle reading dominates by 20–500×, so the unconditional merge
is wrong more often than it is right for these pairs.

## 3. State of the Art in ASR Post-Processing

Surveying mainstream ASR stacks:

- **Kaldi** (https://kaldi-asr.org) — no compound-merge stage.
  Tokenization is determined by the lexicon and the LM. If `airplane` is
  a lexicon entry, it emits whole; otherwise it emits whatever the LM
  prefers.
- **NVIDIA NeMo** (Inverse Text Normalization, Zhang et al. 2021,
  [arXiv:2104.05055](https://arxiv.org/abs/2104.05055)) — WFST-based ITN
  handles numbers, dates, currency, abbreviations. **Does not** merge
  closed compounds.
- **OpenAI Whisper** — `BasicTextNormalizer` and `EnglishTextNormalizer`
  ([source](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py))
  contain a small contraction and spelling map. **No compound merging
  table.** The normalizer exists for WER scoring, not for transcript
  quality.
- **Rev.AI, AssemblyAI, Google Cloud STT, Azure Speech, Deepgram** —
  none document a compound-merging post-processor. Their "smart
  formatting" / "ITN" stages handle punctuation, casing, numerics.
- **wav2vec2, HuBERT, Conformer-based models** — compound spelling comes
  from the LM used in beam search (KenLM or a neural LM), not a
  post-processor.

**No mainstream production ASR pipeline implements a wordlist-based
compound merger.** The industry answer to `air plane` vs `airplane` is
"the LM." batchalign is alone in having this stage.

### 3.1 The MWE literature

Multi-word expression identification is a mature subfield:

- Sag et al. 2002, "Multiword Expressions: A Pain in the Neck for NLP"
  ([ACL Anthology W02-0802](https://aclanthology.org/W02-0802/)) — the
  foundational taxonomy.
- Constant et al. 2017, "Multiword Expression Processing: A Survey"
  ([Computational Linguistics 43:4](https://aclanthology.org/J17-4005/))
  — the canonical modern survey.
- PARSEME shared tasks 2017/2018/2020
  ([multiword.eu/parseme](https://multiword.eu/parseme/)) on verbal MWE
  identification across 14+ languages.
- DiMSUM 2016 ([dimsum16.github.io](https://dimsum16.github.io/)) and
  STREUSLE ([nert-nlp/streusle](https://github.com/nert-nlp/streusle))
  — English MWE and supersense corpora.

Current SOTA for distinguishing closed compounds from phrasal verbs
from free combinations uses **transformer-based sequence labelers**
(fine-tuned BERT/RoBERTa with BIO tags for MWE spans), typically joint
with dependency parsing. English accuracy is around **70–80% F1** —
meaningfully imperfect. This is not a solved problem even with large
models and supervised training. A context-free wordlist is known in
this literature as an anti-pattern that fails on every ambiguous bigram.

### 3.2 Acoustic disambiguation

Compound stress vs. phrasal-verb stress is a well-known English
phonological contrast (Chomsky & Halle 1968; Plag 2006, "The
Variability of Compound Stress"). Compounds bear initial stress
(`COMEback`, `TAKEoff`, `SETup`); phrasal verbs bear final/particle
stress (`come BACK`, `take OFF`, `set UP`). In principle an aligner
with F0/intensity features could disambiguate. **No production ASR
system uses this**; the signal is there in the audio but unexposed by
Whisper/Rev.AI APIs. A separate forced alignment + prosodic analysis
step would be required. High research cost, unclear production payoff.

## 4. CHAT Convention and Empirical Corpus Data

### 4.1 What the CHAT manual says

CHAT Manual **§8.8.2 "Compounds and Linkages"**
(https://talkbank.org/0info/manuals/CHAT.html) prescribes:

- **Closed compounds: single token, no marker.** `birdhouse`,
  `babysitter`. MOR resolves them morphologically. The legacy
  `bird+house` plus-form is still recognized but deprecated.
- **Linkages: underscore.** `Santa_Claus`, `Hong_Kong_University`,
  `how_about`, `how_come`. The underscore indicates "collocational but
  not a true compound."
- **Conventional hyphenation: hyphen.** `cul-de-sac`, `hi-fi`.

The manual does **not** give `comeback`/`come back`, `setup`/`set up`,
`takeoff`/`take off` as worked examples. The convention for those is
implicit: the closed form is the noun, the open form is the verb+particle.

### 4.2 What real corpora do

Main-tier counts from grep (`^\*[A-Z]+:.*\bword\b`) over `.cha` files in
three TalkBank corpora:

| pair (closed / open)          | CHILDES Eng-NA | AphasiaBank | CA           | % closed (avg) |
|-------------------------------|----------------|-------------|--------------|----------------|
| **airplane / air plane**      | 2,765 / 12     | 234 / 0     | 65 / 0       | **99.6%**      |
| **bedroom / bed room**        | 727 / 5        | 238 / 0     | 305 / 10     | **98.8%**      |
| **classroom / class room**    | 186 / 4        | 97 / 0      | 274 / 1      | **99.1%**      |
| **comeback / come back**      | 33 / 2,563     | 2 / 815     | 4 / 890      | **1.7%**       |
| **takeoff / take off**        | 19 / 757       | 3 / 121     | 170 / 99     | ~15% (variable)|
| **setup / set up**            | 30 / 522       | 23 / 168    | 31 / 304     | **7.2%**       |
| **pickup / pick up**          | 56 / 1,730     | 28 / 490    | 26 / 349     | **3.6%**       |
| **lookup / look up**          | 3 / 287        | 0 / 128     | 2 / 99       | **1.0%**       |
| **breakdown / break down**    | 6 / 14         | 8 / 10      | 37 / 43      | ~45% (variable)|
| **ice cream / icecream**      | 2,798 / 757    | 476 / 128   | 99 / 10      | ~20% (CHAT favors OPEN) |
| **high school / highschool**  | 92 / 0         | 555 / 0     | 1,007 / 12   | **0.7%**       |
| **post office / postoffice**  | 36 / 1         | 14 / 3      | 47 / 2       | **6.3%**       |

**Interpretation.** Three tiers:

1. **Unambiguous closed compounds** (`airplane`, `bedroom`,
   `classroom`) are written closed ~99% of the time. ASR merging is
   empirically correct.
2. **Verb-particle pairs** (`come back`, `set up`, `pick up`,
   `look up`, `take off`) are written open 85–99% of the time. ASR
   merging is empirically wrong — by a 20–100× ratio.
3. **Standardly-open compounds** (`ice cream`, `high school`,
   `post office`) are written open. ASR merging is wrong by convention.

The list's coverage straddles all three tiers with no disambiguation. It
is a **selective dictionary with unprincipled membership**: some
categories it gets right by happy accident (tier 1), others it gets
systematically wrong (tiers 2 and 3). The stage's effective accuracy is
a weighted average of those tiers' distributions, not a design target.

## 5. Assessment

**The stage is a hack — but a hack solving a real and narrow
subproblem.** The narrow subproblem is tier 1: ASR providers do
sometimes split lexicalized closed compounds, and no downstream
component currently recovers them. Removing the stage entirely
regresses hundreds of correct `airplane` / `bedroom` / `classroom`
fixes per corpus.

**The fundamental architectural problem:** the stage runs before any
POS or syntactic information is available, so it cannot disambiguate
tiers 1–3. It operates on surface tokens alone, using a static wordlist
whose membership has no inclusion criterion and whose accuracy nobody
has measured against corpus data. It also does not respect the CHAT
manual's linkage underscore convention — when a lexicalized compound is
emitted it comes out as `airplane`, which matches corpus convention for
tier 1, but the same list also merges tier 2/3 pairs to
`comeback`/`icecream`, which violates corpus convention.

**The downstream architecture has what the stage needs.** Stanza runs
later, producing POS tags and dependency structure. A
`VERB + PARTICLE/ADP` bigram is trivially distinguishable from a
`NOUN + NOUN` bigram once POS is available. The architectural mistake
is running compound resolution at stage 1 (pre-Stanza) instead of
staging it after.

## 6. Proposal

**Phased, corpus-driven, empirically auditable.** Each phase is
independently shippable and reversible.

### Phase 1 — Data-driven list audit (short-horizon, low-risk)

Score every pair in `compounds.json` against the full TalkBank CHAT
corpus: `closed_count / (closed_count + open_count)` on main tiers in
the reference data (CHILDES Eng-NA + AphasiaBank + CA + others).
Thresholds:

- **Keep** pairs where closed > 90% in corpus (tier 1: airplane,
  bedroom).
- **Remove** pairs where closed < 50% in corpus (tier 2: comeback,
  setup, pickup, lookup, takeoff, icecream, postoffice, …).
- **Flag for manual review** the 50–90% band (tier 3: breakdown,
  knockout, some ambiguous N+N).

Ship the audit script under `scripts/analysis/audit_compounds.py` so
the list is reproducibly regenerable as corpora grow.

**Effect:** eliminates the reported verb+particle bug AND the silent
failure modes on the rest of tier 2 (dozens of pairs). Preserves the
tier 1 win. List becomes a curated, versioned, auditable dataset
instead of a one-off data dump.

### Phase 2 — POS-gated merging (medium-horizon, structural)

Move the compound-merging stage out of `asr_postprocess` and into a
**post-Stanza** stage. For each candidate bigram in the curated list,
merge only when Stanza's POS tags permit it (`NOUN+NOUN`, `ADJ+NOUN`,
`NOUN+NOUN` with appropriate dependency; never `VERB+PRT`, `VERB+ADP`,
`VERB+ADV`). This requires restructuring: today's post-ASR output goes
straight to CHAT without POS; a POS-aware variant would tag first, then
merge. Transcribe-only workflows (no morphotag) fall back to Phase 1
behavior on the curated list — still correct for tier 1, still safe on
tier 2 because those entries are no longer in the list.

**Effect:** restores correct merging for pairs that *are* closed
compounds in noun contexts (`had a setup` → `had a setup` can fire)
while leaving verb+particle (`set up the table`) untouched. Aligns with
MWE-literature best practice (Constant et al. 2017).

### Phase 3 — CHAT convention alignment (long-horizon, semantic)

For any remaining ambiguous MWE that is collocational but not a true
compound (the `how_about` / `how_come` class), emit the CHAT linkage
underscore form per manual §8.8.2 rather than either surface form.
This requires a classification step beyond POS (distinguishing "true
closed compound emit as `w1w2`" from "linkage emit as `w1_w2`" from
"free combination emit as `w1 w2`"), which is what the MWE literature
does with transformer sequence labelers. Probably not worth it
near-term, but the right target architecture.

### What NOT to do

- **Keep the current list and behavior.** BA2 parity is not a reason
  to ship empirically wrong output.
- **Delete the stage entirely.** Regresses the tier 1 wins; reviewers
  would have to re-merge `air plane` → `airplane` manually across
  every corpus.
- **Prune only the two pairs called out by the immediate report**
  (`come,back`, `put,down`). Fixes the visible complaint but leaves
  the systemic problem (dozens of other verb+particle pairs in the
  list) silently producing wrong output.
- **Acoustic / prosodic disambiguation.** Linguistically principled
  but no production ASR pipeline does this; the signal is unexposed
  by current ASR providers; very high engineering cost for marginal
  gain on an already-narrow problem.

## 7. Recommendation

**Ship Phase 1 next** when this work is picked up. Drive membership
from the existing TalkBank corpora — they are the canonical source of
truth for what CHAT transcripts look like, not an opaque external
scrape heuristic. Land the audit script alongside the data so the
list is regenerable when corpora grow.

**Current disposition (2026-04-20):** deferred. Behavior preserved at
BA2 parity. The two policy-encoding tests at the bottom of
`compounds.rs` (`come_back_not_merged`, `put_down_not_merged`) remain
`#[ignore]`'d as on-the-shelf reminders; they will become the first
RED tests when Phase 1 starts.

**File Phase 2 as a tracked architectural project** with a design note
in this book under `book/src/architecture/`. Do it properly, post-POS,
with tests measuring corpus-level accuracy.

**Phase 3 can wait** until MWE identification in conversational speech
becomes a separately-funded research priority. The CHAT linkage
convention is the right target but the classification is unsolved in
the literature.

## References

### Code
- BA3 merge logic:
  `crates/batchalign-chat-ops/src/asr_postprocess/compounds.rs`
- BA3 data file:
  `crates/batchalign-chat-ops/data/compounds.json`

### Standards
- CHAT Manual: https://talkbank.org/0info/manuals/CHAT.html (§8.8.2
  "Compounds and Linkages")

### Closest public ancestor
- dariusk/corpora compounds.json (closest public match in shape, not
  parent):
  https://github.com/dariusk/corpora/blob/master/data/words/compounds.json
- Moby Words II on Project Gutenberg:
  https://www.gutenberg.org/ebooks/3201

### Literature
- Sag et al. 2002, "Multiword Expressions: A Pain in the Neck for NLP":
  https://aclanthology.org/W02-0802/
- Constant et al. 2017, "Multiword Expression Processing: A Survey":
  https://aclanthology.org/J17-4005/
- PARSEME shared tasks: https://multiword.eu/parseme/
- Zhang et al. 2021, "NeMo Inverse Text Normalization":
  https://arxiv.org/abs/2104.05055
- Plag 2006, "The Variability of Compound Stress in English"

### Industry ASR
- Kaldi documentation: https://kaldi-asr.org/doc/
- Whisper English normalizer:
  https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py
