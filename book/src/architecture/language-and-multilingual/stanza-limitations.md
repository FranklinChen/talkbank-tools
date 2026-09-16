# Stanza Limitations and Their Workarounds

**Status:** Current
**Last updated:** 2026-07-28 13:18 EDT

Stanza is the morphosyntax engine behind `batchalign3 morphotag`. It does not
guarantee Universal Dependencies conformance, and several of its language
models misbehave on the short, single-word utterances that dominate
child-language corpora. This page records every limitation we have found, the
evidence for it, and what the pipeline does about it.

**The governing rule: never trust Stanza's output shape.** Everything it
returns crosses a validation boundary before anything downstream sees it. That
boundary is `RepairedSentence` in `batchalign/inference/morphosyntax.py`:
`batch_infer_morphosyntax` builds each returned item from one, and the only way
to build one is from raw Stanza words, so the step cannot be skipped.

Every relation it rewrites is reported as well as applied. The repair travels
to the server on the analyzed item, and the file's morphotag provenance comment
counts it as `ud_repairs=`, so a repaired transcript says so in itself rather
than only in a worker log.

## Why that boundary is load-bearing (2026-07-28 incident)

The validators existed and were unit-tested for months, and **nothing on the
production path ever called them**. `batch_infer_morphosyntax` took
`doc.to_dict()` straight into the response. The consequences reached published
data:

- `<PAD>` sanitization existed, yet bare `PAD` appears as a `%gra` relation in
  the corpora.
- Non-UD `iob` flowed into `%gra` as `IOB` across many files.

Neither was noticed because nothing checked relations on either side: CLAN
CHECK does not validate `%gra` at all, and chatter only gained the rule (E761)
in v0.4.0, which is what finally surfaced it.

The generalizable lesson, which cost real corpus damage: **a validator that its
own unit tests exercise but no production caller invokes is worse than no
validator, because it produces confident false assurance.** Tests for this
boundary must drive `batch_infer_morphosyntax`, not `UdWord` in isolation.

## Limitation 1: non-UD dependency relations

**Symptom.** Stanza's Italian model emits `deprel="iob"`. Universal
Dependencies has no such relation; the label is `iobj`.

**Evidence** (stanza 1.13.0, verified 2026-07-28):

```text
'attenzione .'   id=2 text='ne'  upos=PRON  head=1  deprel='iob'
```

**Impact.** `2|1|IOB` written into `%gra` across published corpora, oldest
observed provenance `ba3 morphotag | engine=stanza-1.11.1` (2026-05-08).

**Workaround.** `UdWord._normalize_deprel_to_ud` checks the relation HEAD
against `UD_RELATIONS` (the 37 UD v2 relations), maps known aliases via
`UD_DEPREL_ALIASES` (`iob` -> `iobj`), and degrades anything unrecognized to
`dep` with a warning.

Only the HEAD is a closed set. UD defines SUBTYPES as open and
language-specific, and the corpora legitimately use many (`nmod:poss`,
`acl:relcl`, `flat:foreign`), so the subtype is preserved verbatim. This
mirrors exactly what chatter's E761 enforces on the reading side; **the two
vocabularies must not drift apart.**

## Limitation 2: Italian MWT destroys single-word `-ne` nouns

**Symptom.** In ISOLATION, every tested Italian noun ending in `-ne` is split
by the MWT processor into a nonexistent verb plus the clitic `ne`.

**Evidence** (stanza 1.13.0, 2026-07-28), 10/10 mis-split:

```text
attenzione  -> ('attenzi', VERB, root)  ('ne', PRON, iob)
stazione    -> ('stazio',  VERB, root)  ('ne', PRON, iob)
canzone     -> ('canzo',   VERB, root)  ('ne', PRON, iob)
persone     -> ('perso',   VERB, root)  ('ne', PRON, iob)
opinione, ragione, lezione, situazione, televisione, macchine: same shape
```

**In sentence context the same words are analyzed correctly**, so this is
specific to short utterances:

```text
mi piace questa canzone .    -> canzone intact, no iob
ci sono molte persone qui .  -> persone intact, no iob
```

**Why this matters here more than elsewhere.** Single-word utterances are the
norm in child speech (`*CHI:\tmacchine .`), so a defect that only bites short
utterances bites CHILDES hardest. The damage is not limited to the relation
label: `%mor` records `verb|attenzare` for the noun *attenzione*, inventing a
verb that does not exist in Italian.

**Corpus impact** (measured 2026-07-28 over all 106,158 corpus files with a
Rust tool on the typed CHAT AST, resolving language per WORD via the canonical
resolver: `@Languages` primary vs secondary, `[- lang]` precodes, `@s`
markers). 657 files are Italian-PRIMARY, carrying 809,264 MOR-domain words of
which 806,133 resolve to Italian; a further 428 files declare Italian only as
a SECONDARY language and their words default elsewhere. Across both, **1,608
Italian-resolved single-word utterances in 338 files carry a verb+enclitic
`%mor` analysis**: the superset containing all committed mis-split damage plus
the genuine `dammelo`-class imperatives, which regeneration with the fixed
pipeline distinguishes. Word-language resolution left ZERO words unresolved,
and explicitly marked code-switching in Italian-primary files is small (793
`@s`-marked words, 2,279 precoded utterances, top other languages nld/deu).

**Status: FIXED 2026-07-28**, in `batchalign/inference/_italian_mwt.py`.

### Two fixes that do not work

Both are worth recording, because both look right and both were measured wrong.

**A closed-class allowlist.** Italian's preposition+article contractions and
`ecco`+enclitic forms are enumerable, so expansion can be suppressed outside
those sets. This passes every noun assertion and destroys every genuine
imperative, because verb+enclitic is an OPEN class and no surface pattern
separates `giralo` (turn it) from `cavallo` (horse). Worse, it does not even
fail safe: suppressed, `diglielo` comes back as `verb|diglielare`, an invented
verb, so the fix reproduces the exact defect it was written to remove.

**A part-of-speech probe.** Analyze the unsplit form with MWT disabled and allow
the split only when it tags VERB. Scored 29/39 on a discrimination set: it
suppresses `giralo` (NOUN), `aprila` (NOUN) and `eccolo` (ADJ), and it allows
`dondolo`, `viola`, `scivola`, `disegna` and `cancello`, which are ordinary
words. Note also that `tokenize_pretokenized=True` disables MWT expansion
outright, so a probe built that way answers "never splits" for every word: a
result that reads as clean and is a measurement that never ran.

### The fix: validate the split Stanza proposes

The decision is made in the tokenizer postprocessor, which Stanza runs BEFORE
the MWT processor, so the split being judged does not exist yet. A minimal
`tokenize,mwt` probe pipeline (worker-state key `{lang}:mwtprobe`, loaded
lazily, one batched call per batch) previews it. The proposal is then checked
against four facts, and the split is allowed only if all four hold:

| # | Test | Source | Rejects |
|---|---|---|---|
| 0 | the split accounts for every character of the word | structural invariant | `pentolone` -> *pento*+*lo* (drops `ne`), `hallo`, `tagliatelle`, `cavalla` |
| 1 | every non-initial piece is an Italian enclitic | closed class of the language | `gallina` -> *galli*+*na*, `disegna` -> *di*+*se*+*gna* |
| 2 | the base is an attested verb, or `ecco` | Stanza's own shipped lexicon | `bello` -> *ib*+*lo*, `pello` -> *ip*+*lo*, `spaghetti`, `cielo` |
| 3 | the whole form is not itself a dictionary word | Stanza's own shipped lexicon | `cavallo` -> *cava*+*lo*, `pentola`, `cavolo` |

Each is load-bearing; `test_each_of_the_four_tests_is_load_bearing` fails if any
is removed. Test 0 allows the one regular departure from plain concatenation:
only the apocopated monosyllabic imperatives `da'/di'/fa'/sta'/va'` double the
following clitic (`da` + `me` + `lo` = `dammelo`), and `gli` never doubles.
Test 2 restores the elided `e` of an apocopated infinitive, since *caricare* +
*lo* surfaces as `caricar` + `lo`.

The two lexical tests read Stanza's own Italian word list (about 50k surface
forms and 13k verb forms) rather than a table we maintain. That is deliberate:
the April 2026 MWT audit retired five hand-maintained per-language tables
precisely because they had drifted from what the models do. The list lives
behind a private Stanza attribute, so access is isolated in
`extract_stanza_lexicon` and fails loudly; `test_italian_mwt_lexicon.py` pins
its shape so a Stanza upgrade breaks CI instead of a corpus.

Measured before and after on the real corpus words, via the full pipeline:

| utterance | before | after |
|---|---|---|
| `attenzione` | `verb\|attenzare-Inf-Ind-Imp-S2~pron\|ne` | `noun\|attenzione-Fem` |
| `macchine` | `verb\|maccare-Part-Past-P~pron\|ne` | `noun\|macchina-Fem-Plur` |
| `gallina` | `galli` + `na` | `noun\|gallina-Fem` |
| `cavallo` | `cava` + `lo` | `noun\|cavallo-Masc` |
| `bello` | `ib` + `lo` | `adj\|bello` |
| `mucche` | `mu` + `cce` + `he` | `noun\|mucca-Fem-Plur` |
| `eccolo` | `ecco` + `lo` | `adv\|ecco~pron\|lo` (PRESERVED) |
| `dammelo` | `da` + `me` + `lo` | PRESERVED |
| `diglielo` | `di` + `glie` + `lo` | PRESERVED |
| `giralo` | `gira` + `lo` | PRESERVED |

### Residual error, known shapes

Two under-generation shapes are known and accepted. Forms like `pentolo` that
reconstruct, carry real clitics and a plausible base, and are absent from the
lexicon are invisible to every test the rule has; catching them needs a real
morphological analyzer. And reflexive imperatives that Stanza's lexicon lists
as words in their own right (`svegliati`, `vestiti`) are not split, which is
defensible: `vestiti` really is both "get dressed!" and "clothes", and
context-free it is genuinely ambiguous.

Where the rule must guess, it prefers to under-split. Losing a split leaves a
real word coarsely analyzed; a false split invents a verb that does not exist in
Italian, which is the defect this page is about. The asymmetry decides it.

Residual RATES await the language-resolved corpus measurement.

## Limitation 3: Italian MWT over-splits IN CONTEXT

**Symptom.** The same over-splitting as limitation 2, but surviving in full
sentences where no single-word gate can reach it:

```text
la stazione e molto grande .          -> la  = il + i        (DET/DET)
secondo la mia opinione hai ragione . -> hai = ha + i        (VERB/DET)
questa e la mozzarella .              -> mozzarella = mozzar + la
mangiamo le tagliatelle stasera .     -> tagliatelle = tagliate + le
prendi il pennarello rosso .          -> pennarello  = pennar + lo
```

**Corpus impact: not separately quantified.** The language-resolved audit
measures the single-word signature (limitation 2); committed in-context damage
has a different `%mor` shape and awaits its own signature scan. The mechanism
is certain: the five example sentences above are monolingual Italian and
reproduce through the real pipeline, asserted in
`golden_morphotag_ita_multi_word_keeps_genuine_mwts`, and `parla` -> *par* +
*la* is independently attested by the Defect 6 record in the Italian language
page.

**Status: FIXED 2026-07-28**, by the same rule as limitation 2.

### Why one rule covers both

The rule validates the split Stanza proposes against facts about Italian, none
of which mention context, so it answers identically wherever the word appears.
Candidates are exactly the tokens Stanza itself marks `(text, True)` in the
tokenizer postprocessor, which is its documented "expand this" hint. Reading
that marker instead of guessing which words might be multi-word tokens is what
makes the pass cover every context AND cost less: a typical sentence marks two
tokens out of eleven.

Italian has exactly FOUR legitimate multi-word patterns, three of them closed:

| pattern | example | test |
|---|---|---|
| preposition + article | `alla` = *a* + *la* | surface is in the contracted paradigm AND the split is ADP + DET |
| `ecco` + enclitic | `eccolo` = *ecco* + *lo* | reconstruction + real clitics + the presentative host |
| clitic cluster | `glielo` = *glie* + *lo* | reconstruction + every piece is an enclitic |
| verb + enclitic | `giralo` = *gira* + *lo* | reconstruction + real clitics + attested verb base + the whole form is not a dictionary word |

Anything matching none of the four is suppressed.

**The preposition+article test needs BOTH halves.** Validating only the
analysis (is this ADP + DET?) would accept ANY pair the tagger labels that way,
including mangles of non-paradigm surfaces (verified against raw Stanza:
English `well` -> *In* + *l* passes the structural test alone). Validating only
the surface would admit nothing useful, since `la` -> *il* + *i* has to be
rejected on its analysis. The contracted paradigm is a closed, centuries-stable
set, so requiring the surface to belong to it costs nothing and closes the
structural hole. Whether non-Italian material ever reaches the Italian pipeline
in production is a separate, unanswered question about language routing; the
rule is safe either way.

**The probe must see the same context as the pipeline.** This is the one thing
that does not generalize for free, and it cost a full RED-GREEN cycle to find:
`hai` in isolation is left whole, but in `secondo la mia opinione hai ragione .`
it becomes *ha* + *i*. A probe over isolated words therefore reports no split to
judge, and every in-context over-split passes through while the single-word
cases are still caught, which looks like a working fix. The probe runs over
whole utterances for exactly this reason.

## Limitation 4: Italian MWT FAILS to split genuine imperatives (Defects 12/13)

**Symptom.** The mirror image. Stanza declines to expand a real
imperative+enclitic and invents a verb for the whole surface:

```text
aprilo  -> verb|aprilare      leggila -> verb|leggilare
aprila  -> verb|aprilare      finila  -> verb|finilare
```

None of those verbs exist. Milder variants lose the clitic without inventing
anything (`dimmi` comes back as bare `verb|dire`) or leave the surface
unanalyzed (`buttalo` lemmatizes to `buttalo`).

**Prior mitigation and its limit.** `IT_COMPOUND_IMPERATIVES` in `lang_it.rs`
repairs eleven forms downstream: `dammela, dammelo, prendilo, prendila,
prendili, prendile, aprila, aprili, finila, aprilo, leggila`. Verb+enclitic is
an open class, so an eleven-entry list cannot be complete by construction, and
forms outside it (`dimmi`, `buttalo`, `mettilo`, `lascialo` among those
verified through the pipeline) received no repair.

**Status: FIXED 2026-07-28**, by making the same rule bidirectional.

Suppression cannot help here: there is nothing to suppress. But Stanza's
tokenizer hint protocol runs both ways, and its MWT processor OBEYS `(text,
True)`: hinted, `aprilo` yields *apri* + *lo* with the real lemma *aprire*.

So the policy now judges two kinds of candidate with the same four tests:

- tokens Stanza marked `(text, True)`, which may be over-splits;
- tokens it left whole that a cheap lexical pre-filter (`could_be_enclisis`)
  says might be enclisis: peel a maximal enclitic sequence off the end and ask
  whether the remainder can host it. Known dictionary words are excluded up
  front so `cavallo` is not probed on every batch only to be rejected later.

Every candidate is force-hinted in ONE probe pass. That is free for tokens
Stanza was already going to expand, since the hint is what it emitted itself,
and it is the only way to see a split it declines to make. The four tests then
decide, and the real pipeline is told `(text, True)` to force or `(text, False)`
to suppress. A candidate whose forced probe yields no split is left exactly as
Stanza left it, never asserted into an expansion with no proposed shape.

### The constraint that makes forcing safe

Forcing is the dangerous direction: a wrong forced split fabricates structure
instead of merely losing it. The guard is a fact of Italian morphology. The
`e`-form clitics (`me, te, se, ce, ve, glie`) exist ONLY before another clitic:
`mi` becomes `me` in `dammelo` (*me* + *lo*), `ci` becomes `ce` in `metticelo`.
Italian has no word ending in a bare enclitic `me`/`ce`/`ve`, so a split whose
LAST piece is one of them is not a possible Italian word, and a candidate whose
peeled tail ends that way is rejected. Verified against raw Stanza that without
this guard, surfaces like English `face` (-> *fa* + *ce*) would force-split;
with it they cannot, regardless of whether such material ever reaches the
Italian pipeline in production. `ne` is deliberately excluded from the
restriction: it IS a real final clitic (`dammene`, `scegline`).

Verified through the real pipeline after the change (asserted in the golden
tests): `dimmi` -> *dim* + *mi* (dire/mi) and `buttalo` -> *butta* + *lo*
(buttare/lo) recover their analyses, while `la`, `mozzarella` and the nouns
stay whole and `alla` and `dammelo` keep splitting.

## Consequence: the Rust mis-split allowlist is now largely redundant

`IT_MIS_SPLIT_OVERRIDES` in
`crates/batchalign-transform/src/morphosyntax/lang_it.rs` is a 23-entry
hand-curated table that repairs specific known mis-splits downstream, one row
added per production incident (Defects 6 and 7 in the Italian language page).
The rule above prevents that damage at the source, generally, so those Ranges
mostly no longer reach the reconciler.

Verified 2026-07-28: suppressing the split yields the correct analysis directly
for essentially every form in that table, including the ones it hardcodes,
`parla` -> `VERB/parlare`, `coccole` -> `NOUN/coccola` (with the correct plural
lemma), `piccola` -> `ADJ/piccolo`, `cielo` -> `NOUN/cielo`.

**Not removed.** The reconciler fires only on Ranges, so with no Range it simply
no-ops and nothing conflicts; deleting the table is a separate change needing
per-entry verification. Recorded here so the redundancy is known rather than
rediscovered.

## Open work

1. **Re-generate affected corpus files**, scoped by the language-resolved
   audit (338 files with Italian single-word verb+enclitic `%mor`).
   Re-running morphotag BEFORE fixing the generator reproduces the defect
   (verified 2026-07-28), so regeneration follows the fix, and outputs are
   diffed before anything is written into the data repos.
2. **Extend the audit with an in-context damage signature** for limitation 3
   (e.g. `det|il~det|il` and verb+enclitic items on multi-word utterances), so
   in-context committed damage is enumerated the same way.
3. **Characterize what the Italian pipeline actually receives in bilingual
   files**, from the language-resolved data plus the L2 routing code. Whether
   any foreign material reaches the Italian model in production is currently
   unknown; nothing on this page assumes an answer.
4. **Revisit the residual under-splits** only if a better lexical source turns
   up; catching lexicon-invisible forms needs a real morphological analyzer,
   not another heuristic layered on this one.

## Related

- `batchalign/inference/morphosyntax.py`: `UD_RELATIONS`, `UD_DEPREL_ALIASES`,
  `_repaired_relation`, `RepairedSentence`, `batch_infer_morphosyntax`.
- `batchalign/inference/_tokenizer_realign.py`: the MWT-hint postprocessor.
- `batchalign/worker/_stanza_loading.py`: per-language pipeline construction;
  note Italian takes the `tokenize_postprocessor` branch, NOT the
  `tokenize_pretokenized` one, which is why Stanza re-tokenizes CHAT words at
  all.
- chatter error E761 (`%gra` relation head not a UD relation): the reading-side
  rule that surfaced all of this.
