# Utterance Segmentation

**Status:** Current
**Last updated:** 2026-09-16 09:47 EDT

Utterance segmentation splits continuous ASR output into individual utterances
for CHAT transcription. This is a critical step, CHAT requires one utterance
per line, each terminated by a sentence-ending punctuation mark.

One important subtlety: the segmentation language and the upstream ASR request
language are related but not identical. For Rev.AI `--lang auto`, BA3 can reach
the English utterance model in two different ways:

1. Rev language ID succeeds before transcript submission, so the request itself
   becomes the explicit-English path.
2. Rev language ID fails, BA3 submits a true auto request, and only later
   resolves the returned transcript language to English for downstream
   segmentation.

Both branches can eventually run the English BA2 utterance model. Only the
first branch is provider-request-equivalent to explicit `--lang eng`.

## Three Mechanisms

batchalign3 has three utterance segmentation mechanisms:

1. **Pre-CHAT utterance models**: BA2-style token-classification models that
   predict utterance boundaries from typed ASR word lists before CHAT exists.
   Available for 3 language families / 4 supported codes (`eng`, `cmn`, `zho`,
   `yue`).
2. **Punctuation-based fallback**: Rust-side retokenization over typed ASR
   words. Used for unsupported languages and as cleanup after model-backed
   segmentation.
3. **CHAT-text utterance segmentation (`utseg`)**: a second text-task pass over
   already-built CHAT. It uses the configured TalkBank boundary model when one
   exists; Stanza is an explicit fallback for unsupported languages.

```mermaid
flowchart TD
    asr["Typed ASR monologues"]
    revauto{"Rev.AI --lang auto?"}
    langid{"Language ID succeeds\nbefore transcript submit?"}
    efflang["Effective postprocess language"]
    check{"Language has\npre-CHAT model?"}
    bert["BA2 utterance model\nreturns typed assignments"]
    punct["Rust punctuation retokenization\n(. ? ! +... etc.)"]
    utts["Pre-CHAT utterances"]
    chat["CHAT AST"]
    postchat["Second boundary-model pass\non built CHAT text"]

    asr --> revauto
    revauto -->|no or non-Rev engine| efflang
    revauto -->|yes| langid
    langid -->|yes| efflang
    langid -->|no| efflang
    efflang --> check
    check -->|eng/cmn/zho/yue| bert
    check -->|all others| punct
    bert --> punct
    punct --> utts
    utts --> chat
    chat --> postchat
```

## BERT Utterance Models

| Language | Code | Model | Source | Architecture |
|----------|------|-------|--------|--------------|
| English | eng | `talkbank/CHATUtterance-en` | TalkBank fine-tuned | BERT token classification |
| Mandarin | cmn/zho | `talkbank/CHATUtterance-zh_CN` | TalkBank fine-tuned | BERT token classification |
| Cantonese | yue | `PolyU-AngelChanLab/Cantonese-Utterance-Segmentation` | Hong Kong Polytechnic | BERT token classification |

These models predict utterance-boundary actions as a token classification task.
In BA3, Python model inference stays token-based and returns typed
word-assignment groups to Rust; Rust then applies those assignments to the
prepared ASR chunks without round-tripping through ad hoc sentence strings.

The six semantic actions are ordinary, capitalized onset, period boundary,
question boundary, exclamation boundary, and comma. Only the three boundary
actions advance the assignment group. Current postprocessing suppresses the
earlier action whenever two adjacent words both have any non-ordinary action;
the retained evidence records raw and applied actions separately so this
policy can be evaluated without rerunning the model.

The distinction is material. A controlled local replay over 598 retained
English source monologues and 15,141 words found 267 action differences between
the current policy and a policy that suppresses only the earlier of two true
sentence-end actions. Of those differences, 140 restored a sentence-end action
and changed the utterance assignments. This is evidence that the legacy rule
needs human-linked evaluation, not proof that all 140 restored boundaries are
correct. This first census isolates model decoding over retained provider
monologues; production preprocessing and speaker projection can change the
exact model inputs. The production default remains unchanged. See the
developer `utseg` reference for the reproducible, provider-free probe.

Worker protocol V2 also carries the sum of the three sentence-end
probabilities for each classified word at fixed micro precision, along with the
model ID and its exact revision, which is always present: the model is loaded
from a pinned snapshot, so the revision is a required part of its identity
rather than something reported when the library happened to expose it. A normalization omission and a
short input that bypasses model inference are explicit states. Rust refuses
the result unless assignments and evidence exactly parallel the dispatched
words. `transcribe --debug-dir PATH` retains these decisions in separate
versioned pre-CHAT and post-CHAT evidence files; see the transcribe guide.

```mermaid
flowchart LR
    W["Dispatched words"] --> M["Boundary model"]
    M --> R["Raw action + boundary probability"]
    R --> P{"Selected normalization policy"}
    P --> A["Applied action"]
    A --> G["Assignment groups"]
    R --> E["Per-word evidence"]
    A --> E
    G --> V{"Rust shape admission"}
    E --> V
    V -->|"lengths and assignments agree"| D["AdmittedUtsegPrediction"]
    V -->|"mismatch"| F["Typed protocol failure"]
    D --> S["Split prepared chunk or CHAT utterance"]
    O["Normalization omission"] --> V
    C["Model short-circuit"] --> V
```

These are text-model signals, not acoustic confidences. The model sees lexical
context but does not receive pause duration, waveform energy, pitch,
diarization overlap, or CHAT retrace structure. Those signals must be joined
downstream under an explicitly tested policy.

Both standalone `utseg` and `transcribe`'s pre-CHAT segmentation path resolve
through the same manifest table, so `cmn` and `zho` both select
`talkbank/CHATUtterance-zh_CN` at the same commit.

The table above is stated in exactly one place, `UTSEG_BOUNDARY_MODELS` in
`crates/batchalign/src/model_manifest.rs`, which names each model AND pins the
revision it loads. `UtsegRoute::resolve` is the one function that turns a
language plus a fallback policy into a segmenter choice, and it reads
availability from that same table, so a language BA3 offers to segment is by
construction a language it can name a model for. A language with no boundary
model and no authorized Stanza fallback has no segmenter, and that is refused
when the job is submitted (`POST /jobs` answers 400 with a message naming the
language and both ways to authorize the fallback: `--utseg-fallback-stanza` on
the command line, `"utseg_fallback": true` in a job request's options), and
again when a job persisted before that check is planned, before any ASR is
dispatched, rather than at the worker after the transcription has been
produced. A code-switched pair is segmented under its primary language, so
`eng,spa` takes the English model and `spa,eng` needs the same opt-in as
Spanish alone (pairs are withheld from transcription today; the rule holds
for the benchmark and for the route that replaces the pair). Under `--lang auto` the language is not known until ASR
returns, so that case is decided when it is. Until 2026-09-17 the refusal
came only at planning, after the request had been accepted: an API caller saw
a job that failed on every file, and the message named only the command-line
flag and offered "run without utterance segmentation", which `transcribe` has
no way to do.

The Python side holds no language-to-model map at all. An id must be known
before a load in order to pin its revision, so Rust resolves it and sends it to
the worker with the spawn; the worker loads exactly that snapshot and reports
the commit it found on disk.

For Rev.AI `--lang auto`, model selection happens after the effective language
is resolved for post-processing. That means an auto-submitted Rev transcript can
still run through the English BERT utterance model later, even if the original
provider request was not identical to explicit `--lang eng`.

### Cantonese Model Details

The Cantonese model uses character-level tokenization (each Chinese character
is a separate token) and predicts 6 action classes:

| Class | Meaning |
|-------|---------|
| 0 | Normal (continue) |
| 1 | Capitalize next |
| 2 | Period (.) |
| 3 | Question mark (?) |
| 4 | Exclamation mark (!) |
| 5 | Comma (,) |

Before feeding text to the model, Cantonese-specific preprocessing runs:
- Strip punctuation: `.` `,` `!` `！` `？` `。` `，` `?` `（` `）` `：` `＊`
- Split on Cantonese sentence-final particles: 呀, 啦, 喎, 嘞, 㗎喇, 囉, 㗎, 啊, 嗯
- Feed each chunk to the BERT model as character-level tokens

### Memory Footprint

Each utterance model is ~400 MB. In the worker runtime it is loaded alongside
the `utseg` task so transcribe can reuse the same typed text-inference boundary
for both pre-CHAT segmentation and later CHAT-level refinement.

## Punctuation-Based Fallback

For languages without a dedicated utterance model, utterances are split by
punctuation in Rust (`crates/batchalign-transform/src/asr_postprocess/mod.rs`).

### CHAT-Legal Sentence Terminators

```text
.  ?  !  +...  +/.  +//.  +/?  +!?  +"/.  +".  +//?  +..?  +.  ...  (.)
```

### Additional Normalizations

Before splitting:
- Japanese period (。) → `.`
- Spanish inverted punctuation (¿, ¡) → removed
- RTL punctuation (؟, ۔, ،, ؛) → ASCII equivalents

### Split Rules

1. If a word **is** a terminator → flush the current utterance
2. If a word **ends with** a terminator character → split the word, flush
3. If no terminator is found → auto-append `.` at the end
4. Trailing morphological punctuation (‡, „, ,) is stripped before flush

### Long Turn Splitting

Before punctuation-based retokenization, monologues longer than 300 words are
split into chunks of 300. BA3 also applies a long-pause fallback split before
retokenization, but only in a narrow case: a gap of at least 800 ms whose
next word is one of a fixed list of English sentence starters (`and`, `so`,
`what`, and so on: `LONG_PAUSE_SENTENCE_STARTERS` in
`crates/batchalign-transform/src/asr_postprocess/mod.rs`). It never fires for
Chinese, Cantonese or other non-English text, so for those languages a long
unpunctuated run is split only by the utterance model (where one exists) and
by the 300-word cap.

## Stanza Utterance Segmentation (CHAT-text path)

Separately from ASR post-processing, the `utseg` NLP task can also use
**Stanza's constituency parser** to predict utterance boundaries during
standalone `utseg` processing on already-built CHAT text. On the live worker
boundary, Rust freezes a prepared-text batch and dispatches
`execute_v2(task="utseg")`. For model-backed languages, Python may return
direct typed assignments; for the Stanza path it returns raw constituency trees
and Rust computes assignments locally.

**Not all languages have constituency parsing.** Stanza has constituency
models for ~11 languages (en, de, es, it, pt, da, id, ja, tr, vi, zh-hans).
For other languages (e.g. Dutch, Polish, Russian), the utseg config builder
omits the constituency processor and falls back to sentence-boundary
segmentation. This is handled automatically by the Stanza capability table
(`batchalign/worker/_stanza_capabilities.py`), which reads Stanza's
`resources.json` to discover per-language processor availability.

This is a different mechanism from the pre-CHAT utterance models above: it
operates on already-built CHAT text and can refine boundaries using syntactic
structure. The user-facing surface is the standalone `utseg` command and
transcribe's post-CHAT pass. With utterance segmentation enabled, production
transcribe uses a closed two-pass plan: the language model first segments
prepared timed ASR words, CHAT is built, and the model then refines the main
tiers. Disabling utterance segmentation disables both passes. For a language
with a configured TalkBank boundary model, that model remains the primary
path; Stanza is an explicit fallback when no boundary model is configured.

The passes are not independent. A pre-CHAT boundary changes the main-tier
contexts presented after CHAT construction. Offline evaluation therefore
distinguishes an actual one-pass topology from policy isolation that retains
both passes while changing only the pre- or post-CHAT decoder policy. See the
developer transcribe reference for the four typed replay choices.

The second pass also has to project timing onto its new children. BA3 permits
`%wor` partitioning only after equal policy-selected counts pass canonical
lexical corroboration. Which main-tier words hold a `%wor` slot is Chatter's
policy (`WorSlotMembershipPolicy`), asked per word rather than restated here,
so the splitter and the timing binding cannot come to disagree about the
count they are comparing. When every retained child then supplies complete
positive word timing, each new main-tier bullet is rederived as that child's
word-timing hull. If complete per-child evidence is unavailable, no child
receives a main-tier bullet at all: the parent's span measures the whole parent,
not any one of its children, so carrying it onto one of them would present an
unmeasured span as a measured one. This timing projection is downstream of the boundary decision and
must not be interpreted as evidence that the chosen segmentation is
linguistically unique or optimal. See the `%wor` reference for the exact
fallback conditions.

## Why Only 3 Languages Have Models

Training utterance segmentation models requires large amounts of annotated
conversational data with gold-standard utterance boundaries. TalkBank has
this for English (extensive CHILDES/TalkBank corpora), Mandarin (growing
corpus), and Cantonese (PolyU research corpus).

For other languages, the punctuation-based fallback produces acceptable
results because ASR models (especially Whisper) tend to insert punctuation at
natural utterance boundaries. The main limitation is run-on speech without
clear sentence structure, the fallback will produce fewer, longer utterances.

## Adding a New Language Model

To add utterance segmentation for a new language:

1. Collect annotated conversational data with utterance boundaries
2. Fine-tune a BERT token classification model (6 classes: normal, capitalize,
   period, question, exclamation, comma)
3. Upload to HuggingFace Hub
4. Add the language, model id and pinned commit to `UTSEG_BOUNDARY_MODELS` in
   `crates/batchalign/src/model_manifest.rs`. That one edit both makes the
   language routable and pins what it loads; there is no second table to update
5. Add any language-specific preprocessing (e.g., character-level tokenization
   for CJK, particle-based chunking)

## Source Files

| File | Purpose |
|------|---------|
| `crates/batchalign-transform/src/asr_postprocess/mod.rs` | Typed ASR normalization + punctuation retokenization |
| `batchalign/models/utterance/infer.py` | BA2-style utterance model runtime |
| `batchalign/worker/_model_loading/utterance.py` | Utterance model bootstrap |
| `batchalign/inference/utseg.py` | Worker-side utseg dispatch (typed assignments or Stanza trees) |
| `batchalign/models/utterance/evidence.py` | Closed model-action and per-word evidence states |
| `crates/batchalign-types/src/worker_v2/utseg_evidence.rs` | Canonical Rust IPC evidence types |
| `crates/batchalign/src/utseg_evidence.rs` | Versioned pre/post-CHAT experiment artifacts |
