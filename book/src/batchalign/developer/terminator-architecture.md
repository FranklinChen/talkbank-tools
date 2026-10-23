# Terminator Architecture

**Status:** Current
**Last updated:** 2026-10-25 08:12 EDT

This document is a comprehensive reference for how CHAT terminators are created,
stored, defaulted, validated, and propagated through every batchalign3 pipeline.
It also covers the relationship between terminators and bullet placement, the CA
transcript special case, and the differences from batchalign2.

## What Is a Terminator?

A **terminator** is the punctuation mark that ends a CHAT utterance. It indicates
the utterance type or interruption/completion state. CA intonation markers
are separate content separators, not terminators.

**CHAT manual reference:** <https://talkbank.org/0info/manuals/CHAT.html#Utterance_Terminators>

In standard CHAT, every main-tier utterance **must** end with a terminator:

```text
*CHI: I want a cookie .           ← Period (declarative)
*MOT: do you want a cookie ?      ← Question (interrogative)
*CHI: give me cookie !             ← Exclamation
```

**In CA transcripts (`@Options: CA`), terminators are optional.** Many
utterances have no terminator at all, and others use CA-specific intonation
markers alongside their content:

```text
*KE: I like LA the best , LA's quiet →
*CO: yeah
*KE: I'm gonna make a little tiny
```

The `→` is a CA level-pitch separator. None of these examples supplies a
terminator; CA mode permits that absence.

## The 13 Terminator Variants

**Source:** `talkbank-model/src/model/content/terminator.rs`

| Category | Variant | CHAT Token | Unicode |
|----------|---------|-----------|---------|
| **Standard** | `Period` | `.` | |
| | `Question` | `?` | |
| | `Exclamation` | `!` | |
| **Interruption** | `TrailingOff` | `+...` | |
| | `Interruption` | `+/.` | |
| | `SelfInterruption` | `+//.` | |
| | `InterruptedQuestion` | `+/?` | |
| | `SelfInterruptedQuestion` | `+//?` | |
| | `BrokenQuestion` | `+!?` | |
| | `TrailingOffQuestion` | `+..?` | |
| | `BreakForCoding` | `+.` | |
| **Quotation** | `QuotedNewLine` | `+"/.` | |
| | `QuotedPeriodSimple` | `+".` | |

The `Terminator` enum does **not** implement `Default`. An utterance with no
terminator is represented as `Option<Terminator>::None`, never a defaulted
variant.

## Storage in the AST

**Source:** `talkbank-model/src/model/content/tier_content.rs`

```rust,ignore
pub struct TierContent {
    pub linkers: TierLinkers,
    pub language_code: Option<LanguagePrecode>,
    pub content: TierContentItems,           // words, separators, groups, etc.
    pub terminator: Option<Terminator>,  // <-- HERE
    pub postcodes: TierPostcodes,
    pub bullet: Option<Bullet>,       // terminal utterance-level bullet
    pub location: Location,
}
```

Serialization order is critical:

```mermaid
flowchart LR
    L["linkers"] --> LC["language_code"] --> C["content items"] --> T["terminator"] --> PC["postcodes"] --> B["terminal bullet"]
```

When `terminator` is `None`, the serializer skips it entirely. The bullet still
appears at the end, but there is no punctuation mark between content and bullet.

## Terminators, CA Separators and Bullets

The pinned Chatter model represents `⇗ ↗ → ↘ ⇘` and the CA TCU markers
`≋ +≋ ≈ +≈` as `Separator` variants. The former `Terminator::Ca*`
variants are already retired. Their absence from the terminator grammar is
intentional; do not restore them or promote trailing arrows to terminators.

`UtteranceContent::InternalBullet` and `TierContent.bullet` represent distinct
positions. A serialization containing two bullets must be diagnosed from the
actual input and admitted model; the presence of a CA arrow alone is not
proof of a parser defect or authority to move a bullet.

## Where Terminators Are Created

### 1. Parser (Chatter)

The shared model's `Terminator::from_chat_str` accepts the 13 canonical
spellings above. Parser-produced terminators retain their source locations;
programmatically constructed terminators use `Location::Unlocated`.
CA separators remain content items. BA3 consumes the pinned Chatter model
rather than maintaining a separate CHAT terminator inventory.

### 2. ASR post-processing

**File:** `crates/batchalign-transform/src/asr_postprocess/mod.rs`

An utterance's end is a field of the ASR utterance, never one of its words:

```rust,ignore
pub enum UtteranceEnd {
    Observed(Terminator),          // the recognizer's ending punctuation
    Synthesized(SynthesisCause),   // the pipeline closed the utterance
}

pub enum SynthesisCause {
    NoMark,                            // the chunk ran out of words
    UnwritableMark(UnwritableEndMark), // `...` (Ellipsis) or `(.)` (Pause)
}
```

`retokenize()` is the one producer of utterances. A closing mark ends the
current utterance and becomes its end, read once as a `ClosingMark`:
`Observed` when it is a CHAT terminator, `Synthesized(UnwritableMark(..))`
when it is a mark CHAT cannot write as one (a run of two or more periods,
or `(.)`). Stage 3 of preparation keeps such a mark whole as one token, so
it reaches this decision intact rather than as separate periods, the first
of which would read as an observed `.`. When retokenization flushes
remaining words without explicit punctuation, the end is
`Synthesized(NoMark)`. A mark with no words before it closes nothing and
emits no utterance.

CHAT requires a terminator and has no notation for one nobody heard, so a
synthesized end is written as `.` (`UtteranceEnd::terminator`, the one place
that period is chosen). `build_chat` returns the written utterances whose end
was synthesized (`BuiltChat::synthesized_ends`, by position among the file's
utterances), and transcribe counts them in the file's provenance stamp
(`asr_synthesized_terminators=N`), named for the stage it counts: a later
utseg split writes a period at each cut (below), which the stamp does not
count. CA transcripts created from ASR still get these periods.

### 3. Utterance segmentation (utseg)

**File:** `crates/batchalign-transform/src/utseg.rs`

A split's earlier children receive a synthesized period. The final child
inherits the parent's terminator, including a non-period ending; the retained
`utseg_split_inherits_terminator_on_last_child` control exercises that boundary.
The `%wor` terminators are kept in sync with their new main tiers. These cuts
are not included in the ASR assembly's synthesized-end count.

### 4. CHAT building from ASR

**File:** `crates/batchalign-transform/src/build_chat/mod.rs`

Takes the terminator from the description's end (`UtteranceBody::Words { words,
end }`), never from a word: every word is content, and a terminator is not a
word (`ChatWordText` refuses one). Assembly decides nothing about how an
utterance ends.

## Stanza Input and MOR/GRA Output

`stanza_input_terminator` in
`crates/batchalign-transform/src/morphosyntax/payload.rs` is the single owner
of the model-input policy. It copies an existing main-tier terminator; for an
utterance without one, it supplies an unlocated period as Stanza's
sentence-final input signal. This is a deliberate inference policy, not a
claim that the source contained a period. Primary and secondary-language
analysis use the same helper.

Morphotag normally honors `@Options: CA` by passing the document through
without analysis. The explicit analyze policy opts into morphology. The
injected MOR tier receives the payload's typed terminator, and its matching
GRA relation is admitted with it through `MorGraTerminatorSlot` and
`try_align_mor_gra`. The mapper's extra PUNCT relation belongs to that explicit
slot, not to a guessed content word. Empty or inconsistent morphology is
handled by the command's typed completion and refusal policy.

There is no morphology inference cache that could collide on missing versus
present terminators. Incremental equivalence compares the actual optional
main-tier terminator among its reuse inputs.

## Admission and Preservation

Input and output use the command-specific Chatter admission boundaries.
CA policy belongs to those boundaries, not an ad hoc post-command warning
that every utterance must have punctuation. See
[command contracts](../architecture/command-contracts.md) and
[CHAT validation failures](chat-validation-failures.md).

Alignment passes the model between UTR and forced alignment. Word timing and
utterance timing are distinct from the main tier's terminator. Retokenization
preserves the original ending when rebuilding the main tier. Passing an AST
between stages prevents an unnecessary serialization/reparse boundary; it
does not by itself prove every timing or bullet-placement outcome correct.

## Batchalign2 Comparison

BA2 (Python, baseline commit `84ad500b`) handled terminators very differently:

### BA2 data model

BA2's `Utterance` stores content as a flat `List[Form]` where `Form` has
`text: str` and `time: Optional[Tuple[int, int]]`. The terminator is the
**last Form** in the list whose text is in `ENDING_PUNCT`. There is no separate
`terminator` field -- it's just a regular form.

```python
ENDING_PUNCT = [".", "?", "!", "+//.", "+/.", "+...", "+/?", "+!?", ...]
MOR_PUNCT = ["", "", ","]
```

### BA2 parsing

BA2 regex-matches `\x15\d+_\d+\x15` to find bullets. Terminators are detected
by scanning the last word of the content. If no terminator is found, `.` is
the implicit default via the `delim` property.

### BA2 forced alignment

1. Groups utterances into 20-second windows
2. Strips ALL punctuation (both `ENDING_PUNCT` and `MOR_PUNCT`) from the text
   sent to Whisper
3. Whisper returns timestamped words WITHOUT terminators
4. Character-level DP alignment maps words back
5. Terminators remain in the document but get no timing

### BA2 CA handling

BA2 had **no CA-specific terminator handling**. CA intonation arrows were not
in `ENDING_PUNCT` and would have been treated as regular word content.

### Key differences

| Aspect | BA2 | BA3 |
|--------|-----|-----|
| **Terminator storage** | Last Form in content list | Separate `Option<Terminator>` field |
| **Terminator type** | String from `ENDING_PUNCT` | 13-variant enum |
| **CA arrows** | Not recognized as terminators | Typed content separators; never terminators |
| **Missing terminator** | Implicit `.` default | Explicit `None` |
| **Bullet model** | Regex on serialized text | AST: `InternalBullet` vs `TierContent.bullet` |
| **FA terminator treatment** | Strip before Whisper | Exclude from word extraction |
| **Grammar-level distinction** | N/A (no grammar) | `contents` vs `utterance_end` position |

## Implementation Owners

| Owner | Responsibility |
|-------|----------------|
| Chatter `talkbank-model` | Terminator inventory, CA separators, optional main-tier ending and paired MOR/GRA construction |
| `crates/batchalign-transform/src/asr_postprocess/utterance.rs` | Observed or synthesized ASR ending and its cause |
| `crates/batchalign-transform/src/build_chat/utterances.rs` | Consume the description's end; report synthesized assembly endings |
| `crates/batchalign-transform/src/utseg.rs` | Split children, preserve the parent's ending on the final child and synchronize `%wor` |
| `crates/batchalign-transform/src/morphosyntax/payload.rs` | Typed Stanza-input terminator policy |
| `crates/batchalign-transform/src/morphosyntax/sentence_mapping.rs` | Build GRA chunk relations and the terminator relation |
| `crates/batchalign-transform/src/inject.rs` | Admit the MOR/GRA terminator slot before installing tiers |
| `crates/batchalign/src/pipeline/morphosyntax/states.rs` | CA pass-through/analyze policy and document admission |

## Qualification Limits

A synthesized period is a written convention, not observed punctuation or an
acoustically measured boundary. ASR assembly counts and later utseg cuts have
different scopes. CA policy, missing endings, non-period endings and timing
preservation need their own boundary controls; neither an old bug narrative
nor a passing ordinary sentence establishes them all.
