# Transform Taxonomy

**Status:** Current
Classification of the 23 transform commands by operation level.

## Text/Layout Transforms (5)

These transforms operate on serialized CHAT text, not the AST. They handle formatting concerns that have no structural representation in the data model.

| Command | Operation | Why Text-Level |
|---------|-----------|----------------|
| LONGTIER | Remove continuation line wrapping (`\n\t` → single line) | Line wrapping is a display concern |
| LINES | Add/remove line number prefixes | Line numbers have no AST representation |
| INDENT | Align CA overlap markers by column | Column alignment is visual, not structural |
| LOWCASE | Lowercase main tier words | Operates on AST (`Word.text`), but simple enough to be text-like |
| CHSTRING | Find/replace strings using a changes file | Text substitution by design |

## Structured/AST Transforms (15)

These transforms modify the parsed AST and re-serialize. They use typed model fields, not string manipulation.

| Command | Operation | Key AST Types |
|---------|-----------|---------------|
| QUOTES | Extract quoted text to separate utterances | `Utterance`, `Postcode` |
| ORT | Orthographic conversion via dictionary | `Word.text` |
| RETRACE | Copy main tier to `%ret` dependent tier | `MainTier` → `DependentTier` |
| COMBTIER | Merge duplicate dependent tiers | `DependentTier` list |
| POSTMORTEM | Pattern rules on `%mor` items | `MorTier.items` (partially; %mor rewrite pending) |
| FLO | Generate `%flo` tier (simplified fluent output) | `MainTier.content` → `DependentTier::Flo` |
| FIXBULLETS | Repair timing bullet ordering | `Bullet.timing` (start_ms, end_ms) |
| TIERORDER | Reorder dependent tiers to canonical order | `Utterance.dependent_tiers` |
| TRIM | Remove selected dependent tiers | `Utterance.dependent_tiers.retain()` |
| MAKEMOD | Generate `%mod` tier from pronunciation lexicon | `Word` → `DependentTier::Mod` |
| COMPOUND | Normalize compound word formatting (dash → plus) | `Word.text` compound markers |
| REPEAT | Mark utterances with revisions as `[+ rep]` | `Utterance` retrace annotations |
| ROLES | Rename speaker codes | `SpeakerCode` in headers + utterances |
| DATES | Compute participant ages from `@Birth`/`@Date` | `Header::ID` age field |
| DELIM | Add missing terminators | `MainTier.terminator` |

## Mixed/Hybrid Transforms (3)

These transforms use a combination of AST and text-level operations.

| Command | Operation | Notes |
|---------|-----------|-------|
| DATACLEAN | Fix CHAT formatting errors (brackets, tabs, ellipsis) | Text-level by design: fixes sub-token formatting that the parser accepts as-is |
| FIXIT | Normalize via parse-serialize roundtrip | AST transform is a no-op; normalization comes from the serializer's canonical formatting |
| GEM | Extract utterances within `@Bg`/`@Eg` gem boundaries | AST-based: filters `ChatFile.lines` by gem header state |

## Pipeline Patterns

### Standard AST pipeline (most transforms)
```text
read file → parse → validate → transform(AST) → serialize → write
```
Handled by `framework::run_transform()`.

### Text-level pipeline (DATACLEAN, LINES)
```text
read file → parse → validate → serialize → transform(text) → write
```
Uses custom run functions that operate on the serialized string.

### File-argument pipeline (COMPOUND, DATES)
```text
copy to temp → run with file path → read output
```
Some CLAN binaries don't support stdin; these use file arguments.
