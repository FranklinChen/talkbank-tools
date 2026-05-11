# FLUCALC -- Fluency Calculation

**Status:** Current
## Purpose

Detects and quantifies disfluencies in speech transcripts, producing per-speaker counts of stuttering-like disfluencies (SLD) and typical disfluencies (TD). FLUCALC is the standard tool in CLAN for analyzing fluency in stuttering research.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409273) for the original FLUCALC command specification.

## Usage

```bash
chatter clan flucalc file.cha
chatter clan flucalc --speaker CHI file.cha
chatter clan flucalc --format json file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `--exclude-speaker <CODE>` | Exclude speaker |
| `--format <FMT>` | Output format: text, json, csv, clan |

FLUCALC has no command-specific flags beyond the shared
`CommonAnalysisArgs` set. There is no `--syllable-mode` switch in the
current CLI; output measures are word-based.

## Disfluency Categories

### Stuttering-Like Disfluencies (SLD)

| Type | CHAT notation | Example | Status |
|------|--------------|---------|--------|
| Prolongation | `:` within word | `wa:nt` | Implemented |
| Broken word | `^` notation | `base^ball` | Implemented |
| Whole-word repetition | Consecutive identical words | `I I want` | Implemented |
| Part-word repetition | -- | -- | Partial |
| Block | -- | -- | Partial |

### Typical Disfluencies (TD)

| Type | CHAT notation | Example | Status |
|------|--------------|---------|--------|
| Phrase repetition | `[/]` | `I want [/] I want` | Implemented |
| Revision | `[//]` | `I want [//] I need` | Implemented |
| Filled pause | `&-` prefix | `&-uh`, `&-um` | Implemented |
| Phonological fragment | `&+` prefix | `&+fr` | Implemented |

### Output measures

All counts are reported as:
- **Raw values**: Total count per disfluency type
- **Percentages**: Per 100 words

## Algorithm

1. For each utterance, walk the AST content nodes
2. Identify disfluency markers:
   - Retrace groups (`[/]`, `[//]`) from parsed AST annotations
   - Fillers and fragments from word category annotations
   - Prolongations and broken words from within-word notation
   - Whole-word repetitions from consecutive identical countable words
3. Accumulate per-speaker counts by disfluency category
4. Compute percentages relative to total words (or syllables)

## Differences from CLAN

### Detection method

Some categories (specifically `[/]` and `[//]` retrace markers) are currently counted via substring matching on the serialized CHAT text rather than fully through the parsed AST. This produces equivalent results but is a known area for future improvement.

### Part-word repetitions and blocks

Counted via CHAT notation markers rather than acoustic analysis. Full detection of these categories requires audio-linked analysis that is beyond the scope of text-based transcript processing.

### Output formats

Supports text, JSON, and CSV. CLAN produces text only.

### Golden test parity

Verified against CLAN C binary output.
