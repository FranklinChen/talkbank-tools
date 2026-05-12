# Glossary

**Status:** Current
**Last updated:** 2026-05-12 04:22 EDT

## CHAT Format Terms

**CHAT** (Codes for the Human Analysis of Transcripts)
: The transcription format used by TalkBank. Files have `.cha` extension.

**Main tier**
: The primary transcription line beginning with `*SPK:` (e.g., `*CHI:`). Contains the spoken text.

**Dependent tier**
: Lines beginning with `%` that provide annotations for the preceding main tier (e.g., `%mor:` for morphology, `%gra:` for grammar).

**Speaker code**
: Short identifier for a participant — up to seven characters from `A-Z`, `0-9`, `_`, `-`, `'`; three uppercase letters is the convention (e.g., CHI = child, MOT = mother, FAT = father, INV = investigator).

**%mor tier**
: Morphological analysis tier containing POS tags and morpheme breakdowns.

**%gra tier**
: Grammatical relations tier containing dependency structure.

**Gem**
: A marked segment of transcript bounded by `@Bg` (begin gem) and `@Eg` (end gem) headers. See [Filtering gems](../user-guide/filtering-gems.md).

**Terminator**
: Utterance-ending punctuation (`.` period, `?` question, `!` exclamation) that appears at the end of main tier lines.

## Analysis Terms

**MLU** (Mean Length of Utterance)
: Average number of morphemes (or words) per utterance. The primary measure of grammatical development in child language.

**TTR** (Type-Token Ratio)
: Ratio of unique words (types) to total words (tokens). A basic measure of lexical diversity.

**NDW** (Number of Different Words)
: Count of unique word types in a sample.

**D statistic**
: Vocabulary diversity measure computed by VOCD. More stable than TTR because it accounts for sample size.

**DSS** (Developmental Sentence Scoring)
: Clinical scoring system for grammatical complexity (Lee & Canter).

**IPSyn** (Index of Productive Syntax)
: Syntactic complexity score based on the presence of specific grammatical structures.

**SLD** (Stuttering-Like Disfluencies)
: Disfluency types characteristic of stuttering: prolongations, blocks, part-word repetitions.

**TD** (Typical Disfluencies)
: Normal disfluency types: phrase repetitions, revisions, filled pauses.

## Technical Terms

**AST** (Abstract Syntax Tree)
: The typed, structured representation of a CHAT file produced by the parser. All analysis operates on the AST, not raw text.

**Countable word**
: A word with genuine lexical content, as determined by `is_countable_word()`. Excludes fillers, fragments, untranscribed markers, and omissions.

**NormalizedWord**
: Canonical lowercased word form used as frequency map key. Implements `Borrow<str>` for zero-allocation lookups.

**Golden test**
: A test that compares output against the original CLAN C binary to verify parity.

**CUTT** (CLAN Utility Template Toolkit)
: The 17,926-line C framework underlying original CLAN. Replaced by the Rust framework in `src/framework/`.

**Framework**
: The shared infrastructure (`src/framework/`) that handles filtering, command lifecycle, and output formatting for all commands.
