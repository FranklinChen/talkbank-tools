# Command Reference

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Complete reference for all 33 CLAN analysis commands available in the
TalkBank extension. Run any command via right-click **Run CLAN
Analysis...** or the Command Palette. See
[Running CLAN Commands](running-commands.md) for the general workflow.

## All commands

| Command | Category | Description | Requires | Example use case |
|---------|----------|-------------|----------|-----------------|
| **chains** | Coding | Traces code chains and sequences on the `%cod` tier | `%cod` tier | Analyzing sequential patterns in coded behaviors |
| **chip** | Profiling | Child/parent interaction profile -- measures responsiveness and initiation | Multi-speaker file | Studying parent-child conversational dynamics |
| **codes** | Coding | Frequency counts of coding tier codes | `%cod` tier | Summarizing annotation categories across a session |
| **combo** | Search | Boolean keyword search with AND/OR operators | Input: search expression | Finding utterances with specific word combinations (e.g., `want+cookie`) |
| **complexity** | Profiling | Syntactic complexity ratio from the `%gra` dependency tier | `%gra` tier | Measuring subordination and embedding depth |
| **cooccur** | Frequency | Word co-occurrence counting within utterances | -- | Studying lexical associations and collocations |
| **corelex** | Frequency | Core vocabulary analysis -- identifies frequent, high-utility words | -- | Checking whether a child uses age-appropriate core words |
| **dist** | Frequency | Word distribution and dispersion analysis across the transcript | -- | Detecting whether target words cluster in one part of the session |
| **dss** | Profiling | Developmental Sentence Scoring -- weighted grammatical complexity | `%mor` tier; English only | Clinical assessment of grammatical maturity (ages 2;0--6;11) |
| **eval** | Assessment | Combined language evaluation with normative comparison | -- | Adult language assessment against healthy norms |
| **evald** | Assessment | Dementia-specific evaluation using DementiaBank norms | -- | Screening language decline in dementia populations |
| **flucalc** | Other | Fluency calculation -- measures disfluency types and rates | -- | Assessing stuttering severity and disfluency patterns |
| **freq** | Frequency | Word and morpheme frequency counts with type-token ratio | -- | Basic vocabulary analysis, word lists for any transcript |
| **freqpos** | Frequency | Frequency broken down by part-of-speech from `%mor` tier | `%mor` tier | Analyzing grammatical category distribution (e.g., noun-to-verb ratio) |
| **ipsyn** | Profiling | Index of Productive Syntax -- 56 or 100-item structural checklist | `%mor` tier; English only | Assessing range of syntactic structures in child language |
| **keymap** | Coding | Keyword-based contingency mapping between coded events | Input: keyword codes | Studying temporal relationships between coded behaviors |
| **kideval** | Assessment | Child language evaluation with age-normed comparison. Panel auto-detects language and activity from file headers. | -- | Clinical assessment comparing a child to typically developing peers |
| **kwal** | Search | Keyword-in-context search -- finds utterances containing target words | Input: keywords | Finding all instances of a target word with surrounding context |
| **maxwd** | Frequency | Longest words per speaker | -- | Checking lexical sophistication or spotting transcription errors |
| **mlt** | Profiling | Mean Length of Turn -- utterances and words per conversational turn | Multi-speaker file | Measuring conversational participation and verbosity |
| **mlu** | Profiling | Mean Length of Utterance -- morphemes per utterance | -- | The most common measure of grammatical development in children |
| **modrep** | Frequency | Model and replica analysis of imitation patterns | Multi-speaker file | Studying what children imitate from adult speech |
| **mortable** | Morphology | Morpheme frequency table based on a user-defined script | Input: `.cut` script file | Custom morphological analysis (e.g., counting specific inflections) |
| **phonfreq** | Frequency | Phonological segment frequency from the `%pho` tier | `%pho` tier | Phonological inventory analysis, sound acquisition studies |
| **rely** | Other | Inter-rater reliability between two transcriptions | Input: second `.cha` file | Verifying transcription agreement between two coders |
| **script** | Other | Compares a transcript against a template for structural conformance | Input: template `.cha` file | Checking that a transcript follows a required protocol structure |
| **sugar** | Profiling | Sampling Utterances and Grammatical Analysis Elicited in Narrative | -- | Narrative language assessment using the SUGAR protocol |
| **timedur** | Other | Time duration computed from bullet timing marks | Bullet timestamps | Calculating speaking time, pause duration, session length |
| **trnfix** | Other | Compares two dependent tiers for mismatches | Multiple dependent tiers | Quality control: finding alignment errors between `%mor` and `%gra` |
| **uniq** | Frequency | Finds repeated and unique utterances | -- | Detecting echolalia, formulaic language, or duplicate transcription lines |
| **vocd** | Profiling | Vocabulary Diversity -- D statistic (sample-size-corrected TTR) | Minimum ~50 tokens | Comparing lexical diversity across speakers or sessions |
| **wdlen** | Frequency | Word length distribution (character counts) | -- | Studying word length patterns for phonological or literacy research |
| **wdsize** | Frequency | Vocabulary size and character length statistics from `%mor` stems | `%mor` tier | Tracking cumulative vocabulary growth |

## Commands by category

### Frequency (11 commands)

Word counting, vocabulary, and distribution analysis.
See [Frequency & Distribution](frequency.md) for detailed descriptions.

- **freq** -- word frequency counts
- **freqpos** -- frequency by part of speech
- **wdlen** -- word length distribution
- **wdsize** -- vocabulary size
- **maxwd** -- longest words
- **phonfreq** -- phonological segment frequency
- **modrep** -- model and replica (imitation)
- **cooccur** -- word co-occurrence
- **corelex** -- core vocabulary
- **dist** -- word distribution
- **uniq** -- unique/repeated utterances

### Profiling (8 commands)

Structural and developmental measures of language.
See [Profiling Commands](profiling.md) for detailed descriptions.

- **mlu** -- Mean Length of Utterance
- **mlt** -- Mean Length of Turn
- **vocd** -- Vocabulary Diversity (D)
- **dss** -- Developmental Sentence Scoring
- **ipsyn** -- Index of Productive Syntax
- **chip** -- child/parent interaction profile
- **complexity** -- syntactic complexity ratio
- **sugar** -- SUGAR narrative sampling

### Assessment (3 commands)

Normative comparison against reference databases.
See [Assessment Tools](assessment.md) for detailed descriptions.

- **kideval** -- child language assessment
- **eval** -- general language assessment
- **evald** -- dementia assessment

### Search (2 commands)

Finding utterances by keyword or pattern.

- **kwal** -- keyword in context
- **combo** -- Boolean keyword search (AND/OR)

### Coding (3 commands)

Working with coding tiers (`%cod`).

- **codes** -- code frequency counts
- **chains** -- code chains and sequences
- **keymap** -- keyword contingency mapping

### Morphology (1 command)

Custom morphological tabulation.

- **mortable** -- morpheme frequency table from a script

### Other (5 commands)

Specialized tools for specific tasks.

- **flucalc** -- fluency/disfluency calculation
- **rely** -- inter-rater reliability
- **script** -- template comparison
- **timedur** -- timing duration
- **trnfix** -- tier mismatch detection

## Library files used by commands

Several commands use rule files from the `clan-info/lib/` directory.
These files ship with the extension and are discovered automatically
by the language server.

| Command | Library path | Files |
|---------|-------------|-------|
| **dss** | `clan-info/lib/dss/` | `eng.cut`, `engu.cut`, `bss.cut`, `jpn.cut` |
| **ipsyn** | `clan-info/lib/ipsyn/` | `eng.cut`, `eng-100.cut` |
| **kideval** | `clan-info/lib/kideval/` | 15 database files (see [Assessment Tools](assessment.md)) |
| **eval** / **evald** | `clan-info/lib/eval/` | 4 database files (see [Assessment Tools](assessment.md)) |
| **corelex** | `clan-info/lib/corelex/` | `cat.cut`, `cinderella.cut`, `cookie.cut`, `cookie-short.cut`, `cookie_modern.cut`, `sandwich.cut`, `umbrella.cut`, `window.cut` |
| **flucalc** | `clan-info/lib/fluency/` | `ALLflucodes.cut` |

## Next steps

- [Running CLAN Commands](running-commands.md) -- how to run commands
  and export results
- [Profiling Commands](profiling.md) -- detailed guide to MLU, VOCD,
  DSS, IPSyn
- [Frequency & Distribution](frequency.md) -- detailed guide to FREQ,
  WDLEN, and more
- [Assessment Tools](assessment.md) -- KidEval, Eval, Eval-D normative
  comparison
