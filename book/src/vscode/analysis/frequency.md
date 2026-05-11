# Frequency & Distribution

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Frequency and distribution commands count words, morphemes, and
phonological segments in various ways. They answer questions like
"what words appear most often?", "how long are the words?", and
"which parts of speech dominate the transcript?".

All commands in this chapter are available through the standard
analysis workflow described in [Running CLAN Commands](running-commands.md).

## FREQ -- Word Frequency

**What it does:** Counts every word in the transcript and reports the
frequency of each unique word form, organized by speaker.

**Results include:**
- Total word tokens (running words)
- Total word types (unique forms)
- Type-token ratio (TTR)
- Frequency table: each word with its count, sorted by frequency

FREQ is the most general-purpose counting command. It is the starting
point for any lexical analysis.

**Tip:** Use the CSV export to load frequency lists into a spreadsheet
for further analysis. Sort by frequency to find the most common words,
or alphabetically to check for spelling variants.

## FREQPOS -- Frequency by Part of Speech

**What it does:** Counts word frequency broken down by part-of-speech
categories from the `%mor` (morphology) tier.

**Requires:** A `%mor` tier. Files without morphological coding will
produce empty results.

**Results include:**
- Frequency table with columns for word, POS tag, and count
- Summary counts per POS category (nouns, verbs, adjectives, etc.)

**When to use it:** When you need to know not just which words appear,
but what grammatical roles they fill. For example, a child might have
a large vocabulary but use very few verbs -- FREQPOS reveals this.

## WDLEN -- Word Length Distribution

**What it does:** Reports the distribution of word lengths (measured in
characters) across the transcript.

**Results include:**
- Histogram of word lengths (how many 1-character words, 2-character
  words, etc.)
- Mean word length per speaker
- Longest and shortest words

**When to use it:** For phonological or literacy research where word
length is a variable of interest. Comparing word length distributions
across sessions can reveal changes in lexical sophistication.

## WDSIZE -- Vocabulary Size

**What it does:** Reports the cumulative vocabulary size -- the number
of unique word types -- as the transcript progresses. This produces a
vocabulary growth curve.

**Results include:**
- Total vocabulary size per speaker
- Character length statistics from `%mor` stems

**When to use it:** For studying vocabulary growth over the course of
a session or across sessions. The growth curve shows whether a speaker
is still introducing new words or has plateaued.

## MAXWD -- Longest Words

**What it does:** Finds and lists the longest words produced by each
speaker, measured in characters.

**Results include:**
- Longest words per speaker, with their lengths
- Context (the utterance in which each word appeared)

**When to use it:** As a quick check on lexical sophistication, or to
spot transcription errors (unusually long "words" may be coding mistakes
or missing spaces).

## PHONFREQ -- Phonological Frequency

**What it does:** Counts the frequency of phonological segments on the
`%pho` (phonology) tier.

**Requires:** A `%pho` tier with IPA transcription.

**Results include:**
- Frequency of each phoneme
- Consonant vs. vowel breakdown
- Position-in-word analysis (initial, medial, final)

**When to use it:** For phonological research, speech sound disorder
assessment, or studying sound acquisition patterns.

## MODREP -- Model and Replica

**What it does:** Analyzes imitation patterns by comparing a child's
utterance (the replica) to the preceding adult utterance (the model).
Reports what was repeated exactly, what was modified, and what was added
or omitted.

**When to use it:** For studying imitation and language learning
strategies. High exact-imitation rates in older children may indicate
echolalia; selective imitation patterns reveal what structures the child
is actively acquiring.

## COOCCUR -- Co-occurrence

**What it does:** Counts how often pairs of words appear together in the
same utterance.

**Results include:**
- Co-occurrence matrix or list of word pairs with counts
- Statistical association measures

**When to use it:** For studying lexical associations, collocations, and
formulaic language. Useful in discourse analysis and vocabulary research.

## DIST -- Distribution

**What it does:** Analyzes the distribution and dispersion of specific
words or codes across the transcript, showing where they cluster and
where they are absent.

**Results include:**
- Distribution profile across utterance ranges
- Clustering and dispersion statistics

**When to use it:** For studying whether certain words or behaviors are
evenly distributed or concentrated in particular portions of the
interaction.

## UNIQ -- Unique Forms

**What it does:** Identifies repeated and unique utterances in the
transcript. Finds exact duplicates and near-duplicates.

**Results include:**
- List of repeated utterances with their frequencies
- Unique utterances (appearing only once)

**When to use it:** For identifying stereotyped or formulaic language,
echolalia, or transcription artifacts (accidental duplicate lines).

## Working with frequency results

### CSV export for spreadsheet analysis

All frequency commands support CSV export via the **Export CSV** button
in the analysis panel. This is the recommended workflow for quantitative
analysis:

1. Run FREQ (or another frequency command) on a file or directory.
2. Click **Export CSV** in the panel toolbar.
3. Open the CSV in your analysis tool of choice.

### Combining with other commands

Frequency results are most informative when paired with profiling
measures:

- Run **FREQ** to get raw counts, then **VOCD** to get a
  sample-size-corrected diversity measure.
  See [Profiling Commands](profiling.md).
- Run **FREQPOS** to see POS distributions, then **MLU** to relate
  vocabulary composition to utterance length.
- Use **FREQ** on a directory to get corpus-wide word lists for
  normative comparison or stimuli selection.

## Next steps

- [Profiling Commands](profiling.md) -- MLU, MLT, VOCD, DSS, IPSyn
- [Assessment Tools](assessment.md) -- KidEval, Eval, Eval-D
- [Command Reference](command-reference.md) -- all 33 commands in one table
