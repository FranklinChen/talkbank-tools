# Profiling Commands

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Profiling commands measure structural properties of a speaker's language:
how long their utterances are, how diverse their vocabulary is, and how
grammatically complex their sentences are. These are the most commonly
used measures in child language research and clinical assessment.

All profiling commands are available through the standard analysis
workflow described in [Running CLAN Commands](running-commands.md).

## MLU -- Mean Length of Utterance

**What it measures:** The average number of morphemes per utterance.
MLU is the single most widely used index of grammatical development in
young children (Brown, 1973).

**How it works:** The command counts morphemes on the `%mor` tier for
each utterance, then computes the mean across all utterances for each
speaker. If the file has no `%mor` tier, it falls back to counting words
on the main tier.

**When to use it:**
- Tracking grammatical development in children aged 1;6 to 5;0
- Screening for language delay (compare against age norms)
- As a baseline measure in treatment studies

**Results include:**
- MLU value per speaker
- Total morphemes and total utterances
- Standard deviation

**Tip:** MLU is most meaningful for children in Brown's Stages I--V
(roughly MLU 1.0 to 4.5). Beyond MLU 4.5, it becomes less sensitive
to development. For older children, consider [DSS](#dss----developmental-sentence-scoring)
or [IPSyn](#ipsyn----index-of-productive-syntax) instead.

## MLT -- Mean Length of Turn

**What it measures:** The average number of utterances per
conversational turn, along with words per turn. A "turn" is a sequence
of consecutive utterances by the same speaker, bounded by utterances
from other speakers.

**How it works:** The command identifies turn boundaries by detecting
speaker changes, counts the utterances and words in each turn, and
computes the mean.

**When to use it:**
- Measuring conversational participation and verbosity
- Comparing turn-taking patterns between child and adult speakers
- Assessing pragmatic language skills

**Results include:**
- Mean utterances per turn
- Mean words per turn
- Total turns per speaker

## VOCD -- Vocabulary Diversity

**What it measures:** The D statistic, a measure of vocabulary diversity
that is more robust than a simple type-token ratio (TTR). TTR is heavily
influenced by sample size -- longer samples always produce lower TTR.
VOCD corrects for this by computing D from random subsamples of varying
sizes.

**How it works:** The command draws random subsamples of 35 to 50 tokens
from the transcript, computes TTR for each subsample size, and fits a
mathematical curve to the TTR-by-sample-size function. The D parameter
of that curve is the vocabulary diversity score. Higher D indicates
greater diversity.

**When to use it:**
- Comparing lexical diversity across speakers or sessions with
  different sample lengths
- Research on vocabulary development
- Any context where simple TTR would be misleading due to unequal
  sample sizes

**Results include:**
- D statistic per speaker
- Type count and token count
- Type-token ratio (for reference, but prefer D)

**Tip:** VOCD requires a minimum of about 50 tokens to produce a
reliable estimate. Very short transcripts will produce unstable D values.

## DSS -- Developmental Sentence Scoring

**What it measures:** Grammatical complexity of children's sentences,
scored against a weighted system of eight grammatical categories:
indefinite pronouns, personal pronouns, main verbs, secondary verbs,
negatives, conjunctions, interrogative reversals, and wh-questions.
Each sentence receives a total score; the mean across sentences is the
DSS score.

**How it works:** The command applies scoring rules from the DSS rule
files in the `clan-info` library. Only complete sentences with a
subject and predicate are scored. Incomplete utterances, single-word
responses, and imitations are excluded.

**Language support:** DSS rules are available for:

| File | Language |
|------|----------|
| `eng.cut` | American English |
| `engu.cut` | British English |
| `bss.cut` | Bilingual Syntax Score |
| `jpn.cut` | Japanese |

**When to use it:**
- Assessing grammatical maturity in children aged 2;0 to 6;11
- Identifying children with specific language impairment
- Measuring treatment outcomes for grammar intervention

**Results include:**
- DSS score per speaker
- Per-sentence scores
- Category-level breakdown (pronouns, verbs, negatives, etc.)
- Number of sentences scored vs. excluded

**Tip:** DSS is designed for English-speaking children. For non-English
transcripts, consider language-specific profiling tools or use MLU as
a general measure.

## IPSyn -- Index of Productive Syntax

**What it measures:** The range of syntactic and morphological structures
a child produces, based on a checklist of 56 (or 100) language forms
across four categories: noun phrases, verb phrases, questions/negation,
and sentence structure.

**How it works:** The command searches the first 100 utterances for
evidence of each structure on the checklist. Each structure earns
0 points (not found), 1 point (found once), or 2 points (found in
two different utterances). The maximum score is 112 (56-item version)
or 200 (100-item version).

**Rule files:**

| File | Description |
|------|-------------|
| `eng.cut` | Standard 56-item English checklist |
| `eng-100.cut` | Extended 100-item English checklist |

**When to use it:**
- Assessing syntactic development in children aged 1;6 to 4;6
- Complementing MLU with a more detailed structural profile
- Research on syntactic bootstrapping and acquisition order

**Results include:**
- Total IPSyn score per speaker
- Subscale scores (noun phrase, verb phrase, question/negation, sentence)
- Per-item breakdown showing which structures were found

**Tip:** IPSyn samples only the first 100 utterances. If your transcript
is longer, only the first 100 contribute to the score. This is by design
-- the measure assesses productive range, not frequency.

## Choosing the right profiling command

| Question | Use |
|----------|-----|
| How long are the child's utterances? | **MLU** |
| How much does the child talk per turn? | **MLT** |
| How varied is the child's vocabulary? | **VOCD** |
| How grammatically complex are the sentences? | **DSS** |
| What syntactic structures does the child produce? | **IPSyn** |
| Quick overall snapshot? | Run **MLU** + **VOCD** together |
| Detailed clinical profile? | Run all five, then use [KidEval](assessment.md) for normative comparison |

## Next steps

- [Frequency & Distribution](frequency.md) -- word counts and distributions
- [Assessment Tools](assessment.md) -- compare profiling results against
  normative databases with KidEval
- [Command Reference](command-reference.md) -- all 33 commands
