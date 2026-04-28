# Literature Review: Code-Switching Morphosyntactic Analysis

**Status:** Reference
**Last updated:** 2026-04-21 09:01 EDT

This review surveys computational approaches to morphosyntactic analysis
of code-switched text, with particular attention to techniques relevant
to batchalign3's L2 morphotag feature (default-on since 2026-04-15).
The goal is to
situate our approach within the existing literature and identify whether
we are reinventing known techniques or contributing something novel.

## Background: The Problem

Code-switching (CS) is the alternation between two or more languages
within a single conversation, utterance, or even word. In TalkBank's
CHAT format, word-level code-switching is marked with `@s`:

```chat
*EVA: ich möchte film@s studies@s machen .
```

The challenge: morphosyntactic analysis (POS tagging, lemmatization,
dependency parsing) requires language-specific models, but code-switched
text mixes languages at the word level. Monolingual NLP models degrade
at switch points because they encounter out-of-vocabulary words from
the other language.

## Prior Art

### Solorio & Liu (2008) — Two Monolingual Taggers with Supervised Merger

The foundational work on POS tagging for code-switched text. Solorio
and Liu ran separate English and Spanish POS taggers on English-Spanish
code-switched data, then trained an SVM classifier to combine their
outputs. Key finding: using the output of **both** monolingual taggers
as features gave the best results, outperforming either tagger alone.

**Relevance to our work:** This is the closest antecedent to our
two-model approach. The difference: they trained a supervised classifier
to merge tagger outputs (requiring annotated code-switching training
data), while our structural merge uses a rule-based deprel constraint
that requires no code-switching training data at all.

**Citation:** Solorio, T. & Liu, Y. (2008). Part-of-Speech Tagging for
English-Spanish Code-Switched Text. *Proceedings of EMNLP 2008*.

### Bhat et al. (2018) — Neural Stacking for Code-Switching Dependency Parsing

Built the first Hindi-English code-switching Universal Dependencies
treebank and proposed "neural stacking" — base monolingual parsers
(Hindi and English) whose hidden representations are fed to a stacking
parser trained on code-switched data. Achieved 90.5% POS accuracy and
71.0% LAS on code-switched dependency parsing.

**Relevance to our work:** The neural stacking approach is
architecturally related to our structural merge — both use monolingual
models as a foundation and combine their outputs. The critical
difference: Bhat et al. require a code-switching treebank to train the
stacking layer. Our approach requires only monolingual Stanza models
and a hand-crafted deprel-to-POS constraint table.

**Citation:** Bhat, I.A., Bhat, R.A., Shrivastava, M. & Sharma, D.
(2018). Universal Dependency Parsing for Hindi-English Code-switching.
*Proceedings of NAACL-HLT 2018*.

### Soto & Hirschberg (2018) — Joint POS and Language ID Tagging

Demonstrated that jointly predicting language ID and POS improves both
tasks for code-switched text. Adding language ID as a feature to the
POS tagger yields significant accuracy gains, because knowing the
language constrains the set of valid POS tags.

**Relevance to our work:** We effectively have perfect language ID
(the `@s` marker in CHAT), which is a stronger signal than predicted
LID. Our approach exploits this by routing @s words to the correct
language-specific model. Soto & Hirschberg's finding validates that
language-aware POS tagging outperforms language-agnostic approaches.

**Citation:** Soto, V. & Hirschberg, J. (2018). Joint Part-of-Speech
and Language ID Tagging for Code-Switched Data. *Proceedings of the
Third Workshop on Computational Approaches to Code-Switching*.

### LinCE Benchmark (Aguilar et al., 2020)

A centralized benchmark for code-switching NLP evaluation, covering
four language pairs (Spanish-English, Nepali-English, Hindi-English,
MSA-Egyptian Arabic) and four tasks (language identification, NER,
POS tagging, sentiment analysis). Established that multilingual BERT
underperforms specialized approaches on code-switching tasks.

**Relevance to our work:** LinCE confirms that code-switching POS
tagging is a recognized benchmark task with established baselines.
Our approach could be evaluated against LinCE's Spanish-English POS
dataset for quantitative comparison, though our focus is on CHAT
transcript morphology rather than social media text.

**Citation:** Aguilar, G., Kar, S., & Solorio, T. (2020). LinCE: A
Centralized Benchmark for Linguistic Code-switching Evaluation.
*Proceedings of LREC 2020*.

### CS-ELMo (Winata et al., 2020) — Transfer Learning with Morphological Clues

Extended ELMo with a position-aware attention mechanism that enhances
morphological clues from character n-grams. The bottom layers of the
ELMo architecture learn morphological patterns that transfer across
languages, establishing state of the art on code-switching NER and POS
tasks.

**Relevance to our work:** CS-ELMo uses subword/character-level features
to capture cross-language morphological patterns. Our approach operates
at the word level and doesn't exploit subword features — this is a
potential improvement direction.

**Citation:** Winata, G.I., Cahyawijaya, S., Lin, Z., Liu, Z., & Fung,
P. (2020). From English to Code-Switching: Transfer Learning with
Strong Morphological Clues. *Proceedings of ACL 2020*.

### "Parsing the Switch" — BiLingua Parser (2025)

Very recent work using GPT-4 with linguistically-informed prompting to
produce UD annotations for code-switched text. The BiLingua Parser
combines few-shot LLM prompting with expert review. LLM-based
annotations outperform conventional parsers in syntactic accuracy,
particularly at switch points where monolingual models typically fail.

**Relevance to our work:** Confirms that monolingual parsers fail at
switch points — exactly the problem our secondary dispatch addresses.
Their LLM-based approach is more powerful but requires API access to
proprietary models, while our approach uses open Stanza models locally.

**Citation:** (2025). Parsing the Switch: LLM-Based UD Annotation for
Complex Code-Switched and Low-Resource Languages. *Findings of EMNLP
2025*.

### CHILDES MOR Program (Sagae et al., 2010; MacWhinney, 2012)

The direct predecessor to our work within TalkBank. The MOR program
provides automatic morphological analysis for CHILDES transcripts using
hand-crafted rule-based grammars. For bilingual corpora, separate MOR
grammars for each language are applied. MOR achieves 98% accuracy on
adult English corpora and 97% on child language.

**Relevance to our work:** Our L2 morphotag feature is the neural
successor to MOR's bilingual capability. MOR uses hand-crafted lexicons
(limited vocabulary, requires manual maintenance); we use Stanza's
neural models (open vocabulary, no manual lexicon needed). The CHAT
`@s` marker convention that we exploit was designed for MOR's bilingual
processing.

**Citations:**
- Sagae, K., Davis, E., Lavie, A., MacWhinney, B. & Wintner, S. (2010).
  Morphosyntactic annotation of CHILDES transcripts. *J. Child Language*.
- MacWhinney, B. (2012). Morphosyntactic Analysis of the CHILDES and
  TalkBank Corpora. *Proceedings of LREC 2012*.

### Code-Switching UD Treebanks

Several code-switching treebanks exist in the Universal Dependencies
framework, providing gold-standard annotations for evaluation:

- **UD Hindi-English HIENCS** (Bhat et al., 2018) — Hindi-English
  code-switching tweets
- **UD Turkish-German SAGT** (Cetinoglu, 2022) — Turkish-German
  conversational code-switching

These could serve as evaluation resources if we wanted to quantitatively
validate our approach against gold-standard code-switching annotations.

### Matrix Language Frame Model (Myers-Scotton, 1993)

The dominant linguistic theory of code-switching. Defines the "matrix
language" (the language providing the grammatical frame) and the
"embedded language" (the language contributing inserted elements). The
Morpheme Order Principle and System Morpheme Principle govern how
elements from the two languages combine.

**Relevance to our work:** Our architecture aligns naturally with the
MLF model. The primary language in batchalign3 IS the matrix language
(providing the syntactic frame via dependency parsing). The `@s` words
ARE embedded language elements. Our merge algorithm respects the matrix
language's structural frame (deprel, head) while filling in the embedded
language's morphology (lemma, features) — this is essentially a
computational implementation of the MLF's Morpheme Order Principle.

## Situating Our Approach

### What We Share with Prior Work

| Technique | Source | Our implementation |
|-----------|--------|-------------------|
| Two monolingual models combined | Solorio 2008 | Primary (matrix lang) + secondary (embedded lang) Stanza |
| Language ID as POS constraint | Soto 2018 | `@s` marker provides perfect LID |
| Monolingual models as foundation | Bhat 2018 | Stanza models for each language |
| UD framework for output | Bhat 2018, BiLingua 2025 | Full UD-to-CHAT mapping pipeline |

### What Appears Novel

1. **Deprel as cross-linguistic POS constraint.** Using the primary
   model's UD dependency relation to constrain the secondary model's
   POS tag, without any code-switching training data. The UD deprel is
   cross-linguistically valid by design (a word with deprel `advmod`
   must be an adverb in any language), but this property has not been
   exploited for code-switching POS tagging in the literature.

2. **Closed-class function word override.** When the secondary model
   returns a closed-class POS (DET, ADP, SCONJ, CCONJ, AUX, PART,
   PRON), trusting it over the structural constraint. This heuristic
   addresses the specific failure mode where the primary model
   mislabels foreign function words (e.g., parsing Spanish `los` as
   English proper noun "Los"). Not described in prior work.

3. **GRA deprel correction from resolved POS.** Upgrading the primary
   model's structural parse (e.g., `FLAT` to `DET`, `OBL` to `DET`)
   based on the resolved POS. Prior work on code-switching dependency
   parsing focuses on building dedicated CS parsers; we post-correct
   the monolingual parser's output.

4. **Zero code-switching training data requirement.** Most approaches
   (Solorio 2008, Bhat 2018, CS-ELMo 2020) require annotated
   code-switching data for training or fine-tuning. Our approach needs
   only standard monolingual Stanza models plus a hand-crafted
   deprel-to-POS mapping table (a pure function with ~30 lines).

5. **Integration with CHAT/TalkBank ecosystem.** The `@s` marker
   provides reliable word-level language identification without any
   LID model. This is a unique advantage of the CHAT annotation
   framework that prior computational work on code-switching does not
   have access to (they must predict LID from text alone).

### Known Limitations Relative to Prior Art

1. **No subword features.** CS-ELMo (2020) shows that character-level
   morphological features transfer across languages and improve CS POS
   tagging. Our approach operates purely at the word level.

2. **No joint training.** Approaches like neural stacking (Bhat 2018)
   learn to handle switch-point phenomena from code-switching data. Our
   rule-based merge cannot learn from data.

3. **Primary model's structural parse may be unreliable.** The 2025
   BiLingua paper confirms that monolingual parsers degrade at switch
   points. Our function-word override addresses the most common failure
   mode, but content words at switch boundaries may still get wrong
   structural analyses.

4. **No evaluation on standard CS benchmarks.** We have evaluated on
   TalkBank bilingual data but not on LinCE or other standard
   code-switching benchmarks. Quantitative comparison with prior work
   would require evaluation on shared datasets.

## Recommendations

1. **Our approach is viable and reasonably novel.** We are not
   reinventing the wheel — the two-model combination idea is well
   established (Solorio 2008). But our specific implementation (deprel
   constraint + closed-class override + zero CS training data) is a
   new combination of known principles.

2. **Cite Solorio 2008 and Bhat 2018** as the key antecedents. Our
   approach can be described as "a rule-based structural merge variant
   of the two-model pipeline approach (Solorio 2008), exploiting UD's
   cross-linguistic deprel semantics for zero-shot code-switching
   morphosyntax."

3. **Consider evaluation on HIENCS** (Hindi-English CS UD treebank) to
   quantitatively validate our structural merge against gold-standard
   code-switching annotations.

4. **The Matrix Language Frame alignment** should be mentioned in any
   publication — our architecture is a natural computational realization
   of the MLF model's matrix/embedded language distinction.

## Related

- [Experimental L2 Morphotag](l2-morphotag.md) — design
  document for the feature
- [L2 & Language Switching](l2-handling.md) — current behavior reference
