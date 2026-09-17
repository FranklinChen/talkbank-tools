//! One comparison: extraction, one whole-file alignment, and every metric and
//! per-utterance view derived from it.
//!
//! Extraction and POS are owned by a ComparedDocument. Conformed sides borrow
//! that document; one alignment borrows both sides and immediately resolves
//! aligner indices into source-word references. Its render operation consumes
//! those bound steps to produce metrics and views.

use talkbank_model::WriteChat;
use talkbank_model::model::{ChatFile, Line};

use crate::dp_align::{self, AlignResult, MatchMode};

use crate::wer_conform::WerNormalization;

use super::language::{
    Gold, GoldNeighbour, InsertionAttribution, InsertionSite, LanguagedUtterance, LanguagedWord,
    Main, PerUtterance, Side, SiteUtterances, UtteranceIndex, languaged_utterances,
};
use super::metrics::{MetricAccumulator, ScoredWord};
use super::model::{
    CompareStatus, CompareToken, ComparisonBundle, GoldCoverage, GoldWordMatch, UtteranceComparison,
};
use super::pos::{GoldPos, GoldTag, MainTag};

/// A word's position among the compared words of its utterance.
///
/// Counts only words that take part in the comparison, so it is also the
/// position [`GoldWordMatch`] reports. Minted only by [`flatten_side`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ComparePosition(usize);

/// Where a token is shown within one utterance's comparison tier.
///
/// An ordered position, not a numeric key: before the first word, at or just
/// after a word, or the terminator, which sorts after everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    /// Before the utterance's first word.
    BeforeFirst,
    /// At, or just after, the word at this position.
    Word {
        position: ComparePosition,
        placement: WordPlacement,
    },
    /// The utterance terminator.
    Terminator,
}

/// Whether a token takes a word's own slot or the one just after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum WordPlacement {
    At,
    After,
}

impl Slot {
    fn at(position: ComparePosition) -> Self {
        Self::Word {
            position,
            placement: WordPlacement::At,
        }
    }

    fn after(position: ComparePosition) -> Self {
        Self::Word {
            position,
            placement: WordPlacement::After,
        }
    }
}

/// Aligned words borrow their actual sources; no detached token index escapes.
enum AlignStep<'w, 'a> {
    Match {
        key: String,
        main: &'w FlattenedWord<'a, Main>,
        gold: &'w FlattenedWord<'a, Gold>,
    },
    Insertion {
        key: String,
        main: &'w FlattenedWord<'a, Main>,
        site: InsertionSite<&'w FlattenedWord<'a, Gold>>,
    },
    Deletion {
        key: String,
        gold: &'w FlattenedWord<'a, Gold>,
    },
}

/// Extraction and POS are built from one source and cannot be paired separately.
struct ComparedDocument<S: Side> {
    utterances: PerUtterance<S, LanguagedUtterance<S>>,
    pos: GoldPos,
    terminators: Vec<Option<String>>,
}

impl<S: Side> ComparedDocument<S> {
    fn of(file: &ChatFile) -> Self {
        Self {
            utterances: languaged_utterances(file),
            pos: GoldPos::of(file),
            terminators: collect_utterance_terminators(file),
        }
    }
}

/// One compared word: the languaged word itself and where it sits.
#[derive(Debug, Clone)]
struct FlattenedWord<'a, S: Side> {
    word: &'a LanguagedWord<S>,
    position: ComparePosition,
    pos: Option<String>,
}

impl GoldNeighbour for &FlattenedWord<'_, Gold> {
    fn gold_utterance(self) -> UtteranceIndex<Gold> {
        self.word.utterance()
    }
}

/// Punctuation and fillers to exclude from comparison (matching BA2 behavior).
///
/// Terminators are recognized via the typed `Terminator` enum so the set
/// stays in lockstep with the grammar. Separators (`,`, `‡`, `„`) are
/// additionally excluded because BA2's compare skipped them too.
pub(in crate::compare) fn is_punct_or_filler(word: &str) -> bool {
    static FILLERS: &[&str] = &["um", "uhm", "em", "mhm", "uhhm", "eh", "uh", "hm"];

    let w = word.trim();
    talkbank_model::model::content::Terminator::is_chat_terminator(w)
        || matches!(w, "," | "‡" | "„")
        || FILLERS.contains(&w.to_lowercase().as_str())
}

/// One side's compared words and the alignment tokens they conform to.
///
/// One word can conform to several tokens; `word_of_token[j]` is the word
/// `tokens[j]` came from. A word is reached only through [`Self::word`], by a
/// token index of the same side.
struct ConformedSide<'a, S: Side> {
    words: Vec<FlattenedWord<'a, S>>,
    tokens: Vec<String>,
    word_of_token: Vec<usize>,
    document: &'a ComparedDocument<S>,
}

impl<'a, S: Side> ConformedSide<'a, S> {
    fn of(document: &'a ComparedDocument<S>, normalization: WerNormalization) -> Self {
        let words = flatten_side(document);
        let mut tokens = Vec::with_capacity(words.len());
        let mut word_of_token = Vec::with_capacity(words.len());
        for (word_index, word) in words.iter().enumerate() {
            let before = tokens.len();
            normalization.conform_word_into(word.word.extracted().text.as_str(), &mut tokens);
            word_of_token.resize(word_of_token.len() + (tokens.len() - before), word_index);
        }
        Self {
            words,
            tokens,
            word_of_token,
            document,
        }
    }

    /// The word a token came from.
    fn word(&self, token: usize) -> &FlattenedWord<'a, S> {
        &self.words[self.word_of_token[token]]
    }

    /// Every token, in order, with the word it came from.
    fn tokens_with_words(&self) -> impl Iterator<Item = (&str, &FlattenedWord<'a, S>)> {
        self.tokens
            .iter()
            .zip(&self.word_of_token)
            .map(|(token, &word)| (token.as_str(), &self.words[word]))
    }
}

/// Whether a main utterance matched any gold word in the whole-file alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MainUtteranceEvidence {
    Matched,
    Unmatched,
}

/// Which main words a comparison scores as insertions.
enum InsertionScope {
    /// Every one: the gold is a full reference ([`GoldCoverage::Complete`]).
    /// Main utterance boundaries play no part.
    EveryMainWord,
    /// Only those in main utterances that matched some gold word
    /// ([`GoldCoverage::Partial`]); the rest lie outside what the gold covers.
    MatchedMainUtterances(PerUtterance<Main, MainUtteranceEvidence>),
}

/// Whether one unmatched main word is scored.
enum WordCoverage {
    Scored,
    OutsideGold,
}

impl InsertionScope {
    fn coverage(&self, word: &FlattenedWord<'_, Main>) -> WordCoverage {
        match self {
            Self::EveryMainWord => WordCoverage::Scored,
            Self::MatchedMainUtterances(evidence) => match evidence[word.word.utterance()] {
                MainUtteranceEvidence::Matched => WordCoverage::Scored,
                MainUtteranceEvidence::Unmatched => WordCoverage::OutsideGold,
            },
        }
    }
}

/// Where one gold utterance landed, used for utterance language agreement only.
///
/// Three outcomes, not an `Option`: an utterance with no compared words and an
/// utterance none of whose words were matched would both be `None`, and
/// utterance language agreement has to tell them apart. The first had nothing
/// to place; the second is an utterance the transcript lost.
#[derive(Debug, Clone, Copy)]
enum GoldPlacement {
    /// No gold token of the utterance has been seen: it has no compared words.
    NothingToPlace,
    /// It has compared words, and the alignment matched none of them.
    Unplaced,
    /// Some of its words matched.
    Placed(MajorityRun),
}

impl GoldPlacement {
    fn deleted(self) -> Self {
        match self {
            Self::NothingToPlace | Self::Unplaced => Self::Unplaced,
            Self::Placed(run) => Self::Placed(run),
        }
    }

    fn matched(self, main: UtteranceIndex<Main>) -> Self {
        match self {
            Self::NothingToPlace | Self::Unplaced => Self::Placed(MajorityRun::starting(main)),
            Self::Placed(run) => Self::Placed(run.with(main)),
        }
    }
}

/// The main utterance holding most of one gold utterance's matched tokens,
/// reduced as the matches arrive.
///
/// A monotone alignment visits one gold utterance's matches in order and never
/// returns to an earlier main utterance, so each main utterance's matches form
/// one unbroken run and the longest run is the majority. Only a strictly longer
/// run replaces the best, so a tie keeps the earlier main utterance and never
/// reaches forward in the file.
#[derive(Debug, Clone, Copy)]
struct MajorityRun {
    best: Run,
    current: Run,
}

#[derive(Debug, Clone, Copy)]
struct Run {
    main: UtteranceIndex<Main>,
    matched: usize,
}

impl MajorityRun {
    fn starting(main: UtteranceIndex<Main>) -> Self {
        let run = Run { main, matched: 1 };
        Self {
            best: run,
            current: run,
        }
    }

    fn with(self, main: UtteranceIndex<Main>) -> Self {
        let current = match self.current.main == main {
            true => Run {
                main,
                matched: self.current.matched + 1,
            },
            false => Run { main, matched: 1 },
        };
        let best = match current.matched > self.best.matched {
            true => current,
            false => self.best,
        };
        Self { best, current }
    }

    fn main_utterance(self) -> UtteranceIndex<Main> {
        self.best.main
    }
}

/// The ONE alignment a comparison makes, and what follows from it.
///
/// Every main token is aligned against every gold token in a single monotone
/// alignment, and every metric and both per-utterance views are derived from
/// its steps. Under [`GoldCoverage::Complete`] nothing reads main utterance
/// boundaries, so the counts, `cwer`, language attribution and the gold view
/// are the same however the main transcript is split into utterances. Gold
/// utterance placement feeds utterance language agreement only.
///
/// Cost: one alignment over the two files' token counts (Hirschberg, after
/// stripping any common prefix and suffix).
struct WholeFileAlignment<'w, 'a> {
    main: &'w ConformedSide<'a, Main>,
    gold: &'w ConformedSide<'a, Gold>,
    coverage: GoldCoverage,
    alignment: Alignment<'w, 'a>,
}

enum Alignment<'w, 'a> {
    /// The gold has no compared words, so there is nothing to align against.
    NoGoldWords,
    Aligned {
        steps: Vec<AlignStep<'w, 'a>>,
        placements: PerUtterance<Gold, GoldPlacement>,
        scope: InsertionScope,
    },
}

impl<'w, 'a> WholeFileAlignment<'w, 'a> {
    fn of(
        main: &'w ConformedSide<'a, Main>,
        gold: &'w ConformedSide<'a, Gold>,
        coverage: GoldCoverage,
    ) -> Self {
        if gold.tokens.is_empty() {
            return Self {
                main,
                gold,
                coverage,
                alignment: Alignment::NoGoldWords,
            };
        }
        let main_utterances = &main.document.utterances;
        let gold_utterances = &gold.document.utterances;
        let mut consumed = None;
        let steps: Vec<_> = dp_align::align(&main.tokens, &gold.tokens, MatchMode::CaseInsensitive)
            .into_iter()
            .map(|result| match result {
                AlignResult::Match {
                    key,
                    payload_idx,
                    reference_idx,
                } => {
                    consumed = Some(reference_idx);
                    AlignStep::Match {
                        key,
                        main: main.word(payload_idx),
                        gold: gold.word(reference_idx),
                    }
                }
                AlignResult::ExtraReference { key, reference_idx } => {
                    consumed = Some(reference_idx);
                    AlignStep::Deletion {
                        key,
                        gold: gold.word(reference_idx),
                    }
                }
                AlignResult::ExtraPayload { key, payload_idx } => {
                    let site = match consumed {
                        None => InsertionSite::BeforeGold { next: gold.word(0) },
                        Some(previous) if previous + 1 < gold.tokens.len() => {
                            InsertionSite::BetweenGoldWords {
                                previous: gold.word(previous),
                                next: gold.word(previous + 1),
                            }
                        }
                        Some(previous) => InsertionSite::AfterGold {
                            previous: gold.word(previous),
                        },
                    };
                    AlignStep::Insertion {
                        key,
                        main: main.word(payload_idx),
                        site,
                    }
                }
            })
            .collect();

        let mut placements = gold_utterances.map_ref(|_| GoldPlacement::NothingToPlace);
        for step in &steps {
            match step {
                AlignStep::Match {
                    main: main_token,
                    gold: gold_token,
                    ..
                } => {
                    let placement = &mut placements[gold_token.word.utterance()];
                    *placement = placement.matched(main_token.word.utterance());
                }
                AlignStep::Deletion {
                    gold: gold_token, ..
                } => {
                    let placement = &mut placements[gold_token.word.utterance()];
                    *placement = placement.deleted();
                }
                AlignStep::Insertion { .. } => {}
            }
        }

        let scope = match coverage {
            GoldCoverage::Complete => InsertionScope::EveryMainWord,
            GoldCoverage::Partial => {
                let mut evidence = main_utterances.map_ref(|_| MainUtteranceEvidence::Unmatched);
                for step in &steps {
                    match step {
                        AlignStep::Match {
                            main: main_token, ..
                        } => {
                            evidence[main_token.word.utterance()] = MainUtteranceEvidence::Matched;
                        }
                        AlignStep::Insertion { .. } | AlignStep::Deletion { .. } => {}
                    }
                }
                InsertionScope::MatchedMainUtterances(evidence)
            }
        };

        Self {
            main,
            gold,
            coverage,
            alignment: Alignment::Aligned {
                steps,
                placements,
                scope,
            },
        }
    }
}

/// A per-utterance view as it is built: each token with the slot it is shown at.
type PositionedView<S> = PerUtterance<S, Vec<(Slot, CompareToken)>>;

/// Where the main view shows the next deletion.
enum MainAnchor<'w, 'a> {
    /// No main word shown yet. Deletions wait, and are shown before the first
    /// main word shown; if none ever is, they appear in the gold view only.
    NothingShown { waiting: Vec<CompareToken> },
    /// Just after this main word, the last one shown.
    After(&'w FlattenedWord<'a, Main>),
}

impl<'w, 'a> MainAnchor<'w, 'a> {
    /// Show a main word's token; deletions after it follow it.
    fn show(
        &mut self,
        word: &'w FlattenedWord<'a, Main>,
        token: CompareToken,
        view: &mut PositionedView<Main>,
    ) {
        let utterance = &mut view[word.word.utterance()];
        utterance.push((Slot::at(word.position), token));
        match std::mem::replace(self, Self::After(word)) {
            Self::NothingShown { waiting } => {
                utterance.extend(waiting.into_iter().map(|token| (Slot::BeforeFirst, token)));
            }
            Self::After(_) => {}
        }
    }

    fn deletion(&mut self, token: CompareToken, view: &mut PositionedView<Main>) {
        match self {
            Self::NothingShown { waiting } => waiting.push(token),
            Self::After(word) => {
                view[word.word.utterance()].push((Slot::after(word.position), token));
            }
        }
    }
}

/// Where the gold view shows an insertion: in its owner's utterance, just after
/// the gold word before it when that word shares the utterance, otherwise before
/// the utterance's first word.
fn gold_view_slot(site: InsertionSite<&FlattenedWord<'_, Gold>>) -> (UtteranceIndex<Gold>, Slot) {
    match site {
        InsertionSite::BeforeGold { next } => (next.word.utterance(), Slot::BeforeFirst),
        InsertionSite::AfterGold { previous } => {
            (previous.word.utterance(), Slot::after(previous.position))
        }
        InsertionSite::BetweenGoldWords { previous, .. } => match site.utterances() {
            SiteUtterances::Within(utterance) => (utterance, Slot::after(previous.position)),
            SiteUtterances::Across { after, .. } => (after, Slot::BeforeFirst),
        },
    }
}

/// Conform every compared word of one file as a gold side, for tests of the
/// token-to-word mapping. Goes through the same extraction, flattening and
/// normalization choice as [`compare`].
#[cfg(test)]
pub(in crate::compare) fn conform_file_words(chat_file: &ChatFile) -> (Vec<String>, Vec<usize>) {
    let document = ComparedDocument::<Gold>::of(chat_file);
    let ConformedSide {
        tokens,
        word_of_token,
        ..
    } = ConformedSide::of(
        &document,
        WerNormalization::for_declared_languages(&chat_file.languages),
    );
    (tokens, word_of_token)
}

/// Compare a main transcript against a gold-standard reference.
///
/// Both inputs are parsed CHAT files. Words are extracted from the Mor
/// domain (excluding punctuation and fillers), each with its resolved
/// language, normalized with one normalization chosen from the gold, and
/// aligned ONCE over the whole file (see [`WholeFileAlignment`]).
///
/// Returns per-utterance comparison annotations and aggregate metrics.
pub fn compare(
    main_file: &ChatFile,
    gold_file: &ChatFile,
    gold_coverage: GoldCoverage,
) -> ComparisonBundle {
    let main_document = ComparedDocument::<Main>::of(main_file);
    let gold_document = ComparedDocument::<Gold>::of(gold_file);
    let normalization = WerNormalization::for_declared_languages(&gold_file.languages);
    let main = ConformedSide::of(&main_document, normalization);
    let gold = ConformedSide::of(&gold_document, normalization);
    WholeFileAlignment::of(&main, &gold, gold_coverage).render()
}

impl WholeFileAlignment<'_, '_> {
    fn render(self) -> ComparisonBundle {
        let Self {
            main,
            gold,
            coverage: gold_coverage,
            alignment,
        } = self;
        let main_utts = &main.document.utterances;
        let gold_utts = &gold.document.utterances;
        let mut metrics = MetricAccumulator::default();
        let mut main_view: PositionedView<Main> = main_utts.map_ref(|_| Vec::new());
        let mut gold_view: PositionedView<Gold> = gold_utts.map_ref(|_| Vec::new());
        let mut gold_word_matches = Vec::new();

        // 3. One alignment of the whole file, then every step once, in order, into
        // the metrics and both views.
        match alignment {
            Alignment::NoGoldWords => match gold_coverage {
                // Nothing matched, so every main utterance is outside what a
                // partial gold covers.
                GoldCoverage::Partial => {}
                GoldCoverage::Complete => {
                    for (key, word) in main.tokens_with_words() {
                        let token = metrics.record(ScoredWord::inserted(
                            key.to_string(),
                            word.pos.clone(),
                            InsertionAttribution::NoGoldWords,
                        ));
                        main_view[word.word.utterance()].push((Slot::at(word.position), token));
                    }
                }
            },
            Alignment::Aligned {
                steps,
                placements,
                scope,
            } => {
                for utterance in gold_utts {
                    match placements[utterance.utterance_index()] {
                        GoldPlacement::NothingToPlace => {}
                        GoldPlacement::Placed(run) => metrics.placed_utterance(
                            utterance.language(),
                            main_utts[run.main_utterance()].language(),
                        ),
                        GoldPlacement::Unplaced => metrics.unplaced_utterance(utterance.language()),
                    }
                }

                let mut anchor = MainAnchor::NothingShown {
                    waiting: Vec::new(),
                };

                for step in steps {
                    match step {
                        AlignStep::Match {
                            key,
                            main: main_word,
                            gold: gold_word,
                        } => {
                            // A tagged gold side attributes the gold form's POS to
                            // every Match (BA2 compare.py:540-550): the gold
                            // standard is what the reviewer needs to see, not the
                            // transcriber's tag. An UNTAGGED gold side has no tag
                            // to attribute, and the rule for that case belongs to
                            // the gold's `GoldPos` rather than to a `None` read off
                            // this one form. See `super::pos`.
                            let token = metrics.record(ScoredWord::matched(
                                key,
                                gold.document.pos.pos_for_match(
                                    GoldTag(gold_word.pos.as_deref()),
                                    MainTag(main_word.pos.as_deref()),
                                ),
                                gold_word.word,
                                main_word.word.language(),
                            ));
                            gold_view[gold_word.word.utterance()]
                                .push((Slot::at(gold_word.position), token.clone()));
                            anchor.show(main_word, token, &mut main_view);

                            // A word conformed to several tokens matches once per
                            // token, back to back; the structural match is the
                            // word's, recorded once.
                            let structural_match = GoldWordMatch {
                                gold_utterance_index: gold_word.word.utterance().raw(),
                                gold_word_position: gold_word.position.0,
                                main_utterance_index: main_word.word.utterance().raw(),
                                main_word_position: main_word.position.0,
                            };
                            if gold_word_matches.last() != Some(&structural_match) {
                                gold_word_matches.push(structural_match);
                            }
                        }
                        AlignStep::Insertion {
                            key,
                            main: main_word,
                            site,
                        } => match scope.coverage(main_word) {
                            WordCoverage::OutsideGold => {}
                            WordCoverage::Scored => {
                                let token = metrics.record(ScoredWord::inserted(
                                    key,
                                    main_word.pos.clone(),
                                    InsertionAttribution::InGold(site.map(|word| word.word)),
                                ));
                                let (utterance, slot) = gold_view_slot(site);
                                gold_view[utterance].push((slot, token.clone()));
                                anchor.show(main_word, token, &mut main_view);
                            }
                        },
                        AlignStep::Deletion {
                            key,
                            gold: gold_word,
                        } => {
                            let token = metrics.record(ScoredWord::deleted(
                                key,
                                gold_word.pos.clone(),
                                gold_word.word,
                            ));
                            gold_view[gold_word.word.utterance()]
                                .push((Slot::at(gold_word.position), token.clone()));
                            anchor.deletion(token, &mut main_view);
                        }
                    }
                }
            }
        }

        // 4. Append the gold utterance terminator as a PUNCT token so gold-projected
        // `%xsrep` / `%xsmor` lines match batchalign2-master output shape.
        for (utterance, terminator) in gold_utts
            .iter()
            .zip(gold.document.terminators.iter().cloned())
        {
            match terminator {
                Some(terminator) => gold_view[utterance.utterance_index()].push((
                    Slot::Terminator,
                    CompareToken {
                        text: terminator,
                        pos: Some("PUNCT".to_string()),
                        status: CompareStatus::Match,
                    },
                )),
                None => {}
            }
        }

        // 5. Order each utterance's tokens by slot. The sort is stable, so tokens
        // sharing a slot, such as the several tokens of one word, keep alignment
        // order.
        for tokens in main_view.values_mut().chain(gold_view.values_mut()) {
            tokens.sort_by_key(|(slot, _)| *slot);
        }

        ComparisonBundle {
            main_utterances: build_utterance_comparisons(main_utts, main_view),
            gold_utterances: build_utterance_comparisons(gold_utts, gold_view),
            gold_word_matches,
            metrics: metrics.finish(),
        }
    }
}

fn build_utterance_comparisons<S: Side>(
    utterances: &PerUtterance<S, LanguagedUtterance<S>>,
    mut view: PositionedView<S>,
) -> Vec<UtteranceComparison> {
    utterances
        .iter()
        .map(|utterance| UtteranceComparison {
            utterance_index: utterance.utterance_index().raw(),
            speaker: utterance.speaker().as_str().to_string(),
            tokens: std::mem::take(&mut view[utterance.utterance_index()])
                .into_iter()
                .map(|(_, token)| token)
                .collect(),
        })
        .collect()
}

/// Flatten one document's extracted utterances for alignment.
///
/// Returns every word that takes part in the comparison, with its position
/// among its utterance's compared words, and the document's own part-of-speech
/// evidence. Both the punctuation filter and the recorded tag come from that
/// evidence, so the file-level question is asked once, here, rather than
/// rediscovered from a `None` at each consumer.
fn flatten_side<'a, S: Side>(document: &'a ComparedDocument<S>) -> Vec<FlattenedWord<'a, S>> {
    let mut words = Vec::new();
    let pos = &document.pos;

    for utt in &document.utterances {
        let mut compare_position = 0usize;
        let utterance = utt.utterance_index().raw();
        for languaged in utt.words() {
            let extracted = languaged.extracted();
            let word_position = extracted.utterance_word_index.raw();
            if pos.excludes_from_comparison(utterance, word_position, extracted.text.as_str()) {
                continue;
            }
            words.push(FlattenedWord {
                word: languaged,
                position: ComparePosition(compare_position),
                pos: pos.tag(utterance, word_position),
            });
            compare_position += 1;
        }
    }

    words
}

pub(in crate::compare) fn collect_utterance_terminators(
    chat_file: &ChatFile,
) -> Vec<Option<String>> {
    let mut terminators = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            terminators.push(
                utt.main
                    .content
                    .terminator
                    .as_ref()
                    .map(|term| term.to_chat_string()),
            );
        }
    }
    terminators
}
