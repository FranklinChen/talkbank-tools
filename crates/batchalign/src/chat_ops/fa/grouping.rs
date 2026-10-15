//! Utterance grouping for forced alignment time windows.

use talkbank_model::model::{ChatFile, Line, Utterance};
use talkbank_model::{UtteranceIdx, WordIdx};

use batchalign_transform::decisions::{
    DecisionRecord, DecisionStrategy, FaStrategy, RefusedWindow,
};

use super::coordinates::{FaWindow, FileMs, Ms, Recording, WindowFault};
use super::extraction::collect_fa_words;
use super::presence::RecordingPresence;
use super::split::{AnchoredSplit, NonEmptyWords, OverBudgetWindow, Pieces};
use super::utr::AnchorIndex;
use super::{FaWord, TimeSpan};

/// A group of utterances clustered for FA, and the audio it is aligned against.
///
/// A group is the unit of INJECTION: its timings are written back by walking
/// its words with one cursor, utterance by utterance. How it is EXECUTED is
/// its [`GroupSpan`]: one request over one window, or one request per piece of
/// an anchored split. Either way the group's timings come back as one list in
/// word order, so injection cannot tell the two apart.
///
/// Its only field is the span, and each span holds its own words and
/// utterances: an anchored span's pieces own their words, so there is no
/// second copy of the word list for the pieces to disagree with, and its
/// single utterance is the split's own.
#[derive(Debug)]
pub struct FaGroup {
    span: GroupSpan,
}

/// How a group's audio is presented to the aligner.
///
/// Only `Anchored` may exceed the engine budget. Its request pieces each fit;
/// unavailable pieces keep their words but grant no request. `Single` fits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupSpan {
    /// One window, within budget, covering every word of the group.
    Single(SingleSpan),
    /// One over-budget utterance, aligned piece by piece at recovered word
    /// anchors. Never merged with another utterance.
    Anchored(AnchoredSplit),
}

/// One or more within-budget utterances aligned in one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleSpan {
    window: FaWindow,
    words: Vec<FaWord>,
    utterances: Vec<UtteranceIdx>,
}

/// One request's share of a group: the consecutive words it aligns and the
/// window they are aligned against.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GroupUnit<'g> {
    /// The words, a contiguous slice of the group's words.
    pub(crate) words: &'g [FaWord],
    /// The window, within the engine budget.
    pub(crate) window: FaWindow,
}

/// The requests one group is executed as, by shape: a single group's one
/// request, or an anchored group's pieces. A shape rather than a list, so a
/// consumer matches it and a single group's one request needs no allocation.
pub(crate) enum GroupUnits<'g> {
    /// A single group's one request over all its words.
    Whole(GroupUnit<'g>),
    /// An anchored partition, including explicit unavailable word ranges.
    Pieces(Pieces<'g>),
}

/// A group's words in the order injection consumes timings, without
/// allocating.
pub enum GroupWords<'g> {
    /// A single group's word list.
    Single(std::slice::Iter<'g, FaWord>),
    /// An anchored group's pieces' words, piece after piece.
    Anchored {
        /// The pieces not yet started.
        pieces: Pieces<'g>,
        /// The words left in the current piece.
        current: std::slice::Iter<'g, FaWord>,
    },
}

impl<'g> Iterator for GroupWords<'g> {
    type Item = &'g FaWord;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Single(words) => words.next(),
            Self::Anchored { pieces, current } => loop {
                if let Some(word) = current.next() {
                    return Some(word);
                }
                let piece = pieces.next()?;
                *current = piece.words().iter();
            },
        }
    }
}

impl FaGroup {
    /// Supply raw groups only in unit tests of downstream timing/recovery code.
    /// Production callers obtain groups exclusively from `group_utterances`.
    #[cfg(test)]
    pub(crate) fn test_fixture(
        audio_span: TimeSpan,
        words: Vec<FaWord>,
        utterance_indices: Vec<UtteranceIdx>,
    ) -> Self {
        // These downstream fixtures declare a recording ending at their window.
        let recording = Recording::of_duration(Ms(audio_span.end_ms)).expect("fixture recording");
        Self {
            span: GroupSpan::Single(SingleSpan {
                window: FaWindow::within(
                    &recording,
                    FileMs::new(audio_span.start_ms),
                    FileMs::new(audio_span.end_ms),
                )
                .expect("fixture window"),
                words,
                utterances: utterance_indices,
            }),
        }
    }

    /// An anchored group for unit tests of dispatch and injection; the split
    /// still comes from `AnchoredSplit::plan`, so its pieces are real.
    #[cfg(test)]
    pub(crate) fn anchored_test_fixture(split: AnchoredSplit) -> Self {
        Self {
            span: GroupSpan::Anchored(split),
        }
    }

    /// The whole audio the group covers: its one window, or the utterance
    /// window an anchored split partitions.
    pub(crate) fn window(&self) -> FaWindow {
        match &self.span {
            GroupSpan::Single(single) => single.window,
            GroupSpan::Anchored(split) => split.window(),
        }
    }

    /// How the group's audio is presented to the aligner.
    pub fn span(&self) -> &GroupSpan {
        &self.span
    }

    /// The group's words, in the order injection consumes timings.
    pub fn words(&self) -> GroupWords<'_> {
        match &self.span {
            GroupSpan::Single(single) => GroupWords::Single(single.words.iter()),
            GroupSpan::Anchored(split) => GroupWords::Anchored {
                pieces: split.pieces(),
                current: [].iter(),
            },
        }
    }

    /// How many words the group holds.
    pub fn word_count(&self) -> usize {
        match &self.span {
            GroupSpan::Single(single) => single.words.len(),
            GroupSpan::Anchored(split) => split.pieces().map(|piece| piece.words().len()).sum(),
        }
    }

    /// The utterances the group covers, in file order: one for an anchored
    /// group, by construction.
    pub fn utterance_indices(&self) -> &[UtteranceIdx] {
        match &self.span {
            GroupSpan::Single(single) => &single.utterances,
            GroupSpan::Anchored(split) => std::slice::from_ref(split.utterance()),
        }
    }

    /// Whether every utterance of the group may take its timing from its
    /// existing `%wor` tier: the one rule both FA paths reuse a group by.
    pub fn is_reusable_from(&self, reusable_utterances: &std::collections::HashSet<usize>) -> bool {
        let utterances = self.utterance_indices();
        !utterances.is_empty()
            && utterances
                .iter()
                .all(|utterance| reusable_utterances.contains(&utterance.raw()))
    }

    /// Start of the group's whole audio span (ms).
    pub fn audio_start_ms(&self) -> u64 {
        self.window().audio_start().get()
    }

    /// End of the group's whole audio span (ms).
    pub fn audio_end_ms(&self) -> u64 {
        self.window().end().get()
    }

    /// The requests this group is executed as: one for a single group, one
    /// per admitted piece for an anchored one. Unavailable pieces also own
    /// their words; assembly supplies their untimed slots before injection.
    pub(crate) fn units(&self) -> GroupUnits<'_> {
        match &self.span {
            GroupSpan::Single(single) => GroupUnits::Whole(GroupUnit {
                words: &single.words,
                window: single.window,
            }),
            GroupSpan::Anchored(split) => GroupUnits::Pieces(split.pieces()),
        }
    }
}

/// The most label BYTES any one FA group may carry, whatever engine aligns it.
///
/// The number is Whisper's CTC decoder limit, which is stated in TOKENS:
/// exceeding it causes a Python-side `ValueError: Labels' sequence length N
/// cannot exceed the maximum allowed length of 448 tokens`. It is applied to
/// EVERY group, not only Whisper's, and the name says so, because the previous
/// name (`WHISPER_FA_MAX_LABEL_TOKENS`) claimed an engine-specific contract
/// that nothing here can honour: [`group_utterances`] is never told which
/// engine will align its groups. Its two production call sites pass a CHAT
/// file, a millisecond budget and a recording, so gating the cap on the engine
/// is a change to those call sites and to this signature, not to this
/// constant. Until then the tightest engine's limit is applied uniformly,
/// which is conservative for the others: a group smaller than an engine needs
/// is a waste, where one larger than it accepts is a failed request.
///
/// # Why BYTES, and why that is the conservative choice
///
/// The budget is not the same quantity as the limit, so one of the two
/// directions of error is safe and the other is not. Every token of any of
/// these tokenizers occupies at least one UTF-8 byte of the text it covers, so
/// a byte count BOUNDS the token count from above: under the byte cap implies
/// under the token limit, for every script. A CHARACTER count does not bound
/// it, and outside ASCII it is smaller than the byte count, so counting
/// characters LOOSENS the cap against a limit stated in tokens and can let a
/// group through that provokes the very `ValueError` the cap exists to
/// prevent. It was briefly changed to characters on the reasoning that "a
/// token is a character", which is true of no tokenizer any of these engines
/// uses.
///
/// Bytes therefore over-split non-Latin scripts, roughly threefold for
/// Devanagari, CJK and most Indic scripts and twofold for Cyrillic and Greek.
/// That is a known and deliberate cost: more, smaller FA groups still align
/// correctly, where an oversized group is a hard engine failure. Tightening it
/// means asking the engine for its real tokenizer, not swapping one proxy for
/// a looser one.
///
/// # Known limit: this bounds a MERGE, not every group
///
/// The cap is enforced in [`PendingGroup::append`], which is the only place
/// two utterances are joined. A SINGLE utterance whose own labels exceed the
/// cap is never split: it becomes its own group and is sent as it stands. So
/// the guarantee is "merging never creates an oversized group", not "no group
/// exceeds the cap". Splitting one utterance would mean splitting its audio
/// window and its word list at a position nothing here can justify, so it is
/// deliberately left to the engine to refuse.
///
/// The one split grouping does make is for DURATION, not labels: an utterance
/// whose window exceeds the engine budget is cut at words UTR heard (see
/// `chat_ops::fa::split`), where the anchors are the justification. Its
/// pieces carry fewer labels as a side effect, but a within-budget utterance
/// over the label cap is still sent whole.
///
/// The unit is a BYTE. See [`LabelBytes`], which is the only way to produce a
/// value that may be compared against this.
pub const MAX_GROUP_LABEL_BYTES: usize = 448;

/// A count of label BYTES, the unit [`MAX_GROUP_LABEL_BYTES`] is in.
///
/// # Why this is a type and not a `usize`
///
/// The field it replaces was a bare `usize`, so nothing said which quantity it
/// held and nothing stopped a differently-counted number being assigned to it.
/// That is exactly what happened: the field was renamed to `characters` and
/// refilled from `chars().count()`, changing the meaning of the budget without
/// changing the type of anything, and the constant it is compared against went
/// on being a token limit. Naming the variable carefully was what the code did
/// instead of typing it, and that careful naming was the tell that this type
/// was owed.
///
/// [`LabelBytes::of`] is the sole constructor, so a count in any other unit
/// has no route into the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LabelBytes(usize);

impl LabelBytes {
    /// Count one label's UTF-8 bytes. The boundary: raw text in, budget out.
    fn of(text: &str) -> Self {
        Self(text.len())
    }

    /// Sum a group's labels.
    fn total<'a>(labels: impl IntoIterator<Item = &'a String>) -> Self {
        Self(labels.into_iter().map(|label| Self::of(label).0).sum())
    }

    /// The two groups' labels together, saturating rather than wrapping so an
    /// absurd input cannot make an oversized group look small.
    fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Whether this many labels is more than one group may carry.
    fn exceeds_group_cap(self) -> bool {
        self.0 > MAX_GROUP_LABEL_BYTES
    }
}

/// Maximum extension (ms) into the gap after the last utterance in a group.
///
/// When an utterance bullet ends before the next utterance starts, we extend
/// the FA group's audio window into that gap so the FA engine can hear
/// trailing fillers (`&-you_know`, `&-sort_of`) that live between utterances.
/// The extension is capped at this value to avoid bleeding into the next
/// utterance's content.
const TRAILING_GAP_EXTENSION_MS: u64 = 1500;

/// What grouping produced.
///
/// Its decisions are durable evidence, not just log messages. Most are
/// refusals (no request was made for those utterances): missing localization,
/// and windows requiring narrower evidence than this
/// engine can safely align. The rest record an over-budget utterance that
/// WAS aligned, in pieces cut at its recovered word anchors.
pub struct Grouping {
    /// Windows to send to the aligner.
    pub groups: Vec<FaGroup>,
    /// One record per utterance grouping refused or split, for durable
    /// evidence.
    pub decisions: Vec<DecisionRecord>,
    /// Legacy reporting counter, now zero: proportional windows are no longer
    /// produced. Kept for compatibility with existing result records.
    pub windows_clamped: usize,
}

/// Group utterances from a ChatFile into FA segments.
///
/// Every [`GroupSpan::Single`] window fits `max_group_ms`, including trailing
/// padding. A single utterance that cannot fit is split at the word anchors
/// UTR recovered for it. An [`GroupSpan::Anchored`] partition retains bounded
/// requests even when another part is unavailable. Complete partitions record
/// `window_split_at_anchors`; partial ones record each unavailable `anchor_gap`.
/// A whole utterance is still refused as `anchors_unusable`
/// when recovery matched the utterance but gave no usable cut, `over_budget`
/// when recovery has nothing to say about it. Its supplied timing and words
/// are preserved rather than clipped or fabricated.
///
/// Utterances require supplied timing or source-bound UTR search evidence.
/// Word counts and neighbouring bullets cannot locate an unsupported utterance;
/// missing localization is reported in [`Grouping::decisions`].
///
/// * `chat_file` - The parsed CHAT file whose utterances will be grouped.
/// * `max_group_ms` - Maximum audio window duration (in milliseconds) per
///   group. When adding an utterance would push the group past this limit,
///   a new group is started.
/// * `recording` - The audio being aligned against. Required, not optional:
///   when it was an `Option<u64>` a `None` made this function silently SKIP
///   every untimed utterance, so whether a word was ever aligned depended on
///   whether anyone had probed the media.
/// * `anchors` - Word anchors from the UTR pass over this same document, or
///   [`AnchorIndex::not_observed`] when none ran; with no anchors, no
///   utterance is split and every over-budget one is refused `over_budget`.
pub fn group_utterances(
    chat_file: &ChatFile,
    max_group_ms: u64,
    recording: &Recording,
    anchors: &AnchorIndex,
) -> Grouping {
    let total_audio_ms = recording.duration().get();

    let mut groups = Vec::new();
    let mut decisions = Vec::new();
    let mut pending: Option<Pending> = None;
    let budget = Ms(max_group_ms);
    let mut extracted = Vec::new();
    let utterances = chat_file
        .lines
        .iter()
        .enumerate()
        .filter_map(|(line_idx, line)| match line {
            Line::Utterance(utterance) => Some((line_idx, utterance)),
            _ => None,
        });
    for (utt_idx, (line_idx, utt)) in utterances.enumerate() {
        match RecordingPresence::of(utt) {
            RecordingPresence::InRecording => {}
            // Not speech in this recording: no window and no request.
            // Completion records the exclusion separately from refusals.
            RecordingPresence::NotInRecording(_) => continue,
        }
        collect_fa_words(&utt.main.content.content, &mut extracted);
        let label_bytes = LabelBytes::total(extracted.iter());
        let utterance = UtteranceIdx::new(utt_idx);
        // An utterance with no alignable word makes no request.
        let Some(words) = NonEmptyWords::from_vec(
            extracted
                .drain(..)
                .enumerate()
                .map(|(word_idx, text)| FaWord {
                    utterance_index: utterance,
                    utterance_word_index: WordIdx::new(word_idx),
                    text,
                })
                .collect(),
        ) else {
            continue;
        };
        let Some(localized) =
            LocalizedUtterance::from_source(utt, utterance, words, anchors, recording)
        else {
            // Do not merge supported neighbours across an unsupported turn.
            if let Some(previous) = pending.take() {
                groups.push(previous.finish(total_audio_ms));
            }
            decisions.push(DecisionRecord::new_and_trace(
                line_idx,
                utt.main.speaker.as_str().to_owned(),
                DecisionStrategy::Fa(FaStrategy::LocalizationUnavailable),
                "no supplied timing or source-bound UTR search evidence".to_owned(),
                true,
            ));
            continue;
        };
        let span = localized.span;
        let window = match GroupWindow::admit(&localized, budget, recording) {
            Ok(window) => window,
            Err(refusal) => {
                // Neither outcome below joins a pending group: a refused
                // utterance makes no request, and an anchored one is a group
                // of its own (it is over budget by definition, so no
                // neighbour could share its window). So the group before it
                // is finished here, padded into the gap up to its start.
                if let Some(previous) = pending.take() {
                    groups.push(previous.finish(span.start_ms));
                }
                let speaker = utt.main.speaker.as_str().to_owned();
                let refused = match refusal {
                    WindowRefusal::Oversized(over) => {
                        match AnchoredSplit::plan(
                            over,
                            utterance,
                            localized.words,
                            anchors.lookup(utterance),
                        ) {
                            Ok(split) => {
                                let refused: Vec<_> = split
                                    .pieces()
                                    .filter_map(|part| match part {
                                        super::AnchoredPart::Request(_) => None,
                                        super::AnchoredPart::Unavailable(piece) => {
                                            Some(piece.refusal())
                                        }
                                    })
                                    .collect();
                                if refused.is_empty() {
                                    let decision = split.decision();
                                    decisions.push(DecisionRecord::new_and_trace(
                                        line_idx,
                                        speaker,
                                        DecisionStrategy::Fa(FaStrategy::WindowSplitAtAnchors(
                                            decision,
                                        )),
                                        decision.to_string(),
                                        false,
                                    ));
                                } else {
                                    for refused in refused {
                                        decisions.push(DecisionRecord::new_and_trace(
                                            line_idx,
                                            speaker.clone(),
                                            DecisionStrategy::Fa(FaStrategy::WindowRefused(
                                                refused,
                                            )),
                                            refused.to_string(),
                                            true,
                                        ));
                                    }
                                }
                                // Held until the next utterance's start is
                                // known, so its last piece can be padded.
                                pending = Some(Pending::Anchored(split));
                                continue;
                            }
                            Err(refusal) => refusal.into_refused_window(over),
                        }
                    }
                    WindowRefusal::OutsideRecording(fault) => {
                        fault_refusal(fault, FileMs::new(span.start_ms))
                    }
                };
                // Do not clip a long uncertain window or invent word positions.
                // Preserve the supplied CHAT and record why no request was made.
                decisions.push(DecisionRecord::new_and_trace(
                    line_idx,
                    speaker,
                    DecisionStrategy::Fa(FaStrategy::WindowRefused(refused)),
                    refused.to_string(),
                    true,
                ));
                continue;
            }
        };
        let next = PendingGroup {
            window,
            words: localized.words.into_vec(),
            utterances: vec![utterance],
            label_bytes,
        };
        let next_start = next.window.window.audio_start().get();
        pending = Some(match pending.take() {
            None => Pending::Merging(next),
            Some(Pending::Merging(mut previous)) => match previous.append(next) {
                Ok(()) => Pending::Merging(previous),
                Err(next) => {
                    groups.push(previous.finish(next_start));
                    Pending::Merging(next)
                }
            },
            // An anchored group is never merged with a neighbour.
            Some(anchored @ Pending::Anchored(_)) => {
                groups.push(anchored.finish(next_start));
                Pending::Merging(next)
            }
        });
    }
    if let Some(last) = pending {
        groups.push(last.finish(total_audio_ms));
    }
    Grouping {
        groups,
        decisions,
        // Retained reporting field; no proportional windows are produced.
        windows_clamped: 0,
    }
}

/// A source-localized population, before recording/budget admission.
///
/// Owns the words whose UTR evidence was checked. The constructor has no raw
/// fallback-span argument: proportional arithmetic cannot grant localization.
/// Supplied timing and UTR evidence can still be wrong acoustically.
struct LocalizedUtterance {
    words: NonEmptyWords,
    span: TimeSpan,
}

impl LocalizedUtterance {
    fn from_source(
        source: &Utterance,
        utterance: UtteranceIdx,
        words: NonEmptyWords,
        anchors: &AnchorIndex,
        recording: &Recording,
    ) -> Option<Self> {
        let span = match &source.main.content.bullet {
            Some(bullet) => TimeSpan::new(bullet.timing.start_ms, bullet.timing.end_ms),
            None => anchors.search_window(
                utterance,
                words.as_slice(),
                // Permit evidence anywhere in the recording, not only inside
                // a guessed crop. This is a containment bound, never a fallback.
                TimeSpan::new(0, recording.duration().get()),
                recording,
            )?,
        };
        Some(Self { words, span })
    }
}

/// An admitted window carries its maximum end, including any later padding.
/// Only this module can construct or extend it.
struct GroupWindow {
    window: FaWindow,
    end_limit: FileMs,
}

/// Why a window was not admitted as (part of) a single group.
#[derive(Debug)]
enum WindowRefusal {
    OutsideRecording(WindowFault),
    /// A real window inside the recording, longer than the engine budget:
    /// what an anchored split partitions.
    Oversized(OverBudgetWindow),
}

impl From<WindowFault> for WindowRefusal {
    fn from(fault: WindowFault) -> Self {
        Self::OutsideRecording(fault)
    }
}

/// The one conversion from a window fault to its evidence form.
///
/// Every `WindowFault` carries its own figures; only `PastRecording` needs the
/// window's start, which the fault does not record. Exhaustive, so a new fault
/// cannot be recorded without a variant.
fn fault_refusal(fault: WindowFault, window_start: FileMs) -> RefusedWindow {
    match fault {
        WindowFault::Empty { at } => RefusedWindow::Empty { at_ms: at.get() },
        WindowFault::Inverted { start, end } => RefusedWindow::Inverted {
            start_ms: start.get(),
            end_ms: end.get(),
        },
        WindowFault::PastRecording { end, exceeds_by } => RefusedWindow::PastRecording {
            start_ms: window_start.get(),
            end_ms: end.get(),
            exceeds_by_ms: exceeds_by.0,
        },
    }
}

impl GroupWindow {
    fn admit(
        localized: &LocalizedUtterance,
        budget: Ms,
        recording: &Recording,
    ) -> Result<Self, WindowRefusal> {
        let span = localized.span;
        let window = FaWindow::within(
            recording,
            FileMs::new(span.start_ms),
            FileMs::new(span.end_ms),
        )?;
        // Non-emptiness is already proven: `FaWindow::within` refused an empty
        // span above, through `WindowFault::Empty`.
        if let Some(over) = OverBudgetWindow::exceeding(window, budget) {
            return Err(WindowRefusal::Oversized(over));
        }
        Ok(Self {
            window,
            end_limit: FileMs::new(
                span.start_ms
                    .saturating_add(budget.0)
                    .min(recording.duration().get()),
            ),
        })
    }

    fn extend_into_trailing_gap(&mut self, next_start: u64) {
        let gap = next_start.saturating_sub(self.window.end().get());
        let extension = (gap / 2)
            .min(TRAILING_GAP_EXTENSION_MS)
            .min(self.end_limit.get() - self.window.end().get());
        self.window = self.window.extend_by(Ms(extension));
    }
}

/// The group grouping has not finished yet: it waits for the next
/// utterance's start, which its trailing padding is measured against.
enum Pending {
    /// Within-budget utterances, which the next one may still join.
    Merging(PendingGroup),
    /// An anchored group, which nothing joins; only its padding is pending.
    Anchored(AnchoredSplit),
}

impl Pending {
    /// Pad into the silence before `next_start` and close the group.
    fn finish(self, next_start: u64) -> FaGroup {
        match self {
            Self::Merging(group) => group.finish(next_start),
            Self::Anchored(mut split) => {
                split.extend_into_trailing_gap(
                    FileMs::new(next_start),
                    Ms(TRAILING_GAP_EXTENSION_MS),
                );
                FaGroup {
                    span: GroupSpan::Anchored(split),
                }
            }
        }
    }
}

/// A nonempty group of within-budget utterances under construction, before
/// bounded trailing padding. Its words, utterances and window move together
/// when a split is necessary.
struct PendingGroup {
    window: GroupWindow,
    words: Vec<FaWord>,
    utterances: Vec<UtteranceIdx>,
    label_bytes: LabelBytes,
}

impl PendingGroup {
    fn append(&mut self, mut next: Self) -> Result<(), Self> {
        if next.window.window.audio_start().get() < self.window.window.audio_start().get()
            || next.window.window.end().get() > self.window.end_limit.get()
            || self
                .label_bytes
                .saturating_add(next.label_bytes)
                .exceeds_group_cap()
        {
            return Err(next);
        }
        let extension = next
            .window
            .window
            .end()
            .get()
            .saturating_sub(self.window.window.end().get());
        self.window.window = self.window.window.extend_by(Ms(extension));
        self.label_bytes = self.label_bytes.saturating_add(next.label_bytes);
        self.words.append(&mut next.words);
        self.utterances.append(&mut next.utterances);
        Ok(())
    }

    fn finish(mut self, next_start: u64) -> FaGroup {
        self.window.extend_into_trailing_gap(next_start);
        FaGroup {
            span: GroupSpan::Single(SingleSpan {
                window: self.window.window,
                words: self.words,
                utterances: self.utterances,
            }),
        }
    }
}

/// Count utterances with and without timing bullets.
///
/// Returns `(timed, untimed)`: the number of utterances that have a
/// timing bullet and the number that lack one. Non-utterance lines
/// (headers, comments) are not counted, and neither is an utterance not in
/// the recording ([`RecordingPresence::NotInRecording`]): it needs no
/// timing, so it is not untimed work for recovery to do.
pub fn count_utterance_timing(chat_file: &ChatFile) -> (usize, usize) {
    let (mut timed, mut untimed) = (0, 0);
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            match RecordingPresence::of(utt) {
                RecordingPresence::NotInRecording(_) => {}
                RecordingPresence::InRecording => {
                    if utt.main.content.bullet.is_some() {
                        timed += 1;
                    } else {
                        untimed += 1;
                    }
                }
            }
        }
    }
    (timed, untimed)
}
