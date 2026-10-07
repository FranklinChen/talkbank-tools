//! Utterance grouping for forced alignment time windows.

use talkbank_model::model::{ChatFile, Line};
use talkbank_model::{UtteranceIdx, WordIdx};

use batchalign_transform::decisions::{
    DecisionRecord, DecisionStrategy, FaStrategy, RefusedWindow,
};

use super::coordinates::{Clamped, FaWindow, FileMs, Ms, Recording, WindowFault};
use super::extraction::collect_fa_words;
use super::presence::RecordingPresence;
use super::speech_rate::SpeechRate;
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
/// Only `Anchored` may exceed the engine budget, and only through pieces that
/// each fit it; `Single` is always within budget.
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
    /// An anchored group's pieces, in word order.
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
    /// per piece for an anchored one. The units' words concatenate to
    /// [`FaGroup::words`], which is what lets per-unit timings be reassembled
    /// into the group's.
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

/// Where an utterance's audio is, or why it has none.
///
/// Returned per utterance by [`estimate_untimed_boundaries`]. The second
/// variant is the one that matters: an untimed run whose remaining audio could
/// not physically contain its words has no window, and saying so is better than
/// computing one. A `TimeSpan` alone could not express it, so the question went
/// unasked and every run got a window regardless.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    /// Audio this utterance can be aligned against.
    Placed(TimeSpan),
    /// No audio can hold these words; the rate says how badly.
    Unplaceable(SpeechRate),
}

/// What [`estimate_untimed_boundaries`] worked out.
///
/// # Why this is not a bare `Vec<Placement>`
///
/// The pass LEARNS one thing besides where each utterance goes: how many of the
/// windows it computed ran past the end of the recording and had to be cut down
/// to fit. That is a fact about our own gap arithmetic, and the `.min()` it
/// used to be made it unobservable, so nobody could tell a session where every
/// window fitted from one where a dozen were trimmed.
#[derive(Debug, PartialEq)]
pub struct Estimates {
    /// Where each utterance's audio is, in file order.
    pub placements: Vec<Placement>,
    /// How many computed windows overshot the recording and were cut to it.
    pub windows_clamped: usize,
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
/// refusals (no request was made for those utterances): physically
/// unplaceable runs, and windows requiring narrower evidence than this
/// engine can safely align. The rest record an over-budget utterance that
/// WAS aligned, in pieces cut at its recovered word anchors.
pub struct Grouping {
    /// Windows to send to the aligner.
    pub groups: Vec<FaGroup>,
    /// One record per utterance grouping refused or split, for durable
    /// evidence.
    pub decisions: Vec<DecisionRecord>,
    /// How many estimated windows overshot the recording and were cut to it.
    ///
    /// Carried here rather than logged and dropped. It survived one function
    /// boundary in [`Estimates`] and then died in a `tracing::warn!`, which is
    /// the shape this struct's own docstring above spends a paragraph refusing:
    /// a caller could not branch on it, report it, or put it in the artifact.
    /// A session where every window fitted and one where a dozen were trimmed
    /// are different, and that difference is about OUR gap arithmetic.
    pub windows_clamped: usize,
}

/// Group utterances from a ChatFile into FA segments.
///
/// Every [`GroupSpan::Single`] window fits `max_group_ms`, including trailing
/// padding. A single utterance that cannot fit is split at the word anchors
/// UTR recovered for it when they allow every piece to fit (a
/// [`GroupSpan::Anchored`] group, recorded as a `window_split_at_anchors`
/// decision), and otherwise refused with durable evidence: `anchor_gap` when
/// the anchors leave a stretch longer than the budget, `anchors_unusable`
/// when recovery matched the utterance but gave no usable cut, `over_budget`
/// when recovery has nothing to say about it. Its supplied timing and words
/// are preserved rather than clipped or fabricated.
///
/// Utterances with no timing bullet are placed by distributing the surrounding
/// gap across them in proportion to word count, EXCEPT where the audio could not
/// physically contain them: such a run is refused and reported in
/// [`Grouping::decisions`] rather than handed to an aligner.
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
    let Estimates {
        placements: estimates,
        windows_clamped,
    } = estimate_untimed_boundaries(chat_file, recording);
    if windows_clamped > 0 {
        // Logged AND returned: the log is the convenience, `Grouping` is the
        // contract.
        tracing::warn!(
            windows_clamped,
            "estimated FA windows ran past the end of the recording and were cut to it"
        );
    }

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
            // Completion reports it untimed with its cause; the estimates
            // above gave it no share of any gap.
            RecordingPresence::NotInRecording(_) => continue,
        }
        let span = match &utt.main.content.bullet {
            Some(b) => TimeSpan::new(b.timing.start_ms, b.timing.end_ms),
            None => match estimates[utt_idx] {
                Placement::Placed(span) => span,
                Placement::Unplaceable(rate) => {
                    decisions.push(DecisionRecord::new_and_trace(
                        line_idx,
                        utt.main.speaker.as_str().to_owned(),
                        DecisionStrategy::Fa(FaStrategy::UnplaceableRun),
                        rate.to_string(),
                        true,
                    ));
                    continue;
                }
            },
        };
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
        // Original/UTR-admitted bullets remain authoritative. Only an untimed
        // source can use a complete candidate envelope or a producer-bound
        // partial-order corridor. Neither selects uncertain words or grants
        // timing authority. This request still needs containment/budget admission.
        let span = if utt.main.content.bullet.is_none() {
            anchors
                .search_window(utterance, words.as_slice(), span, recording)
                .unwrap_or(span)
        } else {
            span
        };
        let window = match GroupWindow::admit(span, budget, recording) {
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
                        match AnchoredSplit::plan(over, utterance, words, anchors.lookup(utterance))
                        {
                            Ok(split) => {
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
            words: words.into_vec(),
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
        windows_clamped,
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
    fn admit(span: TimeSpan, budget: Ms, recording: &Recording) -> Result<Self, WindowRefusal> {
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

/// Pre-compute interpolated estimates for ALL utterances (indexed by utt_idx).
///
/// For timed utterances the estimate is unused (the real bullet is preferred).
/// For untimed utterances the estimate is interpolated from the nearest
/// neighboring timed utterances, with time distributed proportionally by
/// word count within each gap. Falls back to proportional distribution
/// across the full audio when no timed neighbors exist.
///
/// * `chat_file` - The parsed CHAT file to compute estimates for.
/// * `recording` - The audio being aligned against. Taken as a [`Recording`]
///   rather than as a raw `u64` bound, because a bound pulled out of the type
///   one line into the function is a bound nothing can clamp against safely;
///   `group_utterances` did exactly that before 2026-08-15.
pub fn estimate_untimed_boundaries(chat_file: &ChatFile, recording: &Recording) -> Estimates {
    const BUFFER_MS: u64 = 2000;

    let total_audio_ms = recording.duration().get();
    let mut windows_clamped = 0usize;

    // Collect word counts and existing timing for each utterance. One not in
    // the recording has no words to place and is no anchor: it takes no share
    // of a gap and bounds none, whatever timing it carries.
    let mut info: Vec<(usize, Option<TimeSpan>)> = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            info.push(match RecordingPresence::of(utt) {
                RecordingPresence::NotInRecording(_) => (0, None),
                RecordingPresence::InRecording => {
                    let mut words = Vec::new();
                    collect_fa_words(&utt.main.content.content, &mut words);
                    let span = utt
                        .main
                        .content
                        .bullet
                        .as_ref()
                        .map(|b| TimeSpan::new(b.timing.start_ms, b.timing.end_ms));
                    (words.len(), span)
                }
            });
        }
    }

    if info.is_empty() {
        return Estimates {
            placements: Vec::new(),
            windows_clamped: 0,
        };
    }

    let mut estimates = vec![Placement::Placed(TimeSpan::new(0, 0)); info.len()];

    // Process runs of consecutive untimed utterances between timed anchors.
    // A "run" is a maximal sequence of untimed utterances.
    let mut i = 0;
    while i < info.len() {
        // Skip timed utterances: their estimates are unused.
        if let Some(span) = info[i].1 {
            estimates[i] = Placement::Placed(span);
            i += 1;
            continue;
        }

        // Found start of an untimed run. Find its end.
        let run_start = i;
        while i < info.len() && info[i].1.is_none() {
            i += 1;
        }
        let run_end = i; // exclusive

        // Determine the gap boundaries from neighboring timed utterances.
        let gap_start = if run_start > 0 {
            // Previous timed utterance's end_ms
            info[..run_start]
                .iter()
                .rev()
                .find_map(|(_, span)| span.as_ref())
                .map_or(0, |s| s.end_ms)
        } else {
            0
        };
        let gap_end = if run_end < info.len() {
            // Next timed utterance's start_ms
            info[run_end..]
                .iter()
                .find_map(|(_, span)| span.as_ref())
                .map_or(total_audio_ms, |s| s.start_ms)
        } else {
            total_audio_ms
        };

        // Distribute the gap proportionally by word count.
        let run_words: usize = info[run_start..run_end].iter().map(|(w, _)| w).sum();
        if run_words == 0 {
            // No words: give each utterance a zero-width span at gap_start.
            for est in estimates.iter_mut().take(run_end).skip(run_start) {
                *est = Placement::Placed(TimeSpan::new(gap_start, gap_start));
            }
            continue;
        }

        let gap_duration = gap_end.saturating_sub(gap_start);

        // THE REFUSAL. Distributing words across a gap is arithmetic, and
        // arithmetic will happily produce a window at any density; whether a
        // human could have said those words in that audio is a separate
        // question, and it was never asked. On one real session this placed 175
        // words into 291 ms. The resulting window went to an aligner, which is
        // how invented timings got into a transcript.
        //
        // Refusing is not the same as the skip this function's caller used to
        // perform when the audio length was unknown: that fired on OUR ignorance
        // and silently dropped words. This fires on a stated physical fact,
        // names the rate, and the words stay unaligned because no aligner could
        // have placed them anyway.
        // Measured against the audio the ALIGNER will receive, not the raw gap:
        // each estimate is widened by `BUFFER_MS` at both ends below, so the raw
        // gap understates the window and would refuse runs that are perfectly
        // placeable. An untimed utterance immediately before a timed one has a
        // zero-width raw gap and a two-second real window.
        //
        // It does not rescue the case this check exists for: the measured
        // failure was 175 words in 291 ms, and buffering takes that to 4.3
        // seconds, still 40 words per second and still impossible.
        let available = Ms(gap_duration + 2 * BUFFER_MS);
        let rate = SpeechRate::of(run_words, available);
        if !rate.is_possible() {
            tracing::warn!(
                first_utterance = run_start,
                utterances = run_end - run_start,
                %rate,
                "untimed run cannot fit the audio left for it; leaving it unplaced \
                 rather than handing an impossible window to an aligner"
            );
            for est in estimates.iter_mut().take(run_end).skip(run_start) {
                *est = Placement::Unplaceable(rate);
            }
            continue;
        }

        let mut words_before: usize = 0;
        for idx in run_start..run_end {
            let count = info[idx].0;
            let raw_start =
                gap_start + (words_before as f64 / run_words as f64 * gap_duration as f64) as u64;
            let raw_end = gap_start
                + ((words_before + count) as f64 / run_words as f64 * gap_duration as f64) as u64;

            // The START is clamped too, and this is not symmetry for its own
            // sake. `gap_start` and `gap_end` come from transcript BULLETS,
            // which can lie past the end of the recording; that is the whole
            // phantom-timing story. So a buffered start could exceed the audio
            // while the end below was cut down to it, producing a window whose
            // end precedes its start. `TimeSpan::new`'s doc says "caller is
            // responsible for ensuring start <= end" and this is the caller
            // that could not.
            let start =
                match recording.clamp_bound(FileMs::new(raw_start.saturating_sub(BUFFER_MS))) {
                    Clamped::AsGiven(at) => at.get(),
                    Clamped::ClampedTo { bound } => {
                        windows_clamped += 1;
                        bound.get()
                    }
                };
            // `.min(total_audio_ms)` until 2026-08-15, which is correct
            // arithmetic and invisible behaviour: a window that FITTED and one
            // cut down to fit read identically afterwards. Both arms are named
            // here so the second is acknowledged, and it is counted, because a
            // computed window running past the end of the audio is a fact about
            // OUR gap arithmetic rather than about the recording.
            let end = match recording.clamp_bound(FileMs::new(raw_end + BUFFER_MS)) {
                Clamped::AsGiven(at) => at.get(),
                Clamped::ClampedTo { bound } => {
                    windows_clamped += 1;
                    bound.get()
                }
            };

            // A window with no extent is REFUSED rather than handed on as an
            // inverted or empty `TimeSpan`. `SpeechRate::NoAudio` already means
            // exactly this and already reports `is_possible() == false`, so the
            // case needs no new variant: it is the same answer as the density
            // refusal above, reached a different way.
            estimates[idx] = match end > start {
                true => Placement::Placed(TimeSpan::new(start, end)),
                false => Placement::Unplaceable(SpeechRate::of(count, Ms(0))),
            };
            words_before += count;
        }
    }

    Estimates {
        placements: estimates,
        windows_clamped,
    }
}
