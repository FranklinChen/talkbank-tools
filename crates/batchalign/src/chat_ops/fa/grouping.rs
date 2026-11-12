//! Utterance grouping for forced alignment time windows.

use talkbank_model::model::{ChatFile, Line, Utterance};
use talkbank_model::{UtteranceIdx, WordIdx};

use batchalign_transform::decisions::{
    DecisionRecord, DecisionStrategy, FaStrategy, RefusedWindow,
};

use super::coordinates::{FaWindow, FileMs, Ms, Recording, WindowFault};
use super::extraction::collect_fa_words;
use super::presence::RecordingPresence;
use super::resource_partition::{ResourceParts, ResourceSplit};
#[cfg(test)]
use super::split::OverBudgetWindow;
use super::split::{AnchoredSplit, NonEmptyWords, Pieces};
use super::utr::{AnchorIndex, CorridorCeiling, CorridorFloor, SearchProvenance};
use super::{FaWord, TimeSpan};
use crate::error::AlignmentSourceBindingFailure;
use crate::types::traces::FaOriginalBulletTrace;

mod population;
pub(crate) use population::SourceUtterance;
use population::{GroupPopulation, RecordedUtterance};

/// A working document and the source evidence captured from that same document.
#[derive(Clone)]
pub(crate) struct SourceBoundDocument {
    file: ChatFile,
    source: SourceCensus,
    identity: SourceIdentity,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceIdentity(std::sync::Arc<SourceIdentityMarker>);

#[derive(Debug)]
struct SourceIdentityMarker;

impl SourceIdentity {
    fn new() -> Self {
        Self(std::sync::Arc::new(SourceIdentityMarker))
    }

    pub(crate) fn same_source(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

pub(crate) struct SourceBoundUtrInput<'source> {
    file: &'source ChatFile,
    identity: SourceIdentity,
}

impl SourceBoundUtrInput<'_> {
    pub(crate) fn document(&self) -> &ChatFile {
        self.file
    }

    pub(crate) fn identity(&self) -> &SourceIdentity {
        &self.identity
    }

    #[cfg(test)]
    pub(crate) fn for_test(file: &ChatFile) -> SourceBoundUtrInput<'_> {
        SourceBoundUtrInput {
            file,
            identity: SourceIdentity::new(),
        }
    }
}

impl SourceBoundDocument {
    pub(crate) fn capture(file: ChatFile) -> Self {
        let source = SourceCensus::capture(&file);
        Self {
            file,
            source,
            identity: SourceIdentity::new(),
        }
    }

    pub(crate) fn utr_input(&self) -> SourceBoundUtrInput<'_> {
        SourceBoundUtrInput {
            file: &self.file,
            identity: self.identity.clone(),
        }
    }

    pub(crate) fn group(
        &self,
        target: crate::types::engines::FaAggregationTarget,
        recording: &Recording,
        anchors: &AnchorIndex,
    ) -> Result<Grouping, AlignmentSourceBindingFailure> {
        let bound = self.source.bind(&self.file)?;
        Ok(group_with_policy(
            &self.file,
            WindowPolicy::Aggregation(target),
            recording,
            anchors,
            Some(bound),
        ))
    }

    pub(crate) fn into_document(self) -> ChatFile {
        self.file
    }
}

impl std::ops::Deref for SourceBoundDocument {
    type Target = ChatFile;

    fn deref(&self) -> &Self::Target {
        &self.file
    }
}

impl std::ops::DerefMut for SourceBoundDocument {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.file
    }
}

/// Original input identities and main bullets, captured before recovery.
#[derive(Clone)]
struct SourceCensus {
    utterances: Vec<SourceRecord>,
}

#[derive(Clone)]
struct SourceRecord {
    speaker: talkbank_model::SpeakerCode,
    location: talkbank_model::Location,
    words: Vec<String>,
    bullet: FaOriginalBulletTrace,
}

impl SourceCensus {
    pub(crate) fn capture(chat: &ChatFile) -> Self {
        let utterances = chat
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Utterance(utterance) => Some(utterance),
                _ => None,
            })
            .map(|utterance| {
                let mut words = Vec::new();
                collect_fa_words(&utterance.main.content.content, &mut words);
                let bullet = match utterance.main.content.bullet.as_ref() {
                    None => FaOriginalBulletTrace::Absent,
                    Some(talkbank_model::BulletState::Parsed(bullet)) => {
                        FaOriginalBulletTrace::Parsed {
                            start_ms: bullet.timing.start_ms,
                            end_ms: bullet.timing.end_ms,
                        }
                    }
                    Some(talkbank_model::BulletState::Unparsed { unparsed }) => {
                        FaOriginalBulletTrace::Unparsed {
                            source: unparsed.source().to_owned(),
                        }
                    }
                };
                SourceRecord {
                    speaker: utterance.main.speaker.clone(),
                    location: utterance.main.location,
                    words,
                    bullet,
                }
            })
            .collect();
        Self { utterances }
    }

    fn bind<'a>(
        &'a self,
        chat: &ChatFile,
    ) -> Result<BoundSources<'a>, AlignmentSourceBindingFailure> {
        let mut current = chat.lines.iter().filter_map(|line| match line {
            Line::Utterance(utterance) => Some(utterance),
            _ => None,
        });
        for (index, source) in self.utterances.iter().enumerate() {
            let utterance_index = UtteranceIdx::new(index);
            let utterance =
                current
                    .next()
                    .ok_or(AlignmentSourceBindingFailure::MissingWorkingUtterance {
                        utterance: utterance_index,
                    })?;
            let mut words = Vec::new();
            collect_fa_words(&utterance.main.content.content, &mut words);
            if source.speaker != utterance.main.speaker {
                return Err(AlignmentSourceBindingFailure::SpeakerChanged {
                    utterance: utterance_index,
                });
            }
            if source.location != utterance.main.location {
                return Err(AlignmentSourceBindingFailure::LocationChanged {
                    utterance: utterance_index,
                });
            }
            if source.words != words {
                return Err(AlignmentSourceBindingFailure::WordsChanged {
                    utterance: utterance_index,
                });
            }
        }
        if current.next().is_some() {
            return Err(AlignmentSourceBindingFailure::UnexpectedWorkingUtterance {
                utterance: UtteranceIdx::new(self.utterances.len()),
            });
        }
        Ok(BoundSources(&self.utterances))
    }
}

struct BoundSources<'a>(&'a [SourceRecord]);

enum SourceObservation<'a> {
    Recorded(&'a SourceRecord),
    NotRecorded,
}

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
#[derive(Debug, Clone)]
pub struct FaGroup {
    span: GroupSpan,
}

/// How a group's audio is presented to the aligner.
///
/// These are source-bound request candidates, not resource permits. Each
/// execution needs preparation against its loaded model and a host lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupSpan {
    /// One complete window covering every word of the group.
    Single(SingleSpan),
    /// One over-budget utterance, aligned piece by piece at recovered word
    /// anchors. Never merged with another utterance.
    Anchored(AnchoredSplit),
    /// Actual model/host constraints determined this source-owned partition.
    Resource(ResourceSplit),
}

/// One or more localized utterances proposed as one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleSpan {
    window: FaWindow,
    population: GroupPopulation,
}

impl SingleSpan {
    /// A single span owns exactly one complete nonempty candidate. There is no
    /// optional request to check or missing-request fallback to manufacture.
    pub(crate) fn request(&self) -> GroupUnit<'_> {
        GroupUnit {
            words: self.population.words(),
            window: self.window,
        }
    }
    /// Retained localized source windows, before candidate merging/padding.
    pub(crate) fn source_utterances(&self) -> impl ExactSizeIterator<Item = SourceUtterance<'_>> {
        self.population.sources()
    }
}

/// One request's share of a group: the consecutive words it aligns and the
/// window they are aligned against.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GroupUnit<'g> {
    /// The words, a contiguous slice of the group's words.
    words: &'g [FaWord],
    /// The window, within the engine budget.
    window: FaWindow,
}

impl<'g> GroupUnit<'g> {
    pub(crate) fn from_supplied(source: &'g super::main_bullets::SuppliedRetrySource) -> Self {
        Self {
            words: source.words(),
            window: source.window(),
        }
    }

    pub(crate) fn words(self) -> &'g [FaWord] {
        self.words
    }
    pub(crate) fn window(self) -> FaWindow {
        self.window
    }
    pub(crate) fn from_piece(piece: &'g super::AnchoredPiece) -> Self {
        Self {
            words: piece.request_words(),
            window: piece.window(),
        }
    }
    pub(crate) fn from_resource(request: &'g super::resource_partition::ResourceRequest) -> Self {
        Self {
            words: request.words(),
            window: request.window(),
        }
    }
}

/// The requests one group is executed as, by shape: a single group's one
/// request, or an anchored group's pieces. A shape rather than a list, so a
/// consumer matches it and a single group's one request needs no allocation.
pub(crate) enum GroupUnits<'g> {
    /// A single group's one request over all its words.
    Whole(GroupUnit<'g>),
    /// An anchored partition, including explicit unavailable word ranges.
    Pieces(Pieces<'g>),
    /// Resource-planned requests and explicit unavailable word ranges.
    Resource(ResourceParts<'g>),
}

#[cfg(test)]
pub(crate) enum GroupRequestIter<'g> {
    Whole(std::option::IntoIter<GroupUnit<'g>>),
    Declared(Pieces<'g>),
    Resource(ResourceParts<'g>),
}

#[cfg(test)]
impl<'g> Iterator for GroupRequestIter<'g> {
    type Item = GroupUnit<'g>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Whole(request) => request.next(),
            Self::Declared(parts) => parts.find_map(|part| match part {
                super::AnchoredPart::Request(piece) => Some(GroupUnit {
                    words: piece.request_words(),
                    window: piece.window(),
                }),
                super::AnchoredPart::Unavailable(_) => None,
            }),
            Self::Resource(parts) => parts.find_map(|part| match part {
                super::resource_partition::ResourcePart::Request(piece) => Some(GroupUnit {
                    words: piece.request_words(),
                    window: piece.window(),
                }),
                super::resource_partition::ResourcePart::Unavailable(_) => None,
            }),
        }
    }
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
    /// Source windows retained by grouping, before request padding or splits.
    pub(crate) fn source_windows(&self) -> Option<Vec<crate::types::traces::FaSourceWindowTrace>> {
        match &self.span {
            GroupSpan::Single(single) => Some(
                single
                    .source_utterances()
                    .map(|source| {
                        source_window_trace(
                            source.utterance(),
                            source.first_word(),
                            source.last_word(),
                            source.window(),
                            source.localization(),
                            source.original_bullet().clone(),
                        )
                    })
                    .collect(),
            ),
            GroupSpan::Resource(split) => Some(vec![split.source().source_window_trace()]),
            GroupSpan::Anchored(_) => None,
        }
    }

    /// Actual candidate requests, including their complete borrowed context;
    /// an unavailable word population can never appear as an executable unit.
    #[cfg(test)]
    pub(crate) fn requests(&self) -> GroupRequestIter<'_> {
        match self.units() {
            GroupUnits::Whole(request) => GroupRequestIter::Whole(Some(request).into_iter()),
            GroupUnits::Pieces(parts) => GroupRequestIter::Declared(parts),
            GroupUnits::Resource(parts) => GroupRequestIter::Resource(parts),
        }
    }
    /// Only the resource planner can produce this closed source geometry.
    pub(crate) fn from_resource_split(split: ResourceSplit) -> Self {
        Self {
            span: GroupSpan::Resource(split),
        }
    }
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
        let window = FaWindow::within(
            &recording,
            FileMs::new(audio_span.start_ms),
            FileMs::new(audio_span.end_ms),
        )
        .expect("fixture window");
        Self {
            span: GroupSpan::Single(SingleSpan {
                window,
                population: GroupPopulation::declared_fixture(window, words, utterance_indices),
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
            GroupSpan::Resource(split) => split.window(),
        }
    }

    /// How the group's audio is presented to the aligner.
    pub fn span(&self) -> &GroupSpan {
        &self.span
    }

    /// The group's words, in the order injection consumes timings.
    pub fn words(&self) -> GroupWords<'_> {
        match &self.span {
            GroupSpan::Single(single) => GroupWords::Single(single.population.words().iter()),
            GroupSpan::Resource(split) => GroupWords::Single(split.words().iter()),
            GroupSpan::Anchored(split) => GroupWords::Anchored {
                pieces: split.pieces(),
                current: [].iter(),
            },
        }
    }

    /// How many words the group holds.
    pub fn word_count(&self) -> usize {
        match &self.span {
            GroupSpan::Single(single) => single.population.words().len(),
            GroupSpan::Anchored(split) => split.pieces().map(|piece| piece.words().len()).sum(),
            GroupSpan::Resource(split) => split.words().len(),
        }
    }

    /// The utterances the group covers, in file order: one for an anchored
    /// group, by construction.
    pub fn utterance_indices(&self) -> Vec<UtteranceIdx> {
        match &self.span {
            GroupSpan::Single(single) => single.population.utterance_indices(),
            GroupSpan::Anchored(split) => vec![*split.utterance()],
            GroupSpan::Resource(split) => vec![*split.utterance()],
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
            GroupSpan::Single(single) => GroupUnits::Whole(single.request()),
            GroupSpan::Anchored(split) => GroupUnits::Pieces(split.pieces()),
            GroupSpan::Resource(split) => GroupUnits::Resource(split.pieces()),
        }
    }
}

/// Historical UTF-8 label-byte preference for joining utterances.
///
/// This bounds merges, not individual utterances and not a model's actual
/// tokenizer population. It was introduced to avoid large Whisper joins;
/// conditioning/special labels and backend tokenization must be checked by
/// loaded-model preparation. Bytes are not a proof of token capacity.
///
/// The unit remains bytes, counted through [`LabelBytes`]. Changing it to
/// characters would change the existing merge policy for non-ASCII text.
/// A single utterance above this preference stays one unchanged candidate;
/// neither dropping words nor inventing an audio cut is permitted.
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
pub(super) const TRAILING_GAP_EXTENSION_MS: u64 = 1500;

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
/// The target bounds joins and trailing padding only. A valid utterance above
/// that target remains one unchanged candidate, with no merge or padding
/// capability. Its feasibility belongs to loaded-model preparation and host
/// admission, not the aggregation preference. No word is discarded to fit it.
///
/// Utterances require supplied timing or source-bound UTR search evidence.
/// Word counts and neighbouring bullets cannot locate an unsupported utterance;
/// missing localization is reported in [`Grouping::decisions`].
///
/// * `chat_file` - The parsed CHAT file whose utterances will be grouped.
/// * `target` - Positive preference for joining neighbouring utterances.
/// * `recording` - The audio being aligned against. Required, not optional:
///   when it was an `Option<u64>` a `None` made this function silently SKIP
///   every untimed utterance, so whether a word was ever aligned depended on
///   whether anyone had probed the media.
/// * `anchors` - Word anchors from the UTR pass over this same document, or
///   [`AnchorIndex::not_observed`] when none ran. Absence of anchors cannot
///   turn an aggregation preference into refusal of a supplied utterance.
pub fn group_utterances(
    chat_file: &ChatFile,
    target: crate::types::engines::FaAggregationTarget,
    recording: &Recording,
    anchors: &AnchorIndex,
) -> Grouping {
    group_with_policy(
        chat_file,
        WindowPolicy::Aggregation(target),
        recording,
        anchors,
        None,
    )
}

/// Exercise the anchored partition policy against an explicitly declared
/// fixture capacity. Engine aggregation preferences cannot call this path.
#[cfg(test)]
pub(crate) fn group_with_declared_window_limit(
    chat_file: &ChatFile,
    capacity_ms: u64,
    recording: &Recording,
    anchors: &AnchorIndex,
) -> Grouping {
    group_with_policy(
        chat_file,
        WindowPolicy::DeclaredCapacity(Ms(capacity_ms)),
        recording,
        anchors,
        None,
    )
}

#[derive(Clone, Copy)]
enum WindowPolicy {
    Aggregation(crate::types::engines::FaAggregationTarget),
    #[cfg(test)]
    DeclaredCapacity(Ms),
}

impl WindowPolicy {
    fn target(self) -> Ms {
        match self {
            Self::Aggregation(target) => Ms(target.milliseconds()),
            #[cfg(test)]
            Self::DeclaredCapacity(capacity) => capacity,
        }
    }
}

fn group_with_policy(
    chat_file: &ChatFile,
    policy: WindowPolicy,
    recording: &Recording,
    anchors: &AnchorIndex,
    source: Option<BoundSources<'_>>,
) -> Grouping {
    let total_audio_ms = recording.duration().get();

    let mut groups = Vec::new();
    let mut decisions = Vec::new();
    let mut pending: Option<Pending> = None;
    let mut extracted = Vec::new();
    let utterances = chat_file
        .lines
        .iter()
        .enumerate()
        .filter_map(|(line_idx, line)| match line {
            Line::Utterance(utterance) => Some((line_idx, utterance)),
            _ => None,
        });
    let originals: Box<dyn Iterator<Item = SourceObservation<'_>> + '_> = match source {
        Some(bound) => Box::new(bound.0.iter().map(SourceObservation::Recorded)),
        None => Box::new(std::iter::repeat_with(|| SourceObservation::NotRecorded)),
    };
    for ((utt_idx, (line_idx, utt)), original) in utterances.enumerate().zip(originals) {
        match RecordingPresence::of(utt) {
            RecordingPresence::InRecording => {}
            // Not speech in this recording: no window and no request.
            // Completion records the exclusion separately from refusals.
            RecordingPresence::NotInRecording(_) => continue,
        }
        let original_bullet = match original {
            SourceObservation::Recorded(source) => {
                extracted.extend(source.words.iter().cloned());
                source.bullet.clone()
            }
            SourceObservation::NotRecorded => {
                collect_fa_words(&utt.main.content.content, &mut extracted);
                FaOriginalBulletTrace::NotRecorded
            }
        };
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
        let Some(localized) = LocalizedUtterance::from_source(
            utt,
            utterance,
            words,
            anchors,
            recording,
            original_bullet,
        ) else {
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
        // The request evidence of a searched utterance: where its window came
        // from, recorded before admission so a refused corridor is explained
        // beside its refusal and an admitted one beside its request.
        if let Localization::Searched(SearchProvenance::OrderCorridor { floor, ceiling }) =
            localized.localization
        {
            let corridor = batchalign_transform::decisions::SearchCorridor {
                start_ms: span.start_ms,
                end_ms: span.end_ms,
                floor: floor.into(),
                ceiling: ceiling.into(),
            };
            decisions.push(DecisionRecord::new_and_trace(
                line_idx,
                utt.main.speaker.as_str().to_owned(),
                DecisionStrategy::Fa(FaStrategy::SearchCorridor(corridor)),
                corridor.to_string(),
                false,
            ));
        }
        let (window, source) = match GroupWindow::for_merge(localized, policy, recording) {
            Ok(admitted) => admitted,
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
                    MergeSeparation::Unmerged(source) => {
                        groups.push(FaGroup {
                            span: GroupSpan::Single(SingleSpan {
                                window: source.window(),
                                population: GroupPopulation::one(*source),
                            }),
                        });
                        continue;
                    }
                    #[cfg(test)]
                    MergeSeparation::Oversized { over, source } => {
                        let (utterance, words) = (*source).into_words();
                        match AnchoredSplit::plan(over, utterance, words, anchors.lookup(utterance))
                        {
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
                    MergeSeparation::OutsideRecording(fault) => {
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
            population: GroupPopulation::one(source),
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
            #[cfg(test)]
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
    utterance: UtteranceIdx,
    words: NonEmptyWords,
    span: TimeSpan,
    /// Where `span` came from, for the request evidence.
    localization: Localization,
    original_bullet: FaOriginalBulletTrace,
}

/// How an utterance's span was localized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Localization {
    /// The working document carried a bullet; its original source is unknown here.
    CurrentBullet,
    /// Utterance timing recovery admitted a search window for its words.
    Searched(SearchProvenance),
    /// A legacy or declared test geometry did not retain localization.
    NotRecorded,
}

pub(super) fn source_window_trace(
    utterance: UtteranceIdx,
    first_word: WordIdx,
    last_word: WordIdx,
    window: FaWindow,
    localization: Localization,
    original_bullet: FaOriginalBulletTrace,
) -> crate::types::traces::FaSourceWindowTrace {
    use crate::types::traces::{FaSourceLocalizationTrace, FaSourceWindowTrace};

    let localization = match localization {
        Localization::CurrentBullet => FaSourceLocalizationTrace::WorkingBullet,
        Localization::Searched(SearchProvenance::Candidates) => {
            FaSourceLocalizationTrace::UtrCandidates
        }
        Localization::Searched(SearchProvenance::OrderCorridor { floor, ceiling }) => {
            let floor: batchalign_transform::decisions::CorridorBound = floor.into();
            let ceiling: batchalign_transform::decisions::CorridorBound = ceiling.into();
            FaSourceLocalizationTrace::UtrOrderCorridor {
                floor: floor.into(),
                ceiling: ceiling.into(),
            }
        }
        Localization::NotRecorded => FaSourceLocalizationTrace::NotRecorded,
    };
    FaSourceWindowTrace {
        utterance: utterance.raw(),
        first_word: first_word.raw(),
        last_word: last_word.raw(),
        start_ms: window.audio_start().get(),
        end_ms: window.end().get(),
        original_bullet,
        localization,
    }
}

/// The corridor's edges as the decision record carries them. The decisions
/// crate cannot see this crate's types, so each edge is restated in plain
/// numbers by one exhaustive match per side; the utterance is its zero-based
/// ordinal among main tiers, as `utterance_idx` is throughout the evidence.
impl From<CorridorFloor> for batchalign_transform::decisions::CorridorBound {
    fn from(floor: CorridorFloor) -> Self {
        match floor {
            CorridorFloor::FileStart => Self::RecordingEdge,
            CorridorFloor::MarkedOverlapUnbounded => Self::MarkedOverlapUnbounded,
            CorridorFloor::PrecedingMeasured(bound) => Self::Measured {
                ms: bound.ms,
                utterance: bound.utterance.index(),
            },
        }
    }
}

impl From<CorridorCeiling> for batchalign_transform::decisions::CorridorBound {
    fn from(ceiling: CorridorCeiling) -> Self {
        match ceiling {
            CorridorCeiling::FileEnd => Self::RecordingEdge,
            CorridorCeiling::FollowingOverlaps { utterance } => Self::FollowingOverlaps {
                utterance: utterance.index(),
            },
            CorridorCeiling::FollowingMeasured(bound) => Self::Measured {
                ms: bound.ms,
                utterance: bound.utterance.index(),
            },
        }
    }
}

impl LocalizedUtterance {
    fn from_source(
        source: &Utterance,
        utterance: UtteranceIdx,
        words: NonEmptyWords,
        anchors: &AnchorIndex,
        recording: &Recording,
        original_bullet: FaOriginalBulletTrace,
    ) -> Option<Self> {
        let (span, localization) = match source
            .main
            .content
            .bullet
            .as_ref()
            .and_then(talkbank_model::BulletState::as_parsed)
        {
            Some(bullet) => (
                TimeSpan::new(bullet.timing.start_ms, bullet.timing.end_ms),
                Localization::CurrentBullet,
            ),
            None => {
                let window = anchors.search_window(
                    utterance,
                    words.as_slice(),
                    // Permit evidence anywhere in the recording, not only inside
                    // a guessed crop. This is a containment bound, never a fallback.
                    TimeSpan::new(0, recording.duration().get()),
                    recording,
                )?;
                (window.span, Localization::Searched(window.provenance))
            }
        };
        Some(Self {
            utterance,
            words,
            span,
            localization,
            original_bullet,
        })
    }
}

/// An admitted window carries its maximum end, including any later padding.
/// Only this module can construct or extend it.
struct GroupWindow {
    window: FaWindow,
    end_limit: FileMs,
}

/// Why a window was not admitted as (part of) a single group.
enum MergeSeparation {
    OutsideRecording(WindowFault),
    /// Valid source window that exceeds a joining preference. Kept whole,
    /// without merge or padding capability, for actual resource admission.
    Unmerged(Box<RecordedUtterance>),
    /// A real window inside the recording, longer than the engine budget:
    /// what an anchored split partitions.
    #[cfg(test)]
    Oversized {
        over: OverBudgetWindow,
        source: Box<RecordedUtterance>,
    },
}

impl From<WindowFault> for MergeSeparation {
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
    fn for_merge(
        localized: LocalizedUtterance,
        policy: WindowPolicy,
        recording: &Recording,
    ) -> Result<(Self, RecordedUtterance), MergeSeparation> {
        let span = localized.span;
        let window = FaWindow::within(
            recording,
            FileMs::new(span.start_ms),
            FileMs::new(span.end_ms),
        )?;
        let source = RecordedUtterance::new(
            localized.utterance,
            localized.words,
            window,
            localized.localization,
            localized.original_bullet,
        );
        // Non-emptiness is already proven: `FaWindow::within` refused an empty
        // span above, through `WindowFault::Empty`.
        let budget = policy.target();
        if window.end().get() - window.audio_start().get() > budget.0 {
            return Err(match policy {
                WindowPolicy::Aggregation(_) => MergeSeparation::Unmerged(Box::new(source)),
                #[cfg(test)]
                WindowPolicy::DeclaredCapacity(_) => MergeSeparation::Oversized {
                    over: OverBudgetWindow::exceeding(window, budget)
                        .expect("declared capacity is exceeded"),
                    source: Box::new(source),
                },
            });
        }
        Ok((
            Self {
                window,
                end_limit: FileMs::new(
                    span.start_ms
                        .saturating_add(budget.0)
                        .min(recording.duration().get()),
                ),
            },
            source,
        ))
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
    /// An anchored group retained only by declared-capacity controls.
    #[cfg(test)]
    Anchored(AnchoredSplit),
}

impl Pending {
    /// Pad into the silence before `next_start` and close the group.
    fn finish(self, next_start: u64) -> FaGroup {
        match self {
            Self::Merging(group) => group.finish(next_start),
            #[cfg(test)]
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
    population: GroupPopulation,
    label_bytes: LabelBytes,
}

impl PendingGroup {
    fn append(&mut self, next: Self) -> Result<(), Self> {
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
        self.population.append(next.population);
        Ok(())
    }

    fn finish(mut self, next_start: u64) -> FaGroup {
        self.window.extend_into_trailing_gap(next_start);
        FaGroup {
            span: GroupSpan::Single(SingleSpan {
                window: self.window.window,
                population: self.population,
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
