//! Utterance grouping for forced alignment time windows.

use talkbank_model::model::{ChatFile, Line};
use talkbank_model::{UtteranceIdx, WordIdx};

use batchalign_transform::decisions::{DecisionRecord, DecisionStrategy, FaStrategy};

use super::coordinates::{Clamped, FaWindow, FileMs, Ms, Recording, WindowFault};
use super::extraction::collect_fa_words;
use super::speech_rate::SpeechRate;
use super::{FaWord, TimeSpan};

/// A group of utterances clustered for a single FA call.
#[derive(Debug)]
pub struct FaGroup {
    /// Audio window for this group.
    audio_window: FaWindow,
    /// Words in this group with positional indices.
    pub words: Vec<FaWord>,
    /// Utterance indices included in this group.
    pub utterance_indices: Vec<UtteranceIdx>,
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
            audio_window: FaWindow::within(
                &recording,
                FileMs::new(audio_span.start_ms),
                FileMs::new(audio_span.end_ms),
            )
            .expect("fixture window"),
            words,
            utterance_indices,
        }
    }

    /// The recording-bound window admitted by grouping, used by live and cached inference.
    pub(crate) fn window(&self) -> FaWindow {
        self.audio_window
    }

    /// Start of the audio window (ms).
    pub fn audio_start_ms(&self) -> u64 {
        self.audio_window.audio_start().get()
    }

    /// End of the audio window (ms).
    pub fn audio_end_ms(&self) -> u64 {
        self.audio_window.end().get()
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
/// Refusals are durable evidence, not just log messages: no request was made
/// for those utterances. They include physically unplaceable runs and windows
/// requiring narrower evidence before this engine can safely align them.
pub struct Grouping {
    /// Windows to send to the aligner.
    pub groups: Vec<FaGroup>,
    /// Utterances left unaligned, one record each, for durable evidence.
    pub refusals: Vec<DecisionRecord>,
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
/// Every returned window fits `max_group_ms`, including trailing padding.
/// A single utterance that cannot fit is refused with durable evidence; its
/// supplied timing and words are preserved rather than clipped or fabricated.
///
/// Utterances with no timing bullet are placed by distributing the surrounding
/// gap across them in proportion to word count, EXCEPT where the audio could not
/// physically contain them: such a run is refused and reported in
/// [`Grouping::refusals`] rather than handed to an aligner.
///
/// * `chat_file` - The parsed CHAT file whose utterances will be grouped.
/// * `max_group_ms` - Maximum audio window duration (in milliseconds) per
///   group. When adding an utterance would push the group past this limit,
///   a new group is started.
/// * `recording` - The audio being aligned against. Required, not optional:
///   when it was an `Option<u64>` a `None` made this function silently SKIP
///   every untimed utterance, so whether a word was ever aligned depended on
///   whether anyone had probed the media.
pub fn group_utterances(
    chat_file: &ChatFile,
    max_group_ms: u64,
    recording: &Recording,
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
    let mut refusals = Vec::new();
    let mut pending: Option<PendingGroup> = None;
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
        let span = match &utt.main.content.bullet {
            Some(b) => TimeSpan::new(b.timing.start_ms, b.timing.end_ms),
            None => match estimates[utt_idx] {
                Placement::Placed(span) => span,
                Placement::Unplaceable(rate) => {
                    refusals.push(DecisionRecord::new_and_trace(
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
        if extracted.is_empty() {
            continue;
        }
        let window = match GroupWindow::admit(span, budget, recording) {
            Ok(window) => window,
            Err(reason) => {
                // Do not clip a long uncertain window or invent word positions.
                // Preserve the supplied CHAT and record why no request was made.
                if let Some(previous) = pending.take() {
                    groups.push(previous.finish(span.start_ms));
                }
                refusals.push(DecisionRecord::new_and_trace(
                    line_idx,
                    utt.main.speaker.as_str().to_owned(),
                    DecisionStrategy::Fa(FaStrategy::WindowRefused),
                    reason.to_string(),
                    true,
                ));
                extracted.clear();
                continue;
            }
        };
        let label_bytes = LabelBytes::total(extracted.iter());
        let words = extracted
            .drain(..)
            .enumerate()
            .map(|(word_idx, text)| FaWord {
                utterance_index: UtteranceIdx::new(utt_idx),
                utterance_word_index: WordIdx::new(word_idx),
                text,
            })
            .collect();
        let next = PendingGroup {
            window,
            words,
            utterance_indices: vec![UtteranceIdx::new(utt_idx)],
            label_bytes,
        };
        pending = Some(match pending.take() {
            None => next,
            Some(mut previous) => match previous.append(next) {
                Ok(()) => previous,
                Err(next) => {
                    groups.push(previous.finish(next.window.window.audio_start().get()));
                    next
                }
            },
        });
    }
    if let Some(last) = pending {
        groups.push(last.finish(total_audio_ms));
    }
    Grouping {
        groups,
        refusals,
        windows_clamped,
    }
}

/// An admitted window carries its maximum end, including any later padding.
/// Only this module can construct or extend it.
struct GroupWindow {
    window: FaWindow,
    end_limit: FileMs,
}

#[derive(Debug, thiserror::Error)]
enum WindowRefusal {
    #[error(transparent)]
    OutsideRecording(#[from] WindowFault),
    #[error("audio window at {start} ms has no positive extent")]
    Empty { start: FileMs },
    #[error(
        "audio window duration {duration} ms exceeds alignment budget {budget} ms; narrower evidence is required"
    )]
    Oversized { duration: Ms, budget: Ms },
}

impl GroupWindow {
    fn admit(span: TimeSpan, budget: Ms, recording: &Recording) -> Result<Self, WindowRefusal> {
        let window = FaWindow::within(
            recording,
            FileMs::new(span.start_ms),
            FileMs::new(span.end_ms),
        )?;
        let duration = window.len();
        if duration.0 == 0 {
            return Err(WindowRefusal::Empty {
                start: window.audio_start(),
            });
        }
        if duration.0 > budget.0 {
            return Err(WindowRefusal::Oversized { duration, budget });
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

/// A nonempty group under construction, before bounded trailing padding.
/// Its words, indices and window move together when a split is necessary.
struct PendingGroup {
    window: GroupWindow,
    words: Vec<FaWord>,
    utterance_indices: Vec<UtteranceIdx>,
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
        self.utterance_indices.append(&mut next.utterance_indices);
        Ok(())
    }

    fn finish(mut self, next_start: u64) -> FaGroup {
        self.window.extend_into_trailing_gap(next_start);
        FaGroup {
            audio_window: self.window.window,
            words: self.words,
            utterance_indices: self.utterance_indices,
        }
    }
}

/// Count utterances with and without timing bullets.
///
/// Returns `(timed, untimed)`: the number of utterances that have a
/// timing bullet and the number that lack one. Non-utterance lines
/// (headers, comments) are not counted.
pub fn count_utterance_timing(chat_file: &ChatFile) -> (usize, usize) {
    let (mut timed, mut untimed) = (0, 0);
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            if utt.main.content.bullet.is_some() {
                timed += 1;
            } else {
                untimed += 1;
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

    // Collect word counts and existing timing for each utterance.
    let mut info: Vec<(usize, Option<TimeSpan>)> = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            let mut words = Vec::new();
            collect_fa_words(&utt.main.content.content, &mut words);
            let span = utt
                .main
                .content
                .bullet
                .as_ref()
                .map(|b| TimeSpan::new(b.timing.start_ms, b.timing.end_ms));
            info.push((words.len(), span));
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
