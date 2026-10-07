//! Timestamp injection into the CHAT AST.

use talkbank_model::alignment::helpers::{
    TierDomain, WordItemMut, counts_for_tier, walk_words_mut,
};
use talkbank_model::model::{Bullet, Utterance, Word};

use super::origin::Origin;

use super::{ModelAlignmentScore, WordTiming};

/// Whether every part of a merged CHAT word supplied a model score.
///
/// A plain `(sum, count)` can accidentally average only the scored subset.
/// Once one part is missing, this state cannot return to `Complete`, so a
/// compound score exists only when it honestly describes every aligned part.
enum MergedScore {
    Complete {
        weighted_millionths: u128,
        duration_ms: u128,
    },
    Missing,
}

impl MergedScore {
    fn add(self, timing: &WordTiming) -> Self {
        match (self, timing.model_score()) {
            (
                Self::Complete {
                    weighted_millionths,
                    duration_ms,
                },
                Some(score),
            ) => {
                let duration = u128::from(timing.duration_ms());
                Self::Complete {
                    weighted_millionths: weighted_millionths
                        + u128::from(score.millionths()) * duration,
                    duration_ms: duration_ms + duration,
                }
            }
            _ => Self::Missing,
        }
    }

    fn finish(self) -> Option<ModelAlignmentScore> {
        match self {
            Self::Complete {
                weighted_millionths,
                duration_ms,
            } => ModelAlignmentScore::from_weighted_millionths(weighted_millionths, duration_ms),
            Self::Missing => None,
        }
    }
}

/// Read cursor into a flat array of word timings for an FA group.
///
/// Advances by one for each Wor-alignable word encountered.
pub struct TimingCursor<'a> {
    timings: &'a [Option<WordTiming>],
    pos: usize,
}

impl<'a> TimingCursor<'a> {
    /// Create a new cursor starting at the given offset.
    pub fn with_offset(timings: &'a [Option<WordTiming>], offset: usize) -> Self {
        Self {
            timings,
            pos: offset,
        }
    }

    /// Advance the position and return the timing at the previous position.
    ///
    /// Always advances by one, even past the end, this matches the FA injection
    /// invariant that every alignable word must advance the cursor.
    pub fn take(&mut self) -> Option<&WordTiming> {
        let slot = self.timings.get(self.pos);
        self.pos += 1;
        slot.and_then(|o| o.as_ref())
    }

    /// Current read position.
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// Proof of what was just injected into a specific utterance, per word.
///
/// # Why this exists rather than a plain slice
///
/// A `Bullet` is two integers and cannot carry an `Origin`, so writing timings
/// into the AST DESTROYS their provenance. Post-processing then has to be told
/// what they were, and if that were an ordinary parameter a caller could pass
/// the wrong thing, or pass "read it back off the transcript", which is what
/// the pipeline used to do: it relabelled every invented timing as an
/// observation. Only [`inject_timings_for_utterance`] returns one of these, so
/// possession IS the evidence and there is no constructor a caller can reach.
///
/// # Why it accumulates rather than slicing the cursor
///
/// One entry per word this utterance's `walk_words` visits, in that order,
/// recording what was actually SET. A cursor slice would be misaligned: a
/// compound filler consumes N cursor positions for one word, and a word failing
/// `counts_for_tier` consumes none. Six tests caught that.
#[derive(Debug, Clone)]
pub struct InjectedTimings(Vec<Option<WordTiming>>);

impl InjectedTimings {
    /// The timings, with the provenance the aligner gave them.
    pub fn as_slice(&self) -> &[Option<WordTiming>] {
        &self.0
    }

    /// The spans a transcript already carries, for tests that postprocess a
    /// fixture rather than a fresh alignment.
    ///
    /// `#[cfg(test)]`, which is one of the three sanctioned answers to "how
    /// else can this be obtained". It replaces a `TimingSeed::FromTranscript`
    /// variant that was reachable from production code and therefore made the
    /// read-the-bullets-back route selectable by mistake.
    ///
    /// `Origin::TranscriptBullet` is HONEST here, and that is what makes this
    /// legitimate where the old variant was not: these spans genuinely are in
    /// the fixture's transcript. The defect it replaces was stamping that
    /// origin on values this program had just INVENTED.
    #[cfg(test)]
    pub(crate) fn from_transcript(utterance: &talkbank_model::model::Utterance) -> Self {
        let mut spans: Vec<Option<super::TimeSpan>> = Vec::new();
        super::postprocess::collect_transcript_spans(&utterance.main.content.content, &mut spans);
        Self(
            spans
                .into_iter()
                .map(|span| span.and_then(|s| WordTiming::from_transcript(s).ok()))
                .collect(),
        )
    }
}

/// Inject word-level timings into the AST for a specific utterance.
///
/// `timings` is indexed by the flat word position within the group.
/// Only words that are Wor-alignable get timing (matching the extraction order).
///
/// * `utterance` - The utterance whose words will receive inline timing bullets.
/// * `timings` - Flat array of optional timings for the entire FA group. Each
///   element corresponds to one Wor-alignable word across all utterances in the
///   group.
/// * `timing_offset` - Current read position into `timings`. Advanced by one for
///   each Wor-alignable word encountered in this utterance. The caller should
///   initialize this to 0 for the first utterance in a group and pass the same
///   mutable reference through consecutive utterances.
///
/// Returns [`InjectedTimings`], the per-word record of what was set, which is
/// the only honest input to post-processing.
pub fn inject_timings_for_utterance(
    utterance: &mut Utterance,
    timings: &[Option<WordTiming>],
    timing_offset: &mut usize,
) -> InjectedTimings {
    let mut cursor = TimingCursor::with_offset(timings, *timing_offset);
    let mut per_word: Vec<Option<WordTiming>> = Vec::new();
    // domain=None: recurse into all groups unconditionally (FA needs all words)
    walk_words_mut(
        utterance.main.content.content.as_mut_slice(),
        None,
        &mut |leaf| match leaf {
            WordItemMut::Word(word) => {
                per_word.push(inject_timing_on_word(word, &mut cursor));
            }
            WordItemMut::ReplacedWord(replaced) => {
                // Extraction always sends the original word to FA (not the
                // replacement words).  Injection must mirror that policy:
                // consume exactly one cursor position for the original word
                // and set its inline bullet.  Replacement words are never
                // FA-aligned; they are corrections that the speaker did not
                // actually say.  Using the original word here keeps the
                // cursor in sync with extraction across utterance boundaries.
                per_word.push(inject_timing_on_word(&mut replaced.word, &mut cursor));
            }
            WordItemMut::Separator(_) => {}
        },
    );
    *timing_offset = cursor.position();
    InjectedTimings(per_word)
}

/// Inject timing onto a single CHAT word from the FA timing cursor.
///
/// For compound fillers (`&-you_know`), extraction split the word into N
/// parts for FA. We must consume N timings and merge them into one span.
fn inject_timing_on_word(word: &mut Word, cursor: &mut TimingCursor<'_>) -> Option<WordTiming> {
    if !counts_for_tier(word, TierDomain::Wor) {
        return None;
    }

    let parts = compound_filler_part_count(word);
    if parts <= 1 {
        // Normal word: consume one timing, and keep its provenance intact.
        let t = cursor.take()?;
        word.inline_bullet = Some(t.bullet());
        return Some(t.clone());
    }

    // Compound filler: consume N timings and merge into one span. The parts
    // were measured; the envelope is OUR arithmetic over them and covers any
    // silence between, so it says so rather than inheriting one part's origin.
    let mut min_start: Option<u64> = None;
    let mut max_end: Option<u64> = None;
    let mut merged = 0usize;
    let mut merged_score = MergedScore::Complete {
        weighted_millionths: 0,
        duration_ms: 0,
    };
    for _ in 0..parts {
        if let Some(t) = cursor.take() {
            min_start = Some(min_start.map_or(t.start_ms, |s: u64| s.min(t.start_ms)));
            max_end = Some(max_end.map_or(t.end_ms, |e: u64| e.max(t.end_ms)));
            merged += 1;
            merged_score = merged_score.add(t);
        } else {
            merged_score = MergedScore::Missing;
        }
    }
    let (start, end) = (min_start?, max_end?);
    // The merged envelope is admitted as a positive timing FIRST, and the
    // bullet is built from that proof, never from the two numbers.
    let timing = WordTiming::new(
        start,
        end,
        Origin::MergedFromParts { parts: merged },
        Origin::MergedFromParts { parts: merged },
    )
    .map(|timing| match merged_score.finish() {
        Some(score) => timing.with_model_score(score),
        None => timing,
    })?;
    word.inline_bullet = Some(timing.bullet());
    Some(timing)
}

impl WordTiming {
    /// THE construction of a forced-alignment word bullet: from a positive
    /// word timing only. Every FA site that writes a word bullet (injection,
    /// timing post-processing, the bound clamp) goes through here.
    pub(super) fn bullet(&self) -> Bullet {
        Bullet::new(self.start_ms, self.end_ms)
    }
}

/// Return the number of FA words this CHAT word was split into during extraction.
///
/// Delegates to `split_compound_filler`: the single source of truth for the
/// splitting rule shared between extraction and injection.
fn compound_filler_part_count(word: &Word) -> usize {
    super::split_compound_filler(word).len()
}
