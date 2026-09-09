//! Where a timing number came from, carried with the number.
//!
//! # Why provenance is a type and not a comment
//!
//! A CHAT bullet is a pair of integers. Nothing about `1266565_1286865`
//! distinguishes a moment an engine measured against audio from one this
//! program computed by dividing a gap by a word count, from one a repair pass
//! rebalanced against its neighbour, from one cut down to fit the recording.
//! All four are written identically and all four read downstream as
//! measurements.
//!
//! That is not a hypothetical loss. On one merged corpus roughly 37 percent of
//! the timings were interpolated rather than measured, and the two were
//! indistinguishable in the output, so a comparison against a reference
//! transcript reported 37.2 percent agreement when aligning by time and 76.4
//! percent aligning by text: the gap was almost entirely our own invented
//! timings being scored as though they were observations. There was no way to
//! ask the data which numbers were real, because the data had never been asked
//! to remember.
//!
//! # The rule this module enforces
//!
//! Every timing that reaches a transcript is accompanied by an [`Origin`]
//! saying how it was produced, and the constructors that produce timings
//! require one. A caller cannot mint a number without stating where it came
//! from, because there is no constructor that omits the argument.
//!
//! # The distinction that matters most
//!
//! [`Origin::is_observation`] separates numbers that came from OUTSIDE this
//! program (an engine measuring audio, or a human writing a bullet) from
//! numbers this program DERIVED. Only the first kind may be treated as
//! evidence. Everything else is our own arithmetic, and scoring our arithmetic
//! against a reference measures the arithmetic, not the transcript.

use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::coordinates::{FileMs, Ms};

/// Which alignment engine produced a measurement.
///
/// A newtype rather than a bare string so an engine name cannot be swapped with
/// a language code, a model path, or any of the other short strings that travel
/// beside it through the worker protocol.
///
/// # Why `Cow<'static, str>`
///
/// Every engine name PRODUCED in this workspace is a compile-time literal:
/// `EngineBackend::wire_name` returns `&'static str` and every implementation
/// returns one. So construction borrows and never allocates, which matters
/// because an `Origin` is built per word (twice per word on the interval path,
/// once per token on the onset path) and a `String` here cost a heap
/// allocation at every one of them.
///
/// It is a `Cow` rather than a plain `&'static str` because provenance is
/// SERIALIZED: a timing read back from the FA cache carries an owned name that
/// no literal in this binary corresponds to. Cloning stays free on the live
/// path (a borrowed `Cow` clone copies a pointer), and only cache reads own
/// their string. A type that is expensive to carry is a type people stop
/// carrying, and provenance nobody carries is the defect this module exists to
/// prevent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EngineId(Cow<'static, str>);

impl EngineId {
    /// Name an engine, borrowing a literal.
    pub const fn new(name: &'static str) -> Self {
        Self(Cow::Borrowed(name))
    }
}

impl fmt::Display for EngineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a clamped value was cut down to fit.
///
/// The three bounds this pipeline clamps against are different claims about
/// the world, and collapsing them loses exactly the fact a reader needs. A
/// word cut to the RECORDING means the transcript described more speech than
/// the audio holds, which is a data problem. A word cut to its own UTTERANCE
/// means our word-level arithmetic disagreed with an utterance bullet we were
/// given, which is our problem. A word cut to the NEXT ONSET means an assumed
/// duration ran into the following word, which is neither: it is the assumption
/// working as intended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClampBound {
    /// The end of the audio file.
    RecordingEnd,
    /// The enclosing utterance's bullet.
    UtteranceBullet,
    /// The onset of the following word.
    NextOnset,
}

impl fmt::Display for ClampBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RecordingEnd => f.write_str("the end of the recording"),
            Self::UtteranceBullet => f.write_str("the utterance bullet"),
            Self::NextOnset => f.write_str("the next word's onset"),
        }
    }
}

/// How much of a word and of its labels a character alignment could NOT
/// reconcile.
///
/// # Why this lives with the origins and not with the aligner
///
/// It is part of an [`Origin`]: the counts are the EVIDENCE for an
/// attribution, and "attributed by character alignment" without them is a
/// claim a reader cannot weigh. A word whose characters all matched, one label
/// boundary over, and a numeral that shares not one character with the labels
/// it was handed both come out as `AttributedByCharAlignment`, and only these
/// two numbers tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharEdits {
    /// Characters of the transcript word that no label character matched.
    pub transcript_only: usize,
    /// Characters of this word's labels that no transcript character matched.
    pub label_only: usize,
}

impl CharEdits {
    /// Nothing went unreconciled.
    ///
    /// Deliberately a named constant rather than a `Default`: this is the TRUE
    /// count for a word the alignment matched exactly, never a stand-in for a
    /// count nobody took, so there is no `unwrap_or_default` shaped hole for a
    /// missing tally to fall into.
    pub const ZERO: Self = Self {
        transcript_only: 0,
        label_only: 0,
    };

    /// Whether the word and its labels reconciled character for character.
    ///
    /// This is the fact the in-order stitch proves, so it is also what
    /// separates a word the alignment merely FOLDED from one it GUESSED at.
    /// The alignment route is not that fact: a residue word whose characters
    /// all matched is exactly as well evidenced as a stitched one, and the
    /// count is what says so.
    pub const fn is_exact(self) -> bool {
        self.transcript_only == 0 && self.label_only == 0
    }
}

impl fmt::Display for CharEdits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} transcript and {} label characters unmatched",
            self.transcript_only, self.label_only
        )
    }
}

/// How a timing number was produced.
///
/// Ordered roughly from strongest evidence to weakest. Each derived variant
/// carries the inputs of its own computation, so a reader can reconstruct WHY
/// the number is what it is rather than only that it was derived.
///
/// # A retired variant, and why it must not come back
///
/// `EstimatedFromWordCount { gap, words_before, words_total }` sat here until
/// 2026-09-07 with NO production construction site: the declaration, the
/// classification arms, the `Display` arm, the serialized mirror in
/// `types::traces` and two tests were the whole of it. It described the
/// word-count distribution in `chat_ops::fa::grouping`, and that pass does not
/// produce a timing. It produces an audio WINDOW (a `Placement`), which is
/// then sent to an aligner; the per-word numbers that come back are measured
/// by the engine inside that window, so [`Origin::EngineMeasured`] is the
/// honest answer and nothing is being laundered.
///
/// A variant nothing produces is worse than absent. It reads as a bucket the
/// tally can fill, so `ProvenanceTally::assumed` and
/// `ProvenanceTally::needs_review` looked as though they covered a case they
/// could never see, and the doc comment on `assumed` said so in prose. Before
/// re-adding it, find the code that computes a per-word TIMING from a word
/// count; there is none today, and if one is written it should construct the
/// origin where the number is born.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Origin {
    /// An alignment engine measured this position against the audio.
    ///
    /// The strongest thing we have. Still not ground truth: an engine that
    /// fails to find its words reports positions that are arithmetic rather
    /// than measurement, which is why a measured timing can still be refused
    /// by the containment check.
    EngineMeasured {
        /// The engine that reported it.
        engine: EngineId,
    },

    /// Read from a bullet that was already in the transcript when we received
    /// it.
    ///
    /// An observation in the sense that it did not come from this program, but
    /// its own provenance is whatever produced that transcript, which may
    /// itself have been a machine. Treated as evidence because refusing to is
    /// the same as refusing to accept human transcription.
    TranscriptBullet,

    /// Cut down to fit a bound because it overshot.
    ///
    /// Wraps the origin of the value that was cut, so clamping a measurement
    /// and clamping an estimate stay distinguishable.
    ///
    /// This was `ClampedToRecording` until it was noticed that three call sites
    /// used it for three DIFFERENT bounds: the recording's end, the enclosing
    /// utterance's bullet, and the next word's onset. The variant named one of
    /// them, so two thirds of the provenance it recorded was wrong, and one of
    /// the offending call sites had a docstring saying in prose that its bound
    /// was "not the recording, and the two are different claims". Prose is
    /// gated by nothing; [`ClampBound`] puts the distinction in the value.
    ClampedTo {
        /// Which bound the value was cut down to.
        bound: ClampBound,
        /// Where the value came from before it was cut.
        was: Box<Origin>,
        /// The value before clamping.
        original: FileMs,
        /// How far past the bound it fell.
        overshoot: Ms,
    },

    /// The boundary between this word and its neighbour was moved so that a
    /// word which had collapsed to near-zero duration keeps a usable extent.
    ///
    /// Produced by the two rebalance passes in
    /// `chat_ops::fa::postprocess`, which take milliseconds from an
    /// over-long neighbour and give them to a lexical word the engine
    /// reported as almost instantaneous. BOTH words are re-timed, because one
    /// boundary moved and it belongs to both of them; whichever boundary
    /// actually changed is the one that wraps its previous origin.
    ///
    /// This was `RepairedForOrder` until 2026-09-07, named and documented as
    /// "moved so that timings run in order" and attributed by its own docs to
    /// a repair pass. `chat_ops::fa::repair` produces no origin at all: it
    /// reaches word clamping through `clamp_words_past_bound`, which records
    /// [`Origin::ClampedTo`] with [`ClampBound::UtteranceBullet`]. So the name
    /// pointed at a module that never built one, and described an ordering
    /// motive the only real producer does not have. Restoring order is a
    /// CONSEQUENCE here (the moved boundary stays between the two words), not
    /// the reason.
    RebalancedWithNeighbour {
        /// Where the value came from before the boundary moved.
        was: Box<Origin>,
        /// The value before the rebalance.
        original: FileMs,
    },

    /// Taken from a neighbouring boundary because this word had none of its own.
    ///
    /// The utterance bullet standing in for a word end the engine never
    /// reported. Better evidence than an invented constant, since a human or an
    /// earlier pass placed that bullet, and still not a measurement of THIS
    /// word.
    InheritedFromNeighbour {
        /// The boundary that was copied.
        from: FileMs,
    },

    /// One span covering several separately measured parts.
    ///
    /// A compound filler (`&-you_know`) is sent to the engine as N words and
    /// comes back as N timings, which are merged into the single span the CHAT
    /// word occupies. The same shape arises when an utterance's extent is taken
    /// as the min and max of its words' spans. The parts were measured; the
    /// envelope is our arithmetic over them, and it necessarily covers any
    /// silence between.
    MergedFromParts {
        /// How many measured spans the envelope covers.
        parts: usize,
    },

    /// The end of a word an ONSET-ONLY engine never reported, taken from the
    /// next token's onset.
    ///
    /// Whisper reports when a word starts and never when it stops, so every
    /// word end on that path is inferred from its successor. That is a
    /// reasonable inference and it is still not a measurement: the word may
    /// have ended long before the next one began.
    DerivedFromNextOnset,

    /// The end of the LAST word in a group, where no successor onset exists.
    ///
    /// A fixed duration stood in for a quantity nothing measured. Previously
    /// written as `unwrap_or(start + LAST_WORD_FALLBACK_MS)`, which made an
    /// invented 500 milliseconds indistinguishable from an observed one. A
    /// fallback that cannot be seen is a fabricated measurement.
    FallbackDuration {
        /// The duration that was assumed.
        assumed: Ms,
    },

    /// This word was given its label by CHARACTER ALIGNMENT rather than by
    /// matching it.
    ///
    /// The instant underneath is still the engine's own measurement of the
    /// label it came from, which is why this WRAPS rather than replaces. What
    /// changed is the ATTRIBUTION: an exact in-order stitch proves the label
    /// spells the word, while an edit-distance alignment says only that it
    /// fitted best, and for a numeral spelled out in words ("nineteen ninety
    /// five" for `1995`) it does not even say that, since the neighbours were
    /// placed and this word took what was left between them.
    ///
    /// Classified ASSUMED rather than Observed, because the number now answers
    /// a question the engine did not: WHICH WORD this onset belongs to. Before
    /// this variant existed, a remapped word and a stitched one reached a
    /// reviewer byte-identical, so the guess was indistinguishable from the
    /// proof. `edits` says how far the fit was from exact.
    AttributedByCharAlignment {
        /// How the underlying instant was obtained, before attribution.
        was: Box<Origin>,
        /// What the alignment could not reconcile.
        edits: CharEdits,
    },
}

/// What kind of number an [`Origin`] describes.
///
/// # Why this exists
///
/// `is_observation` and `ProvenanceTally::record` were two exhaustive matches
/// over the same variants, in the same file, kept consistent by nothing but
/// care: the tally's `observed` bucket had to be exactly the set
/// `is_observation` returns true for, and a third reading lived in
/// `WordSpan::is_fully_observed`. A test existed whose only job was to notice
/// the first two drifting apart, which is a standing confession that one of
/// them should not exist.
///
/// Now the classification has ONE owner and both callers ask it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginKind {
    /// An engine measured it, or the transcript already carried it.
    Observed,
    /// Inferred from a neighbouring measurement.
    Derived,
    /// Invented outright.
    Assumed,
    /// Adjusted after the fact.
    Adjusted,
}

impl Origin {
    /// What kind of number this is.
    ///
    /// Exhaustive on purpose: a new variant must be given a kind here, and both
    /// consumers then classify it correctly without being touched.
    pub fn kind(&self) -> OriginKind {
        match self {
            Self::EngineMeasured { .. } | Self::TranscriptBullet => OriginKind::Observed,
            Self::DerivedFromNextOnset
            | Self::InheritedFromNeighbour { .. }
            | Self::MergedFromParts { .. } => OriginKind::Derived,
            Self::FallbackDuration { .. } | Self::AttributedByCharAlignment { .. } => {
                OriginKind::Assumed
            }
            Self::ClampedTo { .. } | Self::RebalancedWithNeighbour { .. } => OriginKind::Adjusted,
        }
    }

    /// Whether this number came from outside this program.
    ///
    /// The one question every downstream consumer actually wants to ask, and
    /// the one that could not be asked at all before provenance was carried.
    /// Only observations may be scored as evidence; derived numbers are our own
    /// arithmetic and scoring them measures us.
    pub fn is_observation(&self) -> bool {
        matches!(self.kind(), OriginKind::Observed)
    }

    /// The origin this one was derived FROM, when it wraps another.
    ///
    /// The single owner of "which variants nest". Both questions below walk the
    /// same chain, and each used to carry its own exhaustive nine-arm match, so
    /// adding a wrapping variant meant remembering to update two places that
    /// nothing kept in step. Now a new variant has exactly one match to answer.
    fn was(&self) -> Option<&Origin> {
        match self {
            Self::ClampedTo { was, .. }
            | Self::RebalancedWithNeighbour { was, .. }
            | Self::AttributedByCharAlignment { was, .. } => Some(was),
            // Written out rather than left to a catch-all so a new variant that
            // wraps another cannot silently report that it wraps nothing.
            Self::EngineMeasured { .. }
            | Self::TranscriptBullet
            | Self::InheritedFromNeighbour { .. }
            | Self::MergedFromParts { .. }
            | Self::DerivedFromNextOnset
            | Self::FallbackDuration { .. } => None,
        }
    }

    /// This origin and every origin it was derived from, outermost first.
    fn chain(&self) -> impl Iterator<Item = &Origin> {
        std::iter::successors(Some(self), |origin| origin.was())
    }

    /// The original observation underneath any adjustments, when there was one.
    ///
    /// Lets a caller ask "was there ever a measurement here?" separately from
    /// "is this number still one?", which are different questions: the first
    /// decides whether re-running alignment could help, the second decides
    /// whether the number may be used as evidence.
    pub fn underlying(&self) -> &Origin {
        // `fold` rather than `last`, because the chain always yields at least
        // `self`, and this way that fact needs no unwrapping to express.
        self.chain().fold(self, |_, origin| origin)
    }

    /// Whether this number was cut down to the end of the recording, at any
    /// depth in its history.
    ///
    /// The one clamp that is a fact about the DATA rather than about our
    /// arithmetic: it means the transcript described speech continuing past the
    /// end of the audio, so either the bullet is wrong or the media is the wrong
    /// file. The other two bounds are routine. Before [`ClampBound`] existed all
    /// three were the same variant, so this question could not be asked and the
    /// case was pooled with the ordinary ones in every count.
    ///
    /// Asks the whole chain, because a value clamped to the recording and then
    /// rebalanced against a neighbour must still report it.
    pub fn overran_recording(&self) -> bool {
        self.chain().any(|origin| {
            matches!(
                origin,
                Self::ClampedTo {
                    bound: ClampBound::RecordingEnd,
                    ..
                }
            )
        })
    }

    /// Whether OUR OWN character alignment decided which word this instant
    /// belongs to, at any depth in its history.
    ///
    /// Different from asking whether [`Self::kind`] is `Assumed`, and that
    /// difference is the whole reason this exists: a clamp or a rebalance
    /// applied afterwards makes the OUTERMOST kind `Adjusted`, so a start the
    /// DP attributed and postprocessing then nudged reported as a routine
    /// adjustment and asked nobody to look at it.
    ///
    /// Walks the chain for the same reason [`Self::overran_recording`] does,
    /// and is deliberately narrower than "was assumed anywhere underneath": a
    /// [`Self::FallbackDuration`] capped at a real neighbouring onset has had
    /// its invented magnitude REPLACED by a measurement, so that case is not
    /// this one. An attribution is not undone by a later adjustment, because
    /// which label belongs to which word is a question no clamp answers.
    pub fn attributed_by_alignment(&self) -> bool {
        self.chain()
            .any(|origin| matches!(origin, Self::AttributedByCharAlignment { .. }))
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EngineMeasured { engine } => write!(f, "measured by {engine}"),
            Self::TranscriptBullet => f.write_str("from the source transcript"),
            Self::ClampedTo {
                bound,
                was,
                original,
                overshoot,
            } => write!(
                f,
                "clamped to {bound} from {original} ({overshoot} over), was {was}"
            ),
            Self::RebalancedWithNeighbour { was, original } => {
                write!(f, "rebalanced with a neighbour from {original}, was {was}")
            }
            Self::MergedFromParts { parts } => write!(f, "merged from {parts} measured parts"),
            Self::InheritedFromNeighbour { from } => write!(f, "inherited from {from}"),
            Self::DerivedFromNextOnset => f.write_str("derived from the next word's onset"),
            Self::FallbackDuration { assumed } => write!(f, "assumed duration of {assumed}"),
            Self::AttributedByCharAlignment { was, edits } => {
                write!(f, "attributed by character alignment ({edits}), was {was}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured() -> Origin {
        Origin::EngineMeasured {
            engine: EngineId::new("whisper-fa-large-v2"),
        }
    }

    #[test]
    fn engine_measurements_and_source_bullets_are_observations() {
        assert!(measured().is_observation());
        assert!(Origin::TranscriptBullet.is_observation());
    }

    #[test]
    fn our_own_arithmetic_is_never_an_observation() {
        // A fallback duration is the invented case that actually reaches a
        // transcript: a constant standing in for a quantity nothing measured.
        assert!(!Origin::FallbackDuration { assumed: Ms(500) }.is_observation());
        assert!(
            !Origin::InheritedFromNeighbour {
                from: FileMs::new(1_000)
            }
            .is_observation()
        );
    }

    #[test]
    fn adjusting_a_measurement_stops_it_being_one_but_remembers_it_was() {
        let clamped = Origin::ClampedTo {
            bound: ClampBound::RecordingEnd,
            was: Box::new(measured()),
            original: FileMs::new(1_288_185),
            overshoot: Ms(28_217),
        };
        assert!(!clamped.is_observation());
        assert_eq!(clamped.underlying(), &measured());
    }

    #[test]
    fn adjustments_compose_without_losing_the_bottom() {
        // Nesting is why `underlying` recurses: a value clamped twice must
        // still name the observation at the bottom, not the previous clamp.
        let stacked = Origin::ClampedTo {
            bound: ClampBound::RecordingEnd,
            was: Box::new(Origin::ClampedTo {
                bound: ClampBound::RecordingEnd,
                was: Box::new(Origin::TranscriptBullet),
                original: FileMs::new(500),
                overshoot: Ms(100),
            }),
            original: FileMs::new(480),
            overshoot: Ms(80),
        };
        assert_eq!(stacked.underlying(), &Origin::TranscriptBullet);
        assert!(!stacked.is_observation());
    }
}

#[cfg(test)]
mod laundering_tests {
    use super::*;

    /// An invented timing must not come back claiming to be an observation.
    ///
    /// This is the regression the 2026-08-15 review found: forced alignment
    /// wrote `WordTiming { span, origin }` into a `Bullet` (two integers), and
    /// post-processing read those bullets back and stamped every one
    /// `TranscriptBullet`, whose `is_observation()` is TRUE. So a 500 ms
    /// fallback this program invented came back labelled as observed, to
    /// exactly the consumer provenance exists for.
    ///
    /// The property is stated here rather than at the seam because it is about
    /// the MEANING of the variants, and it fails loudly if anyone ever makes an
    /// invented origin report as observed.
    #[test]
    fn no_invented_origin_reports_itself_as_an_observation() {
        let invented = [
            Origin::FallbackDuration { assumed: Ms(500) },
            Origin::DerivedFromNextOnset,
            Origin::MergedFromParts { parts: 2 },
            Origin::InheritedFromNeighbour {
                from: FileMs::new(1_000),
            },
            Origin::ClampedTo {
                bound: ClampBound::RecordingEnd,
                was: Box::new(Origin::EngineMeasured {
                    engine: EngineId::new("whisper-fa-large-v2"),
                }),
                original: FileMs::new(1_288_185),
                overshoot: Ms(28_217),
            },
            Origin::RebalancedWithNeighbour {
                was: Box::new(Origin::TranscriptBullet),
                original: FileMs::new(400),
            },
            // The onset underneath WAS measured; which word it belongs to was
            // our guess, so the pair must not read as an observation.
            Origin::AttributedByCharAlignment {
                was: Box::new(Origin::EngineMeasured {
                    engine: EngineId::new("whisper-fa-large-v2"),
                }),
                edits: CharEdits {
                    transcript_only: 4,
                    label_only: 18,
                },
            },
        ];
        for origin in invented {
            assert!(
                !origin.is_observation(),
                "{origin} reported itself as an observation"
            );
        }
    }
}

#[cfg(test)]
mod cache_boundary_tests {
    use super::*;
    use crate::chat_ops::fa::WordTiming;

    /// A stored zero-width timing must not deserialize back into existence.
    ///
    /// This guard was lost, not designed away: adding `origin` to `WordTiming`
    /// replaced `#[serde(try_from = "TimeSpan")]` with a plain derive, and the
    /// docstring next to it went on claiming the check still happened. It is
    /// what makes the FA cache self-cleaning, so losing it silently would have
    /// let a cached `T_T` bullet survive every future run.
    #[test]
    fn a_stored_zero_width_timing_fails_to_load() {
        let stored = r#"{"span":{"start_ms":5,"end_ms":5},"start_origin":"TranscriptBullet","end_origin":"TranscriptBullet"}"#;
        assert!(
            serde_json::from_str::<WordTiming>(stored).is_err(),
            "a zero-width span must not survive a cache round trip"
        );
    }

    /// An entry written before timings carried provenance is retired the same
    /// way, which is why the field needed no migration.
    #[test]
    fn a_pre_provenance_entry_fails_to_load() {
        let stored = r#"{"start_ms":100,"end_ms":600}"#;
        assert!(serde_json::from_str::<WordTiming>(stored).is_err());
    }

    /// A well-formed entry round-trips with its provenance intact, which is the
    /// whole point of storing it.
    #[test]
    fn a_real_timing_round_trips_with_its_origin() {
        // A measured start with an inferred end: the ordinary onset-engine
        // word, and the case a single origin could not express.
        let timing = WordTiming::new(
            100,
            600,
            Origin::EngineMeasured {
                engine: EngineId::new("whisper-fa-large-v2"),
            },
            Origin::DerivedFromNextOnset,
        )
        .expect("has extent");
        let json = serde_json::to_string(&timing).expect("serializes");
        let back: WordTiming = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, timing);
        assert_eq!(back.end_origin(), &Origin::DerivedFromNextOnset);
        assert!(back.start_origin().is_observation());
        assert!(
            !back.end_origin().is_observation(),
            "an inferred end is not an observation"
        );
    }
}

/// How a set of timings was produced, counted by kind.
///
/// # Why this exists
///
/// Provenance that reaches the cache but not the transcript answers nobody's
/// question. A `Bullet` is two integers and cannot carry an `Origin`, so the
/// per-word chain necessarily stops at the write; what CAN cross is a summary,
/// and a summary is what a reader of a corpus actually needs: "of this
/// utterance's sixteen word timings, twelve were measured, three inferred from
/// a neighbour, one invented outright."
///
/// Before this, a consumer scoring our timings against a reference had no way
/// to exclude the ones we made up, which is how a comparison came to report
/// 37.2% agreement by time against 76.4% by text.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ProvenanceTally {
    /// An engine measured it, or the transcript already carried it.
    pub observed: usize,
    /// Inferred from a neighbouring measurement: the next word's onset, the
    /// utterance's own bullet, or several parts merged into one span.
    pub derived: usize,
    /// Our own answer where the engine gave none: a duration standing in for
    /// one nothing measured ([`Origin::FallbackDuration`]), or an onset the
    /// engine measured but attached to a word by our character alignment
    /// rather than by matching ([`Origin::AttributedByCharAlignment`]). This
    /// said "or a word-count distribution" until 2026-09-07, naming a variant
    /// no code produced.
    pub assumed: usize,
    /// Adjusted after the fact: clamped to a bound, or rebalanced with a
    /// neighbouring word.
    pub adjusted: usize,
    /// Of those, how many were cut down to the END OF THE RECORDING.
    ///
    /// A subset of `adjusted`, not a fifth bucket, so the four still sum to
    /// `total`. Counted separately because it is the only adjustment that says
    /// something is wrong with the INPUT rather than with our arithmetic: the
    /// transcript claimed speech past the end of the audio.
    pub overran_recording: usize,
    /// How many timings our own character alignment ATTRIBUTED to their word,
    /// at any depth in their history.
    ///
    /// Cross-cutting rather than a fifth bucket, so it is NOT part of `total`:
    /// an attribution nothing touched afterwards is counted here and in
    /// `assumed`, while one a clamp or a rebalance later wrapped is counted
    /// here and in `adjusted`.
    ///
    /// It exists because the four buckets classify on the OUTERMOST origin,
    /// which is the right answer to "what is this number now" and the wrong
    /// answer to "did we choose which word it belongs to". Until 2026-09-07
    /// only the outermost was asked, so every DP-attributed start that
    /// postprocessing later clamped or rebalanced left `assumed` at zero and
    /// `needs_review` false, which is exactly the population a reviewer is
    /// there to see.
    pub attributed: usize,
}

impl ProvenanceTally {
    /// Count one timing, by the way it most recently came to be.
    ///
    /// The four BUCKETS classify on the OUTERMOST origin, because that is what
    /// the number now is: a measurement that was later clamped is no longer a
    /// measurement. Exhaustive on purpose, so a new `Origin` variant must be
    /// given a bucket rather than silently joining one.
    ///
    /// The two CROSS-CUTTING counts (`overran_recording`, `attributed`) are
    /// taken from the whole chain instead, because each records a fact about
    /// the number's history that a later wrapping does not undo.
    pub fn record(&mut self, origin: &Origin) {
        match origin.kind() {
            OriginKind::Observed => self.observed += 1,
            OriginKind::Derived => self.derived += 1,
            OriginKind::Assumed => self.assumed += 1,
            OriginKind::Adjusted => self.adjusted += 1,
        }
        // Asked of every origin, not only the adjusted ones: a recording clamp
        // can sit underneath a later repair, and the outermost kind would then
        // report `Adjusted` for a reason that is not this one.
        if origin.overran_recording() {
            self.overran_recording += 1;
        }
        // Same reason, and the case that was missing: an attribution the DP
        // made is still ours after a clamp or a rebalance wrapped it, and the
        // outermost kind can no longer say so.
        if origin.attributed_by_alignment() {
            self.attributed += 1;
        }
    }

    /// How many timings were counted.
    pub fn total(self) -> usize {
        self.observed + self.derived + self.assumed + self.adjusted
    }

    /// Whether anything here is not a straightforward observation.
    pub fn any_not_observed(self) -> bool {
        self.total() > self.observed
    }

    /// Whether a human should look at this utterance's timings.
    ///
    /// Three cases earn attention, and they are different complaints. An
    /// INVENTED timing is anchored to nothing, where a derived one is anchored
    /// to a real measurement next door. An ATTRIBUTED one is anchored to a real
    /// measurement that our own character alignment, not the engine, decided
    /// belongs to this word. A timing cut back to the recording's end says the
    /// transcript described speech the audio does not contain, which is a
    /// problem with the delivery rather than with the alignment.
    ///
    /// The last case used to be unaskable: every clamp was one variant, so a
    /// word cut to the end of the file counted the same as one capped at the
    /// next word's onset, and only the routine case was common enough to notice.
    ///
    /// `attributed` rather than `assumed` carries the middle case, because
    /// `assumed` is an outermost-origin bucket and postprocessing routinely
    /// wraps an attributed start in a clamp or a rebalance. Reading the bucket
    /// alone reported "nothing to see" for precisely the words the DP had
    /// guessed at.
    pub fn needs_review(self) -> bool {
        self.assumed > 0 || self.attributed > 0 || self.overran_recording > 0
    }
}

impl fmt::Display for ProvenanceTally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "measured={} derived={} assumed={} adjusted={}",
            self.observed, self.derived, self.assumed, self.adjusted
        )?;
        // Only when it happened, and stated separately from `adjusted` because
        // it is the one adjustment that says something is wrong with the INPUT.
        // Omitting it made an utterance flagged solely for a recording overrun
        // read byte-identically to a routine next-onset clamp that is NOT
        // flagged, so the reviewer was told to look and not told why.
        if self.overran_recording > 0 {
            write!(f, " past-recording-end={}", self.overran_recording)?;
        }
        // Also stated separately, and for the same reason: it is the count a
        // reviewer needs and the only one the four buckets cannot show, since a
        // clamp on top of an attribution moves it out of `assumed`.
        match self.attributed {
            0 => Ok(()),
            n => write!(f, " attributed={n}"),
        }
    }
}

#[cfg(test)]
mod tally_tests {
    use super::*;

    #[test]
    fn each_kind_lands_in_its_own_bucket() {
        let mut tally = ProvenanceTally::default();
        tally.record(&Origin::EngineMeasured {
            engine: EngineId::new("wav2vec_fa"),
        });
        tally.record(&Origin::TranscriptBullet);
        tally.record(&Origin::DerivedFromNextOnset);
        tally.record(&Origin::FallbackDuration { assumed: Ms(500) });
        tally.record(&Origin::ClampedTo {
            bound: ClampBound::RecordingEnd,
            was: Box::new(Origin::TranscriptBullet),
            original: FileMs::new(10),
            overshoot: Ms(5),
        });
        assert_eq!(tally.observed, 2);
        assert_eq!(tally.derived, 1);
        assert_eq!(tally.assumed, 1);
        assert_eq!(tally.adjusted, 1);
        assert_eq!(tally.total(), 5);
    }

    #[test]
    fn only_an_invented_timing_asks_for_review() {
        // A derived end is anchored to a real neighbouring measurement; an
        // assumed one is anchored to nothing, and that is the difference a
        // reviewer's time should be spent on.
        let mut derived = ProvenanceTally::default();
        derived.record(&Origin::DerivedFromNextOnset);
        assert!(derived.any_not_observed());
        assert!(!derived.needs_review());

        let mut assumed = ProvenanceTally::default();
        assumed.record(&Origin::FallbackDuration { assumed: Ms(500) });
        assert!(assumed.needs_review());
    }

    #[test]
    fn all_measured_needs_nothing() {
        let mut tally = ProvenanceTally::default();
        tally.record(&Origin::TranscriptBullet);
        assert!(!tally.any_not_observed());
        assert!(!tally.needs_review());
    }

    #[test]
    fn only_the_recording_clamp_calls_for_a_reviewer() {
        // The three bounds were one variant until 2026-08-15, so this
        // distinction could not be drawn at all. A word capped at the next
        // word's onset is routine; a word cut back to the end of the file means
        // the transcript described speech the audio does not contain.
        let routine = Origin::ClampedTo {
            bound: ClampBound::NextOnset,
            was: Box::new(Origin::FallbackDuration { assumed: Ms(500) }),
            original: FileMs::new(1_500),
            overshoot: Ms(200),
        };
        let overran = Origin::ClampedTo {
            bound: ClampBound::RecordingEnd,
            was: Box::new(Origin::TranscriptBullet),
            original: FileMs::new(1_288_185),
            overshoot: Ms(28_217),
        };

        assert!(!routine.overran_recording());
        assert!(overran.overran_recording());

        // Both are `Adjusted`, which is exactly why the kind alone cannot
        // answer this.
        assert_eq!(routine.kind(), OriginKind::Adjusted);
        assert_eq!(overran.kind(), OriginKind::Adjusted);

        let mut tally = ProvenanceTally::default();
        tally.record(&routine);
        assert_eq!(tally.adjusted, 1);
        assert_eq!(tally.overran_recording, 0);
        // `assumed` is 0 here: the fallback underneath was wrapped by the
        // clamp, so the outermost kind is what counts.
        assert!(!tally.needs_review());

        tally.record(&overran);
        assert_eq!(tally.adjusted, 2);
        assert_eq!(tally.overran_recording, 1);
        assert!(tally.needs_review());
        // The subset does not inflate the total.
        assert_eq!(tally.total(), 2);
    }

    /// RED FIRST (2026-09-07 review, item 2): an attribution a later
    /// adjustment wrapped is still an attribution.
    ///
    /// `record` classifies on the outermost origin, which is right for "what
    /// is this number now" and wrong for "did we choose the word it belongs
    /// to". Postprocessing clamps and rebalances routinely sit on top of a
    /// DP-attributed onset, and every such word left `assumed` at zero and
    /// `needs_review` false: the reviewer was told there was nothing to look
    /// at in exactly the population the DP had guessed at.
    #[test]
    fn an_attributed_onset_a_later_clamp_wrapped_still_asks_for_review() {
        let attributed = Origin::AttributedByCharAlignment {
            was: Box::new(Origin::EngineMeasured {
                engine: EngineId::new("wav2vec_fa"),
            }),
            edits: CharEdits {
                transcript_only: 1,
                label_only: 0,
            },
        };
        let clamped = Origin::ClampedTo {
            bound: ClampBound::NextOnset,
            was: Box::new(attributed.clone()),
            original: FileMs::new(1_500),
            overshoot: Ms(200),
        };
        let rebalanced = Origin::RebalancedWithNeighbour {
            was: Box::new(attributed.clone()),
            original: FileMs::new(1_400),
        };

        // The predicate itself: bare, clamped and rebalanced all report it.
        assert!(attributed.attributed_by_alignment());
        assert!(clamped.attributed_by_alignment());
        assert!(rebalanced.attributed_by_alignment());
        // And it is narrower than "assumed anywhere underneath": a fallback
        // DURATION capped at a real neighbouring onset had its invented
        // magnitude replaced by a measurement, which is not this case.
        assert!(
            !Origin::ClampedTo {
                bound: ClampBound::NextOnset,
                was: Box::new(Origin::FallbackDuration { assumed: Ms(500) }),
                original: FileMs::new(1_500),
                overshoot: Ms(200),
            }
            .attributed_by_alignment()
        );

        let mut tally = ProvenanceTally::default();
        tally.record(&clamped);
        tally.record(&rebalanced);
        // What the numbers ARE now is still an adjustment, so the four buckets
        // are unchanged and still sum to the total.
        assert_eq!(tally.adjusted, 2);
        assert_eq!(tally.assumed, 0);
        assert_eq!(tally.total(), 2);
        // What a reviewer needs is the cross-cutting count.
        assert_eq!(tally.attributed, 2);
        assert!(tally.needs_review());
        assert!(
            tally.to_string().contains("attributed=2"),
            "the summary must say why it is asking, got: {tally}"
        );
    }

    #[test]
    fn a_recording_clamp_survives_a_later_rebalance() {
        // Why `overran_recording` recurses: a rebalance on top of a recording
        // clamp leaves the outermost variant reporting the rebalance, and the
        // overrun would go uncounted if only the top were inspected.
        let repaired = Origin::RebalancedWithNeighbour {
            was: Box::new(Origin::ClampedTo {
                bound: ClampBound::RecordingEnd,
                was: Box::new(Origin::EngineMeasured {
                    engine: EngineId::new("whisper-fa-large-v2"),
                }),
                original: FileMs::new(1_288_185),
                overshoot: Ms(28_217),
            }),
            original: FileMs::new(1_259_968),
        };

        assert!(repaired.overran_recording());
        let mut tally = ProvenanceTally::default();
        tally.record(&repaired);
        assert_eq!(tally.overran_recording, 1);
        assert!(tally.needs_review());
    }
}
