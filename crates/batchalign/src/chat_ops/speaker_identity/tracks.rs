//! Tracks as voices: one vector per speaker code, and how sure the contrast is.
//!
//! # Why a track is scored as a track
//!
//! The mean of per-line cosines is not the cosine of the track's voice. A
//! track whose lines are short and noisy can average low against a reference
//! it plainly is, while a track of long clean lines from another speaker
//! averages higher than it should. Embedding the track's lines as ONE
//! direction (a centroid of unit vectors) and comparing THAT to each enrolled
//! voice is the standard speaker-verification shape, and it is robust to
//! short lines because every line contributes a direction, not a score.
//!
//! # Why the margin carries a p-value and not a floor
//!
//! A fixed margin ("the best track must beat the runner-up by 0.10") is a
//! number somebody chose once for one corpus. The question it stands in for
//! is whether the tracks are distinguishable voices at all: if line-to-track
//! membership were random, how often would a margin this large appear? That
//! is a permutation test over the lines the run actually embedded, it needs
//! no threshold, and its answer is a probability the consumer can rule on.
//!
//! # The type graph
//!
//! ```text
//!   (speaker code, embedded line) --TrackLines::embedded--> TrackLines
//!   (speaker code, refused line)  --TrackLines::refused---/
//!                                          |
//!                                  TrackLines::analyse(voices, plan)
//!                                          |
//!                 +------------------------+--------------------------+
//!                 v                                                   v
//!        Vec<TrackIdentity>                                  Vec<TrackContrast>
//!        (Voiced: centroid + scores | Unvoiced)              (one per enrolled voice)
//! ```
//!
//! `TrackCode` is the only node built from raw text, and only at the
//! transcript boundary; every other value is produced by `analyse`.

use std::collections::BTreeMap;
use std::fmt;
use std::num::{NonZeroU32, NonZeroUsize};

use rand::SeedableRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use super::embedding::{IncomparableEmbeddings, NoCentroid, SpeakerEmbedding};
use super::enrollment::EnrolledLabel;
use super::policy::LabelledScore;

/// The speaker code a track's lines carry in the transcript.
///
/// Built once, at the transcript boundary, from the code the main tier
/// carries. Not a voice: a diarization track is a claim that these lines
/// belong together, and the whole point of this module is to test it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrackCode(String);

/// A speaker code no main tier could carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a track code cannot be empty")]
pub struct EmptyTrackCode;

impl TrackCode {
    /// From the speaker code a scored utterance carries.
    pub fn from_speaker(code: &str) -> Result<Self, EmptyTrackCode> {
        if code.is_empty() {
            return Err(EmptyTrackCode);
        }
        Ok(Self(code.to_owned()))
    }

    /// The code, for display and for matching against transcript tiers.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TrackCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One enrolled label paired with the vector measured from its window.
#[derive(Debug, Clone, PartialEq)]
pub struct EnrolledVoice {
    /// The label the caller gave this voice.
    pub label: EnrolledLabel,
    /// Its embedding, from the same decode as every scored line.
    pub vector: SpeakerEmbedding,
}

/// How many label permutations the contrast draws, and from what seed.
///
/// Both are written into the evidence, so a reader can reproduce every
/// p-value in the file byte for byte. There is no `Default`: the CLI states
/// them, with a documented value the flag's help text names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermutationPlan {
    /// Seed of the shuffle.
    pub seed: u64,
    /// How many shuffles are drawn.
    pub count: PermutationCount,
}

/// The plan the CLI flags name in their help text.
///
/// The ONE owner of the two numbers `--permutations` and `--permutation-seed`
/// default to, so the flag help, the persisted-job fallback and any test
/// cannot each carry their own copy. A thousand draws resolve p to 0.001,
/// which is the finest point the witness grid rules on.
#[must_use]
pub const fn documented_permutation_plan() -> PermutationPlan {
    PermutationPlan {
        seed: 0,
        count: PermutationCount(DOCUMENTED_PERMUTATIONS),
    }
}

/// See [`documented_permutation_plan`]. `NonZeroU32::MIN` is 1, and a
/// checked const addition keeps the value provably non-zero with no panic
/// path; a literal `1000` would need one.
const DOCUMENTED_PERMUTATIONS: NonZeroU32 = NonZeroU32::MIN.saturating_add(999);

/// A positive number of permutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PermutationCount(NonZeroU32);

/// A permutation count of zero, which would make every p-value 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a permutation count must be greater than zero")]
pub struct ZeroPermutations;

impl PermutationCount {
    /// The count as a number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl fmt::Display for PermutationCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<u32> for PermutationCount {
    type Error = ZeroPermutations;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        NonZeroU32::new(value).map(Self).ok_or(ZeroPermutations)
    }
}

/// The gap between the best track's centroid score and the runner-up's.
///
/// # Every route to a value of this type
///
/// [`TrackLines::analyse`], from two centroid scores. A difference of two
/// cosines lies in `[0, 2]` by construction, so nothing validates it.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CentroidMargin(f64);

impl CentroidMargin {
    /// The margin as a number.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// The share of permutations whose margin reached the observed one.
///
/// `(at_or_above + 1) / (permutations + 1)`: the observed labelling counts as
/// one permutation, so the value lies in `(0, 1]` and can never be zero.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PValue(f64);

impl PValue {
    /// The probability as a number.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// One track, as a voice or as a track that never measured one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrackIdentity {
    /// At least one line embedded, so the track has a direction.
    Voiced {
        /// The speaker code.
        track: TrackCode,
        /// Lines whose vectors went into the centroid.
        lines_embedded: NonZeroUsize,
        /// Lines the run could not embed (no bullet, too short, overlapping
        /// an enrollment). Counted, never imputed.
        lines_refused: usize,
        /// The track's voice, so a later cross-session question needs no
        /// re-run.
        centroid: SpeakerEmbedding,
        /// The centroid's similarity to every enrolled voice, for the record.
        /// A contrast's margin is computed by its own single arithmetic route
        /// (see `block_cosines`) and agrees with these to an ulp.
        scores: Vec<LabelledScore>,
    },
    /// Every line of this track was refused; there is no voice to score.
    Unvoiced {
        /// The speaker code.
        track: TrackCode,
        /// Lines the run could not embed.
        lines_refused: usize,
    },
}

/// For one enrolled voice, how far the best track stands out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrackContrast {
    /// Two or more voiced tracks, so a margin exists and was tested.
    Tested {
        /// The enrolled voice.
        label: EnrolledLabel,
        /// The track whose centroid scores highest against it.
        best: TrackCode,
        /// The next best.
        runner_up: TrackCode,
        /// Best minus runner-up, on the labelling the transcript carries.
        observed_margin: CentroidMargin,
        /// The plan that produced the count below.
        permutations: PermutationPlan,
        /// Shuffled labellings whose margin reached the observed one.
        at_or_above: u32,
        /// The share, with the observed labelling counted once.
        p_value: PValue,
    },
    /// One voiced track: nothing to contrast it against.
    OneTrack {
        /// The enrolled voice.
        label: EnrolledLabel,
        /// The only track with a voice.
        track: TrackCode,
    },
    /// No line of any track was embedded.
    NoVoicedTrack {
        /// The enrolled voice.
        label: EnrolledLabel,
    },
}

/// Why the tracks could not be analysed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TrackAnalysisFailure {
    /// A track's lines have no centroid.
    #[error("track {track} has no centroid: {source}")]
    NoCentroid {
        /// Which track.
        track: TrackCode,
        /// Why.
        #[source]
        source: NoCentroid,
    },
    /// A centroid and an enrolled voice cannot be compared.
    #[error("track {track} cannot be compared to enrolled voice {label}: {source}")]
    Incomparable {
        /// Which track.
        track: TrackCode,
        /// Which voice.
        label: EnrolledLabel,
        /// Why.
        #[source]
        source: IncomparableEmbeddings,
    },
    /// The enrolled voice has zero magnitude, so no track can be compared
    /// to it. Named for the voice, because that is where the defect is.
    #[error("enrolled voice {label} has zero magnitude and no direction to compare")]
    EnrolledVoiceWithoutDirection {
        /// Which voice.
        label: EnrolledLabel,
    },
    /// A block of lines summed to nothing, so its cosine has no value and
    /// the count would be a fabrication.
    #[error("a track's lines summed to zero magnitude; the contrast cannot be counted")]
    TrackWithoutDirection,
}

/// What one track accumulated before analysis.
#[derive(Debug, Default)]
struct TrackAccumulator {
    embedded: Vec<SpeakerEmbedding>,
    refused: usize,
}

/// Every scored line, sorted onto its track.
///
/// Filled by the run as it walks the utterances, then consumed by
/// [`TrackLines::analyse`]; there is no other way to obtain the analysis.
#[derive(Debug, Default)]
pub struct TrackLines {
    by_track: BTreeMap<TrackCode, TrackAccumulator>,
}

impl TrackLines {
    /// A line the run embedded.
    pub fn embedded(&mut self, track: TrackCode, vector: SpeakerEmbedding) {
        self.by_track
            .entry(track)
            .or_default()
            .embedded
            .push(vector);
    }

    /// A line the run refused to embed, for whatever reason.
    pub fn refused(&mut self, track: TrackCode) {
        self.by_track.entry(track).or_default().refused += 1;
    }

    /// Score every track as a voice and contrast the tracks per enrolled
    /// voice.
    pub fn analyse(
        self,
        voices: &[EnrolledVoice],
        plan: PermutationPlan,
    ) -> Result<TrackAnalysis, TrackAnalysisFailure> {
        let mut tracks = Vec::with_capacity(self.by_track.len());
        let mut voiced: Vec<VoicedTrack> = Vec::new();
        for (track, lines) in self.by_track {
            let Some(lines_embedded) = NonZeroUsize::new(lines.embedded.len()) else {
                tracks.push(TrackIdentity::Unvoiced {
                    track,
                    lines_refused: lines.refused,
                });
                continue;
            };
            let centroid = SpeakerEmbedding::centroid(&lines.embedded).map_err(|source| {
                TrackAnalysisFailure::NoCentroid {
                    track: track.clone(),
                    source,
                }
            })?;
            let mut scores = Vec::with_capacity(voices.len());
            for voice in voices {
                let score = centroid.similarity_to(&voice.vector).map_err(|source| {
                    TrackAnalysisFailure::Incomparable {
                        track: track.clone(),
                        label: voice.label.clone(),
                        source,
                    }
                })?;
                scores.push(LabelledScore {
                    label: voice.label.clone(),
                    score,
                });
            }
            let mut units = Vec::with_capacity(lines.embedded.len());
            for line in &lines.embedded {
                // The centroid above already refused a directionless line,
                // so this cannot fail; mapping it keeps the proof in the
                // types rather than in an `expect`.
                let unit =
                    line.unit_components()
                        .map_err(|_| TrackAnalysisFailure::NoCentroid {
                            track: track.clone(),
                            source: NoCentroid::SpanWithoutDirection,
                        })?;
                units.push(unit);
            }
            voiced.push(VoicedTrack {
                track: track.clone(),
                units,
            });
            tracks.push(TrackIdentity::Voiced {
                track,
                lines_embedded,
                lines_refused: lines.refused,
                centroid,
                scores,
            });
        }

        let mut contrasts = Vec::with_capacity(voices.len());
        for voice in voices {
            contrasts.push(contrast(voice, &voiced, plan)?);
        }
        Ok(TrackAnalysis { tracks, contrasts })
    }
}

/// The two halves [`TrackLines::analyse`] produces together, so neither can
/// be written without the other.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackAnalysis {
    /// Every track, voiced or not, in speaker-code order.
    pub tracks: Vec<TrackIdentity>,
    /// One contrast per enrolled voice, in enrollment order.
    pub contrasts: Vec<TrackContrast>,
}

/// A voiced track's material for the permutation test.
struct VoicedTrack {
    track: TrackCode,
    units: Vec<Vec<f64>>,
}

/// The cosine of every block's summed unit vectors to the enrolled direction,
/// for one labelling of the lines.
///
/// The ONE arithmetic route for both the observed margin and every permuted
/// one: `order` is the identity for the observed labelling and a shuffle for
/// the null. Two routes to the same quantity would agree only up to an ulp,
/// and the count compares them with an exact `>=`, so a shuffle reproducing
/// the observed labelling could fall on either side of a tie.
fn block_cosines(
    order: &[usize],
    sizes: &[usize],
    lines: &[&[f64]],
    enrolled_unit: &[f64],
    sum: &mut [f64],
    cosines: &mut Vec<f64>,
) -> Result<(), TrackAnalysisFailure> {
    cosines.clear();
    let mut cursor = 0;
    for &size in sizes {
        sum.iter_mut().for_each(|slot| *slot = 0.0);
        for &line_index in &order[cursor..cursor + size] {
            for (slot, value) in sum.iter_mut().zip(lines[line_index]) {
                *slot += value;
            }
        }
        cursor += size;
        let norm = sum.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(TrackAnalysisFailure::TrackWithoutDirection);
        }
        let dot: f64 = sum.iter().zip(enrolled_unit).map(|(a, b)| a * b).sum();
        cosines.push(dot / norm);
    }
    Ok(())
}

/// Best minus second best over a set of cosines. `None` below two values.
fn margin_of(cosines: &[f64]) -> Option<f64> {
    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for &value in cosines {
        if value > best {
            second = best;
            best = value;
        } else if value > second {
            second = value;
        }
    }
    (cosines.len() >= 2).then_some(best - second)
}

/// The contrast for one enrolled voice.
fn contrast(
    voice: &EnrolledVoice,
    voiced: &[VoicedTrack],
    plan: PermutationPlan,
) -> Result<TrackContrast, TrackAnalysisFailure> {
    let label = voice.label.clone();
    match voiced {
        [] => return Ok(TrackContrast::NoVoicedTrack { label }),
        [only] => {
            return Ok(TrackContrast::OneTrack {
                label,
                track: only.track.clone(),
            });
        }
        _ => {}
    }

    let enrolled_unit = voice.vector.unit_components().map_err(|_| {
        TrackAnalysisFailure::EnrolledVoiceWithoutDirection {
            label: label.clone(),
        }
    })?;
    let sizes: Vec<usize> = voiced.iter().map(|track| track.units.len()).collect();
    let lines: Vec<&[f64]> = voiced
        .iter()
        .flat_map(|track| track.units.iter().map(Vec::as_slice))
        .collect();
    let mut sum = vec![0.0_f64; enrolled_unit.len()];
    let mut cosines = Vec::with_capacity(sizes.len());

    // The observed labelling, through the same arithmetic the null uses.
    let mut order: Vec<usize> = (0..lines.len()).collect();
    block_cosines(
        &order,
        &sizes,
        &lines,
        &enrolled_unit,
        &mut sum,
        &mut cosines,
    )?;
    let mut ranked: Vec<(usize, f64)> = cosines.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let (best_index, best_score) = ranked[0];
    let (runner_index, runner_score) = ranked[1];
    let observed_margin = best_score - runner_score;

    // The null: lines keep their vectors, tracks keep their sizes, and
    // membership is shuffled.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(plan.seed);
    let mut at_or_above: u32 = 0;
    for _ in 0..plan.count.get() {
        order.shuffle(&mut rng);
        block_cosines(
            &order,
            &sizes,
            &lines,
            &enrolled_unit,
            &mut sum,
            &mut cosines,
        )?;
        // Two or more voiced tracks were matched above, so a margin exists.
        if let Some(margin) = margin_of(&cosines)
            && margin >= observed_margin
        {
            at_or_above += 1;
        }
    }
    // In f64 from the start: `count` may be `u32::MAX`, and `+ 1` in `u32`
    // would wrap.
    let p_value = (f64::from(at_or_above) + 1.0) / (f64::from(plan.count.get()) + 1.0);
    Ok(TrackContrast::Tested {
        label,
        best: voiced[best_index].track.clone(),
        runner_up: voiced[runner_index].track.clone(),
        observed_margin: CentroidMargin(observed_margin),
        permutations: plan,
        at_or_above,
        p_value: PValue(p_value),
    })
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    fn embedding(components: Vec<f64>) -> SpeakerEmbedding {
        let width = components.len();
        SpeakerEmbedding::from_worker(components, width).expect("test: a legal embedding")
    }

    fn code(text: &str) -> TrackCode {
        TrackCode::from_speaker(text).expect("test: a non-empty code")
    }

    fn voice(label: &str, components: Vec<f64>) -> EnrolledVoice {
        EnrolledVoice {
            label: EnrolledLabel::parse(label).expect("test: a legal label"),
            vector: embedding(components),
        }
    }

    fn plan(seed: u64, count: u32) -> PermutationPlan {
        PermutationPlan {
            seed,
            count: PermutationCount::try_from(count).expect("test: a positive count"),
        }
    }

    /// Two clean tracks, one along the enrolled voice and one orthogonal:
    /// the voiced track wins with the full margin, and a shuffle reaches it
    /// only by reproducing the labelling exactly (or its mirror), which is
    /// 2 in C(20, 10) draws. A tie COUNTS, because the statistic is "at or
    /// above", so the assertion bounds the count rather than pinning zero.
    #[test]
    fn separable_tracks_give_a_p_value_near_the_floor() {
        let mut lines = TrackLines::default();
        for _ in 0..10 {
            lines.embedded(code("PAR0"), embedding(vec![1.0, 0.0]));
            lines.embedded(code("PAR1"), embedding(vec![0.0, 1.0]));
        }
        lines.refused(code("PAR1"));
        let analysis = lines
            .analyse(&[voice("INV", vec![1.0, 0.0])], plan(7, 199))
            .expect("test: analysable tracks");
        match &analysis.contrasts[..] {
            [
                TrackContrast::Tested {
                    best,
                    runner_up,
                    observed_margin,
                    at_or_above,
                    p_value,
                    ..
                },
            ] => {
                assert_eq!(best, &code("PAR0"));
                assert_eq!(runner_up, &code("PAR1"));
                assert!((observed_margin.get() - 1.0).abs() < 1e-12);
                assert!(*at_or_above <= 1, "at_or_above was {at_or_above}");
                assert!(p_value.get() <= 2.0 / 200.0, "p was {}", p_value.get());
            }
            other => panic!("expected one tested contrast, got {other:?}"),
        }
        match &analysis.tracks[1] {
            TrackIdentity::Voiced {
                lines_embedded,
                lines_refused,
                ..
            } => {
                assert_eq!(lines_embedded.get(), 10);
                assert_eq!(*lines_refused, 1);
            }
            other => panic!("expected PAR1 voiced, got {other:?}"),
        }
    }

    /// Two tracks whose lines are the same mixture: the observed margin is
    /// no better than a shuffle, so the p-value is high. This is the case
    /// the fixed 0.10 margin could not distinguish from a real contrast.
    #[test]
    fn indistinguishable_tracks_give_a_high_p_value() {
        let mut lines = TrackLines::default();
        for step in 0..8 {
            let along = if step % 2 == 0 { 1.0 } else { 0.0 };
            let across = 1.0 - along;
            lines.embedded(code("PAR0"), embedding(vec![along, across]));
            lines.embedded(code("PAR1"), embedding(vec![along, across]));
        }
        let analysis = lines
            .analyse(&[voice("INV", vec![1.0, 0.0])], plan(3, 99))
            .expect("test: analysable tracks");
        match &analysis.contrasts[..] {
            [
                TrackContrast::Tested {
                    observed_margin,
                    p_value,
                    ..
                },
            ] => {
                assert!(observed_margin.get().abs() < 1e-12);
                assert!(p_value.get() > 0.5, "p was {}", p_value.get());
            }
            other => panic!("expected one tested contrast, got {other:?}"),
        }
    }

    /// The same plan reproduces the same count; a different seed is a
    /// different draw. This is the reproducibility the evidence promises.
    #[test]
    fn the_plan_reproduces_the_count_byte_for_byte() {
        fn run(seed: u64) -> u32 {
            let mut lines = TrackLines::default();
            for step in 0..5 {
                let wobble = f64::from(step) * 0.1;
                lines.embedded(code("PAR0"), embedding(vec![1.0, wobble]));
                lines.embedded(code("PAR1"), embedding(vec![wobble, 1.0]));
                lines.embedded(code("PAR2"), embedding(vec![0.7, 0.7 + wobble]));
            }
            let analysis = lines
                .analyse(&[voice("INV", vec![0.9, 0.4])], plan(seed, 300))
                .expect("test: analysable tracks");
            match &analysis.contrasts[..] {
                [TrackContrast::Tested { at_or_above, .. }] => *at_or_above,
                other => panic!("expected one tested contrast, got {other:?}"),
            }
        }
        assert_eq!(run(11), run(11));
        assert_ne!(run(11), run(12));
    }

    /// One voiced track and one unvoiced track: no margin exists and the
    /// contrast says so rather than inventing a runner-up.
    #[test]
    fn a_single_voiced_track_has_no_contrast() {
        let mut lines = TrackLines::default();
        lines.embedded(code("PAR0"), embedding(vec![1.0, 0.0]));
        lines.refused(code("CHI"));
        let analysis = lines
            .analyse(&[voice("INV", vec![1.0, 0.0])], plan(1, 10))
            .expect("test: analysable tracks");
        assert_eq!(
            analysis.contrasts,
            vec![TrackContrast::OneTrack {
                label: EnrolledLabel::parse("INV").expect("test: a legal label"),
                track: code("PAR0"),
            }]
        );
        assert!(matches!(
            analysis.tracks[0],
            TrackIdentity::Unvoiced {
                lines_refused: 1,
                ..
            }
        ));
    }

    /// Nothing embedded anywhere: the contrast names that, and there is no
    /// track to call.
    #[test]
    fn no_voiced_track_is_named_as_such() {
        let mut lines = TrackLines::default();
        lines.refused(code("PAR0"));
        let analysis = lines
            .analyse(&[voice("INV", vec![1.0, 0.0])], plan(1, 10))
            .expect("test: analysable tracks");
        assert!(matches!(
            analysis.contrasts[0],
            TrackContrast::NoVoicedTrack { .. }
        ));
    }

    /// A speaker code no main tier could carry is refused at the boundary.
    #[test]
    fn an_empty_track_code_is_refused() {
        assert_eq!(TrackCode::from_speaker(""), Err(EmptyTrackCode));
        assert_eq!(PermutationCount::try_from(0), Err(ZeroPermutations));
    }

    /// The margin helper: below two values there is no margin.
    #[test]
    fn a_margin_needs_two_values() {
        assert_eq!(margin_of(&[0.5]), None);
        assert_eq!(margin_of(&[0.2, 0.9, 0.4]), Some(0.9 - 0.4));
    }
}
