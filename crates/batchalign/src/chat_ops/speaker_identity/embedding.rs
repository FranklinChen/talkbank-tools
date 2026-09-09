//! Speaker embeddings, and the one operation that compares two of them.

use std::num::{NonZeroU32, NonZeroU64};

use serde::{Deserialize, Serialize};

use super::policy::SimilarityScore;

/// Width shared by every embedding in one worker response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingDimension(NonZeroU32);

/// A worker response that claimed zero-width embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a speaker embedding dimension must be greater than zero")]
pub struct ZeroEmbeddingDimension;

impl EmbeddingDimension {
    /// The dimension for wire evidence and vector-width checks.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl TryFrom<u32> for EmbeddingDimension {
    type Error = ZeroEmbeddingDimension;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        NonZeroU32::new(value)
            .map(Self)
            .ok_or(ZeroEmbeddingDimension)
    }
}

/// Smallest span the loaded embedding model can measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MinimumEmbeddingFrames(NonZeroU64);

/// A worker response that claimed a zero-frame model minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a speaker embedding minimum frame count must be greater than zero")]
pub struct ZeroMinimumEmbeddingFrames;

impl MinimumEmbeddingFrames {
    /// The minimum in prepared-audio frames.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl TryFrom<u64> for MinimumEmbeddingFrames {
    type Error = ZeroMinimumEmbeddingFrames;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(ZeroMinimumEmbeddingFrames)
    }
}

/// One fixed-width acoustic vector for one span of audio.
///
/// # Why it is a type and not a `Vec<f64>`
///
/// The invariant it carries is that every component is finite. That is not
/// decoration: the pinned embedding model returns a correctly shaped vector of
/// NaNs for input below its minimum length, and a cosine over NaNs is NaN,
/// which compares false against every threshold and therefore reads as a
/// considered "this is not that speaker". The worker refuses such a span
/// before it becomes a vector at all; this type is the second gate, at the
/// boundary where the wire's `Vec<f64>` becomes a value this crate reasons
/// with. Two independent refusals for one silent failure mode is deliberate:
/// the first is in a different language and a different process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpeakerEmbedding {
    components: Vec<f64>,
}

/// Why a wire vector is not an embedding.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum NotAnEmbedding {
    /// The vector had no components.
    #[error("a speaker embedding cannot be empty")]
    Empty,
    /// Some component was NaN or infinite.
    #[error(
        "a speaker embedding must be finite in every component, and component {index} is {value}"
    )]
    NotFinite {
        /// Which component.
        index: usize,
        /// Its value.
        value: f64,
    },
    /// The vector's width disagreed with the width the worker declared.
    #[error("the worker declared {declared}-component embeddings and sent one with {actual}")]
    WrongWidth {
        /// Width the response envelope declared.
        declared: usize,
        /// Width this vector actually had.
        actual: usize,
    },
}

impl SpeakerEmbedding {
    /// The single route from a wire vector to an embedding.
    ///
    /// `declared_width` comes from the response envelope, which reports the
    /// loaded model's own dimension rather than a constant on this side. A
    /// width mismatch means two different models produced vectors in one
    /// response, which no comparison between them would be meaningful across.
    pub fn from_worker(
        components: Vec<f64>,
        declared_width: usize,
    ) -> Result<Self, NotAnEmbedding> {
        if components.is_empty() {
            return Err(NotAnEmbedding::Empty);
        }
        if components.len() != declared_width {
            return Err(NotAnEmbedding::WrongWidth {
                declared: declared_width,
                actual: components.len(),
            });
        }
        for (index, value) in components.iter().enumerate() {
            if !value.is_finite() {
                return Err(NotAnEmbedding::NotFinite {
                    index,
                    value: *value,
                });
            }
        }
        Ok(Self { components })
    }

    /// How many components this embedding has.
    #[must_use]
    pub fn width(&self) -> usize {
        self.components.len()
    }

    /// The components scaled to unit length, so that a sum of several of
    /// them weights every span equally whatever its magnitude.
    ///
    /// Refused for a vector with no direction, which is the one input a
    /// centroid cannot honestly include.
    pub fn unit_components(&self) -> Result<Vec<f64>, IncomparableEmbeddings> {
        let norm = self.magnitude();
        if norm == 0.0 {
            return Err(IncomparableEmbeddings::NoDirection);
        }
        Ok(self.components.iter().map(|value| value / norm).collect())
    }

    /// The voice of a set of spans as ONE vector: the unit-normalized mean of
    /// their unit vectors, renormalized.
    ///
    /// # Why unit vectors first
    ///
    /// A plain mean weights each span by its magnitude, which the embedding
    /// model does not promise means anything. Normalizing first makes the
    /// centroid a direction agreed on by every span equally, which is the
    /// quantity a cosine against it then measures.
    ///
    /// # Every route to a refusal
    ///
    /// No spans at all; two spans of different widths (two models answered);
    /// a span with no direction; or unit vectors that cancel to nothing,
    /// which is the one case where "the voice of these spans" has no answer.
    pub fn centroid<'a>(
        spans: impl IntoIterator<Item = &'a SpeakerEmbedding>,
    ) -> Result<Self, NoCentroid> {
        let mut sum: Option<Vec<f64>> = None;
        for span in spans {
            let unit = span
                .unit_components()
                .map_err(|_| NoCentroid::SpanWithoutDirection)?;
            match &mut sum {
                None => sum = Some(unit),
                Some(total) => {
                    if total.len() != unit.len() {
                        return Err(NoCentroid::DifferentWidths {
                            first: total.len(),
                            offending: unit.len(),
                        });
                    }
                    for (slot, value) in total.iter_mut().zip(unit) {
                        *slot += value;
                    }
                }
            }
        }
        let total = sum.ok_or(NoCentroid::NoSpans)?;
        let norm = total.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(NoCentroid::SpansCancel);
        }
        Ok(Self {
            components: total.into_iter().map(|value| value / norm).collect(),
        })
    }

    /// The cosine similarity between this embedding and `other`.
    ///
    /// The ONLY producer of a [`SimilarityScore`] in production, which is what
    /// makes "a number that came from somewhere else" unable to reach the
    /// threshold comparison.
    ///
    /// Returns [`IncomparableEmbeddings`] rather than a number when the two
    /// have different widths or either has zero magnitude. A zero-magnitude
    /// vector has no direction, so it has no angle to any other vector; the
    /// arithmetic would divide by zero and produce a NaN that reads as a
    /// verdict.
    pub fn similarity_to(&self, other: &Self) -> Result<SimilarityScore, IncomparableEmbeddings> {
        if self.components.len() != other.components.len() {
            return Err(IncomparableEmbeddings::DifferentWidths {
                left: self.components.len(),
                right: other.components.len(),
            });
        }
        let dot: f64 = self
            .components
            .iter()
            .zip(&other.components)
            .map(|(left, right)| left * right)
            .sum();
        let left_norm = self.magnitude();
        let right_norm = other.magnitude();
        if left_norm == 0.0 || right_norm == 0.0 {
            return Err(IncomparableEmbeddings::NoDirection);
        }

        // Clamped, and the clamp is invisible ON PURPOSE here, unlike
        // `chat_ops::fa::coordinates::Clamped`. There the clamp signalled that
        // a transcript claimed more speech than the recording held, which is a
        // fact about the data. Here the only thing it can signal is that
        // floating-point summation carried a true 1.0 to 1.0000000000000002,
        // which is a fact about IEEE 754 and about nothing else. A variant
        // reporting it would fire on identical audio and mean nothing.
        let cosine = (dot / (left_norm * right_norm)).clamp(-1.0, 1.0);
        SimilarityScore::try_from(cosine).map_err(|_| IncomparableEmbeddings::NoDirection)
    }

    fn magnitude(&self) -> f64 {
        self.components
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }
}

/// Why a set of spans has no centroid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NoCentroid {
    /// Nothing to average.
    #[error("a centroid needs at least one span")]
    NoSpans,
    /// Two spans came from different models.
    #[error("a centroid cannot mix a {first}-component span with a {offending}-component one")]
    DifferentWidths {
        /// Width of the first span seen.
        first: usize,
        /// Width of the span that disagreed.
        offending: usize,
    },
    /// A span has zero magnitude and so no direction to contribute.
    #[error("a span with zero magnitude has no direction to contribute to a centroid")]
    SpanWithoutDirection,
    /// The unit vectors summed to zero.
    #[error("the spans' directions cancel exactly, so they share no voice")]
    SpansCancel,
}

/// Why two embeddings have no similarity.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum IncomparableEmbeddings {
    /// The two came from different models.
    #[error("a {left}-component embedding has no similarity to a {right}-component one")]
    DifferentWidths {
        /// Width of the left operand.
        left: usize,
        /// Width of the right operand.
        right: usize,
    },
    /// One of them has zero magnitude, so it points nowhere.
    #[error("an embedding with zero magnitude has no direction to compare")]
    NoDirection,
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    fn embedding(components: Vec<f64>) -> SpeakerEmbedding {
        let width = components.len();
        match SpeakerEmbedding::from_worker(components, width) {
            Ok(value) => value,
            Err(error) => panic!("a legal embedding: {error}"),
        }
    }

    /// The NaN vector the pinned model returns for too-short input cannot
    /// become an embedding at all, so it can never reach a comparison.
    #[test]
    fn a_non_finite_component_is_refused_with_its_index() {
        match SpeakerEmbedding::from_worker(vec![0.5, f64::NAN, 0.25], 3) {
            Err(NotAnEmbedding::NotFinite { index, .. }) => assert_eq!(index, 1),
            other => panic!("expected a refusal naming the component, got {other:?}"),
        }
        assert!(SpeakerEmbedding::from_worker(vec![f64::INFINITY], 1).is_err());
        assert!(SpeakerEmbedding::from_worker(Vec::new(), 0).is_err());
    }

    /// A vector whose width disagrees with the width the worker declared means
    /// two models answered one request, and is refused rather than compared.
    #[test]
    fn a_width_disagreeing_with_the_envelope_is_refused() {
        assert!(matches!(
            SpeakerEmbedding::from_worker(vec![0.1, 0.2], 3),
            Err(NotAnEmbedding::WrongWidth {
                declared: 3,
                actual: 2
            })
        ));
    }

    /// MEASUREMENT: cosine similarity, at its three fixed points.
    #[test]
    fn cosine_similarity_is_the_ordinary_one() {
        let a = embedding(vec![1.0, 0.0]);
        let b = embedding(vec![0.0, 1.0]);
        let minus_a = embedding(vec![-1.0, 0.0]);

        let same = match a.similarity_to(&a) {
            Ok(score) => score,
            Err(error) => panic!("comparable: {error}"),
        };
        assert!((same.get() - 1.0).abs() < 1e-12);
        let orthogonal = match a.similarity_to(&b) {
            Ok(score) => score,
            Err(error) => panic!("comparable: {error}"),
        };
        assert!(orthogonal.get().abs() < 1e-12);
        let opposite = match a.similarity_to(&minus_a) {
            Ok(score) => score,
            Err(error) => panic!("comparable: {error}"),
        };
        assert!((opposite.get() + 1.0).abs() < 1e-12);
    }

    /// Identical vectors stay inside the cosine range despite floating-point
    /// summation, so the score type's own refusal never fires on real input.
    #[test]
    fn an_identical_pair_stays_inside_the_cosine_range() {
        let a = embedding(vec![0.3; 256]);
        let score = match a.similarity_to(&a) {
            Ok(score) => score,
            Err(error) => panic!("comparable: {error}"),
        };
        assert!(score.get() <= 1.0);
    }

    /// A zero vector points nowhere, so it has no similarity to anything.
    /// The arithmetic would divide by zero and hand back a NaN that reads as
    /// a considered verdict.
    #[test]
    fn a_zero_magnitude_embedding_has_no_similarity() {
        let zero = embedding(vec![0.0, 0.0]);
        let real = embedding(vec![1.0, 0.0]);
        assert_eq!(
            zero.similarity_to(&real),
            Err(IncomparableEmbeddings::NoDirection)
        );
    }

    #[test]
    fn embeddings_of_different_widths_are_incomparable() {
        assert!(matches!(
            embedding(vec![1.0, 0.0]).similarity_to(&embedding(vec![1.0, 0.0, 0.0])),
            Err(IncomparableEmbeddings::DifferentWidths { .. })
        ));
    }

    /// Two spans pointing the same way have that direction as their voice;
    /// magnitude does not weight the answer.
    #[test]
    fn a_centroid_is_the_shared_direction_regardless_of_magnitude() {
        let spans = [embedding(vec![3.0, 0.0]), embedding(vec![0.5, 0.0])];
        let centroid = SpeakerEmbedding::centroid(&spans).expect("test: a legal centroid");
        assert_eq!(centroid.components, vec![1.0, 0.0]);
    }

    /// Opposite directions cancel and have no voice; nothing to average is
    /// refused; a width mismatch names both widths.
    #[test]
    fn a_centroid_refuses_cancelling_empty_and_mixed_width_spans() {
        let opposite = [embedding(vec![1.0, 0.0]), embedding(vec![-1.0, 0.0])];
        assert_eq!(
            SpeakerEmbedding::centroid(&opposite),
            Err(NoCentroid::SpansCancel)
        );
        assert_eq!(
            SpeakerEmbedding::centroid(std::iter::empty()),
            Err(NoCentroid::NoSpans)
        );
        let mixed = [embedding(vec![1.0, 0.0]), embedding(vec![1.0, 0.0, 0.0])];
        assert_eq!(
            SpeakerEmbedding::centroid(&mixed),
            Err(NoCentroid::DifferentWidths {
                first: 2,
                offending: 3
            })
        );
    }
}
