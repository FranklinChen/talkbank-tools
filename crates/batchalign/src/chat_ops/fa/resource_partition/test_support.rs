//! Explicit declared-model port for deterministic policy/cache controls.
//! This produces no production model, allocation or execution proof.
#![allow(clippy::unwrap_used, clippy::panic)]
use super::*;
use crate::api::ReportedEngineName;
use crate::types::worker_v2::{FaConstraintUnitV2, FaPreparationRefusalV2};
use std::{convert::Infallible, num::NonZeroU64};

pub(crate) enum DeclaredModel {
    AudioSamples {
        maximum: NonZeroU64,
    },
    Labels {
        minimum: NonZeroU64,
        maximum: NonZeroU64,
    },
    /// The host can hold a request of at most this many words (zero: none),
    /// and refuses larger ones for host memory.
    HostMemory {
        fits_words: u64,
    },
    /// The host refuses, for host memory, any request holding this word:
    /// one dense spot in an otherwise alignable source.
    HostMemoryAround {
        dense_word: usize,
    },
}

/// The host-memory constraint every declared host refusal carries.
pub(crate) fn declared_host_memory() -> FaResourceConstraintV2 {
    FaResourceConstraintV2::HostMemory {
        engine: ReportedEngineName::try_from("declared-model").unwrap(),
        required_mb: 14_100,
        reserve_mb: 2_048,
        ceiling_mb: 14_000,
    }
}

pub(crate) struct DeclaredPreparation {
    window: FaWindow,
    ids: Vec<String>,
}

impl ResourceProbe for DeclaredModel {
    type Prepared = DeclaredPreparation;
    type Error = Infallible;

    async fn prepare(
        &mut self,
        request: GroupUnit<'_>,
    ) -> Result<ResourceProbeOutcome<Self::Prepared>, Infallible> {
        // A declared host refusal, or the request as the host admits it.
        let host = |refuses: bool| {
            Ok(if refuses {
                ResourceProbeOutcome::Limited(declared_host_memory())
            } else {
                ResourceProbeOutcome::Prepared(DeclaredPreparation {
                    window: request.window(),
                    ids: request.words().iter().map(FaWord::stable_id).collect(),
                })
            })
        };
        let (unit, provided, minimum, maximum) = match self {
            Self::HostMemory { fits_words } => {
                return host(request.words().len() as u64 > *fits_words);
            }
            Self::HostMemoryAround { dense_word } => {
                return host(
                    request
                        .words()
                        .iter()
                        .any(|word| word.utterance_word_index.raw() == *dense_word),
                );
            }
            Self::AudioSamples { maximum } => (
                FaConstraintUnitV2::AudioSamples,
                request.window().len().0 * 16,
                1,
                maximum.get(),
            ),
            Self::Labels { minimum, maximum } => (
                FaConstraintUnitV2::TextLabels,
                request.words().len() as u64,
                minimum.get(),
                maximum.get(),
            ),
        };
        let cause = if provided > maximum {
            Some(FaPreparationRefusalV2::ModelConstraint {
                unit,
                provided,
                maximum,
                engine: ReportedEngineName::try_from("declared-model").unwrap(),
            })
        } else if provided < minimum {
            Some(FaPreparationRefusalV2::InsufficientInput {
                unit,
                provided,
                minimum,
                engine: ReportedEngineName::try_from("declared-model").unwrap(),
            })
        } else {
            None
        };
        Ok(match cause {
            Some(cause) => ResourceProbeOutcome::Limited(FaResourceConstraintV2::Model { cause }),
            None => ResourceProbeOutcome::Prepared(DeclaredPreparation {
                window: request.window(),
                ids: request.words().iter().map(FaWord::stable_id).collect(),
            }),
        })
    }
}

impl DeclaredPreparation {
    pub(crate) fn verify_for(&self, piece: &AnchoredPiece) {
        self.verify_request(GroupUnit::from_piece(piece));
    }
    pub(crate) fn verify_request(&self, request: GroupUnit<'_>) {
        assert_eq!(
            request.window(),
            self.window,
            "dispatch retains the exact prepared window"
        );
        assert_eq!(
            request
                .words()
                .iter()
                .map(FaWord::stable_id)
                .collect::<Vec<_>>(),
            self.ids
        );
    }
}

/// Two disjoint answer ranges may legally borrow the same complete context.
/// Admit both through the real context constructor and probe boundary.
pub(crate) async fn shared_context_partition<P: ResourceProbe>(
    request: GroupUnit<'_>,
    probe: &mut P,
) -> Result<PreparedResourceSplit<P::Prepared>, ResourcePlanError<P::Error>> {
    let words = NonEmptyWords::from_vec(request.words().to_vec())
        .ok_or(ResourcePlanError::InvalidCut(InvalidResourceCut))?;
    let source = ResourceRequest {
        utterance: words.as_slice()[0].utterance_index,
        words,
        window: request.window(),
        source_window: request.window(),
        localization: super::super::grouping::Localization::NotRecorded,
        original_bullet: crate::types::traces::FaOriginalBulletTrace::NotRecorded,
    };
    let first = source.words.first_word();
    let last = source.words.last_word();
    let mid = FileMs::new(source.window.audio_start().get() + source.window.len().0 / 2);
    let (before, after) = source
        .window
        .split_at(mid)
        .map_err(|_| ResourcePlanError::InvalidCut(InvalidResourceCut))?;
    let mut parts = Vec::new();
    for (word, window) in [(first, before), (last, after)] {
        let context = ResourceContext::admit(source.clone(), word, word, window)
            .map_err(ResourcePlanError::InvalidCut)?;
        match context.prepare(probe).await? {
            ResourceProbeOutcome::Prepared(piece) => {
                parts.push(PreparedResourcePart::Request(piece))
            }
            ResourceProbeOutcome::Limited(constraint) => {
                panic!("shared-context fixture needs admitted preparation: {constraint}")
            }
        }
    }
    let last = parts
        .pop()
        .ok_or(ResourcePlanError::InvalidCut(InvalidResourceCut))?;
    Ok(PreparedResourceSplit {
        source,
        earlier: parts,
        last,
    })
}
