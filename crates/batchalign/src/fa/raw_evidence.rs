//! Immutable model-returned forced-alignment evidence.

use serde::{Deserialize, Serialize};

use crate::api::EngineVersion;
use crate::chat_ops::CacheKey;
use crate::types::engines::{FaEngineName, FaTimingResolution};
use crate::types::worker_v2::{ExecuteOutcomeRef, ExecuteResponseV2, TaskResultV2};

const FA_RAW_EVIDENCE_SCHEMA_VERSION: u8 = 2;

/// Where the request-engine namespace attached to raw evidence came from.
///
/// New evidence receives the exact live capability of the worker selected for
/// the request. Schema-1 evidence had no such field; when it is admitted from
/// an exact-version cache lookup, that weaker provenance remains visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FaRequestEngineVersionOrigin {
    SelectedWorkerCapability,
    LegacyCacheNamespace,
}

/// Requested model identity carried with the evidence rather than paired
/// later by convention.
///
/// This is deliberately not called the producing-engine identity: a fallback
/// response may be produced by another engine. The payload proves that
/// effective engine family separately; fully versioning both fallback legs is
/// a distinct cache-key design.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FaRequestEngineIdentity {
    version: EngineVersion,
    origin: FaRequestEngineVersionOrigin,
}

/// Expected main-tier word cardinality attached to one FA request.
///
/// This is a separate type because a raw response's payload cardinality is
/// meaningful only relative to the exact request that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(super) struct ExpectedFaWords(usize);

impl ExpectedFaWords {
    pub(super) fn new(value: usize) -> Self {
        Self(value)
    }

    pub(super) fn get(self) -> usize {
        self.0
    }
}

/// How the effective response relates to the engine selected by the caller.
pub(super) enum FaEvidenceRoute<'a> {
    /// The selected engine produced the response directly.
    Direct,
    /// A retry engine produced the response after a named primary failure.
    Fallback { reason: &'a str },
}

/// Raw worker response admitted against the request facts needed for use.
///
/// A fallback response is valid for the current run but cannot enter the
/// replay cache: its effective Whisper model version is not part of the
/// primary Wave request namespace. Only [`ReplayableFaRawEvidence`] crosses
/// the persistence boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct FaRawEvidence {
    schema_version: u8,
    request_engine_identity: FaRequestEngineIdentity,
    requested_engine: FaEngineName,
    effective_engine: FaEngineName,
    expected_words: ExpectedFaWords,
    cache_key: CacheKey,
    fallback_reason: Option<String>,
    response: ExecuteResponseV2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFaRawEvidence {
    schema_version: u8,
    #[serde(default)]
    request_engine_identity: Option<FaRequestEngineIdentity>,
    requested_engine: FaEngineName,
    effective_engine: FaEngineName,
    expected_words: ExpectedFaWords,
    cache_key: CacheKey,
    fallback_reason: Option<String>,
    response: ExecuteResponseV2,
}

#[derive(Debug, thiserror::Error)]
pub(super) enum FaRawEvidenceError {
    #[error("forced-alignment raw evidence JSON is invalid: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("unsupported forced-alignment raw evidence schema version {0}")]
    SchemaVersion(u8),
    #[error("forced-alignment raw evidence schema {schema_version} has no engine identity")]
    MissingEngineIdentity { schema_version: u8 },
    #[error("legacy forced-alignment raw evidence unexpectedly carries a version identity")]
    UnexpectedLegacyEngineIdentity,
    #[error(
        "cached forced-alignment request namespace {cached} does not match selected worker version {current}"
    )]
    EngineVersionDrift {
        cached: EngineVersion,
        current: EngineVersion,
    },
    #[error(
        "cached forced-alignment evidence requested {cached:?}, not current engine {current:?}"
    )]
    RequestedEngineDrift {
        cached: FaEngineName,
        current: FaEngineName,
    },
    #[error(
        "cached forced-alignment evidence expected {cached} words, not current cardinality {current}"
    )]
    ExpectedWordsDrift { cached: usize, current: usize },
    #[error("cached forced-alignment evidence belongs to a different semantic cache key")]
    CacheKeyDrift,
    #[error(
        "cached forced-alignment evidence claims effective engine {cached:?}, but its payload proves {proven:?}"
    )]
    EffectiveEngineDrift {
        cached: FaEngineName,
        proven: FaEngineName,
    },
    #[error("forced-alignment worker returned an error: {code:?}: {message}")]
    WorkerFailure {
        code: crate::types::worker_v2::ProtocolErrorCodeV2,
        message: String,
    },
    #[error("forced-alignment worker returned the wrong task-result kind")]
    UnexpectedResult,
    #[error("indexed forced-alignment evidence cannot satisfy requested engine {0:?}")]
    IndexedEngineMismatch(FaEngineName),
    #[error(
        "indexed forced-alignment evidence contains {actual} timings for {expected} request words"
    )]
    WordCardinality { expected: usize, actual: usize },
    #[error("direct forced-alignment evidence was produced by a different engine")]
    UnexpectedDirectEngine,
    #[error("forced-alignment fallback did not change the effective engine")]
    IneffectiveFallback,
    #[error("forced-alignment fallback reason must not be empty")]
    EmptyFallbackReason,
    #[error(
        "forced-alignment fallback evidence is valid for this run but lacks a versioned effective-engine cache identity"
    )]
    UnversionedFallbackEvidence,
}

/// Raw evidence proven safe to persist and replay.
///
/// Construction consumes admitted evidence and refuses fallback output. This
/// wrapper is the only raw-evidence type exposed to cache-writing code, so a
/// future caller cannot accidentally persist a response whose effective model
/// version is absent from the cache identity.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub(super) struct ReplayableFaRawEvidence(FaRawEvidence);

impl FaRawEvidence {
    /// Admit a newly returned worker response against its request facts.
    pub(super) fn admit_requested(
        response: &ExecuteResponseV2,
        requested_engine: FaEngineName,
        request_engine_version: &EngineVersion,
        expected_words: ExpectedFaWords,
        cache_key: &CacheKey,
        route: FaEvidenceRoute<'_>,
    ) -> Result<Self, FaRawEvidenceError> {
        let effective_engine = match response.read() {
            ExecuteOutcomeRef::Failed { code, message } => {
                return Err(FaRawEvidenceError::WorkerFailure {
                    code,
                    message: message.to_owned(),
                });
            }
            ExecuteOutcomeRef::Success(TaskResultV2::IndexedWordTimingResult(result)) => {
                // An indexed payload carries a start AND an end per word, so
                // only an engine whose row says it reports word INTERVALS can
                // have produced it. Read from the table rather than compared
                // against a literal `Whisper`: the literal was right only for
                // as long as Whisper stayed the single token-onset engine, and
                // it is the failing engine's own row that the fallback policy
                // next door now reasons about.
                if requested_engine.timing_resolution() == FaTimingResolution::TokenOnsets {
                    return Err(FaRawEvidenceError::IndexedEngineMismatch(requested_engine));
                }
                if result.indexed_timings.len() != expected_words.get() {
                    return Err(FaRawEvidenceError::WordCardinality {
                        expected: expected_words.get(),
                        actual: result.indexed_timings.len(),
                    });
                }
                requested_engine
            }
            // The one literal left, and it is a WIRE fact rather than a
            // policy: `WhisperTokenTimingResult` is Whisper's own protocol
            // shape, named for it, and it is the ONLY token-onset payload the
            // worker protocol defines. Whichever engine was requested, a
            // response of this shape came from Whisper.
            //
            // This is what makes a fallback recognisable at all, since an
            // effective engine differing from the requested one is the whole
            // proof, and it is why the `const` block beside `FA_ENGINES`
            // refuses a fallback target that reports word intervals: such a
            // retry would return the indexed shape above, be credited to the
            // engine that was requested, and have its timings thrown away by
            // the `Fallback` arm below as an ineffective fallback.
            ExecuteOutcomeRef::Success(TaskResultV2::WhisperTokenTimingResult(_)) => {
                FaEngineName::Whisper
            }
            ExecuteOutcomeRef::Success(_) => return Err(FaRawEvidenceError::UnexpectedResult),
        };

        let fallback_reason = match route {
            FaEvidenceRoute::Direct if effective_engine == requested_engine => None,
            FaEvidenceRoute::Direct => return Err(FaRawEvidenceError::UnexpectedDirectEngine),
            FaEvidenceRoute::Fallback { reason: _ } if effective_engine == requested_engine => {
                return Err(FaRawEvidenceError::IneffectiveFallback);
            }
            FaEvidenceRoute::Fallback { reason } if reason.trim().is_empty() => {
                return Err(FaRawEvidenceError::EmptyFallbackReason);
            }
            FaEvidenceRoute::Fallback { reason } => Some(reason.to_owned()),
        };

        Ok(Self {
            schema_version: FA_RAW_EVIDENCE_SCHEMA_VERSION,
            request_engine_identity: FaRequestEngineIdentity {
                version: request_engine_version.clone(),
                origin: FaRequestEngineVersionOrigin::SelectedWorkerCapability,
            },
            requested_engine,
            effective_engine,
            expected_words,
            cache_key: cache_key.clone(),
            fallback_reason,
            response: response.clone(),
        })
    }

    /// Close fresh evidence into the persistence-capable state.
    pub(super) fn into_replayable(self) -> Result<ReplayableFaRawEvidence, FaRawEvidenceError> {
        if self.fallback_reason.is_some() {
            return Err(FaRawEvidenceError::UnversionedFallbackEvidence);
        }
        Ok(ReplayableFaRawEvidence(self))
    }

    pub(super) fn requested_engine(&self) -> FaEngineName {
        self.requested_engine
    }

    pub(super) fn effective_engine(&self) -> FaEngineName {
        self.effective_engine
    }

    #[cfg(test)]
    pub(super) fn expected_words(&self) -> ExpectedFaWords {
        self.expected_words
    }

    pub(super) fn response(&self) -> &ExecuteResponseV2 {
        &self.response
    }

    pub(super) fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }
}

impl ReplayableFaRawEvidence {
    /// Decode persisted evidence only after proving it still matches the request.
    pub(super) fn decode(
        value: serde_json::Value,
        requested_engine: FaEngineName,
        current_engine_version: &EngineVersion,
        expected_words: ExpectedFaWords,
        cache_key: &CacheKey,
    ) -> Result<Self, FaRawEvidenceError> {
        let raw: RawFaRawEvidence = serde_json::from_value(value)?;
        let request_engine_identity = match (raw.schema_version, raw.request_engine_identity) {
            (1, None) => FaRequestEngineIdentity {
                version: current_engine_version.clone(),
                origin: FaRequestEngineVersionOrigin::LegacyCacheNamespace,
            },
            (1, Some(_)) => return Err(FaRawEvidenceError::UnexpectedLegacyEngineIdentity),
            (FA_RAW_EVIDENCE_SCHEMA_VERSION, Some(identity)) => identity,
            (FA_RAW_EVIDENCE_SCHEMA_VERSION, None) => {
                return Err(FaRawEvidenceError::MissingEngineIdentity {
                    schema_version: FA_RAW_EVIDENCE_SCHEMA_VERSION,
                });
            }
            (version, _) => return Err(FaRawEvidenceError::SchemaVersion(version)),
        };
        if &request_engine_identity.version != current_engine_version {
            return Err(FaRawEvidenceError::EngineVersionDrift {
                cached: request_engine_identity.version,
                current: current_engine_version.clone(),
            });
        }
        if raw.requested_engine != requested_engine {
            return Err(FaRawEvidenceError::RequestedEngineDrift {
                cached: raw.requested_engine,
                current: requested_engine,
            });
        }
        if raw.expected_words != expected_words {
            return Err(FaRawEvidenceError::ExpectedWordsDrift {
                cached: raw.expected_words.get(),
                current: expected_words.get(),
            });
        }
        if &raw.cache_key != cache_key {
            return Err(FaRawEvidenceError::CacheKeyDrift);
        }

        let route = match raw.fallback_reason.as_deref() {
            Some(reason) => FaEvidenceRoute::Fallback { reason },
            None => FaEvidenceRoute::Direct,
        };
        let admitted = FaRawEvidence::admit_requested(
            &raw.response,
            requested_engine,
            current_engine_version,
            expected_words,
            cache_key,
            route,
        )?;
        if raw.effective_engine != admitted.effective_engine {
            return Err(FaRawEvidenceError::EffectiveEngineDrift {
                cached: raw.effective_engine,
                proven: admitted.effective_engine,
            });
        }
        admitted.into_replayable().map(|replayable| {
            Self(FaRawEvidence {
                request_engine_identity,
                ..replayable.0
            })
        })
    }

    pub(super) fn into_inner(self) -> FaRawEvidence {
        self.0
    }

    pub(super) fn requested_engine(&self) -> FaEngineName {
        self.0.requested_engine
    }

    pub(super) fn request_engine_version(&self) -> &EngineVersion {
        &self.0.request_engine_identity.version
    }

    pub(super) fn expected_words(&self) -> ExpectedFaWords {
        self.0.expected_words
    }

    pub(super) fn cache_key(&self) -> &CacheKey {
        &self.0.cache_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DurationMs, DurationSeconds};
    use crate::types::engines::FaEngineName;
    use crate::types::worker_v2::{
        ExecuteResponseV2, IndexedWordTimingResultV2, IndexedWordTimingV2, TaskResultV2,
        WhisperTokenTimingResultV2, WhisperTokenTimingV2, WorkerRequestIdV2,
    };

    fn cache_key(label: &str) -> CacheKey {
        CacheKey::from_content(label)
    }

    fn engine_version() -> EngineVersion {
        EngineVersion::from("test-fa-wave-v1")
    }

    #[test]
    fn admitted_raw_evidence_round_trips_without_losing_engine_or_cardinality() {
        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-1"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![Some(IndexedWordTimingV2 {
                    start_ms: DurationMs(10),
                    end_ms: DurationMs(30),
                    confidence: Some(0.75),
                })],
            }),
            DurationSeconds(0.01),
        );

        let evidence = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("roundtrip"),
            FaEvidenceRoute::Direct,
        )
        .expect("valid indexed evidence");
        let encoded = serde_json::to_value(&evidence).expect("serialize evidence");
        let replayed = ReplayableFaRawEvidence::decode(
            encoded,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("roundtrip"),
        )
        .expect("decode evidence")
        .into_inner();

        assert_eq!(replayed.requested_engine(), FaEngineName::Wave2Vec);
        assert_eq!(replayed.effective_engine(), FaEngineName::Wave2Vec);
        assert_eq!(replayed.expected_words().get(), 1);
    }

    #[test]
    fn cached_evidence_for_a_different_requested_engine_is_refused() {
        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-2"),
            TaskResultV2::WhisperTokenTimingResult(WhisperTokenTimingResultV2 {
                tokens: vec![WhisperTokenTimingV2 {
                    text: "hello".to_owned(),
                    time_s: DurationSeconds(0.1),
                }],
            }),
            DurationSeconds(0.01),
        );
        let evidence = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Whisper,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("engine-drift"),
            FaEvidenceRoute::Direct,
        )
        .expect("valid token evidence");

        let error = ReplayableFaRawEvidence::decode(
            serde_json::to_value(evidence).expect("serialize evidence"),
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("engine-drift"),
        )
        .expect_err("a key for another engine cannot admit this evidence");

        assert!(matches!(
            error,
            FaRawEvidenceError::RequestedEngineDrift { .. }
        ));
    }

    #[test]
    fn indexed_raw_evidence_with_wrong_word_cardinality_is_refused() {
        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-3"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![],
            }),
            DurationSeconds(0.01),
        );

        let error = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("cardinality"),
            FaEvidenceRoute::Direct,
        )
        .expect_err("short indexed evidence must not become replayable");

        assert!(matches!(error, FaRawEvidenceError::WordCardinality { .. }));
    }

    #[test]
    fn cached_evidence_for_a_different_semantic_key_is_refused() {
        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-4"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None],
            }),
            DurationSeconds(0.01),
        );
        let evidence = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("first-group"),
            FaEvidenceRoute::Direct,
        )
        .expect("valid indexed evidence");

        let error = ReplayableFaRawEvidence::decode(
            serde_json::to_value(evidence).expect("serialize evidence"),
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("second-group"),
        )
        .expect_err("same-cardinality evidence from another group must not replay");

        assert!(matches!(error, FaRawEvidenceError::CacheKeyDrift));
    }

    #[test]
    fn fallback_route_is_usable_now_but_cannot_enter_the_replay_cache() {
        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-fallback"),
            TaskResultV2::WhisperTokenTimingResult(WhisperTokenTimingResultV2 {
                tokens: vec![WhisperTokenTimingV2 {
                    text: "hello".to_owned(),
                    time_s: DurationSeconds(0.1),
                }],
            }),
            DurationSeconds(0.01),
        );
        let key = cache_key("fallback");
        let evidence = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &key,
            FaEvidenceRoute::Fallback {
                reason: "targets length is too long for CTC",
            },
        )
        .expect("valid fallback evidence");

        assert_eq!(evidence.effective_engine(), FaEngineName::Whisper);
        assert_eq!(
            evidence.fallback_reason(),
            Some("targets length is too long for CTC")
        );
        let error = evidence
            .into_replayable()
            .expect_err("fallback evidence has no effective-model cache identity");
        assert!(matches!(
            error,
            FaRawEvidenceError::UnversionedFallbackEvidence
        ));
    }

    #[test]
    fn direct_route_cannot_hide_an_effective_engine_change() {
        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-unlabelled-fallback"),
            TaskResultV2::WhisperTokenTimingResult(WhisperTokenTimingResultV2 {
                tokens: Vec::new(),
            }),
            DurationSeconds(0.01),
        );

        let error = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("unlabelled-fallback"),
            FaEvidenceRoute::Direct,
        )
        .expect_err("a fallback response needs an explicit fallback route");

        assert!(matches!(error, FaRawEvidenceError::UnexpectedDirectEngine));
    }

    /// An interval payload is refused for every engine the table says reports
    /// token onsets, and admitted for every engine it says reports intervals.
    ///
    /// A ROUNDTRIP between `FA_ENGINES` and the admission that reads it, which
    /// no signature pins: the row states a resolution, and only running the
    /// admission shows that a payload of the other shape is refused rather
    /// than credited to the engine that cannot have produced it.
    ///
    /// Written as a loop over the roster rather than as
    /// `FaEngineName::Whisper`, because naming one engine here would be the
    /// very defect it exists to catch: this refusal WAS a literal `Whisper`
    /// comparison, which was correct only for as long as Whisper stayed the
    /// single token-onset engine.
    #[test]
    fn an_interval_payload_is_refused_for_every_token_onset_engine() {
        use crate::types::engines::{FaTimingResolution, SelectableEngine};

        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-interval-shape"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None],
            }),
            DurationSeconds(0.01),
        );

        for engine in FaEngineName::ALL.iter().copied() {
            let admitted = FaRawEvidence::admit_requested(
                &response,
                engine,
                &engine_version,
                ExpectedFaWords::new(1),
                &cache_key("interval-shape"),
                FaEvidenceRoute::Direct,
            );
            match engine.timing_resolution() {
                FaTimingResolution::TokenOnsets => assert!(
                    matches!(
                        admitted,
                        Err(FaRawEvidenceError::IndexedEngineMismatch(refused))
                            if refused == engine
                    ),
                    "{engine:?} reports token onsets, so an indexed interval payload \
                     cannot be its own work and must not be admitted as it",
                ),
                FaTimingResolution::WordIntervals => {
                    let evidence =
                        admitted.expect("an interval engine's own payload shape must be admitted");
                    assert_eq!(evidence.effective_engine(), engine);
                }
            }
        }
    }

    #[test]
    fn schema_two_evidence_is_refused_after_worker_version_drift() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-version-drift"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None],
            }),
            DurationSeconds(0.01),
        );
        let admitted_version = EngineVersion::from("test-fa-wave-v1");
        let evidence = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &admitted_version,
            ExpectedFaWords::new(1),
            &cache_key("version-drift"),
            FaEvidenceRoute::Direct,
        )
        .expect("fixture evidence is valid");

        let error = ReplayableFaRawEvidence::decode(
            serde_json::to_value(evidence).expect("serialize evidence"),
            FaEngineName::Wave2Vec,
            &EngineVersion::from("test-fa-wave-v2"),
            ExpectedFaWords::new(1),
            &cache_key("version-drift"),
        )
        .expect_err("evidence from another worker version must not replay");

        assert!(matches!(
            error,
            FaRawEvidenceError::EngineVersionDrift { .. }
        ));
    }

    #[test]
    fn schema_one_evidence_retains_its_weaker_cache_namespace_origin() {
        let engine_version = engine_version();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("fa-raw-schema-one"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None],
            }),
            DurationSeconds(0.01),
        );
        let evidence = FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("schema-one"),
            FaEvidenceRoute::Direct,
        )
        .expect("fixture evidence is valid");
        let mut encoded = serde_json::to_value(evidence).expect("serialize evidence");
        let object = encoded.as_object_mut().expect("evidence object");
        object.insert("schema_version".to_owned(), serde_json::json!(1));
        object.remove("request_engine_identity");

        let replayed = ReplayableFaRawEvidence::decode(
            encoded,
            FaEngineName::Wave2Vec,
            &engine_version,
            ExpectedFaWords::new(1),
            &cache_key("schema-one"),
        )
        .expect("schema-one evidence was already selected by exact cache namespace")
        .into_inner();

        assert_eq!(
            replayed.request_engine_identity.origin,
            FaRequestEngineVersionOrigin::LegacyCacheNamespace
        );
    }
}
