//! Durable evidence captured at paid inference boundaries.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

use crate::api::NumSpeakers;
use crate::cache::BuildOwnedNamespace;
use crate::cache::{CacheError, CacheNamespace, InferenceLease, UtteranceCache, tasks};
use crate::chat_ops::CacheKey;
use crate::error::ServerError;
use crate::params::CachePolicy;
use crate::types::worker_v2::{SpeakerBackendV2, SpeakerInferenceEvidenceV2, SpeakerSegmentV2};
use crate::worker::speaker_result_v2::normalize_speaker_evidence_v2;

mod trace;
#[cfg(test)]
use trace::SPEAKER_SEGMENT_DIGEST_REVISION;
#[cfg(test)]
pub(crate) use trace::SpeakerEvidenceCacheOutcome;
pub(crate) use trace::{SpeakerEvidenceTrace, SpeakerProjectionRevision};
use trace::{SpeakerEvidenceTraceSeed, SpeakerSegmentsDigest};

const RAW_SPEAKER_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION: u32 = 1;
const SPEAKER_AUDIO_PREPARATION_REVISION: &str = "mono-16khz-f32le-v1";
const SPEAKER_NORMALIZATION_REVISION: &str = "speaker-evidence-normalizer-v1";
const LOCAL_PYANNOTE_MODEL_MANIFEST: &str =
    include_str!("../../../../batchalign/inference/local_pyannote_model.json");

/// A digest of the media-source bytes passed to canonical audio preparation.
///
/// Unlike the legacy audio identity, this contains neither a path nor an
/// mtime. Copies and renames of identical source media therefore share paid
/// evidence. The preparation recipe has its own revision in the semantic key;
/// changing how source media becomes model-ready PCM must bump that revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
struct SpeakerAudioSourceDigest(String);

impl SpeakerAudioSourceDigest {
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(bytes);
        Self(hasher.finalize().to_hex().to_string())
    }

    async fn from_path(path: &Path) -> Result<Self, SpeakerEvidenceCacheError> {
        let mut file = tokio::fs::File::open(path).await?;
        let mut hasher = blake3::Hasher::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        Ok(Self(hasher.finalize().to_hex().to_string()))
    }
}

/// Source media whose exact bytes established the speaker-evidence identity.
///
/// The source path is retained only so an authorized cache miss can reread and
/// verify those bytes. It is never part of the semantic cache key.
#[derive(Debug, Clone)]
struct PreparedSpeakerProviderMedia {
    source_path: PathBuf,
    source_digest: SpeakerAudioSourceDigest,
}

impl PreparedSpeakerProviderMedia {
    async fn from_source(path: &Path) -> Result<Self, SpeakerEvidenceCacheError> {
        Ok(Self {
            source_path: path.to_owned(),
            source_digest: SpeakerAudioSourceDigest::from_path(path).await?,
        })
    }

    async fn verify(self) -> Result<Vec<u8>, SpeakerEvidenceCacheError> {
        let bytes = tokio::fs::read(&self.source_path).await?;
        if SpeakerAudioSourceDigest::from_bytes(&bytes) != self.source_digest {
            return Err(SpeakerEvidenceCacheError::ProviderMediaDrift(
                self.source_path,
            ));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Serialize)]
struct SpeakerEvidenceKeyMaterial<'a> {
    schema_version: u32,
    audio_source_blake3: &'a SpeakerAudioSourceDigest,
    audio_preparation_revision: &'static str,
    backend: SpeakerBackendV2,
    expected_speakers: Option<u32>,
    model_revision: &'a str,
}

#[derive(Debug, Serialize)]
struct DerivedSpeakerEvidenceKeyMaterial<'a> {
    schema_version: u32,
    raw_evidence_fingerprint: &'a str,
    normalization_revision: &'a str,
}

/// Revision identity for speaker evidence, distinct from an ASR engine's name.
///
/// The private field, whose only production constructor derives it from the
/// speaker backend, and the speaker cache task's namespace type keep any other
/// identity (an ASR engine's included) from scoping speaker-model evidence.
#[derive(Debug, Clone)]
pub(crate) struct SpeakerEvidenceModelRevision(BuildOwnedNamespace);

impl SpeakerEvidenceModelRevision {
    pub(crate) fn for_backend(backend: SpeakerBackendV2) -> Self {
        Self(match backend {
            SpeakerBackendV2::PyannoteAi => BuildOwnedNamespace::literal("pyannote-ai:precision-2"),
            SpeakerBackendV2::Pyannote => BuildOwnedNamespace::derived(format!(
                "pyannote-local-graph:{}",
                blake3::hash(LOCAL_PYANNOTE_MODEL_MANIFEST.as_bytes()).to_hex()
            )),
            SpeakerBackendV2::Nemo => BuildOwnedNamespace::derived(format!(
                "nemo:diar_infer_general:ba-{}",
                env!("CARGO_PKG_VERSION")
            )),
        })
    }

    fn as_str(&self) -> &str {
        self.0.as_str()
    }

    #[cfg(test)]
    fn for_test(label: &str) -> Self {
        Self(BuildOwnedNamespace::derived(label.to_owned()))
    }
}

impl CacheNamespace for SpeakerEvidenceModelRevision {
    fn namespace(&self) -> &str {
        self.as_str()
    }
}

/// Identity of the local raw-evidence projection algorithm.
///
/// This cannot be substituted with a speaker model revision: changing this
/// value invalidates only derived segments and deliberately preserves paid raw
/// evidence. The cache enforces that too: its segments task takes this type.
#[derive(Debug, Clone)]
pub(crate) struct SpeakerNormalizationRevision(BuildOwnedNamespace);

impl CacheNamespace for SpeakerNormalizationRevision {
    fn namespace(&self) -> &str {
        self.as_str()
    }
}

impl SpeakerNormalizationRevision {
    fn current() -> Self {
        Self(BuildOwnedNamespace::literal(SPEAKER_NORMALIZATION_REVISION))
    }

    fn as_str(&self) -> &str {
        self.0.as_str()
    }

    #[cfg(test)]
    fn for_test(label: &str) -> Self {
        Self(BuildOwnedNamespace::derived(label.to_owned()))
    }
}

/// Semantic identity of one speaker-diarization request.
///
/// Construction reads and hashes the prepared audio. All fields that can
/// change inference semantics are owned here so downstream stages cannot
/// accidentally invent partial cache keys.
#[derive(Debug, Clone)]
pub(crate) struct SpeakerEvidenceRequest {
    raw_cache_key: CacheKey,
    derived_cache_key: CacheKey,
    provider_media: PreparedSpeakerProviderMedia,
    backend: SpeakerBackendV2,
    expected_speakers: Option<NumSpeakers>,
    model_revision: SpeakerEvidenceModelRevision,
    normalization_revision: SpeakerNormalizationRevision,
}

impl SpeakerEvidenceRequest {
    pub(crate) async fn from_audio(
        audio_path: &Path,
        backend: SpeakerBackendV2,
        expected_speakers: Option<NumSpeakers>,
        model_revision: &SpeakerEvidenceModelRevision,
    ) -> Result<Self, SpeakerEvidenceCacheError> {
        let provider_media = PreparedSpeakerProviderMedia::from_source(audio_path).await?;
        let key_material = SpeakerEvidenceKeyMaterial {
            schema_version: RAW_SPEAKER_EVIDENCE_SCHEMA_VERSION,
            audio_source_blake3: &provider_media.source_digest,
            audio_preparation_revision: SPEAKER_AUDIO_PREPARATION_REVISION,
            backend,
            expected_speakers: expected_speakers.map(|count| count.0),
            model_revision: model_revision.as_str(),
        };
        let canonical = serde_json::to_string(&key_material)?;
        let raw_cache_key = CacheKey::from_content(&canonical);
        let normalization_revision = SpeakerNormalizationRevision::current();
        let derived_cache_key = derived_cache_key(&raw_cache_key, &normalization_revision)?;
        Ok(Self {
            raw_cache_key,
            derived_cache_key,
            provider_media,
            backend,
            expected_speakers,
            model_revision: model_revision.clone(),
            normalization_revision,
        })
    }

    pub(crate) fn cache_key(&self) -> &CacheKey {
        &self.raw_cache_key
    }

    fn trace_seed(&self) -> SpeakerEvidenceTraceSeed {
        SpeakerEvidenceTraceSeed {
            trace_schema_version: 1,
            source_media_blake3: self.provider_media.source_digest.0.clone(),
            audio_preparation_revision: SPEAKER_AUDIO_PREPARATION_REVISION,
            backend: self.backend,
            expected_speakers: self.expected_speakers.map(|count| count.0),
            model_revision: self.model_revision.as_str().to_owned(),
            raw_evidence_key: self.raw_cache_key.to_string(),
            normalization_revision: self.normalization_revision.as_str().to_owned(),
            derived_evidence_key: self.derived_cache_key.to_string(),
        }
    }

    #[cfg(test)]
    fn with_normalization_revision_for_test(mut self, revision: &str) -> Self {
        self.normalization_revision = SpeakerNormalizationRevision::for_test(revision);
        self.derived_cache_key =
            derived_cache_key(&self.raw_cache_key, &self.normalization_revision)
                .expect("test normalization revision should serialize");
        self
    }

    async fn lookup(
        &self,
        cache: &UtteranceCache,
        policy: CachePolicy,
    ) -> Result<SpeakerEvidenceLookup, SpeakerEvidenceCacheError> {
        let lease = InferenceLease::acquire(&self.raw_cache_key, cache).await;
        match policy {
            CachePolicy::SkipCache if !lease.observed_commit_while_waiting() => {
                return Ok(SpeakerEvidenceLookup::Miss(SpeakerEvidenceMiss {
                    request: self.clone(),
                    reason: SpeakerEvidenceMissReason::ForcedRefresh,
                    _lease: lease,
                }));
            }
            CachePolicy::UseCache | CachePolicy::RequireCache | CachePolicy::SkipCache => {}
        }

        let stored_derived = cache
            .get(
                self.derived_cache_key.as_str(),
                tasks::SPEAKER_DIARIZATION_SEGMENTS,
                &self.normalization_revision,
            )
            .await?;
        if let Some(stored) = stored_derived {
            let envelope: StoredDerivedSpeakerEvidence =
                serde_json::from_value(stored).map_err(|error| {
                    SpeakerEvidenceCacheError::InvalidCachedEvidence(error.to_string())
                })?;
            envelope.validate_for(self)?;
            return Ok(SpeakerEvidenceLookup::DerivedHit(
                ValidatedSpeakerEvidence::new(envelope.segments),
            ));
        }

        let stored_raw = cache
            .get(
                self.raw_cache_key.as_str(),
                tasks::SPEAKER_DIARIZATION_RAW_EVIDENCE,
                &self.model_revision,
            )
            .await?;
        if let Some(stored) = stored_raw {
            let envelope: StoredRawSpeakerEvidence =
                serde_json::from_value(stored).map_err(|error| {
                    SpeakerEvidenceCacheError::InvalidCachedEvidence(error.to_string())
                })?;
            let segments = envelope.validate_and_normalize_for(self)?;
            store_derived_evidence(self, cache, &segments).await?;
            return Ok(SpeakerEvidenceLookup::RawHit(
                ValidatedSpeakerEvidence::new(segments),
            ));
        }

        match policy {
            CachePolicy::RequireCache => {
                return Err(SpeakerEvidenceCacheError::RequiredEvidenceMissing(
                    self.raw_cache_key.clone(),
                ));
            }
            CachePolicy::UseCache | CachePolicy::SkipCache => {}
        }

        Ok(SpeakerEvidenceLookup::Miss(SpeakerEvidenceMiss {
            request: self.clone(),
            reason: SpeakerEvidenceMissReason::NotFound,
            _lease: lease,
        }))
    }

    #[cfg(test)]
    async fn store_unchecked_raw_for_test(
        &self,
        cache: &UtteranceCache,
        value: serde_json::Value,
    ) -> Result<(), SpeakerEvidenceCacheError> {
        cache
            .put(
                self.raw_cache_key.as_str(),
                tasks::SPEAKER_DIARIZATION_RAW_EVIDENCE,
                &self.model_revision,
                &value,
            )
            .await?;
        Ok(())
    }

    #[cfg(test)]
    async fn store_unchecked_derived_for_test(
        &self,
        cache: &UtteranceCache,
        value: serde_json::Value,
    ) -> Result<(), SpeakerEvidenceCacheError> {
        cache
            .put(
                self.derived_cache_key.as_str(),
                tasks::SPEAKER_DIARIZATION_SEGMENTS,
                &self.normalization_revision,
                &value,
            )
            .await?;
        Ok(())
    }
}

fn derived_cache_key(
    raw_cache_key: &CacheKey,
    normalization_revision: &SpeakerNormalizationRevision,
) -> Result<CacheKey, serde_json::Error> {
    let material = DerivedSpeakerEvidenceKeyMaterial {
        schema_version: DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION,
        raw_evidence_fingerprint: raw_cache_key.as_str(),
        normalization_revision: normalization_revision.as_str(),
    };
    serde_json::to_string(&material).map(|canonical| CacheKey::from_content(&canonical))
}

/// Result of checking durable speaker evidence.
///
/// A hit contains only replayable evidence. Only the miss variant contains the
/// state transition that can authorize an inference call.
#[derive(Debug)]
enum SpeakerEvidenceLookup {
    DerivedHit(ValidatedSpeakerEvidence),
    RawHit(ValidatedSpeakerEvidence),
    Miss(SpeakerEvidenceMiss),
}

/// The only service boundary accepted by the production evidence resolver.
///
/// Tests implement this trait with a call-counting fake. Production implements
/// it with the V2 speaker worker adapter. Both therefore exercise the same
/// cache-hit/miss and commit control flow.
#[async_trait::async_trait]
pub(crate) trait SpeakerEvidenceInference: Sync {
    async fn infer(
        &self,
        run: VerifiedSpeakerEvidenceRun,
    ) -> Result<SpeakerInferenceEvidenceV2, ServerError>;
}

/// Stable causal origin of the speaker evidence used for one projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeakerEvidenceSource {
    ReplayedDerived,
    DerivedFromRaw,
    Inferred(SpeakerEvidenceMissReason),
}

/// How one request acquired validated speaker evidence.
#[derive(Debug)]
pub(crate) struct SpeakerEvidenceResolution {
    evidence: ValidatedSpeakerEvidence,
    source: SpeakerEvidenceSource,
    trace_seed: SpeakerEvidenceTraceSeed,
}

impl SpeakerEvidenceResolution {
    pub(crate) fn source(&self) -> SpeakerEvidenceSource {
        self.source
    }

    pub(crate) fn segments(&self) -> &[SpeakerSegmentV2] {
        self.evidence.segments()
    }

    pub(crate) fn into_segments(self) -> Vec<SpeakerSegmentV2> {
        self.evidence.into_segments()
    }
}

/// Resolve evidence through the same cache-or-infer state machine in tests and
/// production.
pub(crate) async fn resolve_speaker_evidence<I: SpeakerEvidenceInference>(
    request: &SpeakerEvidenceRequest,
    cache: &UtteranceCache,
    policy: CachePolicy,
    inference: &I,
) -> Result<SpeakerEvidenceResolution, SpeakerEvidenceResolutionError> {
    let trace_seed = request.trace_seed();
    match request.lookup(cache, policy).await? {
        SpeakerEvidenceLookup::DerivedHit(evidence) => Ok(SpeakerEvidenceResolution {
            evidence,
            source: SpeakerEvidenceSource::ReplayedDerived,
            trace_seed,
        }),
        SpeakerEvidenceLookup::RawHit(evidence) => Ok(SpeakerEvidenceResolution {
            evidence,
            source: SpeakerEvidenceSource::DerivedFromRaw,
            trace_seed,
        }),
        SpeakerEvidenceLookup::Miss(miss) => {
            let authorization = miss.authorize_billable_inference();
            let (run, permit) = authorization.into_run();
            let reason = permit.reason();
            let verified_run = run.verify().await?;
            let raw_evidence = inference.infer(verified_run).await?;
            let evidence = permit.commit(cache, raw_evidence).await?;
            Ok(SpeakerEvidenceResolution {
                evidence,
                source: SpeakerEvidenceSource::Inferred(reason),
                trace_seed,
            })
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SpeakerEvidenceResolutionError {
    #[error(transparent)]
    Evidence(#[from] SpeakerEvidenceCacheError),
    #[error(transparent)]
    Inference(#[from] ServerError),
}

/// Speaker turns that passed cache-envelope and timing validation.
#[derive(Debug, Clone)]
pub(crate) struct ValidatedSpeakerEvidence {
    segments: Vec<SpeakerSegmentV2>,
    segments_digest: SpeakerSegmentsDigest,
}

impl ValidatedSpeakerEvidence {
    fn new(segments: Vec<SpeakerSegmentV2>) -> Self {
        let segments_digest = SpeakerSegmentsDigest::from_segments(&segments);
        Self {
            segments,
            segments_digest,
        }
    }

    pub(crate) fn segments(&self) -> &[SpeakerSegmentV2] {
        &self.segments
    }

    pub(crate) fn into_segments(self) -> Vec<SpeakerSegmentV2> {
        self.segments
    }
}

/// Why no reusable entry was returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeakerEvidenceMissReason {
    NotFound,
    ForcedRefresh,
}

/// A proven cache miss. Its fields are private so other modules cannot forge
/// permission to call the inference service.
#[derive(Debug)]
struct SpeakerEvidenceMiss {
    request: SpeakerEvidenceRequest,
    reason: SpeakerEvidenceMissReason,
    _lease: InferenceLease,
}

impl SpeakerEvidenceMiss {
    fn authorize_billable_inference(self) -> SpeakerInferenceAuthorization {
        SpeakerInferenceAuthorization {
            request: self.request,
            reason: self.reason,
            _lease: self._lease,
        }
    }
}

/// Capability required by the raw speaker-inference function.
///
/// The only constructor consumes [`SpeakerEvidenceMiss`]. A cache hit has no
/// route to this type, so it cannot accidentally fall through to a paid call.
#[derive(Debug)]
struct SpeakerInferenceAuthorization {
    request: SpeakerEvidenceRequest,
    reason: SpeakerEvidenceMissReason,
    _lease: InferenceLease,
}

/// Single-use capability accepted by the live speaker inference boundary.
///
/// Its field is private, so only a proven cache miss can create it. The
/// inference trait consumes the capability, which rules out issuing a second
/// paid request from the same authorization.
#[derive(Debug)]
struct AuthorizedSpeakerEvidenceRun {
    provider_media: PreparedSpeakerProviderMedia,
    backend: SpeakerBackendV2,
    expected_speakers: Option<NumSpeakers>,
}

/// Exact source bytes and request semantics admitted for one speaker run.
///
/// Construction rereads and verifies the source after cache authorization.
/// The live adapter consumes these bytes, not a separately supplied path.
pub(crate) struct VerifiedSpeakerEvidenceRun {
    source_bytes: Vec<u8>,
    backend: SpeakerBackendV2,
    expected_speakers: Option<NumSpeakers>,
}

impl VerifiedSpeakerEvidenceRun {
    pub(super) fn into_worker_input(self) -> (Vec<u8>, SpeakerBackendV2, Option<NumSpeakers>) {
        (self.source_bytes, self.backend, self.expected_speakers)
    }
}

impl AuthorizedSpeakerEvidenceRun {
    async fn verify(self) -> Result<VerifiedSpeakerEvidenceRun, SpeakerEvidenceCacheError> {
        Ok(VerifiedSpeakerEvidenceRun {
            source_bytes: self.provider_media.verify().await?,
            backend: self.backend,
            expected_speakers: self.expected_speakers,
        })
    }
}

impl SpeakerInferenceAuthorization {
    fn into_run(self) -> (AuthorizedSpeakerEvidenceRun, SpeakerEvidenceCommitPermit) {
        let Self {
            request,
            reason,
            _lease,
        } = self;
        let run = AuthorizedSpeakerEvidenceRun {
            provider_media: request.provider_media.clone(),
            backend: request.backend,
            expected_speakers: request.expected_speakers,
        };
        (
            run,
            SpeakerEvidenceCommitPermit {
                request,
                reason,
                _lease,
            },
        )
    }
}

/// Capability to durably commit exactly the evidence produced by one
/// authorized speaker-inference run.
#[derive(Debug)]
struct SpeakerEvidenceCommitPermit {
    request: SpeakerEvidenceRequest,
    reason: SpeakerEvidenceMissReason,
    _lease: InferenceLease,
}

impl SpeakerEvidenceCommitPermit {
    fn reason(&self) -> SpeakerEvidenceMissReason {
        self.reason
    }

    async fn commit(
        self,
        cache: &UtteranceCache,
        raw_evidence: SpeakerInferenceEvidenceV2,
    ) -> Result<ValidatedSpeakerEvidence, SpeakerEvidenceCacheError> {
        validate_evidence_backend(&raw_evidence, self.request.backend)?;
        let segments = normalize_speaker_evidence_v2(&raw_evidence)
            .map_err(SpeakerEvidenceCacheError::InvalidCachedEvidence)?;
        validate_segments(&segments)?;
        let envelope = StoredRawSpeakerEvidence {
            schema_version: RAW_SPEAKER_EVIDENCE_SCHEMA_VERSION,
            request_fingerprint: self.request.raw_cache_key.to_string(),
            evidence: raw_evidence,
        };
        let value = serde_json::to_value(&envelope)?;
        cache
            .put(
                self.request.raw_cache_key.as_str(),
                tasks::SPEAKER_DIARIZATION_RAW_EVIDENCE,
                &self.request.model_revision,
                &value,
            )
            .await?;
        store_derived_evidence(&self.request, cache, &segments).await?;
        self._lease.mark_committed();
        Ok(ValidatedSpeakerEvidence::new(segments))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredRawSpeakerEvidence {
    schema_version: u32,
    request_fingerprint: String,
    evidence: SpeakerInferenceEvidenceV2,
}

impl StoredRawSpeakerEvidence {
    fn validate_and_normalize_for(
        &self,
        request: &SpeakerEvidenceRequest,
    ) -> Result<Vec<SpeakerSegmentV2>, SpeakerEvidenceCacheError> {
        if self.schema_version != RAW_SPEAKER_EVIDENCE_SCHEMA_VERSION {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(format!(
                "schema version {} does not match {}",
                self.schema_version, RAW_SPEAKER_EVIDENCE_SCHEMA_VERSION
            )));
        }
        if self.request_fingerprint != request.raw_cache_key.as_str() {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(
                "request fingerprint does not match cache key".to_owned(),
            ));
        }
        validate_evidence_backend(&self.evidence, request.backend)?;
        let segments = normalize_speaker_evidence_v2(&self.evidence)
            .map_err(SpeakerEvidenceCacheError::InvalidCachedEvidence)?;
        validate_segments(&segments)?;
        Ok(segments)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredDerivedSpeakerEvidence {
    schema_version: u32,
    raw_evidence_fingerprint: String,
    normalization_revision: String,
    segments: Vec<SpeakerSegmentV2>,
}

impl StoredDerivedSpeakerEvidence {
    fn validate_for(
        &self,
        request: &SpeakerEvidenceRequest,
    ) -> Result<(), SpeakerEvidenceCacheError> {
        if self.schema_version != DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(format!(
                "derived schema version {} does not match {}",
                self.schema_version, DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION
            )));
        }
        if self.raw_evidence_fingerprint != request.raw_cache_key.as_str() {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(
                "derived raw-evidence fingerprint does not match request".to_owned(),
            ));
        }
        if self.normalization_revision != request.normalization_revision.as_str() {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(
                "derived normalization revision does not match request".to_owned(),
            ));
        }
        validate_segments(&self.segments)
    }
}

async fn store_derived_evidence(
    request: &SpeakerEvidenceRequest,
    cache: &UtteranceCache,
    segments: &[SpeakerSegmentV2],
) -> Result<(), SpeakerEvidenceCacheError> {
    validate_segments(segments)?;
    let envelope = StoredDerivedSpeakerEvidence {
        schema_version: DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION,
        raw_evidence_fingerprint: request.raw_cache_key.to_string(),
        normalization_revision: request.normalization_revision.as_str().to_owned(),
        segments: segments.to_vec(),
    };
    cache
        .put(
            request.derived_cache_key.as_str(),
            tasks::SPEAKER_DIARIZATION_SEGMENTS,
            &request.normalization_revision,
            &serde_json::to_value(envelope)?,
        )
        .await?;
    Ok(())
}

fn validate_evidence_backend(
    evidence: &SpeakerInferenceEvidenceV2,
    backend: SpeakerBackendV2,
) -> Result<(), SpeakerEvidenceCacheError> {
    let matches = matches!(
        (evidence, backend),
        (
            SpeakerInferenceEvidenceV2::PyannoteAi { .. },
            SpeakerBackendV2::PyannoteAi
        ) | (
            SpeakerInferenceEvidenceV2::Pyannote { .. },
            SpeakerBackendV2::Pyannote
        ) | (
            SpeakerInferenceEvidenceV2::Nemo { .. },
            SpeakerBackendV2::Nemo
        )
    );
    if matches {
        Ok(())
    } else {
        Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(format!(
            "speaker evidence provenance does not match requested backend {backend:?}"
        )))
    }
}

fn validate_segments(segments: &[SpeakerSegmentV2]) -> Result<(), SpeakerEvidenceCacheError> {
    let mut previous_start = 0_u64;
    for (index, segment) in segments.iter().enumerate() {
        if segment.speaker.trim().is_empty() {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(format!(
                "segment {index} has an empty speaker label"
            )));
        }
        if segment.end_ms.0 < segment.start_ms.0 {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(format!(
                "segment {index} has an inverted interval {}..{}",
                segment.start_ms.0, segment.end_ms.0
            )));
        }
        if index > 0 && segment.start_ms.0 < previous_start {
            return Err(SpeakerEvidenceCacheError::InvalidCachedEvidence(format!(
                "segment {index} starts before the preceding segment"
            )));
        }
        previous_start = segment.start_ms.0;
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SpeakerEvidenceCacheError {
    #[error("could not read prepared audio for speaker evidence: {0}")]
    Io(#[from] std::io::Error),
    #[error("speaker evidence cache failed: {0}")]
    Cache(#[from] CacheError),
    #[error("speaker evidence key/envelope serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid cached speaker evidence: {0}")]
    InvalidCachedEvidence(String),
    #[error("required speaker evidence is missing for cache key {0}")]
    RequiredEvidenceMissing(CacheKey),
    #[error("speaker source media changed after speaker evidence preparation: {0}")]
    ProviderMediaDrift(PathBuf),
}

#[cfg(test)]
mod tests;
