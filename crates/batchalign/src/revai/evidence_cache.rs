//! Durable raw Rev.AI transcript evidence and paid-call typestate.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::api::{EngineVersion, LanguageCode3, LanguageSpec, NumSpeakers};
use crate::cache::{CacheBackend, CacheError, InferenceLease, UtteranceCache};
use crate::chat_ops::{CacheKey, CacheTaskName};
use crate::error::ServerError;
use crate::params::CachePolicy;

use super::Transcript;
use super::types::{RevTranscriptEvidence, RevTranscriptFidelity};

const REV_ASR_EVIDENCE_STORAGE_SCHEMA_VERSION: u32 = 3;
const REV_ASR_REQUEST_IDENTITY_REVISION: u32 = 2;
const REV_ASR_REQUEST_POLICY_REVISION: &str = "langid-skip-itn-v1";
const REV_ASR_TRACE_SCHEMA_VERSION: u32 = 2;

/// A fetched provider response is retained even when it cannot be admitted.
pub(crate) enum RevAsrInferenceOutcome {
    Completed(CompletedRevAsrEvidence),
    UnresolvedLanguage(RejectedRevLanguageEvidence),
}

#[derive(Debug, Serialize)]
pub(crate) struct RejectedRevLanguageEvidence {
    pub(crate) transcript_evidence: RevTranscriptEvidence,
    pub(crate) requested_language: LanguageSpec,
    pub(crate) effective_language: LanguageSpec,
    pub(crate) detected_language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
struct RevProviderMediaDigest(String);

impl RevProviderMediaDigest {
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(bytes);
        Self(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RevMediaPreparationRecipe {
    /// Preserve source bytes and the historical `audio/mpeg` multipart type.
    SourceBytesLegacyAudioMpegV1,
}

impl RevMediaPreparationRecipe {
    fn upload_mime(self) -> &'static str {
        match self {
            Self::SourceBytesLegacyAudioMpegV1 => "audio/mpeg",
        }
    }

    fn prepare_provider_bytes(self, source_bytes: Vec<u8>) -> Vec<u8> {
        match self {
            Self::SourceBytesLegacyAudioMpegV1 => source_bytes,
        }
    }
}

/// Complete provider-visible presentation used by both request identity and
/// observability.
///
/// Keeping this relationship in one value prevents the cache key and trace
/// from independently reassembling a MIME/name/metadata tuple that could
/// drift while still describing the same provider bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RevProviderPresentation {
    provider_media_blake3: RevProviderMediaDigest,
    preparation_recipe: RevMediaPreparationRecipe,
    upload_file_name: String,
    upload_mime: &'static str,
    upload_metadata: String,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRevProviderMedia {
    source_path: PathBuf,
    source_digest: RevProviderMediaDigest,
    presentation: RevProviderPresentation,
}

impl PreparedRevProviderMedia {
    pub(crate) async fn from_source(path: &Path) -> Result<Self, RevAsrEvidenceCacheError> {
        let bytes = tokio::fs::read(path).await?;
        let source_digest = RevProviderMediaDigest::from_bytes(&bytes);
        let recipe = RevMediaPreparationRecipe::SourceBytesLegacyAudioMpegV1;
        let provider_media_blake3 =
            RevProviderMediaDigest::from_bytes(&recipe.prepare_provider_bytes(bytes));
        let upload_extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .filter(|extension| {
                !extension.is_empty() && extension.bytes().all(|byte| byte.is_ascii_alphanumeric())
            })
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "media".to_owned());
        let upload_file_name = format!("provider-media.{upload_extension}");
        let upload_metadata = format!("batchalign3_{}", &provider_media_blake3.0[..16]);
        Ok(Self {
            source_path: path.to_owned(),
            source_digest,
            presentation: RevProviderPresentation {
                provider_media_blake3,
                preparation_recipe: recipe,
                upload_file_name,
                upload_mime: recipe.upload_mime(),
                upload_metadata,
            },
        })
    }

    pub(super) fn verify(self) -> Result<VerifiedRevProviderMedia, RevAsrEvidenceCacheError> {
        let source_bytes = std::fs::read(&self.source_path)?;
        if RevProviderMediaDigest::from_bytes(&source_bytes) != self.source_digest {
            return Err(RevAsrEvidenceCacheError::ProviderMediaDrift(
                self.source_path,
            ));
        }
        let bytes = self
            .presentation
            .preparation_recipe
            .prepare_provider_bytes(source_bytes);
        if RevProviderMediaDigest::from_bytes(&bytes) != self.presentation.provider_media_blake3 {
            return Err(RevAsrEvidenceCacheError::ProviderPreparationDrift(
                self.source_path,
            ));
        }
        Ok(VerifiedRevProviderMedia {
            bytes,
            upload_file_name: self.presentation.upload_file_name,
            upload_mime: self.presentation.upload_mime,
            metadata: self.presentation.upload_metadata,
        })
    }
}

pub(super) struct VerifiedRevProviderMedia {
    pub(super) bytes: Vec<u8>,
    pub(super) upload_file_name: String,
    pub(super) upload_mime: &'static str,
    pub(super) metadata: String,
}

#[derive(Debug, Serialize)]
struct RevAsrEvidenceKeyMaterial<'a> {
    domain: &'static str,
    #[serde(rename = "schema_version")]
    request_identity_revision: u32,
    #[serde(flatten)]
    provider_presentation: &'a RevProviderPresentation,
    requested_language: String,
    expected_speakers: Option<u32>,
    request_policy_revision: &'static str,
    model_revision: &'a str,
}

/// Provider revision identity distinct from worker ASR engine versions.
#[derive(Debug, Clone)]
pub(crate) struct RevAsrModelRevision(EngineVersion);

impl RevAsrModelRevision {
    pub(crate) fn current() -> Self {
        Self(EngineVersion::from("revai:asynchronous-transcript-v1"))
    }

    fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RevAsrEvidenceRequest {
    cache_key: CacheKey,
    model_revision: RevAsrModelRevision,
    provider_media: PreparedRevProviderMedia,
    requested_language: LanguageSpec,
    expected_speakers: Option<NumSpeakers>,
}

impl RevAsrEvidenceRequest {
    pub(crate) fn new(
        provider_media: PreparedRevProviderMedia,
        requested_language: &LanguageSpec,
        expected_speakers: impl Into<Option<NumSpeakers>>,
        model_revision: &RevAsrModelRevision,
    ) -> Result<Self, RevAsrEvidenceCacheError> {
        let expected_speakers = expected_speakers.into();
        if expected_speakers.is_some_and(|count| count.0 == 0) {
            return Err(RevAsrEvidenceCacheError::InvalidRequest(
                "expected speaker count must be positive or automatic".to_owned(),
            ));
        }
        if requested_language.is_per_file() {
            return Err(RevAsrEvidenceCacheError::InvalidRequest(
                "Rev.AI transcribe evidence cannot use per-file language routing".to_owned(),
            ));
        }
        let material = RevAsrEvidenceKeyMaterial {
            domain: "rev_asr_evidence",
            request_identity_revision: REV_ASR_REQUEST_IDENTITY_REVISION,
            provider_presentation: &provider_media.presentation,
            requested_language: requested_language.to_string(),
            expected_speakers: expected_speakers.map(|count| count.0),
            request_policy_revision: REV_ASR_REQUEST_POLICY_REVISION,
            model_revision: model_revision.as_str(),
        };
        let canonical = serde_json::to_string(&material)?;
        Ok(Self {
            cache_key: CacheKey::from_content(&canonical),
            model_revision: model_revision.clone(),
            provider_media,
            requested_language: requested_language.clone(),
            expected_speakers,
        })
    }

    pub(crate) fn cache_key(&self) -> &CacheKey {
        &self.cache_key
    }

    fn trace_seed(&self) -> RevAsrEvidenceTraceSeed {
        RevAsrEvidenceTraceSeed {
            trace_schema_version: REV_ASR_TRACE_SCHEMA_VERSION,
            source_media_blake3: self.provider_media.source_digest.0.clone(),
            provider_presentation: self.provider_media.presentation.clone(),
            requested_language: self.requested_language.to_string(),
            expected_speakers: self.expected_speakers.map(|count| count.0),
            request_policy_revision: REV_ASR_REQUEST_POLICY_REVISION,
            model_revision: self.model_revision.as_str().to_owned(),
            raw_evidence_key: self.cache_key.to_string(),
        }
    }

    #[cfg(test)]
    pub(crate) async fn from_audio(
        audio_path: &Path,
        requested_language: &LanguageSpec,
        expected_speakers: impl Into<Option<NumSpeakers>>,
        model_revision: &RevAsrModelRevision,
    ) -> Result<Self, RevAsrEvidenceCacheError> {
        Self::new(
            PreparedRevProviderMedia::from_source(audio_path).await?,
            requested_language,
            expected_speakers,
            model_revision,
        )
    }

    async fn lookup(
        &self,
        cache: &UtteranceCache,
        policy: CachePolicy,
    ) -> Result<RevAsrEvidenceLookup, RevAsrEvidenceCacheError> {
        let lease = InferenceLease::acquire(&self.cache_key, cache).await;
        match policy {
            CachePolicy::SkipCache if !lease.observed_commit_while_waiting() => {
                return Ok(RevAsrEvidenceLookup::Miss(Box::new(RevAsrEvidenceMiss {
                    request: self.clone(),
                    reason: RevAsrEvidenceMissReason::ForcedRefresh,
                    _lease: lease,
                })));
            }
            CachePolicy::UseCache | CachePolicy::RequireCache | CachePolicy::SkipCache => {}
        }
        let stored = cache
            .get(
                self.cache_key.as_str(),
                CacheTaskName::RevAsrEvidence.as_str(),
                self.model_revision.as_str(),
            )
            .await?;
        let Some(stored) = stored else {
            match policy {
                CachePolicy::RequireCache => {
                    return Err(RevAsrEvidenceCacheError::RequiredEvidenceMissing(
                        self.cache_key.clone(),
                    ));
                }
                CachePolicy::UseCache | CachePolicy::SkipCache => {}
            }
            return Ok(RevAsrEvidenceLookup::Miss(Box::new(RevAsrEvidenceMiss {
                request: self.clone(),
                reason: RevAsrEvidenceMissReason::NotFound,
                _lease: lease,
            })));
        };
        Ok(RevAsrEvidenceLookup::Hit(StoredRevAsrEvidence::decode_for(
            stored, self,
        )?))
    }

    #[cfg(test)]
    async fn store_unchecked_for_test(
        &self,
        cache: &UtteranceCache,
        value: serde_json::Value,
    ) -> Result<(), RevAsrEvidenceCacheError> {
        cache
            .put(
                self.cache_key.as_str(),
                CacheTaskName::RevAsrEvidence.as_str(),
                self.model_revision.as_str(),
                env!("CARGO_PKG_VERSION"),
                &value,
            )
            .await?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompletedRevAsrEvidence {
    pub(crate) transcript_evidence: RevTranscriptEvidence,
    pub(crate) resolved_language: LanguageCode3,
}

fn validate_evidence(evidence: &CompletedRevAsrEvidence) -> Result<(), RevAsrEvidenceCacheError> {
    for (monologue_index, monologue) in evidence
        .transcript_evidence
        .transcript()
        .monologues
        .iter()
        .enumerate()
    {
        if monologue.speaker < 0 {
            return Err(RevAsrEvidenceCacheError::InvalidEvidence(format!(
                "monologue {monologue_index} has negative speaker index"
            )));
        }
        for (element_index, element) in monologue.elements.iter().enumerate() {
            for (label, value) in [("start", element.ts), ("end", element.end_ts)] {
                if let Some(value) = value
                    && (!value.is_finite() || value < 0.0)
                {
                    return Err(RevAsrEvidenceCacheError::InvalidEvidence(format!(
                        "monologue {monologue_index} element {element_index} has invalid {label}"
                    )));
                }
            }
            if let (Some(start), Some(end)) = (element.ts, element.end_ts)
                && end < start
            {
                return Err(RevAsrEvidenceCacheError::InvalidEvidence(format!(
                    "monologue {monologue_index} element {element_index} ends before it starts"
                )));
            }
            if let Some(confidence) = element.confidence
                && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
            {
                return Err(RevAsrEvidenceCacheError::InvalidEvidence(format!(
                    "monologue {monologue_index} element {element_index} has invalid confidence"
                )));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredRevAsrEvidence {
    schema_version: u32,
    request_fingerprint: String,
    evidence: CompletedRevAsrEvidence,
}

impl StoredRevAsrEvidence {
    fn decode_for(
        value: serde_json::Value,
        request: &RevAsrEvidenceRequest,
    ) -> Result<CompletedRevAsrEvidence, RevAsrEvidenceCacheError> {
        let schema_version = value
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                RevAsrEvidenceCacheError::InvalidEvidence(
                    "stored evidence has no numeric schema version".to_owned(),
                )
            })?;
        let evidence = match schema_version {
            3 => {
                let envelope: Self = serde_json::from_value(value).map_err(|error| {
                    RevAsrEvidenceCacheError::InvalidEvidence(error.to_string())
                })?;
                validate_request_fingerprint(&envelope.request_fingerprint, request)?;
                envelope.evidence
            }
            2 => {
                let envelope: StoredRevAsrEvidenceV2 =
                    serde_json::from_value(value).map_err(|error| {
                        RevAsrEvidenceCacheError::InvalidEvidence(error.to_string())
                    })?;
                validate_request_fingerprint(&envelope.request_fingerprint, request)?;
                CompletedRevAsrEvidence {
                    transcript_evidence: RevTranscriptEvidence::from_legacy_transcript(
                        envelope.evidence.transcript,
                    ),
                    resolved_language: envelope.evidence.resolved_language,
                }
            }
            unsupported => {
                return Err(RevAsrEvidenceCacheError::InvalidEvidence(format!(
                    "unsupported stored evidence schema version {unsupported}"
                )));
            }
        };
        validate_evidence(&evidence)?;
        Ok(evidence)
    }
}

#[derive(Debug, Deserialize)]
struct StoredRevAsrEvidenceV2 {
    request_fingerprint: String,
    evidence: CompletedRevAsrEvidenceV2,
}

#[derive(Debug, Deserialize)]
struct CompletedRevAsrEvidenceV2 {
    transcript: Transcript,
    resolved_language: LanguageCode3,
}

fn validate_request_fingerprint(
    request_fingerprint: &str,
    request: &RevAsrEvidenceRequest,
) -> Result<(), RevAsrEvidenceCacheError> {
    if request_fingerprint != request.cache_key.as_str() {
        return Err(RevAsrEvidenceCacheError::InvalidEvidence(
            "request fingerprint does not match cache key".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RevAsrEvidenceMissReason {
    NotFound,
    ForcedRefresh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RevAsrEvidenceCacheOutcome {
    Replayed,
    InferredNotFound,
    InferredForcedRefresh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum RevAsrProjectionRevision {
    #[serde(rename = "rev-transcript-to-asr-response-v1")]
    AsrResponseV1,
    #[serde(rename = "rev-transcript-to-utr-asr-response-v1")]
    UtrAsrResponseV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct RevAsrEvidenceTrace {
    #[serde(flatten)]
    request: RevAsrEvidenceTraceSeed,
    cache_outcome: RevAsrEvidenceCacheOutcome,
    transcript_fidelity: RevTranscriptFidelity,
    projection_revision: RevAsrProjectionRevision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RevAsrEvidenceTraceSeed {
    trace_schema_version: u32,
    source_media_blake3: String,
    #[serde(flatten)]
    provider_presentation: RevProviderPresentation,
    requested_language: String,
    expected_speakers: Option<u32>,
    request_policy_revision: &'static str,
    model_revision: String,
    raw_evidence_key: String,
}

impl RevAsrEvidenceTraceSeed {
    fn finish(
        self,
        cache_outcome: RevAsrEvidenceCacheOutcome,
        transcript_fidelity: RevTranscriptFidelity,
        projection_revision: RevAsrProjectionRevision,
    ) -> RevAsrEvidenceTrace {
        RevAsrEvidenceTrace {
            request: self,
            cache_outcome,
            transcript_fidelity,
            projection_revision,
        }
    }
}

/// Stable meaning of a Rev projection, excluding only how its evidence was
/// obtained for this run.
///
/// A cold inference and a durable replay should compare equal here while
/// retaining distinct causal receipts in [`RevAsrEvidenceTrace`]. Borrowing
/// the request identity prevents tests and callers from manufacturing a second,
/// subtly divergent trace-shaped record.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RevAsrEvidenceSemanticProjection<'a> {
    request: &'a RevAsrEvidenceTraceSeed,
    transcript_fidelity: RevTranscriptFidelity,
    projection_revision: RevAsrProjectionRevision,
}

impl RevAsrEvidenceTrace {
    #[cfg(test)]
    pub(crate) fn cache_outcome(&self) -> RevAsrEvidenceCacheOutcome {
        self.cache_outcome
    }

    #[cfg(test)]
    pub(crate) fn semantic_projection(&self) -> RevAsrEvidenceSemanticProjection<'_> {
        RevAsrEvidenceSemanticProjection {
            request: &self.request,
            transcript_fidelity: self.transcript_fidelity,
            projection_revision: self.projection_revision,
        }
    }
}

#[derive(Debug)]
enum RevAsrEvidenceLookup {
    Hit(CompletedRevAsrEvidence),
    Miss(Box<RevAsrEvidenceMiss>),
}

#[derive(Debug)]
struct RevAsrEvidenceMiss {
    request: RevAsrEvidenceRequest,
    reason: RevAsrEvidenceMissReason,
    _lease: InferenceLease,
}

impl RevAsrEvidenceMiss {
    fn authorize(self) -> RevAsrInferenceAuthorization {
        RevAsrInferenceAuthorization {
            request: self.request,
            reason: self.reason,
            _lease: self._lease,
        }
    }
}

#[derive(Debug)]
struct RevAsrInferenceAuthorization {
    request: RevAsrEvidenceRequest,
    reason: RevAsrEvidenceMissReason,
    _lease: InferenceLease,
}

pub(crate) struct AuthorizedRevEvidenceRun {
    pub(super) provider_media: PreparedRevProviderMedia,
    pub(super) requested_language: LanguageSpec,
    pub(super) expected_speakers: Option<NumSpeakers>,
}

impl RevAsrInferenceAuthorization {
    fn into_run(self) -> (AuthorizedRevEvidenceRun, RevAsrEvidenceCommitPermit) {
        let Self {
            request,
            reason,
            _lease,
        } = self;
        let RevAsrEvidenceRequest {
            cache_key,
            model_revision,
            provider_media,
            requested_language,
            expected_speakers,
        } = request;
        (
            AuthorizedRevEvidenceRun {
                provider_media,
                requested_language,
                expected_speakers,
            },
            RevAsrEvidenceCommitPermit {
                cache_key,
                model_revision,
                reason,
                _lease,
            },
        )
    }
}

struct RevAsrEvidenceCommitPermit {
    cache_key: CacheKey,
    model_revision: RevAsrModelRevision,
    reason: RevAsrEvidenceMissReason,
    _lease: InferenceLease,
}

impl RevAsrEvidenceCommitPermit {
    async fn retain_rejected_language(
        self,
        cache: &UtteranceCache,
        request: &RevAsrEvidenceTraceSeed,
        rejected: RejectedRevLanguageEvidence,
    ) -> Result<String, RevAsrEvidenceCacheError> {
        let envelope = serde_json::json!({
            "diagnostic_schema_version": 1,
            "admission": "rejected_unresolved_language",
            "request": request,
            "response": rejected,
        });
        // Distinct responses cannot overwrite each other, and normal evidence
        // lookup never reads this namespace. Do not mark the success lease.
        let bytes = serde_json::to_vec(&envelope)?;
        let key = format!("rev-asr-rejected-language-{}", blake3::hash(&bytes).to_hex());
        cache.put(&key, CacheTaskName::RevAsrEvidence.as_str(), self.model_revision.as_str(),
            env!("CARGO_PKG_VERSION"), &envelope).await.map_err(|source| RevAsrEvidenceCacheError::RejectedEvidencePersistence {
                diagnostic_key: key.clone(), source,
            })?;
        Ok(key)
    }

    async fn commit(
        self,
        cache: &UtteranceCache,
        evidence: CompletedRevAsrEvidence,
    ) -> Result<CompletedRevAsrEvidence, RevAsrEvidenceCacheError> {
        validate_evidence(&evidence)?;
        let envelope = StoredRevAsrEvidence {
            schema_version: REV_ASR_EVIDENCE_STORAGE_SCHEMA_VERSION,
            request_fingerprint: self.cache_key.to_string(),
            evidence,
        };
        let value = serde_json::to_value(&envelope)?;
        cache
            .put(
                self.cache_key.as_str(),
                CacheTaskName::RevAsrEvidence.as_str(),
                self.model_revision.as_str(),
                env!("CARGO_PKG_VERSION"),
                &value,
            )
            .await?;
        self._lease.mark_committed();
        Ok(envelope.evidence)
    }
}

#[async_trait::async_trait]
pub(crate) trait RevAsrEvidenceInference: Sync {
    async fn infer(
        &self,
        run: AuthorizedRevEvidenceRun,
    ) -> Result<RevAsrInferenceOutcome, ServerError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RevAsrEvidenceSource {
    Replayed,
    Inferred(RevAsrEvidenceMissReason),
}

#[derive(Debug)]
pub(crate) struct RevAsrEvidenceResolution {
    evidence: CompletedRevAsrEvidence,
    source: RevAsrEvidenceSource,
    trace_seed: RevAsrEvidenceTraceSeed,
}

impl RevAsrEvidenceResolution {
    fn cache_outcome(&self) -> RevAsrEvidenceCacheOutcome {
        match self.source {
            RevAsrEvidenceSource::Replayed => RevAsrEvidenceCacheOutcome::Replayed,
            RevAsrEvidenceSource::Inferred(RevAsrEvidenceMissReason::NotFound) => {
                RevAsrEvidenceCacheOutcome::InferredNotFound
            }
            RevAsrEvidenceSource::Inferred(RevAsrEvidenceMissReason::ForcedRefresh) => {
                RevAsrEvidenceCacheOutcome::InferredForcedRefresh
            }
        }
    }

    pub(crate) fn trace(
        &self,
        projection_revision: RevAsrProjectionRevision,
    ) -> RevAsrEvidenceTrace {
        self.trace_seed.clone().finish(
            self.cache_outcome(),
            self.evidence.transcript_evidence.fidelity(),
            projection_revision,
        )
    }

    pub(crate) fn source(&self) -> RevAsrEvidenceSource {
        self.source
    }

    pub(crate) fn into_evidence(self) -> CompletedRevAsrEvidence {
        self.evidence
    }

    #[cfg(test)]
    pub(crate) fn replayed_for_test(
        request: &RevAsrEvidenceRequest,
        evidence: CompletedRevAsrEvidence,
    ) -> Self {
        Self {
            evidence,
            source: RevAsrEvidenceSource::Replayed,
            trace_seed: request.trace_seed(),
        }
    }
}

pub(crate) async fn resolve_rev_asr_evidence<I: RevAsrEvidenceInference + ?Sized>(
    request: &RevAsrEvidenceRequest,
    cache: &UtteranceCache,
    policy: CachePolicy,
    inference: &I,
) -> Result<RevAsrEvidenceResolution, RevAsrEvidenceResolutionError> {
    let trace_seed = request.trace_seed();
    match request.lookup(cache, policy).await? {
        RevAsrEvidenceLookup::Hit(evidence) => Ok(RevAsrEvidenceResolution {
            evidence,
            source: RevAsrEvidenceSource::Replayed,
            trace_seed,
        }),
        RevAsrEvidenceLookup::Miss(miss) => {
            let authorization = (*miss).authorize();
            let (run, permit) = authorization.into_run();
            let reason = permit.reason;
            let evidence = match inference.infer(run).await? {
                RevAsrInferenceOutcome::Completed(evidence) => evidence,
                RevAsrInferenceOutcome::UnresolvedLanguage(rejected) => {
                    let key = permit.retain_rejected_language(cache, &trace_seed, rejected).await?;
                    return Err(ServerError::Validation(format!(
                        "Rev.AI returned no usable detected language. Raw response retained as diagnostic {key}; not admitted for replay. Supply an explicitly verified language before further processing; no English default was applied."
                    )).into());
                }
            };
            let evidence = permit.commit(cache, evidence).await?;
            Ok(RevAsrEvidenceResolution {
                evidence,
                source: RevAsrEvidenceSource::Inferred(reason),
                trace_seed,
            })
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum RevAsrEvidenceResolutionError {
    #[error(transparent)]
    Evidence(#[from] RevAsrEvidenceCacheError),
    #[error(transparent)]
    Inference(#[from] ServerError),
}

/// Preserve cache-only precondition failures as typed, actionable server
/// errors instead of flattening them into an internal persistence failure.
pub(crate) fn rev_asr_resolution_error_to_server_error(
    error: RevAsrEvidenceResolutionError,
) -> ServerError {
    match error {
        RevAsrEvidenceResolutionError::Evidence(
            RevAsrEvidenceCacheError::RequiredEvidenceMissing(cache_key),
        ) => {
            ServerError::RequiredEvidenceUnavailable(crate::error::MissingRequiredEvidence::RevAsr(
                crate::error::MissingRevAsrEvidence::new(cache_key),
            ))
        }
        RevAsrEvidenceResolutionError::Evidence(error) => {
            ServerError::Persistence(error.to_string())
        }
        RevAsrEvidenceResolutionError::Inference(error) => error,
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum RevAsrEvidenceCacheError {
    #[error("failed to retain rejected Rev.AI response at diagnostic key {diagnostic_key}: {source}")]
    RejectedEvidencePersistence { diagnostic_key: String, source: CacheError },
    #[error("could not read Rev.AI provider media: {0}")]
    Io(#[from] std::io::Error),
    #[error("Rev.AI evidence cache failed: {0}")]
    Cache(#[from] CacheError),
    #[error("Rev.AI evidence serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Rev.AI evidence request: {0}")]
    InvalidRequest(String),
    #[error("Rev.AI provider media changed after preparation: {0}")]
    ProviderMediaDrift(PathBuf),
    #[error("Rev.AI provider-media recipe produced different bytes after preparation: {0}")]
    ProviderPreparationDrift(PathBuf),
    #[error("invalid cached Rev.AI evidence: {0}")]
    InvalidEvidence(String),
    #[error("required Rev.AI evidence is missing for cache key {0}")]
    RequiredEvidenceMissing(CacheKey),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LanguageCode3;
    use crate::cache::CacheStats;
    use crate::revai::types::{Element, Monologue};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sample_evidence() -> CompletedRevAsrEvidence {
        let transcript = Transcript {
            monologues: vec![Monologue {
                speaker: 0,
                elements: vec![Element {
                    element_type: "text".to_owned(),
                    value: "hello".to_owned(),
                    ts: Some(0.1),
                    end_ts: Some(0.5),
                    confidence: Some(0.9),
                }],
            }],
        };
        CompletedRevAsrEvidence {
            transcript_evidence: RevTranscriptEvidence::from_provider_json(
                serde_json::to_string(&transcript).expect("fixture transcript JSON"),
            )
            .expect("fixture provider transcript"),
            resolved_language: LanguageCode3::eng(),
        }
    }

    struct CountingRevService {
        calls: AtomicUsize,
        delay_ms: u64,
    }

    struct FailingCommitBackend;

    #[derive(Clone, Default)]
    struct RecordingBackend(std::sync::Arc<std::sync::Mutex<HashMap<String, serde_json::Value>>>);

    #[async_trait::async_trait]
    impl CacheBackend for RecordingBackend {
        async fn get(&self, key: &str, _task: &str, _engine_version: &str) -> Result<Option<serde_json::Value>, CacheError> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        async fn get_batch(&self, _keys: &[String], _task: &str, _engine_version: &str) -> Result<HashMap<String, serde_json::Value>, CacheError> { unreachable!("single entry boundary") }
        async fn put(&self, key: &str, _task: &str, _engine_version: &str, _ba_version: &str, data: &serde_json::Value) -> Result<(), CacheError> {
            self.0.lock().unwrap().insert(key.to_owned(), data.clone());
            Ok(())
        }
        async fn put_batch(&self, _entries: &[(String, serde_json::Value)], _task: &str, _engine_version: &str, _ba_version: &str) -> Result<(), CacheError> { unreachable!("single entry boundary") }
        async fn delete_batch(&self, _keys: &[String], _task: &str) -> Result<usize, CacheError> { unreachable!("no deletion") }
        async fn stats(&self) -> Result<CacheStats, CacheError> { unreachable!("no statistics") }
    }

    struct LanguageResponseService {
        raw: &'static str,
        detected: Option<&'static str>,
        effective: LanguageSpec,
    }

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for LanguageResponseService {
        async fn infer(&self, run: AuthorizedRevEvidenceRun) -> Result<RevAsrInferenceOutcome, ServerError> {
            Ok(super::super::asr::classify_language_response(
                RevTranscriptEvidence::from_provider_json(self.raw.to_owned()).unwrap(),
                run.requested_language, self.effective.clone(), self.detected.map(str::to_owned),
            ))
        }
    }

    fn language_request() -> RevAsrEvidenceRequest {
        let digest = RevProviderMediaDigest::from_bytes(b"provider media");
        let media = PreparedRevProviderMedia {
            source_path: PathBuf::from("fixture.wav"), source_digest: digest.clone(),
            presentation: RevProviderPresentation {
                provider_media_blake3: digest,
                preparation_recipe: RevMediaPreparationRecipe::SourceBytesLegacyAudioMpegV1,
                upload_file_name: "fixture.wav".to_owned(), upload_mime: "audio/mpeg", upload_metadata: "fixture".to_owned(),
            },
        };
        RevAsrEvidenceRequest::new(media, &LanguageSpec::Auto, None, &RevAsrModelRevision::current()).unwrap()
    }

    #[tokio::test]
    async fn rejected_language_response_is_exact_retained_and_never_replayed() {
        let backend = RecordingBackend::default();
        let cache = UtteranceCache::from_backend(Box::new(backend.clone()));
        let request = language_request();
        let empty = "{ \"monologues\":[], \"future\":{\"value\":17} }";
        let spoken = r#"{"monologues":[{"speaker":0,"elements":[{"type":"text","value":"hello","ts":0.1,"end_ts":0.2}]}],"future":[1,2]}"#;
        for (detected, raw) in [(None, empty), (Some("unmapped-provider-language"), spoken)] {
            let service = LanguageResponseService { raw, detected, effective: LanguageSpec::Auto };
            let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::SkipCache, &service).await.unwrap_err();
            let message = error.to_string();
            assert!(message.contains("rev-asr-rejected-language-"));
            assert!(matches!(error, RevAsrEvidenceResolutionError::Inference(ServerError::Validation(_))));
            let records = backend.0.lock().unwrap();
            assert!(!records.contains_key(request.cache_key.as_str()));
            let (key, diagnostic) = records.iter().find(|(_, value)| value["response"]["detected_language"] == serde_json::to_value(detected).unwrap()).unwrap();
            assert!(message.contains(key));
            assert_eq!(diagnostic["request"], serde_json::to_value(request.trace_seed()).unwrap());
            assert_eq!(diagnostic["response"]["transcript_evidence"]["raw_json"], raw);
            assert_eq!(diagnostic["response"]["requested_language"], serde_json::to_value(LanguageSpec::Auto).unwrap());
            assert_eq!(diagnostic["response"]["effective_language"], serde_json::to_value(LanguageSpec::Auto).unwrap());
            assert!(StoredRevAsrEvidence::decode_for(diagnostic.clone(), &request).is_err());
        }
        assert_eq!(backend.0.lock().unwrap().len(), 2);
        let service = LanguageResponseService { raw: empty, detected: None, effective: LanguageSpec::Auto };
        assert!(matches!(resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service).await,
            Err(RevAsrEvidenceResolutionError::Evidence(RevAsrEvidenceCacheError::RequiredEvidenceMissing(_)))));
        // Auto -> successfully identified language remains admitted, even if
        // transcription omitted its own detected-language field.
        let service = LanguageResponseService { raw: spoken, detected: None, effective: LanguageSpec::Resolved(LanguageCode3::eng()) };
        resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service).await.unwrap();
        let replay = resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service).await.unwrap();
        assert_eq!(replay.source(), RevAsrEvidenceSource::Replayed);
        assert_eq!(replay.evidence.transcript_evidence.exact_provider_json(), Some(spoken));
    }

    #[tokio::test]
    async fn rejected_language_response_persistence_failure_is_not_masked() {
        let cache = UtteranceCache::from_backend(Box::new(FailingCommitBackend));
        let request = language_request();
        let service = LanguageResponseService { raw: r#"{"monologues":[]}"#, detected: None, effective: LanguageSpec::Auto };
        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service).await.unwrap_err();
        assert!(error.to_string().contains("rev-asr-rejected-language-"));
        assert!(error.to_string().contains("injected durable commit failure"));
        assert!(matches!(rev_asr_resolution_error_to_server_error(error), ServerError::Persistence(_)));
    }

    #[async_trait::async_trait]
    impl CacheBackend for FailingCommitBackend {
        async fn get(
            &self,
            _key: &str,
            _task: &str,
            _engine_version: &str,
        ) -> Result<Option<serde_json::Value>, CacheError> {
            Ok(None)
        }

        async fn get_batch(
            &self,
            _keys: &[String],
            _task: &str,
            _engine_version: &str,
        ) -> Result<HashMap<String, serde_json::Value>, CacheError> {
            Ok(HashMap::new())
        }

        async fn put(
            &self,
            _key: &str,
            _task: &str,
            _engine_version: &str,
            _ba_version: &str,
            _data: &serde_json::Value,
        ) -> Result<(), CacheError> {
            Err(CacheError::Io(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "injected durable commit failure",
            )))
        }

        async fn put_batch(
            &self,
            _entries: &[(String, serde_json::Value)],
            _task: &str,
            _engine_version: &str,
            _ba_version: &str,
        ) -> Result<(), CacheError> {
            unreachable!("evidence commits are single-entry writes")
        }

        async fn delete_batch(&self, _keys: &[String], _task: &str) -> Result<usize, CacheError> {
            unreachable!("evidence resolution does not delete cache entries")
        }

        async fn stats(&self) -> Result<CacheStats, CacheError> {
            unreachable!("evidence resolution does not request cache statistics")
        }
    }

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for CountingRevService {
        async fn infer(
            &self,
            _run: AuthorizedRevEvidenceRun,
        ) -> Result<RevAsrInferenceOutcome, ServerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            Ok(RevAsrInferenceOutcome::Completed(sample_evidence()))
        }
    }

    #[test]
    fn auto_speakers_cache_identity_is_distinct_and_trace_records_null() {
        // Request identity admission is pure: no media upload or temporary files.
        let digest = RevProviderMediaDigest::from_bytes(b"provider media");
        let media = PreparedRevProviderMedia {
            source_path: PathBuf::from("fixture.wav"),
            source_digest: digest.clone(),
            presentation: RevProviderPresentation {
                provider_media_blake3: digest,
                preparation_recipe: RevMediaPreparationRecipe::SourceBytesLegacyAudioMpegV1,
                upload_file_name: "fixture.wav".to_owned(),
                upload_mime: "audio/mpeg",
                upload_metadata: "fixture".to_owned(),
            },
        };
        let language = LanguageSpec::Resolved(LanguageCode3::eng());
        let revision = RevAsrModelRevision::current();
        let automatic = RevAsrEvidenceRequest::new(media.clone(), &language, None, &revision).expect("automatic request");
        let exact = RevAsrEvidenceRequest::new(media.clone(), &language, NumSpeakers(2), &revision).expect("exact request");
        assert_ne!(automatic.cache_key(), exact.cache_key());
        assert!(serde_json::to_value(automatic.trace_seed()).expect("trace")["expected_speakers"].is_null());
        assert_eq!(serde_json::to_value(exact.trace_seed()).expect("trace")["expected_speakers"], 2);
        assert!(RevAsrEvidenceRequest::new(media, &language, NumSpeakers(0), &revision).is_err());
    }

    #[tokio::test]
    async fn rev_key_reuses_identical_media_but_invalidates_semantic_changes() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let first = tempdir.path().join("first.wav");
        let renamed = tempdir.path().join("renamed.wav");
        let renamed_mp3 = tempdir.path().join("renamed.mp3");
        tokio::fs::write(&first, b"provider media")
            .await
            .expect("write first");
        tokio::fs::write(&renamed, b"provider media")
            .await
            .expect("write renamed");
        tokio::fs::write(&renamed_mp3, b"provider media")
            .await
            .expect("write renamed mp3");
        let revision = RevAsrModelRevision::current();

        let baseline = RevAsrEvidenceRequest::from_audio(
            &first,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &revision,
        )
        .await
        .expect("baseline request");
        let copied = RevAsrEvidenceRequest::from_audio(
            &renamed,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &revision,
        )
        .await
        .expect("copied request");
        let different_presentation = RevAsrEvidenceRequest::from_audio(
            &renamed_mp3,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &revision,
        )
        .await
        .expect("different presentation request");
        let auto_language = RevAsrEvidenceRequest::from_audio(
            &first,
            &LanguageSpec::Auto,
            NumSpeakers(2),
            &revision,
        )
        .await
        .expect("auto-language request");
        let three_speakers = RevAsrEvidenceRequest::from_audio(
            &first,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(3),
            &revision,
        )
        .await
        .expect("three-speaker request");
        tokio::fs::write(&renamed, b"changed provider media")
            .await
            .expect("change copied media");
        let changed_media = RevAsrEvidenceRequest::from_audio(
            &renamed,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &revision,
        )
        .await
        .expect("changed-media request");

        assert_eq!(baseline.cache_key(), copied.cache_key());
        assert_eq!(
            baseline.cache_key().as_str(),
            "c3590a1d2a28ea2f9bacb3b781bc798a8ba02c69483134f9f24227eb12f1ed23"
        );
        assert_ne!(baseline.cache_key(), different_presentation.cache_key());
        assert_ne!(baseline.cache_key(), auto_language.cache_key());
        assert_ne!(baseline.cache_key(), three_speakers.cache_key());
        assert_ne!(baseline.cache_key(), changed_media.cache_key());
    }

    #[tokio::test]
    async fn storage_upgrade_preserves_request_key_and_replays_schema_two_evidence() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");

        let key_material = RevAsrEvidenceKeyMaterial {
            domain: "rev_asr_evidence",
            request_identity_revision: REV_ASR_REQUEST_IDENTITY_REVISION,
            provider_presentation: &request.provider_media.presentation,
            requested_language: request.requested_language.to_string(),
            expected_speakers: request.expected_speakers.map(|count| count.0),
            request_policy_revision: REV_ASR_REQUEST_POLICY_REVISION,
            model_revision: request.model_revision.as_str(),
        };
        let key_json = serde_json::to_value(key_material).expect("key material JSON");
        assert_eq!(key_json["schema_version"], 2);
        assert!(key_json.get("request_identity_revision").is_none());

        request
            .store_unchecked_for_test(
                &cache,
                serde_json::json!({
                    "schema_version": 2,
                    "request_fingerprint": request.cache_key().as_str(),
                    "evidence": {
                        "transcript": {
                            "monologues": [{
                                "speaker": 0,
                                "elements": [{
                                    "type": "text",
                                    "value": "legacy",
                                    "ts": 0.1,
                                    "end_ts": 0.5,
                                    "confidence": 0.9
                                }]
                            }]
                        },
                        "resolved_language": "eng"
                    }
                }),
            )
            .await
            .expect("seed schema-two evidence");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };

        let replay =
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service)
                .await
                .expect("schema-two evidence should remain replayable");
        let trace = serde_json::to_value(replay.trace(RevAsrProjectionRevision::AsrResponseV1))
            .expect("trace JSON");

        assert_eq!(trace["cache_outcome"], "replayed");
        assert_eq!(trace["transcript_fidelity"], "legacy_typed_projection");
        assert_eq!(service.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn prepared_media_refuses_bytes_changed_before_authorized_submission() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.mp3");
        tokio::fs::write(&audio, b"prepared bytes")
            .await
            .expect("write audio");
        let prepared = PreparedRevProviderMedia::from_source(&audio)
            .await
            .expect("prepare");
        tokio::fs::write(&audio, b"changed after preparation")
            .await
            .expect("change audio");

        assert!(matches!(
            prepared.verify(),
            Err(RevAsrEvidenceCacheError::ProviderMediaDrift(path)) if path == audio
        ));
    }

    #[tokio::test]
    async fn verified_media_has_stable_provider_visible_presentation() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("Original.WAV");
        let bytes = b"provider media";
        tokio::fs::write(&audio, bytes).await.expect("write audio");

        let verified = PreparedRevProviderMedia::from_source(&audio)
            .await
            .expect("prepare")
            .verify()
            .expect("verify");
        let digest = RevProviderMediaDigest::from_bytes(bytes);

        assert_eq!(verified.bytes, bytes);
        assert_eq!(verified.upload_file_name, "provider-media.wav");
        assert_eq!(verified.upload_mime, "audio/mpeg");
        assert_eq!(
            verified.metadata,
            format!("batchalign3_{}", &digest.0[..16])
        );
    }

    #[tokio::test]
    async fn evidence_trace_joins_media_request_cache_and_projection_identity() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("Original.WAV");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");

        let resolution = RevAsrEvidenceResolution {
            evidence: sample_evidence(),
            source: RevAsrEvidenceSource::Inferred(RevAsrEvidenceMissReason::NotFound),
            trace_seed: request.trace_seed(),
        };
        let value = serde_json::to_value(resolution.trace(RevAsrProjectionRevision::AsrResponseV1))
            .expect("trace JSON");

        assert_eq!(value["trace_schema_version"], 2);
        assert_eq!(
            value["preparation_recipe"],
            "source_bytes_legacy_audio_mpeg_v1"
        );
        assert_eq!(value["source_media_blake3"], value["provider_media_blake3"]);
        assert_eq!(value["upload_file_name"], "provider-media.wav");
        assert_eq!(value["upload_mime"], "audio/mpeg");
        assert!(
            value["upload_metadata"]
                .as_str()
                .expect("metadata string")
                .starts_with("batchalign3_")
        );
        assert_eq!(value["requested_language"], "eng");
        assert_eq!(value["expected_speakers"], 2);
        assert_eq!(value["cache_outcome"], "inferred_not_found");
        assert_eq!(value["transcript_fidelity"], "exact_provider_json");
        assert_eq!(value["raw_evidence_key"], request.cache_key().as_str());
        assert_eq!(
            value["projection_revision"],
            "rev-transcript-to-asr-response-v1"
        );
        assert!(
            !value
                .to_string()
                .contains(tempdir.path().to_string_lossy().as_ref()),
            "trace identity should not leak a machine-local source path"
        );
    }

    #[tokio::test]
    async fn durable_replay_does_not_cross_rev_service_again() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache_dir = tempdir.path().join("cache");
        let cache = UtteranceCache::sqlite(Some(cache_dir.clone()))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let cold = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("cold");
        assert!(matches!(cold.source(), RevAsrEvidenceSource::Inferred(_)));
        let cold_evidence = cold.into_evidence();
        let cold_response = crate::revai::rev_evidence_to_asr_response(&cold_evidence);
        drop(cache);

        let reopened = UtteranceCache::sqlite(Some(cache_dir))
            .await
            .expect("reopen");
        let warm = resolve_rev_asr_evidence(&request, &reopened, CachePolicy::UseCache, &service)
            .await
            .expect("warm");
        assert_eq!(warm.source(), RevAsrEvidenceSource::Replayed);
        let warm_evidence = warm.into_evidence();
        let warm_response = crate::revai::rev_evidence_to_asr_response(&warm_evidence);
        assert_eq!(
            serde_json::to_value(cold_response).expect("cold response JSON"),
            serde_json::to_value(warm_response).expect("warm response JSON")
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn required_cache_refuses_cold_rev_service_call() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };

        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service)
            .await
            .expect_err("a cold required-cache lookup must refuse Rev.AI inference");

        assert!(error.to_string().contains("required Rev.AI evidence"));
        let server_error = rev_asr_resolution_error_to_server_error(error);
        let ServerError::RequiredEvidenceUnavailable(
            crate::error::MissingRequiredEvidence::RevAsr(missing),
        ) = server_error
        else {
            panic!("required Rev evidence must remain an actionable precondition refusal");
        };
        assert_eq!(missing.cache_key(), request.cache_key());
        assert_eq!(service.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn required_cache_replays_warm_rev_evidence() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("seed evidence");

        let replay =
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service)
                .await
                .expect("required-cache replay");

        assert_eq!(replay.source(), RevAsrEvidenceSource::Replayed);
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn concurrent_rev_resolvers_cross_service_once() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 25,
        };

        let (first, second) = tokio::join!(
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service),
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service),
        );
        let first = first.expect("first resolution");
        let second = second.expect("second resolution");
        assert!(matches!(
            (first.source(), second.source()),
            (
                RevAsrEvidenceSource::Inferred(_),
                RevAsrEvidenceSource::Replayed
            ) | (
                RevAsrEvidenceSource::Replayed,
                RevAsrEvidenceSource::Inferred(_)
            )
        ));
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn concurrent_forced_refreshes_share_one_fresh_service_result() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 25,
        };

        let (first, second) = tokio::join!(
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::SkipCache, &service),
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::SkipCache, &service),
        );
        let first = first.expect("first refresh");
        let second = second.expect("second refresh");

        assert!(matches!(
            (first.source(), second.source()),
            (
                RevAsrEvidenceSource::Inferred(_),
                RevAsrEvidenceSource::Replayed
            ) | (
                RevAsrEvidenceSource::Replayed,
                RevAsrEvidenceSource::Inferred(_)
            )
        ));
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn corrupt_rev_evidence_fails_without_service_call() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        request
            .store_unchecked_for_test(&cache, serde_json::json!({"evidence": "broken"}))
            .await
            .expect("seed");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect_err("corruption must fail");
        assert!(error.to_string().contains("invalid cached Rev.AI evidence"));
        assert_eq!(service.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn successful_inference_with_failed_commit_is_not_reported_as_resolved() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let cache = UtteranceCache::from_backend(Box::new(FailingCommitBackend));
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };

        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect_err("commit failure must fail the resolution");

        assert!(
            error
                .to_string()
                .contains("injected durable commit failure")
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }
}
