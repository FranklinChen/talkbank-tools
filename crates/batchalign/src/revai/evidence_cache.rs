//! Durable raw Rev.AI transcript evidence and paid-call typestate.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::api::{LanguageCode3, NumSpeakers, RevAiJobId, TranscriptLanguage};
use crate::cache::{CacheError, CacheNamespace, InferenceLease, UtteranceCache, tasks};
use crate::chat_ops::CacheKey;
use crate::error::ServerError;
use crate::media::probe::{MediaContainer, MediaProbe, ProbeError};
use crate::params::CachePolicy;
use crate::types::revai_language::RevLanguage;

use super::Transcript;
use super::poll::{UnfinishedCause, UnfinishedRevJob};
use super::types::{RevTranscriptEvidence, RevTranscriptFidelity};

mod inspection;
use inspection::AdmittedRevReplayRequest;
pub(crate) use inspection::{inspect_retained_rev_cache, inspect_retained_utr_cache};

const REV_ASR_EVIDENCE_STORAGE_SCHEMA_VERSION: u32 = 3;
const REV_ASR_REQUEST_IDENTITY_REVISION: u32 = 2;
const REV_ASR_REQUEST_POLICY_REVISION: &str = "langid-skip-itn-v1";
const REV_ASR_TRACE_SCHEMA_VERSION: u32 = 2;
const REV_ASR_PENDING_JOB_SCHEMA_VERSION: u32 = 1;

/// A fetched provider response is retained even when it cannot be admitted,
/// and a job the wait did not see finish is retained by name.
pub(crate) enum RevAsrInferenceOutcome {
    Fetched(FetchedRevAsrEvidence),
    UnresolvedLanguage(RejectedRevLanguageEvidence),
    /// The wait ended (deadline, or transient poll budget) with the job still
    /// Rev.AI's and paid for. Recorded against the request so the next run
    /// resumes the wait instead of uploading again.
    Unfinished(UnfinishedRevSubmission),
}

/// What a fetched transcript classifies as: exactly the two things it can be.
/// The wider [`RevAsrInferenceOutcome`] also holds an unfinished job, which a
/// transcript in hand can never be, so classification cannot return it.
pub(crate) enum ClassifiedRevTranscript {
    /// The transcript's language resolved; the evidence is admissible.
    Admitted(FetchedRevAsrEvidence),
    /// Rev.AI named no language BA3 maps; the response is retained, not
    /// admitted.
    UnresolvedLanguage(RejectedRevLanguageEvidence),
}

impl From<ClassifiedRevTranscript> for RevAsrInferenceOutcome {
    /// A classified transcript is an outcome. The third outcome, an unfinished
    /// job, has no transcript to classify and so no route through here.
    fn from(classified: ClassifiedRevTranscript) -> Self {
        match classified {
            ClassifiedRevTranscript::Admitted(fetched) => Self::Fetched(fetched),
            ClassifiedRevTranscript::UnresolvedLanguage(rejected) => {
                Self::UnresolvedLanguage(rejected)
            }
        }
    }
}

/// An unfinished job together with the language it was submitted under, which
/// a resumed wait needs in order to classify the transcript it finally gets
/// (under detection, language identification may have chosen it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnfinishedRevSubmission {
    pub(crate) job: UnfinishedRevJob,
    pub(crate) effective_language: RevLanguage,
}

/// How an authorized run reaches Rev.AI's transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RevJobRoute {
    /// Upload the media and wait for the new job.
    Submit,
    /// An earlier run of this request submitted a job and left it unfinished:
    /// wait for that job, upload nothing.
    Resume(PendingRevJob),
}

/// A paid job an earlier run left unfinished, as the cache recorded it.
///
/// Built only by [`PendingRevJob::from_stored`], which admits the record
/// against the request it is read for, so a run cannot resume another
/// request's job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PendingRevJob {
    pub(crate) job: RevAiJobId,
    pub(crate) effective_language: RevLanguage,
}

/// The durable record of an unfinished job: schema, the request it belongs
/// to, the job, the language it was submitted under, and why the wait ended.
#[derive(Debug, Serialize, Deserialize)]
struct StoredPendingRevJob {
    pending_schema_version: u32,
    request_fingerprint: String,
    job: RevAiJobId,
    effective_language: String,
    cause: UnfinishedCause,
}

/// Where a request's pending-job record lives: beside its evidence, under a
/// key no evidence lookup reads.
fn pending_job_key(cache_key: &CacheKey) -> String {
    format!("rev-asr-pending-job-{cache_key}")
}

impl PendingRevJob {
    /// Admit a stored record for `request_key`, or refuse it by name. A record
    /// that cannot be read is refused rather than ignored: ignoring it would
    /// upload and pay again, which is the defect the record exists to prevent.
    fn from_stored(
        value: serde_json::Value,
        request_key: &CacheKey,
    ) -> Result<Self, RevAsrEvidenceCacheError> {
        let key = pending_job_key(request_key);
        let refuse = |reason: String| RevAsrEvidenceCacheError::InvalidPendingJob {
            key: key.clone(),
            reason,
        };
        let stored: StoredPendingRevJob =
            serde_json::from_value(value).map_err(|error| refuse(error.to_string()))?;
        if stored.pending_schema_version != REV_ASR_PENDING_JOB_SCHEMA_VERSION {
            return Err(refuse(format!(
                "unsupported pending-job schema version {}",
                stored.pending_schema_version
            )));
        }
        if stored.request_fingerprint != request_key.as_str() {
            return Err(refuse(
                "request fingerprint does not match cache key".to_owned(),
            ));
        }
        // The same route a request's language takes from text: the spec, the
        // ASR request, then Rev.AI's admission of it.
        let spec = crate::api::LanguageSpec::try_from(stored.effective_language.as_str())
            .map_err(|error| refuse(format!("effective language: {error}")))?;
        let request = crate::api::AsrLanguageRequest::try_from(&spec)
            .map_err(|error| refuse(format!("effective language: {error}")))?;
        let effective_language = RevLanguage::admit(&request)
            .map_err(|error| refuse(format!("effective language: {error}")))?;
        Ok(Self {
            job: stored.job,
            effective_language,
        })
    }
}

/// Proof that an unfinished paid job was durably recorded against its
/// request, so a rerun resumes the wait instead of re-submitting. Only the
/// evidence commit permit can build one; a job id alone is not a record.
#[derive(Debug, thiserror::Error)]
#[error(
    "{unfinished}; a rerun of this request resumes the wait for it instead of re-submitting (pending record {pending_key})"
)]
pub struct RetainedPendingRevJob {
    unfinished: UnfinishedRevJob,
    pending_key: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct RejectedRevLanguageEvidence {
    pub(crate) transcript_evidence: RevTranscriptEvidence,
    pub(crate) requested_language: RevLanguage,
    pub(crate) effective_language: RevLanguage,
    pub(crate) detected_language: Option<String>,
}

/// Proof that the exact rejected provider response was durably retained.
/// Only the evidence commit operation can construct this result; a candidate
/// cache key or a failed persistence attempt is not retained evidence.
#[derive(Debug, thiserror::Error)]
#[error(
    "Rev.AI returned no usable detected language. Raw response retained as diagnostic {diagnostic_key}; not admitted for replay. Supply an explicitly verified language before further processing; no English default was applied."
)]
pub struct RetainedRevLanguageRejection {
    diagnostic_key: String,
}

impl RetainedRevLanguageRejection {
    /// Address of the committed diagnostic, not a successful ASR cache entry.
    pub fn diagnostic_key(&self) -> &str {
        &self.diagnostic_key
    }
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

/// How the source bytes become the bytes uploaded to Rev.AI, and under which
/// declaration. Part of every request's identity, so a change in what the
/// provider receives is a new evidence identity, never a silent reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RevMediaPreparationRecipe {
    /// Source bytes preserved, declared `audio/mpeg`: the presentation every
    /// upload had before 2026-11-03, which for MPEG audio is also the true one.
    /// MPEG audio keeps it, so every existing MP3 request key is unchanged.
    SourceBytesLegacyAudioMpegV1,
    /// Source bytes preserved, declared by their probed container. Every
    /// other container uploads under this recipe; before it, a WAV or FLAC
    /// was declared `audio/mpeg` and named `provider-media.wav`, which worked
    /// only because Rev.AI sniffs content.
    SourceBytesDeclaredContainerV1,
}

impl RevMediaPreparationRecipe {
    fn prepare_provider_bytes(self, source_bytes: Vec<u8>) -> Vec<u8> {
        match self {
            Self::SourceBytesLegacyAudioMpegV1 | Self::SourceBytesDeclaredContainerV1 => {
                source_bytes
            }
        }
    }
}

/// The container of media as uploaded to Rev.AI: the one source of the
/// upload's declared type, its file name and its preparation recipe.
///
/// Built from the file's probed [`MediaContainer`] (its header, never its
/// extension) by [`PreparedRevProviderMedia::from_source`], and read back from
/// a retained trace's file name by replay inspection. A container Rev.AI is
/// not sent as is (Matroska, ASF: BA3 converts those to WAV before any engine)
/// is refused by [`TryFrom`], never declared as something else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderMediaContainer {
    Mp3,
    Wav,
    Mp4,
    Flac,
    Ogg,
    Aac,
}

/// A measured container BA3 does not upload to Rev.AI as is.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "{container:?} media is not uploaded to Rev.AI as is; convert it to WAV first (BA3's media conversion does so before transcription)"
)]
pub(crate) struct RevProviderContainerRefusal {
    container: MediaContainer,
}

impl TryFrom<MediaContainer> for ProviderMediaContainer {
    type Error = RevProviderContainerRefusal;

    fn try_from(container: MediaContainer) -> Result<Self, Self::Error> {
        match container {
            MediaContainer::MpegAudio => Ok(Self::Mp3),
            MediaContainer::Wav => Ok(Self::Wav),
            MediaContainer::IsoMedia => Ok(Self::Mp4),
            MediaContainer::Flac => Ok(Self::Flac),
            MediaContainer::Ogg => Ok(Self::Ogg),
            MediaContainer::Adts => Ok(Self::Aac),
            MediaContainer::Matroska | MediaContainer::Asf => {
                Err(RevProviderContainerRefusal { container })
            }
        }
    }
}

impl ProviderMediaContainer {
    /// The multipart part's declared media type: true of the bytes.
    pub(crate) const fn mime(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Mp4 => "audio/mp4",
            Self::Flac => "audio/flac",
            Self::Ogg => "audio/ogg",
            Self::Aac => "audio/aac",
        }
    }

    /// The extension of the multipart file name `provider-media.<ext>`.
    pub(crate) const fn extension(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Mp4 => "mp4",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
            Self::Aac => "aac",
        }
    }

    /// The recipe this container uploads under. MPEG audio keeps the
    /// historical recipe because its declaration never changed; the rest
    /// take the one that declares them truthfully.
    const fn recipe(self) -> RevMediaPreparationRecipe {
        match self {
            Self::Mp3 => RevMediaPreparationRecipe::SourceBytesLegacyAudioMpegV1,
            Self::Wav | Self::Mp4 | Self::Flac | Self::Ogg | Self::Aac => {
                RevMediaPreparationRecipe::SourceBytesDeclaredContainerV1
            }
        }
    }

    /// The inverse of [`Self::extension`], for reading a retained trace's
    /// upload file name back into the container that produced it.
    fn from_upload_extension(extension: &str) -> Option<Self> {
        [
            Self::Mp3,
            Self::Wav,
            Self::Mp4,
            Self::Flac,
            Self::Ogg,
            Self::Aac,
        ]
        .into_iter()
        .find(|container| container.extension() == extension)
    }
}

/// Complete provider-visible presentation used by both request identity and
/// observability.
///
/// Keeping this relationship in one value prevents the cache key and trace
/// from independently reassembling a MIME/name/metadata tuple that could
/// drift while still describing the same provider bytes. The fields are
/// serialized flat into every request key, so their names and spellings are
/// fixed; [`Self::for_container`] is the only constructor, so a name, type
/// and recipe that disagree cannot be assembled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RevProviderPresentation {
    provider_media_blake3: RevProviderMediaDigest,
    preparation_recipe: RevMediaPreparationRecipe,
    upload_file_name: String,
    upload_mime: &'static str,
    upload_metadata: String,
}

impl RevProviderPresentation {
    /// The presentation of `provider_media_blake3`'s bytes in `container`.
    fn for_container(
        container: ProviderMediaContainer,
        provider_media_blake3: RevProviderMediaDigest,
    ) -> Self {
        Self::assemble(
            provider_media_blake3,
            container.recipe(),
            container.extension(),
            container.mime(),
        )
    }

    /// The presentation every upload had before 2026-11-03, for the one use
    /// it still has: finding evidence a non-MPEG recording was cached under
    /// then. Declared `audio/mpeg`, named after the SOURCE path's extension
    /// (lower-cased, alphanumeric; `media` when the path had none), the same
    /// bytes. Never sent: a value of this shape names a key, not an upload.
    fn legacy_for(source_path: &Path, provider_media_blake3: RevProviderMediaDigest) -> Self {
        let upload_extension = source_path
            .extension()
            .and_then(|extension| extension.to_str())
            .filter(|extension| {
                !extension.is_empty() && extension.bytes().all(|byte| byte.is_ascii_alphanumeric())
            })
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "media".to_owned());
        Self::assemble(
            provider_media_blake3,
            RevMediaPreparationRecipe::SourceBytesLegacyAudioMpegV1,
            &upload_extension,
            "audio/mpeg",
        )
    }

    /// The one spelling of a presentation's file name and metadata, so the
    /// current and the legacy constructors cannot drift apart in either.
    fn assemble(
        provider_media_blake3: RevProviderMediaDigest,
        preparation_recipe: RevMediaPreparationRecipe,
        upload_extension: &str,
        upload_mime: &'static str,
    ) -> Self {
        let upload_metadata = format!("batchalign3_{}", &provider_media_blake3.0[..16]);
        Self {
            provider_media_blake3,
            preparation_recipe,
            upload_file_name: format!("provider-media.{upload_extension}"),
            upload_mime,
            upload_metadata,
        }
    }
}

/// The presentation a recording's evidence was keyed under before 2026-11-03,
/// where that differs from today's.
///
/// Decided by comparing the two presentations themselves, in the one
/// constructor ([`PreparedRevProviderMedia::new`]), never by container: MPEG
/// audio keeps its historical recipe, so an MPEG file at a `.mp3` path has one
/// key then and now, while the same bytes at an `.mpga` path, or at a path
/// with no extension, were keyed by that name then and by `mp3` now, and
/// consult the old key like any other container.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LegacyPresentation {
    /// The current presentation is the historical one: one key then and now.
    SameAsCurrent,
    /// The historical presentation, which keys differently from today's.
    Distinct(RevProviderPresentation),
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRevProviderMedia {
    source_path: PathBuf,
    source_digest: RevProviderMediaDigest,
    presentation: RevProviderPresentation,
    /// How this recording's evidence was keyed before 2026-11-03, when that
    /// differs from `presentation`.
    legacy_presentation: LegacyPresentation,
}

impl PreparedRevProviderMedia {
    /// Read and hash the media, and probe its container, which decides the
    /// declaration Rev.AI receives. The probe reads the header only; a file
    /// in a container this crate has not measured, or one BA3 does not upload
    /// as is, is refused here, naming the file, never declared `audio/mpeg`.
    pub(crate) async fn from_source(path: &Path) -> Result<Self, RevAsrEvidenceCacheError> {
        let bytes = tokio::fs::read(path).await?;
        let source_digest = RevProviderMediaDigest::from_bytes(&bytes);
        let container = ProviderMediaContainer::try_from(MediaProbe::new(path).container().await?)
            .map_err(
                |refusal| RevAsrEvidenceCacheError::UnsupportedProviderContainer {
                    path: path.to_owned(),
                    refusal,
                },
            )?;
        let provider_media_blake3 =
            RevProviderMediaDigest::from_bytes(&container.recipe().prepare_provider_bytes(bytes));
        // Both recipes preserve the bytes, so the digest the legacy key
        // carried is this one.
        Ok(Self::new(
            path.to_owned(),
            source_digest,
            container,
            provider_media_blake3,
        ))
    }

    /// The one way to pair a recording with its presentations. The legacy
    /// presentation is derived here from the same path and digest and
    /// compared with the current one, so `SameAsCurrent` is true by
    /// construction and no caller can pair a current presentation with a
    /// legacy one of another recording.
    fn new(
        source_path: PathBuf,
        source_digest: RevProviderMediaDigest,
        container: ProviderMediaContainer,
        provider_media_blake3: RevProviderMediaDigest,
    ) -> Self {
        let presentation =
            RevProviderPresentation::for_container(container, provider_media_blake3.clone());
        let legacy = RevProviderPresentation::legacy_for(&source_path, provider_media_blake3);
        let legacy_presentation = if legacy == presentation {
            LegacyPresentation::SameAsCurrent
        } else {
            LegacyPresentation::Distinct(legacy)
        };
        Self {
            source_path,
            source_digest,
            presentation,
            legacy_presentation,
        }
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
pub(crate) struct RevAsrModelRevision(crate::cache::BuildOwnedNamespace);

impl RevAsrModelRevision {
    pub(crate) fn current() -> Self {
        Self(crate::cache::BuildOwnedNamespace::literal(
            "revai:asynchronous-transcript-v1",
        ))
    }

    fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl CacheNamespace for RevAsrModelRevision {
    fn namespace(&self) -> &str {
        self.as_str()
    }
}

/// The key this request's evidence was stored under before 2026-11-03, where
/// that is not today's key. Derived once with the current key, from the same
/// material, so the two keys name the same request.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LegacyEvidenceKey {
    /// One key then and now (MPEG audio at a `.mp3` path): nothing to
    /// consult.
    SameAsCurrent,
    /// Evidence cached before the declaration changed sits under this key.
    Distinct(CacheKey),
}

/// What a non-MPEG request found under its legacy key, decided in one place
/// ([`RevAsrEvidenceRequest::legacy_lookup`]).
enum LegacyLookup {
    /// Evidence of this request, stored under the old key, read and admitted.
    Hit(CompletedRevAsrEvidence),
    /// Nothing there, or no legacy key to consult.
    Miss,
}

/// Where a cache hit's evidence was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HitSource {
    /// Under the request's own key.
    CurrentKey,
    /// Under the key the request had before 2026-11-03; it has been stored
    /// under the current key too, so the next run hits that directly.
    MigratedFromLegacyKey,
}

#[derive(Debug, Clone)]
pub(crate) struct RevAsrEvidenceRequest {
    cache_key: CacheKey,
    legacy_key: LegacyEvidenceKey,
    model_revision: RevAsrModelRevision,
    provider_media: PreparedRevProviderMedia,
    requested_language: RevLanguage,
    expected_speakers: Option<NumSpeakers>,
}

impl RevAsrEvidenceRequest {
    pub(crate) fn new(
        provider_media: PreparedRevProviderMedia,
        requested_language: &RevLanguage,
        speakers: crate::transcribe::SpeakerCount,
        model_revision: &RevAsrModelRevision,
    ) -> Result<Self, RevAsrEvidenceCacheError> {
        // The provider's parameter: `None` is "infer", never a sentinel. Zero
        // is unrepresentable in the count, so nothing is re-checked here.
        let expected_speakers = speakers.known();
        // One formula for both keys: the material differs in the
        // presentation alone, so a legacy key is exactly what this build
        // would have written for the same request before 2026-11-03.
        let key_for = |presentation: &RevProviderPresentation| {
            let material = RevAsrEvidenceKeyMaterial {
                domain: "rev_asr_evidence",
                request_identity_revision: REV_ASR_REQUEST_IDENTITY_REVISION,
                provider_presentation: presentation,
                requested_language: requested_language.to_string(),
                expected_speakers: expected_speakers.map(|count| count.0),
                request_policy_revision: REV_ASR_REQUEST_POLICY_REVISION,
                model_revision: model_revision.as_str(),
            };
            serde_json::to_string(&material).map(|canonical| CacheKey::from_content(&canonical))
        };
        let cache_key = key_for(&provider_media.presentation)?;
        let legacy_key = match &provider_media.legacy_presentation {
            LegacyPresentation::SameAsCurrent => LegacyEvidenceKey::SameAsCurrent,
            LegacyPresentation::Distinct(legacy) => LegacyEvidenceKey::Distinct(key_for(legacy)?),
        };
        Ok(Self {
            cache_key,
            legacy_key,
            model_revision: model_revision.clone(),
            provider_media,
            requested_language: requested_language.clone(),
            expected_speakers,
        })
    }

    /// Whether this request has a legacy key distinct from its own, for the
    /// tests that pin which containers consult one.
    #[cfg(test)]
    fn has_distinct_legacy_key(&self) -> bool {
        matches!(self.legacy_key, LegacyEvidenceKey::Distinct(_))
    }

    pub(crate) fn cache_key(&self) -> &CacheKey {
        &self.cache_key
    }

    /// The key this request's evidence was stored under before 2026-11-03,
    /// where that is not today's key; for the log line that names a
    /// migration.
    pub(crate) fn legacy_key(&self) -> Option<&CacheKey> {
        match &self.legacy_key {
            LegacyEvidenceKey::SameAsCurrent => None,
            LegacyEvidenceKey::Distinct(key) => Some(key),
        }
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
        requested_language: &RevLanguage,
        speakers: crate::transcribe::SpeakerCount,
        model_revision: &RevAsrModelRevision,
    ) -> Result<Self, RevAsrEvidenceCacheError> {
        Self::new(
            PreparedRevProviderMedia::from_source(audio_path).await?,
            requested_language,
            speakers,
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
                return self
                    .miss(cache, RevAsrEvidenceMissReason::ForcedRefresh, lease)
                    .await;
            }
            CachePolicy::UseCache | CachePolicy::RequireCache | CachePolicy::SkipCache => {}
        }
        let stored = cache
            .get(
                self.cache_key.as_str(),
                tasks::REV_ASR_EVIDENCE,
                &self.model_revision,
            )
            .await?;
        let Some(stored) = stored else {
            // Evidence a recording was cached under before its declaration
            // changed is found under the key it had then, stored under today's
            // key (re-encoded in the current envelope schema), and used: the
            // one consultation of the old key, under both policies that read
            // the cache (a forced refresh has already left), since a required
            // cache that holds the evidence has it.
            if let LegacyLookup::Hit(evidence) = self.legacy_lookup(cache).await? {
                store_evidence(cache, &self.cache_key, &self.model_revision, &evidence).await?;
                lease.mark_committed();
                // A job an earlier run submitted under today's key and did not
                // see finish is superseded by the evidence just stored, so its
                // record goes, as a commit's does: after the evidence, never
                // before.
                forget_pending_job_record(cache, &self.cache_key).await?;
                return Ok(RevAsrEvidenceLookup::Hit(
                    evidence,
                    HitSource::MigratedFromLegacyKey,
                ));
            }
            match policy {
                CachePolicy::RequireCache => {
                    return Err(RevAsrEvidenceCacheError::RequiredEvidenceMissing(
                        self.cache_key.clone(),
                    ));
                }
                CachePolicy::UseCache | CachePolicy::SkipCache => {}
            }
            return self
                .miss(cache, RevAsrEvidenceMissReason::NotFound, lease)
                .await;
        };
        Ok(RevAsrEvidenceLookup::Hit(
            StoredRevAsrEvidence::decode_for(stored, self)?,
            HitSource::CurrentKey,
        ))
    }

    /// Read this request's evidence under the key it had before 2026-11-03,
    /// when that key is distinct from today's. MPEG audio has one key and is
    /// never looked up twice. The stored envelope names the legacy key as its
    /// fingerprint, so it is admitted against that key.
    async fn legacy_lookup(
        &self,
        cache: &UtteranceCache,
    ) -> Result<LegacyLookup, RevAsrEvidenceCacheError> {
        let Some(admitted) = AdmittedRevReplayRequest::for_legacy_key(self) else {
            return Ok(LegacyLookup::Miss);
        };
        let stored = cache
            .get(
                admitted.key.as_str(),
                tasks::REV_ASR_EVIDENCE,
                &self.model_revision,
            )
            .await?;
        match stored {
            None => Ok(LegacyLookup::Miss),
            Some(value) => Ok(LegacyLookup::Hit(StoredRevAsrEvidence::decode_for_replay(
                value, &admitted,
            )?)),
        }
    }

    /// A miss, carrying the record of any job an earlier run of this request
    /// left unfinished, so the authorized run resumes that job rather than
    /// uploading and paying again. A forced refresh reads it too: the record
    /// is not evidence to bypass, it is a submission already in flight.
    async fn miss(
        &self,
        cache: &UtteranceCache,
        reason: RevAsrEvidenceMissReason,
        lease: InferenceLease,
    ) -> Result<RevAsrEvidenceLookup, RevAsrEvidenceCacheError> {
        let pending = self.pending_job(cache).await?;
        Ok(RevAsrEvidenceLookup::Miss(Box::new(RevAsrEvidenceMiss {
            request: self.clone(),
            reason,
            pending,
            _lease: lease,
        })))
    }

    /// The pending-job record for this request, admitted, if one exists.
    async fn pending_job(
        &self,
        cache: &UtteranceCache,
    ) -> Result<Option<PendingRevJob>, RevAsrEvidenceCacheError> {
        let stored = cache
            .get(
                &pending_job_key(&self.cache_key),
                tasks::REV_ASR_EVIDENCE,
                &self.model_revision,
            )
            .await?;
        stored
            .map(|value| PendingRevJob::from_stored(value, &self.cache_key))
            .transpose()
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
                tasks::REV_ASR_EVIDENCE,
                &self.model_revision,
                &value,
            )
            .await?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FetchedRevAsrEvidence {
    pub(crate) transcript_evidence: RevTranscriptEvidence,
    pub(crate) resolved_language: TranscriptLanguage,
}

/// Evidence admitted against its request, never deserialized directly.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub(crate) struct CompletedRevAsrEvidence(FetchedRevAsrEvidence);

impl CompletedRevAsrEvidence {
    fn admit(
        raw: FetchedRevAsrEvidence,
        language: &RevLanguage,
    ) -> Result<Self, RevAsrEvidenceCacheError> {
        validate_evidence(&raw)?;
        language
            .check_stored_resolution(&raw.resolved_language)
            .map_err(|error| RevAsrEvidenceCacheError::InvalidEvidence(error.to_string()))?;
        Ok(Self(raw))
    }

    pub(crate) fn transcript_evidence(&self) -> &RevTranscriptEvidence {
        &self.0.transcript_evidence
    }
    pub(crate) fn resolved_language(&self) -> &TranscriptLanguage {
        &self.0.resolved_language
    }

    #[cfg(test)]
    pub(crate) fn admit_for_test(raw: FetchedRevAsrEvidence, language: &RevLanguage) -> Self {
        Self::admit(raw, language).expect("admitted fixture evidence")
    }
}

fn validate_evidence(evidence: &FetchedRevAsrEvidence) -> Result<(), RevAsrEvidenceCacheError> {
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
            // Each bound is a valid position by type, admitted when the
            // provider's JSON was read; only their order is left to check.
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
struct StoredRevAsrEvidence<E = FetchedRevAsrEvidence> {
    schema_version: u32,
    request_fingerprint: String,
    evidence: E,
}

impl StoredRevAsrEvidence {
    fn decode_for(
        value: serde_json::Value,
        request: &RevAsrEvidenceRequest,
    ) -> Result<CompletedRevAsrEvidence, RevAsrEvidenceCacheError> {
        Self::decode_for_replay(value, &AdmittedRevReplayRequest::from_production(request))
    }

    fn decode_for_replay(
        value: serde_json::Value,
        request: &AdmittedRevReplayRequest,
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
                validate_request_fingerprint(&envelope.request_fingerprint, &request.key)?;
                envelope.evidence
            }
            2 => {
                let envelope: StoredRevAsrEvidenceV2 =
                    serde_json::from_value(value).map_err(|error| {
                        RevAsrEvidenceCacheError::InvalidEvidence(error.to_string())
                    })?;
                validate_request_fingerprint(&envelope.request_fingerprint, &request.key)?;
                FetchedRevAsrEvidence {
                    transcript_evidence: RevTranscriptEvidence::from_legacy_transcript(
                        envelope.evidence.transcript,
                    ),
                    // Schema 2 predates code-switched requests: one language.
                    resolved_language: TranscriptLanguage::One(envelope.evidence.resolved_language),
                }
            }
            unsupported => {
                return Err(RevAsrEvidenceCacheError::InvalidEvidence(format!(
                    "unsupported stored evidence schema version {unsupported}"
                )));
            }
        };
        CompletedRevAsrEvidence::admit(evidence, &request.language)
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
    request_key: &CacheKey,
) -> Result<(), RevAsrEvidenceCacheError> {
    if request_fingerprint != request_key.as_str() {
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
    /// Replayed from evidence cached under the request's pre-2026-11-03 key
    /// and stored under today's key by this run: the migration a reader of
    /// the trace sees once per request identity (recording, language,
    /// speaker count, model revision).
    ReplayedFromLegacyKey,
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
    Hit(CompletedRevAsrEvidence, HitSource),
    Miss(Box<RevAsrEvidenceMiss>),
}

#[derive(Debug)]
struct RevAsrEvidenceMiss {
    request: RevAsrEvidenceRequest,
    reason: RevAsrEvidenceMissReason,
    /// The job an earlier run left unfinished, when the cache records one.
    pending: Option<PendingRevJob>,
    _lease: InferenceLease,
}

impl RevAsrEvidenceMiss {
    fn authorize(self) -> RevAsrInferenceAuthorization {
        RevAsrInferenceAuthorization {
            request: self.request,
            reason: self.reason,
            pending: self.pending,
            _lease: self._lease,
        }
    }
}

#[derive(Debug)]
struct RevAsrInferenceAuthorization {
    request: RevAsrEvidenceRequest,
    reason: RevAsrEvidenceMissReason,
    pending: Option<PendingRevJob>,
    _lease: InferenceLease,
}

pub(crate) struct AuthorizedRevEvidenceRun {
    pub(super) provider_media: PreparedRevProviderMedia,
    pub(super) requested_language: RevLanguage,
    pub(super) expected_speakers: Option<NumSpeakers>,
    /// Submit, or resume the job an earlier run left unfinished.
    pub(super) route: RevJobRoute,
}

/// Whether the cache holds a pending-job record for the request a permit
/// commits, which decides whether finishing the request must also remove it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingRecord {
    Absent,
    Present,
}

impl RevAsrInferenceAuthorization {
    fn into_run(self) -> (AuthorizedRevEvidenceRun, RevAsrEvidenceCommitPermit) {
        let Self {
            request,
            reason,
            pending,
            _lease,
        } = self;
        let RevAsrEvidenceRequest {
            cache_key,
            // An authorized run stores under the current key alone.
            legacy_key: _,
            model_revision,
            provider_media,
            requested_language,
            expected_speakers,
        } = request;
        let (route, pending_record) = match pending {
            Some(pending) => (RevJobRoute::Resume(pending), PendingRecord::Present),
            None => (RevJobRoute::Submit, PendingRecord::Absent),
        };
        (
            AuthorizedRevEvidenceRun {
                provider_media,
                requested_language: requested_language.clone(),
                expected_speakers,
                route,
            },
            RevAsrEvidenceCommitPermit {
                requested_language,
                cache_key,
                model_revision,
                reason,
                pending_record,
                _lease,
            },
        )
    }
}

struct RevAsrEvidenceCommitPermit {
    requested_language: RevLanguage,
    cache_key: CacheKey,
    model_revision: RevAsrModelRevision,
    reason: RevAsrEvidenceMissReason,
    pending_record: PendingRecord,
    _lease: InferenceLease,
}

impl RevAsrEvidenceCommitPermit {
    /// Record a job the wait did not see finish, against its request.
    ///
    /// The lease is NOT marked committed: no evidence exists. A concurrent
    /// waiter on the same key looks up again, finds no evidence, finds this
    /// record, and resumes the job rather than submitting its own.
    async fn retain_pending_job(
        self,
        cache: &UtteranceCache,
        unfinished: UnfinishedRevSubmission,
    ) -> Result<RetainedPendingRevJob, RevAsrEvidenceCacheError> {
        let key = pending_job_key(&self.cache_key);
        let record = StoredPendingRevJob {
            pending_schema_version: REV_ASR_PENDING_JOB_SCHEMA_VERSION,
            request_fingerprint: self.cache_key.to_string(),
            job: unfinished.job.job.clone(),
            effective_language: unfinished.effective_language.to_string(),
            cause: unfinished.job.cause.clone(),
        };
        let value = serde_json::to_value(&record)?;
        cache
            .put(&key, tasks::REV_ASR_EVIDENCE, &self.model_revision, &value)
            .await
            .map_err(|source| RevAsrEvidenceCacheError::PendingJobPersistence {
                key: key.clone(),
                source,
            })?;
        Ok(RetainedPendingRevJob {
            unfinished: unfinished.job,
            pending_key: key,
        })
    }

    /// Remove the pending-job record once the job it names is over: its
    /// transcript committed or rejected, or Rev.AI's terminal verdict on it.
    /// Nothing to do when no record was read for this request.
    async fn forget_pending_job(
        &self,
        cache: &UtteranceCache,
    ) -> Result<(), RevAsrEvidenceCacheError> {
        match self.pending_record {
            PendingRecord::Absent => Ok(()),
            PendingRecord::Present => forget_pending_job_record(cache, &self.cache_key).await,
        }
    }

    /// Inference failed. A resumed job that Rev.AI judged terminally (the job
    /// failed, or it is gone) is over, so its record goes; a transient or
    /// account-level failure leaves the job, and its record, in place for the
    /// next run to resume.
    async fn inference_failed(
        self,
        cache: &UtteranceCache,
        error: ServerError,
    ) -> RevAsrEvidenceResolutionError {
        if let ServerError::AsrProvider {
            disposition: crate::error::AsrProviderDisposition::Terminal,
            ..
        } = &error
            && let Err(forget) = self.forget_pending_job(cache).await
        {
            return RevAsrEvidenceResolutionError::Evidence(forget);
        }
        RevAsrEvidenceResolutionError::Inference(error)
    }

    async fn retain_rejected_language(
        self,
        cache: &UtteranceCache,
        request: &RevAsrEvidenceTraceSeed,
        rejected: RejectedRevLanguageEvidence,
    ) -> Result<RetainedRevLanguageRejection, RevAsrEvidenceCacheError> {
        let envelope = serde_json::json!({
            "diagnostic_schema_version": 1,
            "admission": "rejected_unresolved_language",
            "request": request,
            "response": rejected,
        });
        // Distinct responses cannot overwrite each other, and normal evidence
        // lookup never reads this namespace. Do not mark the success lease.
        let bytes = serde_json::to_vec(&envelope)?;
        let key = format!(
            "rev-asr-rejected-language-{}",
            blake3::hash(&bytes).to_hex()
        );
        cache
            .put(
                &key,
                tasks::REV_ASR_EVIDENCE,
                &self.model_revision,
                &envelope,
            )
            .await
            .map_err(
                |source| RevAsrEvidenceCacheError::RejectedEvidencePersistence {
                    diagnostic_key: key.clone(),
                    source,
                },
            )?;
        Ok(RetainedRevLanguageRejection {
            diagnostic_key: key,
        })
    }

    fn admit(
        self,
        raw: FetchedRevAsrEvidence,
    ) -> Result<AdmittedRevAsrCommit, RevAsrEvidenceCacheError> {
        let evidence = CompletedRevAsrEvidence::admit(raw, &self.requested_language)?;
        Ok(AdmittedRevAsrCommit {
            permit: self,
            evidence,
        })
    }
}

/// The one write of admitted evidence under a request's key: the envelope has
/// one shape and one writer, whether the evidence was just inferred
/// ([`AdmittedRevAsrCommit::commit`]) or found under the request's legacy key
/// (`RevAsrEvidenceRequest::lookup`). The legacy entry, in the second case,
/// is left as it is: it is still true evidence of the request as it was then.
async fn store_evidence(
    cache: &UtteranceCache,
    cache_key: &CacheKey,
    model_revision: &RevAsrModelRevision,
    evidence: &CompletedRevAsrEvidence,
) -> Result<(), RevAsrEvidenceCacheError> {
    let envelope = StoredRevAsrEvidence {
        schema_version: REV_ASR_EVIDENCE_STORAGE_SCHEMA_VERSION,
        request_fingerprint: cache_key.to_string(),
        evidence,
    };
    let value = serde_json::to_value(&envelope)?;
    cache
        .put(
            cache_key.as_str(),
            tasks::REV_ASR_EVIDENCE,
            model_revision,
            &value,
        )
        .await?;
    Ok(())
}

/// Remove the pending-job record under `cache_key`; an absent record is
/// nothing to remove. Called only after the request's evidence is stored.
async fn forget_pending_job_record(
    cache: &UtteranceCache,
    cache_key: &CacheKey,
) -> Result<(), RevAsrEvidenceCacheError> {
    let key = pending_job_key(cache_key);
    cache
        .delete_batch(std::slice::from_ref(&key), tasks::REV_ASR_EVIDENCE.name())
        .await
        .map(|_| ())
        .map_err(|source| RevAsrEvidenceCacheError::PendingJobForget { key, source })
}

/// The commit operation owns both the request permit and its admitted evidence.
struct AdmittedRevAsrCommit {
    permit: RevAsrEvidenceCommitPermit,
    evidence: CompletedRevAsrEvidence,
}

impl AdmittedRevAsrCommit {
    async fn commit(
        self,
        cache: &UtteranceCache,
    ) -> Result<CompletedRevAsrEvidence, RevAsrEvidenceCacheError> {
        let Self { permit, evidence } = self;
        store_evidence(cache, &permit.cache_key, &permit.model_revision, &evidence).await?;
        permit._lease.mark_committed();
        // After the evidence, never before: were the record removed first and
        // the evidence write then fail, the next run would upload and pay
        // again. Failing here after a successful write costs nothing but this
        // run's report, since the retry replays the committed evidence.
        permit.forget_pending_job(cache).await?;
        Ok(evidence)
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
    /// Replayed from the legacy key, and re-stored under the current one.
    MigratedFromLegacyKey,
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
            RevAsrEvidenceSource::MigratedFromLegacyKey => {
                RevAsrEvidenceCacheOutcome::ReplayedFromLegacyKey
            }
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
            self.evidence.transcript_evidence().fidelity(),
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
        evidence: FetchedRevAsrEvidence,
    ) -> Self {
        Self {
            evidence: CompletedRevAsrEvidence::admit_for_test(
                evidence,
                &request.requested_language,
            ),
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
        RevAsrEvidenceLookup::Hit(evidence, hit) => Ok(RevAsrEvidenceResolution {
            evidence,
            source: match hit {
                HitSource::CurrentKey => RevAsrEvidenceSource::Replayed,
                HitSource::MigratedFromLegacyKey => RevAsrEvidenceSource::MigratedFromLegacyKey,
            },
            trace_seed,
        }),
        RevAsrEvidenceLookup::Miss(miss) => {
            let authorization = (*miss).authorize();
            let (run, permit) = authorization.into_run();
            let reason = permit.reason;
            let outcome = match inference.infer(run).await {
                Ok(outcome) => outcome,
                Err(error) => return Err(permit.inference_failed(cache, error).await),
            };
            let evidence = match outcome {
                RevAsrInferenceOutcome::Fetched(evidence) => evidence,
                RevAsrInferenceOutcome::UnresolvedLanguage(rejected) => {
                    // The job finished (its transcript was fetched), so any
                    // record of it as unfinished is over.
                    permit.forget_pending_job(cache).await?;
                    let retained = permit
                        .retain_rejected_language(cache, &trace_seed, rejected)
                        .await?;
                    return Err(RevAsrEvidenceResolutionError::UnresolvedLanguage(retained));
                }
                RevAsrInferenceOutcome::Unfinished(unfinished) => {
                    let retained = permit.retain_pending_job(cache, unfinished).await?;
                    return Err(RevAsrEvidenceResolutionError::Unfinished(retained));
                }
            };
            let evidence = permit.admit(evidence)?.commit(cache).await?;
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
    UnresolvedLanguage(RetainedRevLanguageRejection),
    /// The paid job is still Rev.AI's, recorded for the next run to resume.
    #[error(transparent)]
    Unfinished(RetainedPendingRevJob),
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
        RevAsrEvidenceResolutionError::UnresolvedLanguage(retained) => {
            ServerError::UnresolvedAsrLanguage(retained)
        }
        // Transient on purpose: the control plane retries a transient provider
        // failure, and the retry resumes the recorded job at no new cost.
        RevAsrEvidenceResolutionError::Unfinished(retained) => ServerError::AsrProvider {
            disposition: crate::error::AsrProviderDisposition::Transient,
            message: retained.to_string(),
        },
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
    #[error(
        "failed to retain rejected Rev.AI response at diagnostic key {diagnostic_key}: {source}"
    )]
    RejectedEvidencePersistence {
        diagnostic_key: String,
        source: CacheError,
    },
    #[error("could not read Rev.AI provider media: {0}")]
    Io(#[from] std::io::Error),
    /// The media's container could not be established, so no declaration
    /// for it is true. The probe's error names the file and the reason.
    #[error("could not establish the container of Rev.AI provider media: {0}")]
    ProviderMediaProbe(#[from] ProbeError),
    #[error("Rev.AI provider media {path}: {refusal}")]
    UnsupportedProviderContainer {
        path: PathBuf,
        refusal: RevProviderContainerRefusal,
    },
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
    #[error("invalid Rev.AI pending-job record {key}: {reason}")]
    InvalidPendingJob { key: String, reason: String },
    #[error("failed to record unfinished Rev.AI job at {key}: {source}")]
    PendingJobPersistence { key: String, source: CacheError },
    #[error("failed to remove the Rev.AI pending-job record {key}: {source}")]
    PendingJobForget { key: String, source: CacheError },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{AsrLanguageRequest, AudioPositionSeconds};
    use crate::cache::{CacheBackend, CacheStats};
    use crate::media::probe::fixtures::{write_pcm_wav, write_unpadded_cbr_mp3};
    use crate::media::tools::MediaTool;

    fn english() -> RevLanguage {
        RevLanguage::admit(&AsrLanguageRequest::One(LanguageCode3::eng()))
            .expect("Rev.AI recognizes English")
    }

    /// An exact speaker count, as a request admits it.
    fn speakers(count: u32) -> crate::transcribe::SpeakerCount {
        crate::transcribe::SpeakerCount::admit(false, NumSpeakers(count)).expect("test: a count")
    }
    use crate::revai::types::{Element, Monologue};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sample_evidence() -> FetchedRevAsrEvidence {
        let transcript = Transcript {
            monologues: vec![Monologue {
                speaker: 0,
                elements: vec![Element {
                    element_type: "text".to_owned(),
                    value: "hello".to_owned(),
                    ts: Some(AudioPositionSeconds::try_from(0.1).expect("fixture position")),
                    end_ts: Some(AudioPositionSeconds::try_from(0.5).expect("fixture position")),
                    confidence: Some(0.9),
                }],
            }],
        };
        FetchedRevAsrEvidence {
            transcript_evidence: RevTranscriptEvidence::from_provider_json(
                serde_json::to_string(&transcript).expect("fixture transcript JSON"),
            )
            .expect("fixture provider transcript"),
            resolved_language: TranscriptLanguage::One(LanguageCode3::eng()),
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
        async fn get(
            &self,
            key: &str,
            _task: &str,
            _engine_version: &str,
        ) -> Result<Option<serde_json::Value>, CacheError> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        async fn get_batch(
            &self,
            _keys: &[String],
            _task: &str,
            _engine_version: &str,
        ) -> Result<HashMap<String, serde_json::Value>, CacheError> {
            unreachable!("single entry boundary")
        }
        async fn put(
            &self,
            key: &str,
            _task: &str,
            _engine_version: &str,
            _ba_version: &str,
            data: &serde_json::Value,
        ) -> Result<(), CacheError> {
            self.0.lock().unwrap().insert(key.to_owned(), data.clone());
            Ok(())
        }
        async fn put_batch(
            &self,
            _entries: &[(String, serde_json::Value)],
            _task: &str,
            _engine_version: &str,
            _ba_version: &str,
        ) -> Result<(), CacheError> {
            unreachable!("single entry boundary")
        }
        async fn delete_batch(&self, _keys: &[String], _task: &str) -> Result<usize, CacheError> {
            unreachable!("no deletion")
        }
        async fn stats(&self) -> Result<CacheStats, CacheError> {
            unreachable!("no statistics")
        }
    }

    struct LanguageResponseService {
        raw: &'static str,
        detected: Option<&'static str>,
        effective: RevLanguage,
    }

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for LanguageResponseService {
        async fn infer(
            &self,
            run: AuthorizedRevEvidenceRun,
        ) -> Result<RevAsrInferenceOutcome, ServerError> {
            Ok(RevAsrInferenceOutcome::from(
                super::super::asr::classify_language_response(
                    RevTranscriptEvidence::from_provider_json(self.raw.to_owned()).unwrap(),
                    run.requested_language,
                    self.effective.clone(),
                    self.detected.map(str::to_owned),
                ),
            ))
        }
    }

    fn language_request() -> RevAsrEvidenceRequest {
        let digest = RevProviderMediaDigest::from_bytes(b"provider media");
        let media = PreparedRevProviderMedia::new(
            PathBuf::from("fixture.wav"),
            digest.clone(),
            ProviderMediaContainer::Wav,
            digest,
        );
        RevAsrEvidenceRequest::new(
            media,
            &RevLanguage::detect(),
            crate::transcribe::SpeakerCount::Auto,
            &RevAsrModelRevision::current(),
        )
        .unwrap()
    }

    /// A code-switched request is its own evidence: it keys apart from each
    /// of its languages and from detection, and its trace names the pair.
    #[test]
    fn a_language_pair_request_keys_apart_from_its_languages() {
        let pair = RevLanguage::admit(&AsrLanguageRequest::Pair(
            crate::api::LanguagePair::new(LanguageCode3::eng(), LanguageCode3::spa())
                .expect("two languages"),
        ))
        .expect("Rev.AI takes English/Spanish");
        let spanish = RevLanguage::admit(&AsrLanguageRequest::One(LanguageCode3::spa()))
            .expect("Rev.AI recognizes Spanish");
        let media = language_request().provider_media;
        let key = |language: &RevLanguage| {
            RevAsrEvidenceRequest::new(
                media.clone(),
                language,
                crate::transcribe::SpeakerCount::Auto,
                &RevAsrModelRevision::current(),
            )
            .expect("request")
        };
        let pair_request = key(&pair);
        for other in [english(), spanish, RevLanguage::detect()] {
            assert_ne!(pair_request.cache_key(), key(&other).cache_key(), "{other}");
        }
        assert_eq!(pair_request.trace_seed().requested_language, "eng,spa");
    }

    #[tokio::test]
    async fn rejected_language_response_is_exact_retained_and_never_replayed() {
        let backend = RecordingBackend::default();
        let cache = UtteranceCache::from_backend(Box::new(backend.clone()));
        let request = language_request();
        let empty = "{ \"monologues\":[], \"future\":{\"value\":17} }";
        let spoken = r#"{"monologues":[{"speaker":0,"elements":[{"type":"text","value":"hello","ts":0.1,"end_ts":0.2}]}],"future":[1,2]}"#;
        for (detected, raw) in [(None, empty), (Some("unmapped-provider-language"), spoken)] {
            let service = LanguageResponseService {
                raw,
                detected,
                effective: RevLanguage::detect(),
            };
            let error =
                resolve_rev_asr_evidence(&request, &cache, CachePolicy::SkipCache, &service)
                    .await
                    .unwrap_err();
            let message = error.to_string();
            assert!(message.contains("rev-asr-rejected-language-"));
            let server_error = rev_asr_resolution_error_to_server_error(error);
            assert_eq!(
                crate::runner::util::classify_server_error(&server_error),
                crate::scheduling::FailureCategory::ProviderTerminal
            );
            let ServerError::UnresolvedAsrLanguage(retained) = server_error else {
                panic!("retained language rejection lost its state");
            };
            assert!(message.contains(retained.diagnostic_key()));
            #[cfg(feature = "server")]
            {
                use axum::response::IntoResponse;
                let key = retained.diagnostic_key().to_owned();
                let response = ServerError::UnresolvedAsrLanguage(retained).into_response();
                assert_eq!(response.status(), axum::http::StatusCode::BAD_GATEWAY);
                let bytes = axum::body::to_bytes(response.into_body(), 16 * 1024)
                    .await
                    .unwrap();
                let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(body["diagnostic_key"], key);
                assert_eq!(body["admission"], "rejected_unresolved_language");
            }
            let records = backend.0.lock().unwrap();
            assert!(!records.contains_key(request.cache_key.as_str()));
            let (key, diagnostic) = records
                .iter()
                .find(|(_, value)| {
                    value["response"]["detected_language"]
                        == serde_json::to_value(detected).unwrap()
                })
                .unwrap();
            assert!(message.contains(key));
            assert_eq!(
                diagnostic["request"],
                serde_json::to_value(request.trace_seed()).unwrap()
            );
            assert_eq!(
                diagnostic["response"]["transcript_evidence"]["raw_json"],
                raw
            );
            assert_eq!(
                diagnostic["response"]["requested_language"],
                serde_json::to_value(RevLanguage::detect()).unwrap()
            );
            assert_eq!(
                diagnostic["response"]["effective_language"],
                serde_json::to_value(RevLanguage::detect()).unwrap()
            );
            assert!(StoredRevAsrEvidence::decode_for(diagnostic.clone(), &request).is_err());
        }
        assert_eq!(backend.0.lock().unwrap().len(), 2);
        let service = LanguageResponseService {
            raw: empty,
            detected: None,
            effective: RevLanguage::detect(),
        };
        assert!(matches!(
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service).await,
            Err(RevAsrEvidenceResolutionError::Evidence(
                RevAsrEvidenceCacheError::RequiredEvidenceMissing(_)
            ))
        ));
        // Auto -> successfully identified language remains admitted, even if
        // transcription omitted its own detected-language field.
        let service = LanguageResponseService {
            raw: spoken,
            detected: None,
            effective: english(),
        };
        resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .unwrap();
        let replay =
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service)
                .await
                .unwrap();
        assert_eq!(replay.source(), RevAsrEvidenceSource::Replayed);
        assert_eq!(
            replay.evidence.transcript_evidence().exact_provider_json(),
            Some(spoken)
        );
    }

    #[tokio::test]
    async fn rejected_language_response_persistence_failure_is_not_masked() {
        let cache = UtteranceCache::from_backend(Box::new(FailingCommitBackend));
        let request = language_request();
        let service = LanguageResponseService {
            raw: r#"{"monologues":[]}"#,
            detected: None,
            effective: RevLanguage::detect(),
        };
        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("rev-asr-rejected-language-"));
        assert!(
            error
                .to_string()
                .contains("injected durable commit failure")
        );
        assert!(matches!(
            rev_asr_resolution_error_to_server_error(error),
            ServerError::Persistence(_)
        ));
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
            Ok(RevAsrInferenceOutcome::Fetched(sample_evidence()))
        }
    }

    #[test]
    fn auto_speakers_cache_identity_is_distinct_and_trace_records_null() {
        // Request identity admission is pure: no media upload or temporary files.
        let digest = RevProviderMediaDigest::from_bytes(b"provider media");
        let media = PreparedRevProviderMedia::new(
            PathBuf::from("fixture.wav"),
            digest.clone(),
            ProviderMediaContainer::Wav,
            digest,
        );
        let language = english();
        let revision = RevAsrModelRevision::current();
        let automatic = RevAsrEvidenceRequest::new(
            media.clone(),
            &language,
            crate::transcribe::SpeakerCount::Auto,
            &revision,
        )
        .expect("automatic request");
        let exact = RevAsrEvidenceRequest::new(media, &language, speakers(2), &revision)
            .expect("exact request");
        assert_ne!(automatic.cache_key(), exact.cache_key());
        assert!(
            serde_json::to_value(automatic.trace_seed()).expect("trace")["expected_speakers"]
                .is_null()
        );
        assert_eq!(
            serde_json::to_value(exact.trace_seed()).expect("trace")["expected_speakers"],
            2
        );
    }

    #[tokio::test]
    async fn rev_key_reuses_identical_media_but_invalidates_semantic_changes() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let first = tempdir.path().join("first.wav");
        let renamed = tempdir.path().join("renamed.wav");
        let misnamed = tempdir.path().join("renamed.mp3");
        let other_container = tempdir.path().join("tone.mp3");
        write_pcm_wav(&first, 0);
        write_pcm_wav(&renamed, 0);
        write_pcm_wav(&misnamed, 0);
        write_unpadded_cbr_mp3(&other_container);
        let revision = RevAsrModelRevision::current();

        let baseline =
            RevAsrEvidenceRequest::from_audio(&first, &english(), speakers(2), &revision)
                .await
                .expect("baseline request");
        let copied =
            RevAsrEvidenceRequest::from_audio(&renamed, &english(), speakers(2), &revision)
                .await
                .expect("copied request");
        // The same WAV bytes under an `.mp3` name: the container is probed
        // from the bytes and the name is not consulted, so the provider sees
        // the same upload and the identity is the same.
        let misnamed_copy =
            RevAsrEvidenceRequest::from_audio(&misnamed, &english(), speakers(2), &revision)
                .await
                .expect("misnamed copy request");
        let other_container =
            RevAsrEvidenceRequest::from_audio(&other_container, &english(), speakers(2), &revision)
                .await
                .expect("other-container request");
        let auto_language = RevAsrEvidenceRequest::from_audio(
            &first,
            &RevLanguage::detect(),
            speakers(2),
            &revision,
        )
        .await
        .expect("auto-language request");
        let three_speakers =
            RevAsrEvidenceRequest::from_audio(&first, &english(), speakers(3), &revision)
                .await
                .expect("three-speaker request");
        write_pcm_wav(&renamed, 1);
        let changed_media =
            RevAsrEvidenceRequest::from_audio(&renamed, &english(), speakers(2), &revision)
                .await
                .expect("changed-media request");

        assert_eq!(baseline.cache_key(), copied.cache_key());
        assert_eq!(baseline.cache_key(), misnamed_copy.cache_key());
        // The key is the hash of this exact spelling of the request; the
        // recipe and declaration of a WAV are the ones introduced 2026-11-03.
        let digest = &baseline.provider_media.presentation.provider_media_blake3.0;
        assert_eq!(
            baseline.cache_key().as_str(),
            blake3::hash(
                legacy_key_material(
                    digest,
                    "source_bytes_declared_container_v1",
                    "wav",
                    "audio/wav"
                )
                .as_bytes()
            )
            .to_hex()
            .as_str()
        );
        assert_ne!(baseline.cache_key(), other_container.cache_key());
        assert_ne!(baseline.cache_key(), auto_language.cache_key());
        assert_ne!(baseline.cache_key(), three_speakers.cache_key());
        assert_ne!(baseline.cache_key(), changed_media.cache_key());
    }

    /// The request-key material for English, two speakers, at request
    /// identity revision 2, spelled as the wire has it: an independent
    /// restatement of the formula, so a drift in the serialized shape is a
    /// changed key the test sees.
    fn legacy_key_material(digest: &str, recipe: &str, extension: &str, mime: &str) -> String {
        format!(
            r#"{{"domain":"rev_asr_evidence","schema_version":2,"provider_media_blake3":"{digest}","preparation_recipe":"{recipe}","upload_file_name":"provider-media.{extension}","upload_mime":"{mime}","upload_metadata":"batchalign3_{}","requested_language":"eng","expected_speakers":2,"request_policy_revision":"langid-skip-itn-v1","model_revision":"revai:asynchronous-transcript-v1"}}"#,
            &digest[..16]
        )
    }

    /// Every container Rev.AI is sent declares its own type and file name,
    /// and MPEG audio alone keeps the historical recipe, because its
    /// declaration never changed.
    #[test]
    fn each_provider_container_declares_its_own_type_and_name() {
        use RevMediaPreparationRecipe::{
            SourceBytesDeclaredContainerV1, SourceBytesLegacyAudioMpegV1,
        };
        for (container, mime, extension, recipe) in [
            (
                ProviderMediaContainer::Mp3,
                "audio/mpeg",
                "mp3",
                SourceBytesLegacyAudioMpegV1,
            ),
            (
                ProviderMediaContainer::Wav,
                "audio/wav",
                "wav",
                SourceBytesDeclaredContainerV1,
            ),
            (
                ProviderMediaContainer::Mp4,
                "audio/mp4",
                "mp4",
                SourceBytesDeclaredContainerV1,
            ),
            (
                ProviderMediaContainer::Flac,
                "audio/flac",
                "flac",
                SourceBytesDeclaredContainerV1,
            ),
            (
                ProviderMediaContainer::Ogg,
                "audio/ogg",
                "ogg",
                SourceBytesDeclaredContainerV1,
            ),
            (
                ProviderMediaContainer::Aac,
                "audio/aac",
                "aac",
                SourceBytesDeclaredContainerV1,
            ),
        ] {
            assert_eq!(container.mime(), mime, "{container:?}");
            assert_eq!(container.extension(), extension, "{container:?}");
            assert_eq!(container.recipe(), recipe, "{container:?}");
            assert_eq!(
                ProviderMediaContainer::from_upload_extension(extension),
                Some(container)
            );
            let presentation = RevProviderPresentation::for_container(
                container,
                RevProviderMediaDigest::from_bytes(b"bytes"),
            );
            assert_eq!(
                presentation.upload_file_name,
                format!("provider-media.{extension}")
            );
            assert_eq!(presentation.upload_mime, mime);
            assert_eq!(presentation.preparation_recipe, recipe);
        }
        assert_eq!(ProviderMediaContainer::from_upload_extension("media"), None);
        for container in [MediaContainer::Matroska, MediaContainer::Asf] {
            assert!(
                ProviderMediaContainer::try_from(container).is_err(),
                "{container:?} is converted before any engine, never uploaded as is"
            );
        }
    }

    /// RED FIRST: the declaration came from the recipe, which said
    /// `audio/mpeg` for everything, and the name from the source's extension.
    /// Both come from the probed container: a WAV named `.mp3` is declared a
    /// WAV and an MP3 named `.wav` an MP3, and MPEG audio's request key is,
    /// byte for byte, the legacy formula's, so no MP3 evidence is re-bought.
    #[tokio::test]
    async fn provider_media_is_declared_by_its_probed_container_not_its_name() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let wav_named_mp3 = tempdir.path().join("recording.mp3");
        write_pcm_wav(&wav_named_mp3, 0);
        let mp3_named_wav = tempdir.path().join("recording.wav");
        write_unpadded_cbr_mp3(&mp3_named_wav);

        let wav = PreparedRevProviderMedia::from_source(&wav_named_mp3)
            .await
            .expect("a WAV")
            .verify()
            .expect("unchanged");
        assert_eq!(wav.upload_mime, "audio/wav");
        assert_eq!(wav.upload_file_name, "provider-media.wav");

        let mp3 = PreparedRevProviderMedia::from_source(&mp3_named_wav)
            .await
            .expect("an MP3");
        assert_eq!(mp3.presentation.upload_mime, "audio/mpeg");
        assert_eq!(mp3.presentation.upload_file_name, "provider-media.mp3");
        assert_eq!(
            mp3.presentation.preparation_recipe,
            RevMediaPreparationRecipe::SourceBytesLegacyAudioMpegV1
        );
        let digest = mp3.presentation.provider_media_blake3.0.clone();
        let request = RevAsrEvidenceRequest::new(
            mp3,
            &english(),
            speakers(2),
            &RevAsrModelRevision::current(),
        )
        .expect("request");
        assert_eq!(
            request.cache_key().as_str(),
            blake3::hash(
                legacy_key_material(
                    &digest,
                    "source_bytes_legacy_audio_mpeg_v1",
                    "mp3",
                    "audio/mpeg"
                )
                .as_bytes()
            )
            .to_hex()
            .as_str(),
            "an MP3's key is the pre-2026-11-03 key"
        );
    }

    /// Encode a short tone with ffmpeg into `path`, in whatever container
    /// `codec_args` and the extension select. Requires ffmpeg, as the probe's
    /// own boundary tests do.
    fn encode_tone(path: &Path, codec_args: &[&str]) {
        let status = MediaTool::Ffmpeg
            .command()
            .args(["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("sine=frequency=440:sample_rate=16000:duration=200ms")
            .args(codec_args)
            .arg(path)
            .status()
            .expect("test requires ffmpeg");
        assert!(status.success(), "could not encode {}", path.display());
    }

    /// Media in a container the probe has not measured (AIFF), or one BA3
    /// does not upload as is (Matroska), is refused at the boundary naming
    /// the file; nothing is declared `audio/mpeg` for it.
    #[tokio::test]
    async fn a_container_not_sent_to_rev_is_refused_naming_the_file() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let aiff = tempdir.path().join("tone.aiff");
        encode_tone(&aiff, &["-acodec", "pcm_s16be"]);
        match PreparedRevProviderMedia::from_source(&aiff).await {
            Err(RevAsrEvidenceCacheError::ProviderMediaProbe(
                ProbeError::UnmeasuredContainer { input, demuxer },
            )) => {
                assert_eq!(demuxer, "aiff");
                assert!(input.ends_with("tone.aiff"), "{input}");
            }
            Err(other) => panic!("expected an unmeasured-container refusal, got {other}"),
            Ok(_) => panic!("AIFF has no true declaration here"),
        }

        let matroska = tempdir.path().join("tone.mka");
        encode_tone(&matroska, &["-acodec", "pcm_s16le", "-f", "matroska"]);
        match PreparedRevProviderMedia::from_source(&matroska).await {
            Err(RevAsrEvidenceCacheError::UnsupportedProviderContainer { path, refusal }) => {
                assert_eq!(path, matroska);
                assert_eq!(refusal.container, MediaContainer::Matroska);
                assert!(refusal.to_string().contains("WAV"), "{refusal}");
            }
            Err(other) => panic!("expected a container refusal, got {other}"),
            Ok(_) => panic!("Matroska is not uploaded to Rev.AI as is"),
        }
    }

    #[tokio::test]
    async fn storage_upgrade_preserves_request_key_and_replays_schema_two_evidence() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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

    /// Evidence whose transcript language its request could not have resolved
    /// to is refused at replay, not handed to the pipeline: an English request
    /// never yields a Spanish transcript.
    #[tokio::test]
    async fn stored_evidence_in_a_language_its_request_cannot_resolve_to_is_refused() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        request
            .store_unchecked_for_test(
                &cache,
                serde_json::json!({
                    "schema_version": 2,
                    "request_fingerprint": request.cache_key().as_str(),
                    "evidence": {
                        "transcript": {"monologues": []},
                        "resolved_language": "spa"
                    }
                }),
            )
            .await
            .expect("seed contradictory evidence");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };

        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect_err("a Spanish transcript cannot answer an English request");
        assert!(
            error.to_string().contains("cannot come from a request"),
            "{error}"
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn contradictory_fresh_evidence_is_refused_before_commit() {
        struct Contradictory;
        #[async_trait::async_trait]
        impl RevAsrEvidenceInference for Contradictory {
            async fn infer(
                &self,
                _run: AuthorizedRevEvidenceRun,
            ) -> Result<RevAsrInferenceOutcome, ServerError> {
                let mut evidence = sample_evidence();
                evidence.resolved_language = TranscriptLanguage::One(LanguageCode3::spa());
                Ok(RevAsrInferenceOutcome::Fetched(evidence))
            }
        }
        let mut request = language_request();
        // Build the request through admission so its fingerprint is English too.
        request = RevAsrEvidenceRequest::new(
            request.provider_media,
            &english(),
            crate::transcribe::SpeakerCount::Auto,
            &RevAsrModelRevision::current(),
        )
        .unwrap();
        let backend = RecordingBackend::default();
        let cache = UtteranceCache::from_backend(Box::new(backend.clone()));
        let error =
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &Contradictory)
                .await
                .unwrap_err();
        assert!(
            error.to_string().contains("cannot come from a request"),
            "{error}"
        );
        assert!(
            backend.0.lock().unwrap().is_empty(),
            "contradictory evidence must not become a cache hit"
        );
        // Rejection releases the miss lease, and a valid response can still complete.
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .unwrap();
        resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service)
            .await
            .unwrap();
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn prepared_media_refuses_bytes_changed_before_authorized_submission() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        write_pcm_wav(&audio, 0);
        let prepared = PreparedRevProviderMedia::from_source(&audio)
            .await
            .expect("prepare");
        write_pcm_wav(&audio, 1);

        assert!(matches!(
            prepared.verify(),
            Err(RevAsrEvidenceCacheError::ProviderMediaDrift(path)) if path == audio
        ));
    }

    /// The presentation is the file's: the bytes as they are, declared by
    /// the probed container (not the `.WAV` spelling of the name), and named
    /// and labelled from the digest of those bytes.
    #[tokio::test]
    async fn verified_media_has_stable_provider_visible_presentation() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("Original.WAV");
        write_pcm_wav(&audio, 0);
        let bytes = std::fs::read(&audio).expect("fixture bytes");

        let verified = PreparedRevProviderMedia::from_source(&audio)
            .await
            .expect("prepare")
            .verify()
            .expect("verify");
        let digest = RevProviderMediaDigest::from_bytes(&bytes);

        assert_eq!(verified.bytes, bytes);
        assert_eq!(verified.upload_file_name, "provider-media.wav");
        assert_eq!(verified.upload_mime, "audio/wav");
        assert_eq!(
            verified.metadata,
            format!("batchalign3_{}", &digest.0[..16])
        );
    }

    #[tokio::test]
    async fn evidence_trace_joins_media_request_cache_and_projection_identity() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("Original.WAV");
        write_pcm_wav(&audio, 0);
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");

        let resolution = RevAsrEvidenceResolution {
            evidence: CompletedRevAsrEvidence::admit(sample_evidence(), &english()).unwrap(),
            source: RevAsrEvidenceSource::Inferred(RevAsrEvidenceMissReason::NotFound),
            trace_seed: request.trace_seed(),
        };
        let value = serde_json::to_value(resolution.trace(RevAsrProjectionRevision::AsrResponseV1))
            .expect("trace JSON");

        assert_eq!(value["trace_schema_version"], 2);
        assert_eq!(
            value["preparation_recipe"],
            "source_bytes_declared_container_v1"
        );
        assert_eq!(value["source_media_blake3"], value["provider_media_blake3"]);
        assert_eq!(value["upload_file_name"], "provider-media.wav");
        assert_eq!(value["upload_mime"], "audio/wav");
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
        write_pcm_wav(&audio, 0);
        let cache_dir = tempdir.path().join("cache");
        let cache = UtteranceCache::sqlite(Some(cache_dir.clone()))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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
        let cold_response = crate::revai::rev_evidence_to_asr_response(&cold_evidence)
            .expect("an admissible fixture");
        drop(cache);

        let reopened = UtteranceCache::sqlite(Some(cache_dir))
            .await
            .expect("reopen");
        let warm = resolve_rev_asr_evidence(&request, &reopened, CachePolicy::UseCache, &service)
            .await
            .expect("warm");
        assert_eq!(warm.source(), RevAsrEvidenceSource::Replayed);
        let warm_evidence = warm.into_evidence();
        let warm_response = crate::revai::rev_evidence_to_asr_response(&warm_evidence)
            .expect("an admissible fixture");
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
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::from_backend(Box::new(FailingCommitBackend));
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
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

    /// What a scripted Rev boundary answers each authorized run with.
    enum ScriptedInference {
        /// The wait ended with the job still Rev.AI's.
        Unfinished(UnfinishedRevJob),
        /// The transcript, fetched and admissible.
        Fetched,
        /// A provider failure with this verdict.
        Failed(crate::error::AsrProviderDisposition),
    }

    /// A Rev boundary that answers from a script and records the route each
    /// run was authorized with, so a test can see whether it was told to
    /// submit or to resume.
    struct RoutedService {
        script: std::sync::Mutex<std::collections::VecDeque<ScriptedInference>>,
        routes: std::sync::Mutex<Vec<RevJobRoute>>,
    }

    impl RoutedService {
        fn new(script: Vec<ScriptedInference>) -> Self {
            Self {
                script: std::sync::Mutex::new(script.into()),
                routes: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn routes(&self) -> Vec<RevJobRoute> {
            self.routes.lock().expect("test: routes").clone()
        }
    }

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for RoutedService {
        async fn infer(
            &self,
            run: AuthorizedRevEvidenceRun,
        ) -> Result<RevAsrInferenceOutcome, ServerError> {
            self.routes
                .lock()
                .expect("test: routes")
                .push(run.route.clone());
            let next = self
                .script
                .lock()
                .expect("test: script")
                .pop_front()
                .expect("test: the script covers every run");
            match next {
                ScriptedInference::Unfinished(job) => Ok(RevAsrInferenceOutcome::Unfinished(
                    UnfinishedRevSubmission {
                        job,
                        effective_language: run.requested_language,
                    },
                )),
                ScriptedInference::Fetched => {
                    Ok(RevAsrInferenceOutcome::Fetched(sample_evidence()))
                }
                ScriptedInference::Failed(disposition) => Err(ServerError::AsrProvider {
                    disposition,
                    message: "scripted provider failure".to_owned(),
                }),
            }
        }
    }

    fn unfinished_job(id: &str) -> UnfinishedRevJob {
        UnfinishedRevJob {
            job: RevAiJobId::try_new(id).expect("a job id"),
            cause: UnfinishedCause::DeadlineExceeded {
                deadline: super::super::poll::PollDeadline::of(std::time::Duration::from_secs(1)),
                elapsed: std::time::Duration::from_secs(2),
            },
        }
    }

    /// The pending-job record for `request`, if the cache holds one.
    async fn pending_record(
        cache: &UtteranceCache,
        request: &RevAsrEvidenceRequest,
    ) -> Option<serde_json::Value> {
        cache
            .get(
                &pending_job_key(request.cache_key()),
                tasks::REV_ASR_EVIDENCE,
                &RevAsrModelRevision::current(),
            )
            .await
            .expect("test: cache read")
    }

    async fn pending_fixture() -> (tempfile::TempDir, UtteranceCache, RevAsrEvidenceRequest) {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        (tempdir, cache, request)
    }

    /// RED FIRST: a wait that ended before the job finished left no record, so
    /// the next run of the same request uploaded and paid again. The job is
    /// recorded by name, the failure is reported transient (so the control
    /// plane retries), the retry is told to RESUME that job, and once its
    /// transcript is committed the record is gone and later runs replay.
    #[tokio::test]
    async fn an_unfinished_job_is_recorded_and_resumed_by_the_next_run() {
        let (_tempdir, cache, request) = pending_fixture().await;
        let service = RoutedService::new(vec![
            ScriptedInference::Unfinished(unfinished_job("job-42")),
            ScriptedInference::Fetched,
        ]);

        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect_err("the wait ended with the job unfinished");
        let RevAsrEvidenceResolutionError::Unfinished(retained) = error else {
            panic!("an unfinished job is its own resolution error, got {error}");
        };
        assert!(retained.to_string().contains("job-42"), "{retained}");
        assert!(
            pending_record(&cache, &request).await.is_some(),
            "the job is recorded against its request"
        );
        match rev_asr_resolution_error_to_server_error(RevAsrEvidenceResolutionError::Unfinished(
            retained,
        )) {
            ServerError::AsrProvider {
                disposition: crate::error::AsrProviderDisposition::Transient,
                message,
            } => assert!(message.contains("job-42"), "{message}"),
            other => panic!("an unfinished job is a transient provider failure, got {other}"),
        }

        let resumed = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("the resumed job finishes");
        assert_eq!(
            resumed.source(),
            RevAsrEvidenceSource::Inferred(RevAsrEvidenceMissReason::NotFound)
        );
        assert_eq!(
            service.routes(),
            vec![
                RevJobRoute::Submit,
                RevJobRoute::Resume(PendingRevJob {
                    job: RevAiJobId::try_new("job-42").expect("a job id"),
                    effective_language: english(),
                }),
            ],
            "the second run resumes the recorded job instead of submitting"
        );
        assert!(
            pending_record(&cache, &request).await.is_none(),
            "a committed transcript ends the record"
        );

        let replayed = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("replay");
        assert_eq!(replayed.source(), RevAsrEvidenceSource::Replayed);
        assert_eq!(service.routes().len(), 2, "replay crosses no boundary");
    }

    /// Rev.AI's terminal verdict on a resumed job (failed, or gone) ends the
    /// record, so the run after that submits afresh; a transient failure of
    /// the resumed wait keeps the record, so the run after that resumes again.
    #[tokio::test]
    async fn a_resumed_jobs_record_follows_the_providers_verdict() {
        let (_tempdir, cache, request) = pending_fixture().await;
        let service = RoutedService::new(vec![
            ScriptedInference::Unfinished(unfinished_job("job-7")),
            ScriptedInference::Failed(crate::error::AsrProviderDisposition::Transient),
            ScriptedInference::Failed(crate::error::AsrProviderDisposition::Terminal),
            ScriptedInference::Fetched,
        ]);
        let resolve =
            || resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service);

        resolve().await.expect_err("unfinished");
        resolve()
            .await
            .expect_err("transient failure of the resumed wait");
        assert!(
            pending_record(&cache, &request).await.is_some(),
            "a transient failure leaves the job, and its record, in place"
        );
        resolve()
            .await
            .expect_err("terminal verdict on the resumed job");
        assert!(
            pending_record(&cache, &request).await.is_none(),
            "a terminal verdict ends the record"
        );
        resolve().await.expect("a fresh submission");
        let routes = service.routes();
        assert_eq!(routes[0], RevJobRoute::Submit);
        assert!(matches!(routes[1], RevJobRoute::Resume(_)));
        assert!(matches!(routes[2], RevJobRoute::Resume(_)));
        assert_eq!(routes[3], RevJobRoute::Submit);
    }

    /// A pending-job record this build cannot read, or one for another
    /// request, is refused by name without a provider call: ignoring it would
    /// upload and pay again, which is what the record exists to prevent.
    #[tokio::test]
    async fn a_corrupt_pending_record_is_refused_without_a_provider_call() {
        let (_tempdir, cache, request) = pending_fixture().await;
        cache
            .put(
                &pending_job_key(request.cache_key()),
                tasks::REV_ASR_EVIDENCE,
                &RevAsrModelRevision::current(),
                &serde_json::json!({
                    "pending_schema_version": 1,
                    "request_fingerprint": "another request",
                    "job": "job-1",
                    "effective_language": "eng",
                    "cause": {"kind": "deadline_exceeded", "deadline": {"secs": 1, "nanos": 0}, "elapsed": {"secs": 2, "nanos": 0}}
                }),
            )
            .await
            .expect("test: seed the record");
        let service = RoutedService::new(vec![ScriptedInference::Fetched]);

        let error = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect_err("a record for another request is refused");
        assert!(
            matches!(
                error,
                RevAsrEvidenceResolutionError::Evidence(
                    RevAsrEvidenceCacheError::InvalidPendingJob { .. }
                )
            ),
            "{error}"
        );
        assert!(service.routes().is_empty(), "no provider call");
    }

    /// Evidence a WAV recording was cached under before 2026-11-03 (declared
    /// `audio/mpeg`, named after the path) is found under that key on the
    /// first run under the new key, stored under the new key, and used
    /// without a provider call; the run's trace says it was replayed from the
    /// legacy key. The second run hits the new key directly.
    #[tokio::test]
    async fn a_wav_cached_under_the_legacy_key_is_found_once_and_then_hit_directly() {
        let (_tempdir, cache, request) = legacy_fixture().await;
        let revision = RevAsrModelRevision::current();
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let migrated = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("the legacy entry serves the request");
        assert_eq!(
            migrated.source(),
            RevAsrEvidenceSource::MigratedFromLegacyKey
        );
        assert_eq!(
            migrated
                .trace(RevAsrProjectionRevision::AsrResponseV1)
                .cache_outcome(),
            RevAsrEvidenceCacheOutcome::ReplayedFromLegacyKey
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 0, "no provider call");
        let under_new_key = cache
            .get(
                request.cache_key().as_str(),
                tasks::REV_ASR_EVIDENCE,
                &revision,
            )
            .await
            .expect("cache read")
            .expect("the evidence is stored under the current key");
        assert_eq!(
            under_new_key["request_fingerprint"],
            request.cache_key().to_string(),
            "with the current key as its fingerprint"
        );

        let direct =
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service)
                .await
                .expect("the second run needs no fallback");
        assert_eq!(direct.source(), RevAsrEvidenceSource::Replayed);
        assert_eq!(service.calls.load(Ordering::SeqCst), 0);
    }

    /// MPEG audio kept its key, so it has no legacy key to consult: a cold
    /// MP3 request goes to the provider once, consulting nothing else.
    #[tokio::test]
    async fn an_mp3_request_has_no_legacy_key_to_consult() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("recording.mp3");
        write_unpadded_cbr_mp3(&audio);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &english(),
            speakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        assert!(
            !request.has_distinct_legacy_key(),
            "an MP3's key is the key it always had"
        );
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let cold = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("cold inference");
        assert_eq!(
            cold.source(),
            RevAsrEvidenceSource::Inferred(RevAsrEvidenceMissReason::NotFound)
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    }

    /// Whether a second key exists is decided by comparing presentations, not
    /// by container: MPEG audio at a `.mp3` path has one key then and now,
    /// while the same bytes at an `.mpga` path, or at a path with no
    /// extension, were keyed by that name before and by `mp3` since.
    #[test]
    fn an_mpeg_file_not_named_mp3_consults_the_key_its_name_gave_it() {
        let digest = RevProviderMediaDigest::from_bytes(b"provider media");
        let request = |path: &str| {
            RevAsrEvidenceRequest::new(
                PreparedRevProviderMedia::new(
                    PathBuf::from(path),
                    digest.clone(),
                    ProviderMediaContainer::Mp3,
                    digest.clone(),
                ),
                &english(),
                speakers(2),
                &RevAsrModelRevision::current(),
            )
            .unwrap()
        };
        assert!(!request("recording.mp3").has_distinct_legacy_key());
        assert!(
            !request("recording.MP3").has_distinct_legacy_key(),
            "the legacy name lower-cased the extension"
        );
        assert!(request("recording.mpga").has_distinct_legacy_key());
        assert!(request("recording").has_distinct_legacy_key());
    }

    /// A required cache that holds the evidence under the legacy key has it:
    /// the first run under the new key is a hit, not a refusal.
    #[tokio::test]
    async fn a_legacy_entry_satisfies_a_required_cache_on_the_first_run() {
        let (_tempdir, cache, request) = legacy_fixture().await;
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let migrated =
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::RequireCache, &service)
                .await
                .expect("a required cache holding the evidence under its old key has it");
        assert_eq!(
            migrated.source(),
            RevAsrEvidenceSource::MigratedFromLegacyKey
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 0, "no provider call");
    }

    /// A forced refresh bypasses the cache, the legacy key included: the
    /// provider is called, and the legacy entry is neither read nor moved.
    #[tokio::test]
    async fn a_forced_refresh_consults_no_legacy_key() {
        let (_tempdir, cache, request) = legacy_fixture().await;
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let refreshed =
            resolve_rev_asr_evidence(&request, &cache, CachePolicy::SkipCache, &service)
                .await
                .expect("a forced refresh infers");
        assert_eq!(
            refreshed.source(),
            RevAsrEvidenceSource::Inferred(RevAsrEvidenceMissReason::ForcedRefresh)
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 1, "one provider call");
    }

    /// A job an earlier run submitted under today's key and did not see finish
    /// is superseded by evidence found under the legacy key: the record goes
    /// with the migration, as it goes with a commit.
    #[tokio::test]
    async fn a_migration_supersedes_a_pending_job_record_under_the_current_key() {
        let (_tempdir, cache, request) = legacy_fixture().await;
        cache
            .put(
                &pending_job_key(request.cache_key()),
                tasks::REV_ASR_EVIDENCE,
                &RevAsrModelRevision::current(),
                &serde_json::json!({
                    "pending_schema_version": 1,
                    "request_fingerprint": request.cache_key().to_string(),
                    "job": "job-1",
                    "effective_language": "eng",
                    "cause": {"kind": "deadline_exceeded", "deadline": {"secs": 1, "nanos": 0}, "elapsed": {"secs": 2, "nanos": 0}}
                }),
            )
            .await
            .expect("pending record");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
            delay_ms: 0,
        };
        let migrated = resolve_rev_asr_evidence(&request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("the legacy entry serves the request");
        assert_eq!(
            migrated.source(),
            RevAsrEvidenceSource::MigratedFromLegacyKey
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 0, "no provider call");
        assert!(
            pending_record(&cache, &request).await.is_none(),
            "the superseded job's record is gone"
        );
    }

    /// A WAV request whose evidence a run before 2026-11-03 stored: under the
    /// legacy key's material (declared `audio/mpeg`, named after the path), in
    /// an envelope fingerprinted with that key.
    async fn legacy_fixture() -> (tempfile::TempDir, UtteranceCache, RevAsrEvidenceRequest) {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("recording.wav");
        write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let revision = RevAsrModelRevision::current();
        let request = RevAsrEvidenceRequest::from_audio(&audio, &english(), speakers(2), &revision)
            .await
            .expect("request");
        assert!(
            request.has_distinct_legacy_key(),
            "a WAV was keyed differently before"
        );
        let digest = &request.provider_media.presentation.provider_media_blake3.0;
        let legacy_key = CacheKey::from_content(&legacy_key_material(
            digest,
            "source_bytes_legacy_audio_mpeg_v1",
            "wav",
            "audio/mpeg",
        ));
        assert_ne!(&legacy_key, request.cache_key());
        assert_eq!(request.legacy_key(), Some(&legacy_key));
        let stored = serde_json::to_value(StoredRevAsrEvidence {
            schema_version: REV_ASR_EVIDENCE_STORAGE_SCHEMA_VERSION,
            request_fingerprint: legacy_key.to_string(),
            evidence: sample_evidence(),
        })
        .expect("envelope");
        cache
            .put(
                legacy_key.as_str(),
                tasks::REV_ASR_EVIDENCE,
                &revision,
                &stored,
            )
            .await
            .expect("legacy entry");
        (tempdir, cache, request)
    }
}
