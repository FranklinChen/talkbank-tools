//! Read-only replay inspection. No prepared media, inference capability,
//! writable cache backend or schema migration enters this module.

use super::*;
use crate::api::{AsrLanguageRequest, LanguageSpec};
use sqlx::{ConnectOptions, Connection, Row, SqliteConnection};

/// A request identity validated against this build, not authority to upload.
pub(super) struct AdmittedRevReplayRequest {
    pub(super) key: CacheKey,
    pub(super) language: RevLanguage,
}

#[derive(Deserialize)]
struct RetainedRequestTrace {
    trace_schema_version: u32,
    source_media_blake3: String,
    provider_media_blake3: String,
    preparation_recipe: String,
    upload_file_name: String,
    upload_mime: String,
    upload_metadata: String,
    requested_language: String,
    #[serde(deserialize_with = "required_speaker_field")]
    expected_speakers: Option<u32>,
    request_policy_revision: String,
    model_revision: String,
    raw_evidence_key: CacheKey,
}

fn required_speaker_field<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u32>, D::Error> {
    Option::<u32>::deserialize(deserializer)
}

impl AdmittedRevReplayRequest {
    pub(super) fn from_production(request: &RevAsrEvidenceRequest) -> Self {
        Self {
            key: request.cache_key.clone(),
            language: request.requested_language.clone(),
        }
    }

    fn from_trace(bytes: &[u8]) -> Result<Self, RevAsrEvidenceCacheError> {
        let trace: RetainedRequestTrace = serde_json::from_slice(bytes)?;
        let refuse = |reason: &str| RevAsrEvidenceCacheError::InvalidRequest(reason.to_owned());
        let valid_digest = |digest: &str| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        let recipe = RevMediaPreparationRecipe::SourceBytesLegacyAudioMpegV1;
        let extension = trace
            .upload_file_name
            .strip_prefix("provider-media.")
            .filter(|ext| {
                !ext.is_empty()
                    && ext
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            });
        if trace.trace_schema_version != REV_ASR_TRACE_SCHEMA_VERSION
            || !valid_digest(&trace.source_media_blake3)
            || trace.source_media_blake3 != trace.provider_media_blake3
            || serde_json::to_value(&trace.preparation_recipe)? != serde_json::to_value(recipe)?
            || trace.upload_mime != recipe.upload_mime()
            || extension.is_none()
            || trace.upload_metadata
                != format!("batchalign3_{}", &trace.provider_media_blake3[..16])
            || trace.request_policy_revision != REV_ASR_REQUEST_POLICY_REVISION
            || trace.model_revision != RevAsrModelRevision::current().as_str()
            || trace.expected_speakers == Some(0)
        {
            return Err(refuse(
                "retained presentation is incompatible with this build",
            ));
        }
        let spec: LanguageSpec =
            serde_json::from_value(serde_json::Value::String(trace.requested_language.clone()))
                .map_err(|_| refuse("invalid retained language"))?;
        let language_request = AsrLanguageRequest::try_from(&spec)
            .map_err(|_| refuse("retained language is not an ASR request"))?;
        let language = RevLanguage::admit(&language_request)
            .map_err(|_| refuse("unsupported retained Rev language"))?;
        if language.to_string() != trace.requested_language {
            return Err(refuse("retained language is not canonical"));
        }
        let presentation = RevProviderPresentation {
            provider_media_blake3: RevProviderMediaDigest(trace.provider_media_blake3),
            preparation_recipe: recipe,
            upload_file_name: trace.upload_file_name,
            upload_mime: recipe.upload_mime(),
            upload_metadata: trace.upload_metadata,
        };
        let revision = RevAsrModelRevision::current();
        let material = RevAsrEvidenceKeyMaterial {
            domain: "rev_asr_evidence",
            request_identity_revision: REV_ASR_REQUEST_IDENTITY_REVISION,
            provider_presentation: &presentation,
            requested_language: language.to_string(),
            expected_speakers: trace.expected_speakers,
            request_policy_revision: REV_ASR_REQUEST_POLICY_REVISION,
            model_revision: revision.as_str(),
        };
        let key = CacheKey::from_content(&serde_json::to_string(&material)?);
        if key != trace.raw_evidence_key {
            return Err(refuse(
                "retained raw key does not match the current request fingerprint",
            ));
        }
        Ok(Self { key, language })
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum InspectedEvidence {
    Missing,
    NamespaceMismatch { recorded: String, expected: String },
    Refused { reason: String },
    Replayable { fidelity: RevTranscriptFidelity },
    DerivedReplayable,
}

#[derive(Debug, Serialize)]
pub(crate) struct RevCacheInspection {
    raw_evidence_key: String,
    raw: InspectedEvidence,
    derived_utr: Option<UtrCacheInspection>,
    /// Retained trace compatibility is not a fresh media-content attestation.
    recording_bytes_revalidated: bool,
}

impl RevCacheInspection {
    pub(crate) fn raw_is_replayable(&self) -> bool {
        matches!(self.raw, InspectedEvidence::Replayable { .. })
    }
}

/// A metadata-key inspection is not a recording-content attestation.
#[derive(Debug, Serialize)]
pub(crate) struct UtrCacheInspection {
    cache_key: String,
    namespace: String,
    audio_identity: String,
    evidence: InspectedEvidence,
    recording_bytes_revalidated: bool,
}

impl UtrCacheInspection {
    pub(crate) fn is_replayable(&self) -> bool {
        matches!(self.evidence, InspectedEvidence::DerivedReplayable)
    }
}

async fn inspect_utr_response(
    connection: &mut SqliteConnection,
    audio: &Path,
    language: &LanguageCode3,
    engine: &crate::options::UtrEngine,
) -> Result<UtrCacheInspection, crate::cli::error::CliError> {
    use crate::cli::error::CliError;
    let path = audio.canonicalize()?;
    let metadata = path.metadata()?;
    if !metadata.is_file() {
        return Err(CliError::InvalidArgument(
            "UTR audio path must identify a regular file".into(),
        ));
    }
    let modified = metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| CliError::InvalidArgument(error.to_string()))?
        .as_secs();
    let path_text = path
        .to_str()
        .ok_or_else(|| CliError::InvalidArgument("audio path is not UTF-8".into()))?;
    let identity =
        crate::chat_ops::fa::AudioIdentity::from_metadata(path_text, modified, metadata.len());
    let models = crate::model_manifest::utr_pinned_models(engine, language)
        .map_err(|error| CliError::InvalidArgument(error.to_string()))?;
    let crate::cache::UtrAsrCacheEligibility::Pinned(namespace) =
        crate::cache::UtrAsrCacheNamespace::for_pinned_plan(engine, &models)
    else {
        return Err(CliError::InvalidArgument(
            "floating UTR model plan cannot admit a cache row".into(),
        ));
    };
    let key = crate::chat_ops::fa::utr_asr_cache_key(&identity, engine, language);
    let evidence = match read_row(
        connection,
        &key,
        crate::chat_ops::CacheTaskName::UtrAsr.as_str(),
        namespace.namespace(),
    )
    .await?
    {
        Err(status) => status,
        Ok(value) => match serde_json::from_value::<crate::transcribe::AsrResponse>(value) {
            Ok(_) => InspectedEvidence::DerivedReplayable,
            Err(error) => InspectedEvidence::Refused {
                reason: error.to_string(),
            },
        },
    };
    Ok(UtrCacheInspection {
        cache_key: key.to_string(),
        namespace: namespace.namespace().to_owned(),
        audio_identity: identity.to_string(),
        evidence,
        recording_bytes_revalidated: false,
    })
}

/// Inspect derived UTR evidence independently: no raw trace need exist when
/// production can already replay a compatible derived response.
pub(crate) async fn inspect_retained_utr_cache(
    audio: &Path,
    db: &Path,
    language: &str,
    engine: &crate::options::UtrEngine,
) -> Result<UtrCacheInspection, crate::cli::error::CliError> {
    use crate::cli::error::CliError;
    let language: LanguageSpec =
        serde_json::from_value(serde_json::Value::String(language.to_owned()))
            .map_err(|error| CliError::InvalidArgument(error.to_string()))?;
    let LanguageSpec::Resolved(language) = language else {
        return Err(CliError::InvalidArgument(
            "UTR inspection requires one explicit language".into(),
        ));
    };
    let mut connection = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(db)
        .read_only(true)
        .create_if_missing(false)
        .connect()
        .await?;
    let report = inspect_utr_response(&mut connection, audio, &language, engine).await?;
    connection.close().await?;
    Ok(report)
}

async fn read_row(
    connection: &mut SqliteConnection,
    key: &CacheKey,
    task: &str,
    namespace: &str,
) -> Result<Result<serde_json::Value, InspectedEvidence>, sqlx::Error> {
    let row =
        sqlx::query("SELECT data, engine_version FROM cache_entries WHERE key = ? AND task = ?")
            .bind(key.as_str())
            .bind(task)
            .fetch_optional(connection)
            .await?;
    let Some(row) = row else {
        return Ok(Err(InspectedEvidence::Missing));
    };
    let recorded: String = row.try_get("engine_version")?;
    if recorded != namespace {
        return Ok(Err(InspectedEvidence::NamespaceMismatch {
            recorded,
            expected: namespace.to_owned(),
        }));
    }
    let bytes: String = row.try_get("data")?;
    Ok(
        serde_json::from_str(&bytes).map_err(|error| InspectedEvidence::Refused {
            reason: error.to_string(),
        }),
    )
}

/// Read a retained trace and existing cache only. An optional audio path is
/// inspected for filesystem metadata, never read/hashed or decoded. Its content
/// binding to the trace remains an external provenance requirement.
pub(crate) async fn inspect_retained_rev_cache(
    trace: &Path,
    db: &Path,
    utr_audio: Option<&Path>,
) -> Result<RevCacheInspection, crate::cli::error::CliError> {
    use crate::cli::error::CliError;
    let request = AdmittedRevReplayRequest::from_trace(&std::fs::read(trace)?)
        .map_err(|error| CliError::InvalidArgument(error.to_string()))?;
    let mut connection = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(db)
        .read_only(true)
        .create_if_missing(false)
        .connect()
        .await?;
    let raw = match read_row(
        &mut connection,
        &request.key,
        crate::chat_ops::CacheTaskName::RevAsrEvidence.as_str(),
        RevAsrModelRevision::current().as_str(),
    )
    .await?
    {
        Err(status) => status,
        Ok(value) => match StoredRevAsrEvidence::decode_for_replay(value, &request) {
            Ok(evidence) => InspectedEvidence::Replayable {
                fidelity: evidence.transcript_evidence().fidelity(),
            },
            Err(error) => InspectedEvidence::Refused {
                reason: error.to_string(),
            },
        },
    };
    let derived_utr = if let Some(audio) = utr_audio {
        let AsrLanguageRequest::One(language) = request.language.request() else {
            return Err(CliError::InvalidArgument(
                "UTR inspection requires one explicit language".into(),
            ));
        };
        Some(
            inspect_utr_response(
                &mut connection,
                audio,
                &language,
                &crate::options::UtrEngine::RevAi,
            )
            .await?,
        )
    } else {
        None
    };
    connection.close().await?;
    Ok(RevCacheInspection {
        raw_evidence_key: request.key.to_string(),
        raw,
        derived_utr,
        recording_bytes_revalidated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    async fn fixture(root: &Path) -> (RevAsrEvidenceRequest, PathBuf, PathBuf) {
        let audio = root.join("audio.wav");
        std::fs::write(&audio, b"synthetic media identity fixture").unwrap();
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &RevLanguage::admit(&AsrLanguageRequest::One(LanguageCode3::eng())).unwrap(),
            NumSpeakers(1),
            &RevAsrModelRevision::current(),
        )
        .await
        .unwrap();
        let mut trace = serde_json::to_value(request.trace_seed()).unwrap();
        trace["cache_outcome"] = json!("inferred_not_found");
        trace["transcript_fidelity"] = json!("exact_provider_json");
        trace["projection_revision"] = json!("rev-transcript-to-utr-asr-response-v1");
        let trace_path = root.join("trace.json");
        std::fs::write(&trace_path, serde_json::to_vec(&trace).unwrap()).unwrap();
        (request, trace_path, audio)
    }

    async fn database(path: &Path, request: &RevAsrEvidenceRequest, namespace: &str, value: Value) {
        let mut db = sqlx::sqlite::SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .connect()
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE cache_entries (key TEXT, task TEXT, data TEXT, engine_version TEXT)",
        )
        .execute(&mut db)
        .await
        .unwrap();
        sqlx::query("INSERT INTO cache_entries VALUES (?, 'rev_asr_evidence', ?, ?)")
            .bind(request.cache_key.as_str())
            .bind(value.to_string())
            .bind(namespace)
            .execute(&mut db)
            .await
            .unwrap();
        db.close().await.unwrap();
    }

    fn envelope(request: &RevAsrEvidenceRequest) -> Value {
        serde_json::to_value(StoredRevAsrEvidence {
            schema_version: REV_ASR_EVIDENCE_STORAGE_SCHEMA_VERSION,
            request_fingerprint: request.cache_key.to_string(),
            evidence: FetchedRevAsrEvidence {
                transcript_evidence: RevTranscriptEvidence::from_provider_json(
                    r#"{"monologues":[],"unknown":"preserved"}"#.into(),
                )
                .unwrap(),
                resolved_language: TranscriptLanguage::One(LanguageCode3::eng()),
            },
        })
        .unwrap()
    }

    #[tokio::test]
    async fn retained_cache_inspection_is_read_only_and_media_need_not_exist() {
        let root = tempfile::tempdir().unwrap();
        let (request, trace, audio) = fixture(root.path()).await;
        let db = root.path().join("cache.db");
        database(
            &db,
            &request,
            request.model_revision.as_str(),
            envelope(&request),
        )
        .await;
        let original = std::fs::read(&db).unwrap();
        std::fs::remove_file(audio).unwrap();
        let report = inspect_retained_rev_cache(&trace, &db, None).await.unwrap();
        assert!(report.raw_is_replayable());
        assert_eq!(std::fs::read(&db).unwrap(), original);
        assert_eq!(report.raw_evidence_key, request.cache_key.to_string());
        assert!(!report.recording_bytes_revalidated);
        assert!(report.derived_utr.is_none());
        insta::assert_json_snapshot!(report.raw, @r#"
        {
          "status": "replayable",
          "fidelity": "exact_provider_json"
        }
        "#);
    }

    #[tokio::test]
    async fn retained_cache_inspection_refuses_incompatible_traces_without_opening_database() {
        let root = tempfile::tempdir().unwrap();
        let (_, trace, _) = fixture(root.path()).await;
        let value: Value = serde_json::from_slice(&std::fs::read(trace).unwrap()).unwrap();
        for (field, replacement) in [
            ("trace_schema_version", json!(1)),
            ("upload_mime", json!("audio/wav")),
            ("upload_file_name", json!("original.wav")),
            ("upload_metadata", json!("other")),
            ("request_policy_revision", json!("old")),
            ("model_revision", json!("other")),
            ("expected_speakers", json!(0)),
            ("expected_speakers", json!(2)),
            ("raw_evidence_key", json!("0".repeat(64))),
            ("source_media_blake3", json!("short")),
            ("requested_language", json!("fra")),
        ] {
            let mut changed = value.clone();
            changed[field] = replacement;
            assert!(
                AdmittedRevReplayRequest::from_trace(&serde_json::to_vec(&changed).unwrap())
                    .is_err(),
                "{field}"
            );
        }
        let mut missing = value;
        missing.as_object_mut().unwrap().remove("expected_speakers");
        assert!(
            AdmittedRevReplayRequest::from_trace(&serde_json::to_vec(&missing).unwrap()).is_err()
        );
    }

    #[tokio::test]
    async fn retained_cache_inspection_distinguishes_missing_namespace_and_corruption() {
        let root = tempfile::tempdir().unwrap();
        let (request, trace, _) = fixture(root.path()).await;
        let absent = root.path().join("absent.db");
        assert!(
            inspect_retained_rev_cache(&trace, &absent, None)
                .await
                .is_err()
        );
        assert!(!absent.exists());
        for (index, namespace, value) in [
            (0, "old", envelope(&request)),
            (
                1,
                request.model_revision.as_str(),
                json!({"schema_version": 999}),
            ),
            (2, request.model_revision.as_str(), {
                let mut value = envelope(&request);
                value["request_fingerprint"] = json!("0".repeat(64));
                value
            }),
        ] {
            let db = root.path().join(format!("cache{index}.db"));
            database(&db, &request, namespace, value).await;
            let before = std::fs::read(&db).unwrap();
            let report = inspect_retained_rev_cache(&trace, &db, None).await.unwrap();
            assert!(!report.raw_is_replayable());
            assert_eq!(std::fs::read(&db).unwrap(), before);
            if index == 0 {
                assert!(matches!(
                    report.raw,
                    InspectedEvidence::NamespaceMismatch { .. }
                ));
            } else {
                assert!(matches!(report.raw, InspectedEvidence::Refused { .. }));
            }
        }
    }

    #[tokio::test]
    async fn retained_cache_inspection_keeps_derived_metadata_lookup_separate() {
        let root = tempfile::tempdir().unwrap();
        let (request, trace, audio) = fixture(root.path()).await;
        let db = root.path().join("cache.db");
        database(
            &db,
            &request,
            request.model_revision.as_str(),
            envelope(&request),
        )
        .await;
        let report = inspect_retained_rev_cache(&trace, &db, Some(&audio))
            .await
            .unwrap();
        assert!(matches!(
            report.derived_utr.as_ref().map(|report| &report.evidence),
            Some(InspectedEvidence::Missing)
        ));
        let path = audio.canonicalize().unwrap();
        let metadata = path.metadata().unwrap();
        let mtime = metadata
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let identity = crate::chat_ops::fa::AudioIdentity::from_metadata(
            path.to_str().unwrap(),
            mtime,
            metadata.len(),
        );
        let engine = crate::options::UtrEngine::RevAi;
        let models =
            crate::model_manifest::utr_pinned_models(&engine, &LanguageCode3::eng()).unwrap();
        let crate::cache::UtrAsrCacheEligibility::Pinned(namespace) =
            crate::cache::UtrAsrCacheNamespace::for_pinned_plan(&engine, &models)
        else {
            panic!("pinned Rev model")
        };
        let key = crate::chat_ops::fa::utr_asr_cache_key(&identity, &engine, "eng");
        let mut connection = sqlx::sqlite::SqliteConnectOptions::new()
            .filename(&db)
            .connect()
            .await
            .unwrap();
        sqlx::query("INSERT INTO cache_entries VALUES (?, 'utr_asr', ?, ?)")
            .bind(key.as_str())
            .bind(json!({"tokens": [], "lang": "eng"}).to_string())
            .bind(namespace.namespace())
            .execute(&mut connection)
            .await
            .unwrap();
        connection.close().await.unwrap();
        let before = std::fs::read(&db).unwrap();
        let report = inspect_retained_rev_cache(&trace, &db, Some(&audio))
            .await
            .unwrap();
        assert!(report.raw_is_replayable());
        assert!(matches!(
            report.derived_utr.as_ref().map(|report| &report.evidence),
            Some(InspectedEvidence::DerivedReplayable)
        ));
        assert!(!report.recording_bytes_revalidated);
        assert_eq!(std::fs::read(&db).unwrap(), before);
        let independent = inspect_retained_utr_cache(&audio, &db, "eng", &engine)
            .await
            .unwrap();
        assert!(independent.is_replayable());
        assert_eq!(independent.cache_key, key.to_string());
        assert_eq!(std::fs::read(&db).unwrap(), before);
    }
}
