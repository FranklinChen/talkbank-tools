use super::*;
use crate::api::NumSpeakers;
use crate::cache::UtteranceCache;
use crate::error::ServerError;
use crate::types::worker_v2::{
    ExecuteResponseV2, SpeakerBackendV2, SpeakerInferenceEvidenceV2, SpeakerProviderJobIdV2,
    SpeakerResultV2, SpeakerSegmentV2, TaskResultV2, WorkerRequestIdV2,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingSpeakerService {
    calls: AtomicUsize,
    segment: SpeakerSegmentV2,
}

impl CountingSpeakerService {
    fn standard() -> Self {
        Self::with_segment(segment("SPEAKER_00"))
    }

    fn with_segment(segment: SpeakerSegmentV2) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            segment,
        }
    }
}

#[async_trait::async_trait]
impl SpeakerEvidenceInference for CountingSpeakerService {
    async fn infer(
        &self,
        run: VerifiedSpeakerEvidenceRun,
    ) -> Result<SpeakerInferenceReply, ServerError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let segment = &self.segment;
        let evidence = match run.backend {
            SpeakerBackendV2::PyannoteAi => raw_evidence(std::slice::from_ref(segment)),
            SpeakerBackendV2::Pyannote => serde_json::from_value(serde_json::json!({
                "kind": "pyannote",
                "segments": [segment],
                "loaded_graph": {
                    "pipeline_config_sha256": "1".repeat(64),
                    "segmentation_sha256": "2".repeat(64),
                    "embedding_sha256": "3".repeat(64),
                }
            }))
            .expect("valid observed graph"),
        };
        Ok(reply(call, evidence))
    }
}

fn reply(call: usize, evidence: SpeakerInferenceEvidenceV2) -> SpeakerInferenceReply {
    let response = ExecuteResponseV2::success(
        WorkerRequestIdV2::from(format!("fixture-speaker-request-{call}")),
        TaskResultV2::SpeakerResult(SpeakerResultV2 { evidence }),
        crate::api::NonNegativeSeconds::try_from(0.01).expect("elapsed"),
    );
    SpeakerInferenceReply::admit(response).expect("speaker reply admitted")
}

fn segment(speaker: &str) -> SpeakerSegmentV2 {
    SpeakerSegmentV2 {
        interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 750)
            .expect("ordered"),
        speaker: speaker.to_owned(),
    }
}

fn raw_evidence(segments: &[SpeakerSegmentV2]) -> SpeakerInferenceEvidenceV2 {
    SpeakerInferenceEvidenceV2::PyannoteAi {
        job_id: SpeakerProviderJobIdV2::from("job-test"),
        output: serde_json::from_value(serde_json::json!({
            "exclusiveDiarization": segments
                .iter()
                .map(|segment| serde_json::json!({
                    "start": segment.interval.start_ms() as f64 / 1000.0,
                    "end": segment.interval.end_ms() as f64 / 1000.0,
                    "speaker": segment.speaker,
                }))
                .collect::<Vec<_>>()
        }))
        .expect("provider output object"),
        warning: None,
    }
}

/// Guards: Speaker cache keys follow bytes, count and model configuration while allowing relocation.
/// Kind: contract
/// Not a type: Hashing actual media and serializing request choices determine cross-run identity compatibility.
/// State: Private byte fixtures share or change exact content and vary request policy numerically.
/// Concurrency: Private directory and awaited reads after writes.
#[tokio::test]
async fn semantic_request_changes_invalidate_the_key() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    let renamed = tempdir.path().join("renamed.wav");
    let changed = tempdir.path().join("changed.wav");
    for (path, bytes) in [
        (&audio, b"prepared audio".as_slice()),
        (&renamed, b"prepared audio"),
        (&changed, b"other audio"),
    ] {
        tokio::fs::write(path, bytes).await.expect("write audio");
    }

    async fn request(
        path: &std::path::Path,
        speakers: Option<NumSpeakers>,
        model: &str,
    ) -> SpeakerEvidenceRequest {
        SpeakerEvidenceRequest::from_audio(
            path,
            SpeakerBackendV2::PyannoteAi,
            speakers,
            &SpeakerEvidenceModelRevision::for_test(model),
        )
        .await
        .expect("speaker request")
    }

    let two = Some(NumSpeakers(2));
    let baseline = request(&audio, two, "precision-2").await;
    let relocated = request(&renamed, two, "precision-2").await;
    assert_eq!(baseline.cache_key(), relocated.cache_key());
    for altered in [
        request(&audio, Some(NumSpeakers(3)), "precision-2").await,
        request(&audio, two, "precision-3").await,
        request(&changed, two, "precision-2").await,
    ] {
        assert_ne!(baseline.cache_key(), altered.cache_key());
    }
    let auto = request(&audio, None, "precision-2").await;
    assert_ne!(baseline.cache_key(), auto.cache_key());
    assert_eq!(auto.trace_seed().expected_speakers, None);
}

/// Guards: Concurrent speaker lookups wait for the first commit and use its stored projection.
/// Kind: contract
/// Not a type: Lock and notification ordering determine if the waiter sees the newly persisted evidence.
/// State: Private SQLite and one held miss capability witness the real cache handoff without a provider.
/// Concurrency: Spawned waiter overlaps the held lease and is joined after commit.
#[tokio::test]
async fn concurrent_identical_lookup_waits_for_first_inference_then_hits() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = std::sync::Arc::new(
        UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache"),
    );
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let first_miss = match request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect("first lookup")
    {
        SpeakerEvidenceLookup::DerivedHit(_) | SpeakerEvidenceLookup::RawHit(_) => {
            panic!("empty cache must miss")
        }
        SpeakerEvidenceLookup::Miss(miss) => miss,
    };

    let second_request = request.clone();
    let second_cache = cache.clone();
    let mut second = tokio::spawn(async move {
        resolve_speaker_evidence(
            &second_request,
            &second_cache,
            CachePolicy::UseCache,
            &CountingSpeakerService::standard(),
        )
        .await
    });
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut second)
            .await
            .is_err(),
        "a duplicate lookup must wait while the first miss owns inference"
    );

    first_miss
        .execute(&CountingSpeakerService::standard())
        .await
        .expect("commit first inference");
    let second = second
        .await
        .expect("second task")
        .expect("second resolution");
    assert_eq!(second.source(), SpeakerEvidenceSource::ReplayedDerived);
}

/// Guards: A hosted speaker receipt preserves original worker, build and provider job across durable required-cache replay.
/// Kind: journey
/// Not a type: The external receipt and reopened SQLite can lose observed producer identity or report current work instead.
/// State: Private source, SQLite and debug directory exercise the resolver and writer with a counted synthetic reply.
/// Concurrency: Private directory and ordered cold dispatch and reopened replay.
#[tokio::test]
async fn production_resolver_crosses_billable_boundary_once_then_replays() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache_dir = tempdir.path().join("cache");
    let cache = UtteranceCache::sqlite(Some(cache_dir.clone()))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let service = CountingSpeakerService::standard();

    let cold = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect("cold resolution");
    let cold_trace = cold.trace(SpeakerProjectionRevision::SegmentsV1);
    assert_eq!(
        cold_trace.cache_outcome(),
        SpeakerEvidenceCacheOutcome::InferredNotFound
    );
    drop(cache);
    let reopened_cache = UtteranceCache::sqlite(Some(cache_dir))
        .await
        .expect("reopened cache");
    let warm = resolve_speaker_evidence(
        &request,
        &reopened_cache,
        CachePolicy::RequireCache,
        &service,
    )
    .await
    .expect("warm resolution");
    let warm_trace = warm.trace(SpeakerProjectionRevision::SegmentsV1);
    assert_eq!(
        warm_trace.cache_outcome(),
        SpeakerEvidenceCacheOutcome::ReplayedDerived
    );
    assert_eq!(
        cold_trace.semantic_projection(),
        warm_trace.semantic_projection()
    );
    let debug_dir = tempdir.path().join("debug");
    let dumper = crate::runner::debug_dumper::DebugDumper::new(Some(&debug_dir));
    let export = |identity, trace| {
        use crate::runner::debug_dumper::SpeakerEvidenceDumpOutcome;
        let outcome = dumper
            .dump_speaker_evidence(identity, trace)
            .expect("trace write");
        let SpeakerEvidenceDumpOutcome::Written(path) = outcome else {
            panic!("enabled writer must publish the receipt");
        };
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(path).expect("receipt"))
            .expect("speaker trace JSON")
    };
    let trace_json = export("cold/audio.wav", &cold_trace);
    let warm_json = export("warm/audio.wav", &warm_trace);
    assert_eq!(warm_json["cache_outcome"], "replayed_derived");
    assert_eq!(trace_json["producer"]["kind"], "worker_reply");
    assert_eq!(
        trace_json["producer"]["invocation"]["request_id"],
        "fixture-speaker-request-0"
    );
    assert_eq!(
        trace_json["producer"]["invocation"]["server_build"],
        crate::build_hash()
    );
    assert_eq!(
        trace_json["producer"]["model"],
        serde_json::json!({"kind":"hosted_job","provider_job_id":"job-test"})
    );
    assert_eq!(warm_json["producer"], trace_json["producer"]);
    assert_eq!(trace_json["trace_schema_version"], 2);
    assert_eq!(trace_json["cache_key_algorithm"], "blake3");
    assert_eq!(trace_json["backend"], "pyannote_ai");
    assert_eq!(trace_json["expected_speakers"], 2);
    assert_eq!(
        trace_json["audio_preparation_revision"],
        SPEAKER_AUDIO_PREPARATION_REVISION
    );
    assert_eq!(
        trace_json["projection_revision"],
        "speaker-evidence-to-segments-v1"
    );
    assert_eq!(
        trace_json["segment_digest_revision"],
        SPEAKER_SEGMENT_DIGEST_REVISION
    );
    assert_eq!(trace_json["projected_segment_count"], 1);
    assert_eq!(
        trace_json["projected_segments_blake3"]
            .as_str()
            .expect("segment digest string")
            .len(),
        64
    );
    assert!(trace_json.get("source_path").is_none());
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
}

/// Guards: Speaker projection digest changes with label or timing under its named identity.
/// Kind: wire
/// Not a type: The digest recipe can omit a field even when it is present in the segment type.
/// State: Pure ordered typed segments vary only one semantic field at a time.
/// Concurrency: none
#[test]
fn segment_projection_digest_changes_with_timing_or_speaker() {
    let baseline = SpeakerSegmentsDigest::from_segments(&[segment("SPEAKER_00")]);
    let mut shifted_segment = segment("SPEAKER_00");
    shifted_segment.interval =
        batchalign_types::interval::AdmittedInterval::admit_millis(0, 1_001).expect("ordered");
    let shifted = SpeakerSegmentsDigest::from_segments(&[shifted_segment]);
    let relabeled = SpeakerSegmentsDigest::from_segments(&[segment("SPEAKER_01")]);

    assert_ne!(baseline, shifted);
    assert_ne!(baseline, relabeled);
}

/// Guards: Changing the speaker source after identity computation refuses worker inference.
/// Kind: boundary
/// Not a type: A file may change while its earlier content identity remains held.
/// State: Private source bytes are rewritten before admission and a counted worker double detects submission.
/// Concurrency: Fixture rewrite completes before lookup and verification.
#[tokio::test]
async fn source_drift_after_cache_identity_refuses_speaker_inference() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"bytes used for the cache identity")
        .await
        .expect("write original audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    tokio::fs::write(&audio, b"different bytes at inference time")
        .await
        .expect("replace audio");
    let service = CountingSpeakerService::standard();

    let error = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect_err("changed source bytes must not cross the inference boundary");

    assert!(
        error
            .to_string()
            .contains("changed after speaker evidence preparation")
    );
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
}

/// Guards: Cold required speaker cache reports a missing precondition without a worker call.
/// Kind: policy
/// Not a type: Current mutable storage may lack both raw and derived evidence for an admitted request.
/// State: Private source and empty cache with a counted synthetic worker.
/// Concurrency: Private directory and awaited lookup.
#[tokio::test]
async fn required_cache_refuses_cold_speaker_inference() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let service = CountingSpeakerService::standard();

    let error = resolve_speaker_evidence(&request, &cache, CachePolicy::RequireCache, &service)
        .await
        .expect_err("a cold required-cache lookup must refuse inference");

    assert!(error.to_string().contains("required speaker evidence"));
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
}

/// Guards: Old raw speaker rows retain their hosted job and old derived rows retain unknown producer origin.
/// Kind: journey
/// Not a type: Legacy cache envelopes can omit fields and must preserve exactly their weaker historical evidence.
/// State: Private SQLite stores actual legacy envelopes and a counted worker double detects paid paths.
/// Concurrency: Private directory with ordered legacy writes and lookups.
#[tokio::test]
async fn required_cache_replays_warm_speaker_evidence() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let service = CountingSpeakerService::standard();
    request
        .store_unchecked_derived_for_test(
            &cache,
            serde_json::json!({
                "schema_version": DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION,
                "raw_evidence_fingerprint": request.raw_cache_key.as_str(),
                "normalization_revision": request.normalization_revision.as_str(),
                "segments": [segment("SPEAKER_00")],
            }),
        )
        .await
        .expect("seed legacy derived evidence");

    let replay = resolve_speaker_evidence(&request, &cache, CachePolicy::RequireCache, &service)
        .await
        .expect("required-cache replay");

    assert_eq!(replay.source(), SpeakerEvidenceSource::ReplayedDerived);
    let trace = serde_json::to_value(replay.trace(SpeakerProjectionRevision::SegmentsV1)).unwrap();
    assert_eq!(trace["producer"], serde_json::json!({"kind":"unrecorded"}));

    request
        .store_unchecked_raw_for_test(
            &cache,
            serde_json::json!({
                "schema_version": RAW_SPEAKER_EVIDENCE_SCHEMA_VERSION,
                "request_fingerprint": request.raw_cache_key.as_str(),
                "evidence": raw_evidence(&[segment("SPEAKER_00")]),
            }),
        )
        .await
        .expect("seed legacy raw evidence with a provider observation");
    let revised = request.with_normalization_revision_for_test("legacy-control-projection");
    for source in [
        SpeakerEvidenceSource::DerivedFromRaw,
        SpeakerEvidenceSource::ReplayedDerived,
    ] {
        let replay =
            resolve_speaker_evidence(&revised, &cache, CachePolicy::RequireCache, &service)
                .await
                .unwrap();
        assert_eq!(replay.source(), source);
        let trace =
            serde_json::to_value(replay.trace(SpeakerProjectionRevision::SegmentsV1)).unwrap();
        assert_eq!(
            trace["producer"],
            serde_json::json!({
                "kind":"raw_without_invocation",
                "model":{"kind":"hosted_job","provider_job_id":"job-test"},
            })
        );
    }
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
}

/// Guards: A new normalizer reuses raw speaker evidence and carries the same exact loaded graph through both replay paths.
/// Kind: journey
/// Not a type: Stored bytes and resolver transitions can lose local-model or invocation evidence when projection changes.
/// State: Private source and SQLite with a counted local reply carrying valid graph digests.
/// Concurrency: Private directory and ordered raw then derived lookups.
#[tokio::test]
async fn a_new_normalizer_revision_reuses_raw_evidence_without_another_service_call() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::Pyannote,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("requested-local-graph"),
    )
    .await
    .expect("request");
    let revised_request = request
        .clone()
        .with_normalization_revision_for_test("speaker-normalizer-v-next");
    let service = CountingSpeakerService::standard();

    let original = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect("cold resolution");
    let original =
        serde_json::to_value(original.trace(SpeakerProjectionRevision::SegmentsV1)).unwrap();
    assert_eq!(original["model_revision"], "requested-local-graph");
    assert_eq!(
        original["producer"]["model"],
        serde_json::json!({
            "kind": "local_graph",
            "graph": {
                "pipeline_config_sha256": "1".repeat(64),
                "segmentation_sha256": "2".repeat(64),
                "embedding_sha256": "3".repeat(64),
            },
        })
    );
    for source in [
        SpeakerEvidenceSource::DerivedFromRaw,
        SpeakerEvidenceSource::ReplayedDerived,
    ] {
        let replay = resolve_speaker_evidence(
            &revised_request,
            &cache,
            CachePolicy::RequireCache,
            &service,
        )
        .await
        .expect("re-normalize raw evidence");
        assert_eq!(replay.source(), source);
        let trace =
            serde_json::to_value(replay.trace(SpeakerProjectionRevision::SegmentsV1)).unwrap();
        assert_eq!(trace["producer"], original["producer"]);
        assert_eq!(trace["raw_evidence_key"], original["raw_evidence_key"]);
        assert_ne!(
            trace["derived_evidence_key"],
            original["derived_evidence_key"]
        );
    }
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
}

/// Guards: Corrupt speaker raw evidence fails closed without a replacement call.
/// Kind: boundary
/// Not a type: An existing SQLite row can defeat serde or request association before inference authorization.
/// State: Private source and SQLite retain malformed evidence and a counted double witnesses no resubmission.
/// Concurrency: Private directory and awaited cache operations.
#[tokio::test]
async fn corrupt_cached_evidence_fails_closed_instead_of_rebilling() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    request
        .store_unchecked_raw_for_test(&cache, serde_json::json!({"evidence": "invalid"}))
        .await
        .expect("seed corrupt entry");

    let service = CountingSpeakerService::standard();
    let error = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect_err("corruption must not become a miss");
    assert!(
        error
            .to_string()
            .contains("invalid cached speaker evidence")
    );
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
}

/// Guards: An explicit forced refresh enters the forced miss state even with stored evidence.
/// Kind: policy
/// Not a type: Runtime cache policy must choose a fresh route when both computed entries exist.
/// State: Private SQLite and one synthetic committed reply materialize the warm state.
/// Concurrency: Private directory and initial commit awaited before refresh.
#[tokio::test]
async fn forced_refresh_is_a_typed_miss_even_when_evidence_exists() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let service = CountingSpeakerService::standard();
    let first = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect("commit the first inference");
    let refresh = resolve_speaker_evidence(&request, &cache, CachePolicy::SkipCache, &service)
        .await
        .expect("refresh resolution");
    assert_eq!(
        refresh.source(),
        SpeakerEvidenceSource::Inferred(SpeakerEvidenceMissReason::ForcedRefresh)
    );
    let request_id = |resolution: &SpeakerEvidenceResolution| {
        serde_json::to_value(resolution.trace(SpeakerProjectionRevision::SegmentsV1))
            .expect("trace")["producer"]["invocation"]["request_id"]
            .clone()
    };
    assert_ne!(request_id(&first), request_id(&refresh));
    assert_eq!(service.calls.load(Ordering::SeqCst), 2);
}

/// Guards: Derived cache admission rejects reversed interval bounds and evidence naming a different producer backend.
/// Kind: boundary
/// Not a type: Persisted derived bytes can violate projection invariants or name the wrong provider.
/// State: Private source and SQLite store invalid rows and a counted double witnesses no resubmission.
/// Concurrency: Private directory and each serial adversarial row is checked after writing.
#[tokio::test]
async fn invalid_cached_timing_fails_closed() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let invalid = serde_json::json!({
        "schema_version": DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION,
        "raw_evidence_fingerprint": request.raw_cache_key.as_str(),
        "normalization_revision": request.normalization_revision.as_str(),
        "segments": [{
            "start_ms": 900,
            "end_ms": 100,
            "speaker": "SPEAKER_00"
        }]
    });
    request
        .store_unchecked_derived_for_test(&cache, invalid)
        .await
        .expect("seed invalid timing");

    let service = CountingSpeakerService::standard();
    let error = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect_err("invalid timing must not become a miss");
    assert!(
        error.to_string().contains("before its start"),
        "the interval admission refuses the stored segment: {error}"
    );

    request.store_unchecked_derived_for_test(&cache, serde_json::json!({
        "schema_version": DERIVED_SPEAKER_EVIDENCE_SCHEMA_VERSION,
        "raw_evidence_fingerprint": request.raw_cache_key.as_str(),
        "normalization_revision": request.normalization_revision.as_str(),
        "segments": [segment("SPEAKER_00")],
        "producer": {"kind":"raw_without_invocation","model":{"kind":"local_graph_unreported"}},
    })).await.unwrap();
    let error = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect_err("foreign backend");
    assert!(
        error
            .to_string()
            .contains("does not match requested backend"),
        "{error}"
    );
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
}

/// Guards: A zero-duration speaker segment survives typed worker admission, durable commit and cache replay.
/// Kind: contract
/// Not a type: A worker-valid equal-endpoint interval can be silently dropped by normalization or storage.
/// State: Private source and SQLite receive a synthetic worker result through the actual resolver.
/// Concurrency: Private directory and ordered commit then replay.
#[tokio::test]
async fn zero_duration_segment_allowed_by_worker_protocol_round_trips() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let request = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let zero_duration = SpeakerSegmentV2 {
        interval: batchalign_types::interval::AdmittedInterval::admit_millis(500, 500)
            .expect("ordered"),
        speaker: "SPEAKER_00".to_owned(),
    };

    let service = CountingSpeakerService::with_segment(zero_duration.clone());
    let committed = resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect("worker-valid zero-duration evidence should commit");
    assert_eq!(committed.segments(), std::slice::from_ref(&zero_duration));

    let replay = resolve_speaker_evidence(&request, &cache, CachePolicy::RequireCache, &service)
        .await
        .expect("committed evidence must replay");
    assert_eq!(replay.source(), SpeakerEvidenceSource::ReplayedDerived);
    assert_eq!(replay.segments(), &[zero_duration]);
}
