use super::*;
use crate::api::NumSpeakers;
use crate::cache::UtteranceCache;
use crate::error::ServerError;
use crate::types::worker_v2::{
    SpeakerBackendV2, SpeakerInferenceEvidenceV2, SpeakerProviderJobIdV2, SpeakerSegmentV2,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingSpeakerService {
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl SpeakerEvidenceInference for CountingSpeakerService {
    async fn infer(
        &self,
        _run: VerifiedSpeakerEvidenceRun,
    ) -> Result<SpeakerInferenceEvidenceV2, ServerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(raw_evidence(&[segment("SPEAKER_00")]))
    }
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

#[tokio::test]
async fn identical_audio_bytes_share_a_key_across_paths() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let first = tempdir.path().join("first.wav");
    let renamed = tempdir.path().join("renamed.wav");
    tokio::fs::write(&first, b"same prepared audio")
        .await
        .expect("write first");
    tokio::fs::write(&renamed, b"same prepared audio")
        .await
        .expect("write renamed");

    let first_request = SpeakerEvidenceRequest::from_audio(
        &first,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("first request");
    let renamed_request = SpeakerEvidenceRequest::from_audio(
        &renamed,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("renamed request");

    assert_eq!(first_request.cache_key(), renamed_request.cache_key());
}

#[tokio::test]
async fn semantic_request_changes_invalidate_the_key() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"prepared audio")
        .await
        .expect("write audio");

    let two_speakers = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("two-speaker request");
    let three_speakers = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(3)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("three-speaker request");
    let revised_model = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-3"),
    )
    .await
    .expect("revised-model request");

    assert_ne!(two_speakers.cache_key(), three_speakers.cache_key());
    assert_ne!(two_speakers.cache_key(), revised_model.cache_key());
}

#[tokio::test]
async fn only_a_miss_can_become_billable_and_commit_evidence() {
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

    let lookup = request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect("lookup");
    let miss = match lookup {
        SpeakerEvidenceLookup::DerivedHit(_) | SpeakerEvidenceLookup::RawHit(_) => {
            panic!("empty cache must miss")
        }
        SpeakerEvidenceLookup::Miss(miss) => miss,
    };
    let authorization = miss.authorize_billable_inference();
    let (run, permit) = authorization.into_run();
    // The live boundary must consume this capability.  Keeping commit
    // permission separate means the inference adapter cannot reuse the
    // same cache miss for a second paid call.
    let _: AuthorizedSpeakerEvidenceRun = run;
    let committed = permit
        .commit(&cache, raw_evidence(&[segment("SPEAKER_00")]))
        .await
        .expect("commit");
    assert_eq!(committed.segments(), &[segment("SPEAKER_00")]);

    let hit = request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect("lookup hit");
    match hit {
        SpeakerEvidenceLookup::DerivedHit(evidence) => {
            assert_eq!(evidence.segments(), &[segment("SPEAKER_00")]);
        }
        SpeakerEvidenceLookup::RawHit(_) => {
            panic!("committed derived evidence should hit directly")
        }
        SpeakerEvidenceLookup::Miss(_) => panic!("committed evidence must hit"),
    }
}

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
    let first_authorization = match request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect("first lookup")
    {
        SpeakerEvidenceLookup::DerivedHit(_) | SpeakerEvidenceLookup::RawHit(_) => {
            panic!("empty cache must miss")
        }
        SpeakerEvidenceLookup::Miss(miss) => miss.authorize_billable_inference(),
    };
    let (_run, first_permit) = first_authorization.into_run();

    let second_request = request.clone();
    let second_cache = cache.clone();
    let mut second = tokio::spawn(async move {
        second_request
            .lookup(&second_cache, CachePolicy::UseCache)
            .await
    });
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut second)
            .await
            .is_err(),
        "a duplicate lookup must wait while the first miss owns inference"
    );

    first_permit
        .commit(&cache, raw_evidence(&[segment("SPEAKER_00")]))
        .await
        .expect("commit first inference");
    let second_lookup = second.await.expect("second task").expect("second lookup");
    assert!(matches!(
        second_lookup,
        SpeakerEvidenceLookup::DerivedHit(_)
    ));
}

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
    let service = CountingSpeakerService {
        calls: AtomicUsize::new(0),
    };

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
    let warm = resolve_speaker_evidence(&request, &reopened_cache, CachePolicy::UseCache, &service)
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
    let trace_json = serde_json::to_value(&cold_trace).expect("speaker trace JSON");
    assert_eq!(trace_json["trace_schema_version"], 1);
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

#[test]
fn segment_projection_digest_changes_with_timing_or_speaker() {
    let baseline = ValidatedSpeakerEvidence::new(vec![segment("SPEAKER_00")]);
    let mut shifted_segment = segment("SPEAKER_00");
    shifted_segment.interval =
        batchalign_types::interval::AdmittedInterval::admit_millis(0, 1_001).expect("ordered");
    let shifted = ValidatedSpeakerEvidence::new(vec![shifted_segment]);
    let relabeled = ValidatedSpeakerEvidence::new(vec![segment("SPEAKER_01")]);

    assert_ne!(baseline.segments_digest, shifted.segments_digest);
    assert_ne!(baseline.segments_digest, relabeled.segments_digest);
}

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
    let service = CountingSpeakerService {
        calls: AtomicUsize::new(0),
    };

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
    let service = CountingSpeakerService {
        calls: AtomicUsize::new(0),
    };

    let error = resolve_speaker_evidence(&request, &cache, CachePolicy::RequireCache, &service)
        .await
        .expect_err("a cold required-cache lookup must refuse inference");

    assert!(error.to_string().contains("required speaker evidence"));
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
}

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
    let service = CountingSpeakerService {
        calls: AtomicUsize::new(0),
    };
    resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect("seed evidence");

    let replay = resolve_speaker_evidence(&request, &cache, CachePolicy::RequireCache, &service)
        .await
        .expect("required-cache replay");

    assert_eq!(replay.source(), SpeakerEvidenceSource::ReplayedDerived);
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn auto_detected_speaker_count_has_distinct_replay_identity() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let audio = tempdir.path().join("audio.wav");
    tokio::fs::write(&audio, b"same audio")
        .await
        .expect("audio fixture");
    let revision = SpeakerEvidenceModelRevision::for_test("precision-2");

    let auto =
        SpeakerEvidenceRequest::from_audio(&audio, SpeakerBackendV2::PyannoteAi, None, &revision)
            .await
            .expect("auto request");
    let exactly_two = SpeakerEvidenceRequest::from_audio(
        &audio,
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &revision,
    )
    .await
    .expect("two-speaker request");

    assert_ne!(auto.cache_key(), exactly_two.cache_key());
    let trace = auto.trace_seed();
    assert_eq!(trace.expected_speakers, None);
}

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
        SpeakerBackendV2::PyannoteAi,
        Some(NumSpeakers(2)),
        &SpeakerEvidenceModelRevision::for_test("precision-2"),
    )
    .await
    .expect("request");
    let revised_request = request
        .clone()
        .with_normalization_revision_for_test("speaker-normalizer-v-next");
    let service = CountingSpeakerService {
        calls: AtomicUsize::new(0),
    };

    resolve_speaker_evidence(&request, &cache, CachePolicy::UseCache, &service)
        .await
        .expect("cold resolution");
    let replay =
        resolve_speaker_evidence(&revised_request, &cache, CachePolicy::UseCache, &service)
            .await
            .expect("re-normalize raw evidence");

    assert_eq!(replay.source(), SpeakerEvidenceSource::DerivedFromRaw);
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
}

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

    let error = request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect_err("corruption must not become a miss");
    assert!(
        error
            .to_string()
            .contains("invalid cached speaker evidence")
    );
}

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
    let miss = match request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect("initial miss")
    {
        SpeakerEvidenceLookup::DerivedHit(_) | SpeakerEvidenceLookup::RawHit(_) => {
            panic!("empty cache must miss")
        }
        SpeakerEvidenceLookup::Miss(miss) => miss,
    };
    let (_run, permit) = miss.authorize_billable_inference().into_run();
    permit
        .commit(&cache, raw_evidence(&[segment("SPEAKER_00")]))
        .await
        .expect("commit");

    let refresh = request
        .lookup(&cache, CachePolicy::SkipCache)
        .await
        .expect("refresh lookup");
    let authorization = match refresh {
        SpeakerEvidenceLookup::DerivedHit(_) | SpeakerEvidenceLookup::RawHit(_) => {
            panic!("refresh must not hit")
        }
        SpeakerEvidenceLookup::Miss(miss) => miss.authorize_billable_inference(),
    };
    let (_run, permit) = authorization.into_run();
    assert_eq!(permit.reason(), SpeakerEvidenceMissReason::ForcedRefresh);
}

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

    let error = request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect_err("invalid timing must not become a miss");
    assert!(
        error.to_string().contains("before its start"),
        "the interval admission refuses the stored segment: {error}"
    );
}

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

    let miss = match request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect("initial lookup")
    {
        SpeakerEvidenceLookup::DerivedHit(_) | SpeakerEvidenceLookup::RawHit(_) => {
            panic!("empty cache must miss")
        }
        SpeakerEvidenceLookup::Miss(miss) => miss,
    };
    let (_run, permit) = miss.authorize_billable_inference().into_run();
    permit
        .commit(&cache, raw_evidence(std::slice::from_ref(&zero_duration)))
        .await
        .expect("worker-valid zero-duration evidence should commit");

    let hit = request
        .lookup(&cache, CachePolicy::UseCache)
        .await
        .expect("replay");
    let SpeakerEvidenceLookup::DerivedHit(evidence) = hit else {
        panic!("committed evidence must replay");
    };
    assert_eq!(evidence.segments(), &[zero_duration]);
}
