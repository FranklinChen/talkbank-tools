use super::*;
use crate::api::AudioPositionSeconds;
use crate::cache::UtteranceCache;
use crate::pipeline::post_validate::OutputReport;
use crate::revai::{
    AuthorizedRevEvidenceRun, RevAsrEvidenceCacheOutcome, RevAsrEvidenceInference,
    RevTranscriptEvidence,
};
use crate::transcribe::replay::{
    LegacyProjectedAsrProducer, LegacyReplayManifestRequest, admit_legacy_replay_manifest,
    write_legacy_replay_manifest,
};
use crate::transcribe::{AsrBackend, AsrToken};
use crate::types::worker_v2::SpeakerBackendV2;
use crate::worker::pool::{PoolConfig, WorkerPool};
use std::sync::atomic::{AtomicUsize, Ordering};

impl Built {
    /// The assembled bytes, whichever way the producer judged them.
    fn as_str(&self) -> &str {
        match self {
            Self::Admitted(ready) => ready.document.as_str(),
            Self::Diagnosed(diagnosed) => diagnosed.document.as_str(),
        }
    }

    /// The speaker evidence, carried on both arms.
    fn speaker_assignments(&self) -> &SpeakerAssignmentOutcome {
        match self {
            Self::Admitted(ready) => &ready.speaker_assignments,
            Self::Diagnosed(diagnosed) => &diagnosed.speaker_assignments,
        }
    }
}

/// A fixture position; every literal in this module is a valid one.
fn at(seconds: f64) -> Option<AudioPositionSeconds> {
    Some(AudioPositionSeconds::try_from(seconds).expect("fixture position"))
}

#[test]
fn transcribe_stage_progress_labels_are_stable() {
    assert_eq!(
        progress_stage_for_stage(StageId::AsrInfer),
        FileStage::Transcribing
    );
    assert_eq!(
        progress_stage_for_stage(StageId::SpeakerDiarization),
        FileStage::PostProcessing
    );
    assert_eq!(
        progress_stage_for_stage(StageId::AsrPostprocess),
        FileStage::PostProcessing
    );
    assert_eq!(
        progress_stage_for_stage(StageId::BuildChat),
        FileStage::BuildingChat
    );
    assert_eq!(
        progress_stage_for_stage(StageId::OptionalUtseg),
        FileStage::SegmentingUtterances
    );
    assert_eq!(
        progress_stage_for_stage(StageId::OptionalMorphosyntax),
        FileStage::AnalyzingMorphosyntax
    );
    assert_eq!(
        progress_stage_for_stage(StageId::Serialize),
        FileStage::Finalizing
    );
}

fn test_transcribe_options(speaker_backend: Option<SpeakerBackendV2>) -> TranscribeOptions {
    test_transcribe_options_in(speaker_backend, LanguageCode3::eng().into())
}

fn test_transcribe_options_in(
    speaker_backend: Option<SpeakerBackendV2>,
    language: crate::api::LanguageSpec,
) -> TranscribeOptions {
    TranscribeOptions {
        plan: crate::transcribe::types::AdmittedTranscribePlan::admit(
            crate::transcribe::TranscribeAsrPlan::from_request(
                AsrBackend::RustRevAi,
                false,
                2,
                &std::collections::BTreeMap::new(),
                &language,
            )
            .unwrap(),
            false,
            UtsegFallbackPolicy::Refuse,
        )
        .unwrap(),
        diarize: true,
        speaker_backend,
        with_morphosyntax: false,
        cache_policies: crate::transcribe::TranscribeCachePolicies::uniform(
            crate::params::CachePolicy::UseCache,
        ),
        write_wor: false,
        media_name: Some("sample".into()),
        engine_extras: std::collections::BTreeMap::new(),
    }
}

struct CountingRevInference {
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl RevAsrEvidenceInference for CountingRevInference {
    async fn infer(
        &self,
        _run: AuthorizedRevEvidenceRun,
    ) -> Result<crate::revai::RevAsrInferenceOutcome, ServerError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(crate::revai::RevAsrInferenceOutcome::Fetched(FetchedRevAsrEvidence {
            transcript_evidence: RevTranscriptEvidence::from_provider_json(
                r#"{"monologues":[{"speaker":0,"elements":[{"type":"text","value":"hello","ts":0.1,"end_ts":0.5,"confidence":0.9},{"type":"punct","value":".","ts":null,"end_ts":null,"confidence":null}]},{"speaker":1,"elements":[{"type":"text","value":"there","ts":0.6,"end_ts":1.0,"confidence":0.8},{"type":"punct","value":"?","ts":null,"end_ts":null,"confidence":null}]}]}"#
                    .to_owned(),
            )
            .expect("valid provider transcript fixture"),
            resolved_language: TranscriptLanguage::One(LanguageCode3::eng()),
        }))
    }
}

fn only_debug_artifact(dir: &Path, suffix: &str) -> std::path::PathBuf {
    let matches = std::fs::read_dir(dir)
        .expect("read debug directory")
        .map(|entry| entry.expect("debug directory entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(suffix))
        })
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "expected one {suffix} artifact");
    matches.into_iter().next().expect("one debug artifact")
}

/// Assert that two transcribe outputs differ, if at all, only in the
/// execution timestamp of an otherwise identical provenance receipt.
fn assert_same_transcribe_semantics(left: &str, right: &str) {
    let left_provenance = crate::provenance::extract_provenance(left).expect("left stamps parse");
    let right_provenance =
        crate::provenance::extract_provenance(right).expect("right stamps parse");
    assert_eq!(
        left_provenance.len(),
        1,
        "expected one left provenance receipt"
    );
    assert_eq!(
        right_provenance.len(),
        1,
        "expected one right provenance receipt"
    );
    assert_eq!(left_provenance[0].command, "transcribe");
    assert_eq!(right_provenance[0].command, "transcribe");
    assert_eq!(left_provenance[0].fields, right_provenance[0].fields);
    assert!(
        left == right
            || crate::provenance::is_provenance_only_difference(
                left,
                right,
                crate::api::ReleasedCommand::Transcribe,
            ),
        "transcribe outputs differ outside the execution timestamp"
    );
}

mod rev_replay;
mod stages;

/// An optional stage whose OWN output is refused is a shortfall, and the
/// caller keeps its admitted predecessor; every other failure is still the
/// file's error, so the runner's retry and failure policy apply unchanged.
/// A judged refusal of hundreds of findings travels with the shortfall as a
/// bounded record: the count, the count per code and the first findings.
#[test]
fn a_judged_stage_refusal_is_recorded_bounded() {
    let finding = |n: usize| crate::api::OutputFindingRecord {
        code: Some("E220".to_owned()),
        level: crate::api::FindingLevel::StructurallyComplete,
        message: format!("finding {n}"),
    };
    let refused = not_applied::<crate::pipeline::post_validate::PostValidated>(
        OptionalStage::UtteranceSegmentation,
        Err(ServerError::OutputAdmission {
            command: crate::api::ReleasedCommand::Utseg,
            details: crate::error::OutputAdmissionRefusal::Judged {
                bar: crate::api::JudgementBar::Construction,
                first: finding(0),
                rest: (1..500).map(finding).collect(),
            },
        }),
    )
    .expect("a refused stage output is not the file's error");
    let StageOutcome::NotApplied(Shortfall::StageNotApplied {
        refusal:
            crate::api::StageRefusalRecord::Judged {
                finding_count,
                findings_by_code,
                first_findings,
                ..
            },
        ..
    }) = refused
    else {
        panic!("a judged refusal is recorded as judged");
    };
    assert_eq!(finding_count, 500);
    assert_eq!(
        findings_by_code,
        vec![crate::api::FindingCodeCount {
            code: Some("E220".to_owned()),
            count: 500
        }]
    );
    assert_eq!(
        first_findings,
        (0..crate::api::FileOutputDiagnostics::FIRST_FINDINGS)
            .map(finding)
            .collect::<Vec<_>>(),
        "only the first findings, in order, travel with the shortfall"
    );
}

#[test]
fn only_a_refused_stage_output_is_a_shortfall() {
    let refused = not_applied::<crate::pipeline::post_validate::PostValidated>(
        OptionalStage::Morphosyntax,
        Err(ServerError::OutputAdmission {
            command: crate::api::ReleasedCommand::Morphotag,
            details: crate::error::OutputAdmissionRefusal::unestablished("E999 synthetic refusal"),
        }),
    )
    .expect("a refused stage output is not the file's error");
    let StageOutcome::NotApplied(shortfall) = refused else {
        panic!("a refused stage output is recorded, not applied");
    };
    assert_eq!(
        shortfall,
        Shortfall::StageNotApplied {
            stage: OptionalStage::Morphosyntax,
            refusal: crate::api::StageRefusalRecord::Unestablished {
                reason: "E999 synthetic refusal".to_owned(),
            },
        }
    );
    assert!(
        shortfall
            .to_string()
            .starts_with("morphosyntax not applied")
    );

    let worker = not_applied::<crate::pipeline::post_validate::PostValidated>(
        OptionalStage::UtteranceSegmentation,
        Err(ServerError::Validation("worker went away".to_owned())),
    );
    assert!(
        matches!(worker, Err(ServerError::Validation(_))),
        "any other failure stays the file's error"
    );
}
