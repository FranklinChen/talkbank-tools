//! Synthetic request evidence through the real FA pipeline, not model accuracy.

use super::*;
use crate::api::{NonNegativeSeconds, OutputShortfallRecord};
use crate::cache::{UtteranceCache, tasks::FORCED_ALIGNMENT};
use crate::chat_ops::fa::coordinates::{Ms, Recording};
use crate::chat_ops::fa::utr::{AsrTimingToken, inject_spaced};
use crate::chat_ops::fa::{
    AnchoredPart, AudioIdentity, EndOverlapPolicy, ExistingWorBoundaryPolicy, MainBulletPolicy,
    ReviewLevel, WordGapHealing, WordTiming, group_utterances,
};
use crate::types::engines::FaEngineName;
use crate::types::params::{CachePolicy, WorTierPolicy};
use crate::types::worker_v2::{
    ExecuteResponseV2, IndexedWordTimingResultV2, TaskResultV2, WorkerRequestIdV2,
};
use crate::worker::pool::{PoolConfig, WorkerPool};

/// Prefix, interior and suffix gaps must keep the same original five-word
/// population through dispatch, assembly, injection and output admission.
#[tokio::test]
async fn request_context_preserves_partial_partition_output() {
    let controls = [
        (
            "context_suffix",
            [3_000, 6_000, 9_000, 14_800, 15_200],
            vec![],
        ),
        (
            "short_anchored_tail",
            [1_000, 2_000, 16_080, 17_000, 19_000],
            vec![0, 1],
        ),
        ("prefix", [20_000, 22_000, 24_000, 26_000, 30_000], vec![0]),
        ("interior", [2_000, 4_000, 30_000, 32_000, 34_000], vec![2]),
        ("suffix", [2_000, 4_000, 6_000, 8_000, 50_000], vec![4]),
        (
            "all",
            [20_000, 40_000, 60_000, 80_000, 100_000],
            vec![0, 1, 2, 3, 4],
        ),
        (
            "anchored_tail",
            [1_000, 2_000, 16_000, 17_000, 19_000],
            vec![0, 1],
        ),
    ];
    let mut summaries = Vec::new();
    for (label, ends, expected_missing) in controls {
        let words = ["one", "two", "three", "four", "five"];
        let source = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n\
             @ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n\
             *SPK:\tone two three four five . \u{15}0_{}\u{15}\n@End\n",
            ends[4]
        );
        let mut working = read_fa_source_named(
            &source,
            talkbank_model::model::TranscriptName::for_path(std::path::Path::new("sample.cha")),
            MainBulletPolicy::DeriveFromWords,
        )
        .expect("admitted fixture");
        let tokens: Vec<_> = words
            .iter()
            .zip(ends)
            .enumerate()
            .filter(|(index, _)| !label.ends_with("anchored_tail") || *index >= 2)
            .map(|(index, (word, end))| AsrTimingToken {
                text: (*word).to_owned(),
                start_ms: end
                    - match (label, index) {
                        ("short_anchored_tail", 2) => 80,
                        ("context_suffix", 4) => 400,
                        _ => 500,
                    },
                end_ms: end,
            })
            .collect();
        working
            .active_mut()
            .expect("active")
            .recover_timing(async |chat| Ok(inject_spaced(chat, &tokens)))
            .await
            .expect("synthetic UTR anchors");
        let FaInputDocument::Active(active) = working.attempt() else {
            panic!("active")
        };
        let recording = Recording::of_duration(Ms(300_000)).expect("recording");
        let grouping = group_utterances(&active.chat_file, 15_000, &recording, active.anchors);
        assert_eq!(grouping.groups.len(), 1, "{label}");
        let crate::chat_ops::fa::GroupSpan::Anchored(partition) = grouping.groups[0].span() else {
            panic!("partition")
        };
        let missing: Vec<_> = partition
            .pieces()
            .filter_map(|part| match part {
                AnchoredPart::Unavailable(piece) => Some(piece.words()),
                AnchoredPart::Request(_) => None,
            })
            .flatten()
            .map(|word| word.utterance_word_index.raw())
            .collect();
        assert_eq!(missing, expected_missing, "{label}");
        assert_eq!(
            grouping.groups[0]
                .words()
                .map(|word| word.utterance_word_index.raw())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );

        let dir = tempfile::tempdir().expect("directory");
        let audio_path = dir.path().join("sample.mp3");
        crate::media::probe::fixtures::write_unpadded_cbr_mp3(&audio_path);
        let identity = AudioIdentity::from_metadata("sample.mp3", 0, 0);
        let audio = AudioContext {
            audio_path: &audio_path,
            audio_identity: &identity,
            audio_duration: None,
        };
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let cache = UtteranceCache::sqlite(Some(dir.path().join("cache")))
            .await
            .expect("cache");
        let namespace = FaCacheNamespace::for_test("test-fa-wave-v1");
        let plan = units::DispatchPlan::build(
            &grouping.groups,
            &identity,
            crate::chat_ops::fa::WhisperTextMode::SpaceJoined,
            FaEngineName::Wave2Vec,
        );
        let mut expected_timings = vec![None; 5];
        if label == "short_anchored_tail" {
            assert_eq!(
                plan.units().count(),
                1,
                "the rescued word must share its following context"
            );
        }
        if label == "context_suffix" {
            let last = plan.units().last().expect("suffix request");
            assert!(
                last.words().len() > 1,
                "the suffix must borrow lexical and acoustic context"
            );
        }
        let requested = partition.pieces().filter_map(|part| match part {
            AnchoredPart::Request(piece) => Some(piece),
            AnchoredPart::Unavailable(_) => None,
        });
        for (unit, piece) in plan.units().zip(requested) {
            assert!(unit.window().len().0 <= 15_000);
            assert_eq!(unit.words(), piece.request_words());
            assert!(unit.window().audio_start() <= piece.ownership_window().audio_start());
            assert!(unit.window().end() >= piece.ownership_window().end());
            for word in unit.words() {
                assert!(
                    !expected_missing.contains(&word.utterance_word_index.raw()),
                    "context must not cross an unavailable range"
                );
            }
            for word in piece.words() {
                let end = ends[word.utterance_word_index.raw()];
                expected_timings[word.utterance_word_index.raw()] =
                    WordTiming::fixture(end - 80, end);
            }
            let owned_only_key = crate::chat_ops::fa::cache_key(
                &piece
                    .words()
                    .iter()
                    .map(|word| word.text.clone())
                    .collect::<Vec<_>>(),
                &identity,
                piece.ownership_window().audio_start().get(),
                piece.ownership_window().end().get(),
                crate::chat_ops::fa::WhisperTextMode::SpaceJoined,
                FaEngineName::Wave2Vec,
            );
            if piece.words() != piece.request_words() {
                assert_ne!(
                    unit.cache_key(),
                    &owned_only_key,
                    "expanded requests cannot reuse old request evidence"
                );
            }
            let timings: Vec<_> = unit
                .words()
                .iter()
                .map(|word| {
                    let end = ends[word.utterance_word_index.raw()];
                    let borrowed = !piece.words().contains(word);
                    // Borrowed words deliberately disagree with their owning
                    // request. Only the owner's result may reach final CHAT.
                    WordTiming::fixture(end - if borrowed { 40 } else { 80 }, end)
                })
                .collect();
            let response = ExecuteResponseV2::success(
                WorkerRequestIdV2::from("partition-fixture"),
                TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                    indexed_timings: vec![None; timings.len()],
                }),
                NonNegativeSeconds::try_from(0.01).expect("elapsed"),
            );
            let raw = raw_evidence::FaRawEvidence::admit_requested(
                &response,
                FaEngineName::Wave2Vec,
                &namespace,
                raw_evidence::ExpectedFaWords::new(timings.len()),
                unit.cache_key(),
                raw_evidence::FaEvidenceRoute::Direct,
            )
            .expect("fixture response admission")
            .into_replayable()
            .expect("direct");
            let value = AdmittedCachedFaTimings::encode_from_raw(timings, &raw)
                .expect("versioned fixture cache");
            cache
                .put_batch(
                    &[(unit.cache_key().as_str().to_owned(), value)],
                    FORCED_ALIGNMENT,
                    &namespace,
                )
                .await
                .expect("store fixture evidence");
        }
        let mut params = FaParams {
            gap_healing: WordGapHealing::PreserveMeasured,
            whisper_text: crate::chat_ops::fa::WhisperTextMode::SpaceJoined,
            existing_wor_boundaries: ExistingWorBoundaryPolicy::Preserve,
            end_overlap_policy: EndOverlapPolicy::PreserveCrossSpeaker,
            main_bullets: MainBulletPolicy::DeriveFromWords,
            engine: FaEngineName::Wave2Vec,
            cache_policy: CachePolicy::RequireCache,
            wor_tier: WorTierPolicy::Include,
            bullet_repair: false,
            review_level: ReviewLevel::None,
        };
        let result = run_fa_from_ast(
            working.attempt(),
            &audio,
            &crate::api::LanguageCode3::eng(),
            FaServices {
                pipeline: PipelineServices::new(&pool, &cache),
                cache_namespace: &namespace,
                gpu: &crate::worker::pool::GpuModelPlan::alignment(FaEngineName::Wave2Vec, None),
                attention_capture_dir: None,
            },
            &params,
            None,
        )
        .await
        .expect("real FA pipeline without worker inference");
        let (output, trace) = result.into_document_and_timeline();
        let crate::types::traces::FaGroupSpanTrace::Anchored {
            pieces,
            unrequested,
        } = &trace.groups[0].span
        else {
            panic!("anchored trace")
        };
        assert_eq!(pieces.len(), plan.units().count());
        assert_eq!(trace.evidence_schema_version, 8);
        for (trace_piece, unit) in pieces.iter().zip(plan.units()) {
            let context = trace_piece.context.as_ref().expect("schema 8 context");
            assert_eq!(
                context.first_word,
                unit.words()[0].utterance_word_index.raw()
            );
            assert_eq!(
                context.last_word,
                unit.words().last().unwrap().utterance_word_index.raw()
            );
            assert!(context.first_word <= trace_piece.first_word);
            assert!(context.last_word >= trace_piece.last_word);
        }
        assert_eq!(
            unrequested
                .iter()
                .flat_map(|piece| piece.first_word..=piece.last_word)
                .collect::<Vec<_>>(),
            expected_missing
        );
        let wire = serde_json::to_value(&trace.groups[0].span).expect("wire");
        let decoded: crate::types::traces::FaGroupSpanTrace =
            serde_json::from_value(wire.clone()).expect("trace roundtrip");
        assert_eq!(serde_json::to_value(decoded).expect("rewritten wire"), wire);
        let reread = read_fa_source_named(
            output.document.as_str(),
            talkbank_model::model::TranscriptName::for_path(std::path::Path::new("sample.cha")),
            MainBulletPolicy::DeriveFromWords,
        )
        .expect("written CHAT readback");
        let utterance = reread
            .document()
            .utterances()
            .next()
            .expect("retained utterance");
        let mut retained_words = Vec::new();
        crate::chat_ops::fa::collect_fa_words(&utterance.main.content.content, &mut retained_words);
        assert_eq!(retained_words, words, "all original words retained");
        // Parsed main-tier words do not inherit dependent-tier bullets. Check
        // the serialized %wor boundary itself rather than treating those empty
        // main-tier slots as lost timing.
        let expected_wor = format!(
            "%wor:\t{} .",
            words
                .iter()
                .zip(&expected_timings)
                .map(|(word, timing)| match timing {
                    Some(timing) =>
                        format!("{word} \u{15}{}_{}\u{15}", timing.start_ms, timing.end_ms),
                    None => (*word).to_owned(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        let written_wor = output
            .document
            .as_str()
            .lines()
            .find(|line| line.starts_with("%wor:"));
        if expected_timings.iter().any(Option::is_some) || written_wor.is_some() {
            assert_eq!(
                written_wor,
                Some(expected_wor.as_str()),
                "{label}: no missing-range shift\n{}",
                output.document.as_str()
            );
        }
        if expected_missing.is_empty() {
            assert!(
                output.shortfalls.is_empty(),
                "{label}: {:?}",
                output.shortfalls
            );
        } else {
            assert!(
                matches!(output.shortfalls.as_slice(), [OutputShortfallRecord::TimingIncomplete { required_words: 5, untimed_words, .. }] if *untimed_words == expected_missing.len() as u64),
                "{label}: {:?}",
                output.shortfalls
            );
            assert!(
                matches!(
                    crate::pipeline::post_validate::OutputReport::of(
                        &crate::pipeline::post_validate::ProducedOutput::from(output.document),
                        &output.shortfalls,
                    ),
                    crate::pipeline::post_validate::OutputReport::Diagnosed(_)
                ),
                "partial alignment is never clean success"
            );
        }
        if label == "prefix" {
            // Real explicit --nowor path over the same synthetic cache-only
            // evidence. No provider/model work or compensating tier creation.
            params.wor_tier = WorTierPolicy::Omit;
            let omitted = run_fa_from_ast(
                working.attempt(),
                &audio,
                &crate::api::LanguageCode3::eng(),
                FaServices {
                    pipeline: PipelineServices::new(&pool, &cache),
                    cache_namespace: &namespace,
                    gpu: &crate::worker::pool::GpuModelPlan::alignment(
                        FaEngineName::Wave2Vec,
                        None,
                    ),
                    attention_capture_dir: None,
                },
                &params,
                None,
            )
            .await
            .expect("cache-only omitted-tier control")
            .into_document();
            let (parsed, errors) = batchalign_transform::parse::parse_lenient(
                &crate::chat_parser(),
                omitted.document.as_str(),
            );
            assert!(errors.is_empty(), "{errors:?}");
            assert!(
                parsed
                    .utterances()
                    .all(|utterance| utterance.wor_tier().is_none())
            );
            assert!(
                matches!(
                    omitted.shortfalls.as_slice(),
                    [OutputShortfallRecord::TimingIncomplete {
                        required_words: 5,
                        untimed_words: 5,
                        ..
                    }]
                ),
                "no durable word timing: {:?}",
                omitted.shortfalls
            );
        }
        summaries.push((label, missing, plan.units().count()));
    }
    insta::assert_debug_snapshot!(summaries, @r#"
    [
        (
            "context_suffix",
            [],
            2,
        ),
        (
            "short_anchored_tail",
            [
                0,
                1,
            ],
            1,
        ),
        (
            "prefix",
            [
                0,
            ],
            1,
        ),
        (
            "interior",
            [
                2,
            ],
            2,
        ),
        (
            "suffix",
            [
                4,
            ],
            1,
        ),
        (
            "all",
            [
                0,
                1,
                2,
                3,
                4,
            ],
            0,
        ),
        (
            "anchored_tail",
            [
                0,
                1,
            ],
            1,
        ),
    ]
    "#);
}
