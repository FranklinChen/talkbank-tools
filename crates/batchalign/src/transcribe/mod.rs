//! Server-side transcribe orchestrator.
//!
//! Owns the full audio-to-CHAT lifecycle:
//! raw ASR inference → Rust normalization → post-processing → CHAT assembly
//! → optional utseg → optional morphosyntax.
//!
//! Split into submodules:
//! - [`types`]: ASR response types, backend selection, transcribe options
//! - [`infer`]: ASR and speaker inference dispatch to worker backends
//! - [`asr_output`]: ASR response conversion, participant IDs, CHAT helpers

mod asr_output;
mod evidence_cache;
mod infer;
pub(crate) mod replay;
pub mod types;

// Re-export the public API so callers don't need to know about the split.
pub(crate) use asr_output::*;
pub(crate) use evidence_cache::*;
pub(crate) use infer::*;
pub use types::*;

use std::path::Path;

use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::pipeline::transcribe::run_transcribe_pipeline;
use crate::runner::util::ProgressSender;

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Process a single audio file through the transcribe pipeline.
///
/// Returns the final serialized CHAT text.
///
/// # Pipeline stages
///
/// 1. **ASR inference**: invoke the selected ASR backend, get raw tokens
/// 2. **Post-processing**: compound merging, number expansion, retokenization
/// 3. **CHAT assembly**: build `ChatFile` AST from utterances
/// 4. **Utterance segmentation** (optional), BERT-based re-segmentation
/// 5. **Morphosyntax** (optional): POS/dependency tagging
pub(crate) async fn process_transcribe(
    audio_path: &Path,
    services: PipelineServices<'_>,
    opts: &TranscribeOptions,
    progress: Option<ProgressSender>,
    debug_dir: Option<&Path>,
) -> Result<String, ServerError> {
    run_transcribe_pipeline(audio_path, services, opts, progress, debug_dir).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DurationSeconds, LanguageCode3};
    use batchalign_transform::asr_postprocess::{self, SpeakerIndex};
    use batchalign_transform::build_chat;
    use batchalign_transform::serialize::to_chat_string;

    #[test]
    fn asr_backend_mapping_distinguishes_live_v2_worker_modes() {
        use crate::options::UtrEngine;
        use crate::types::engines::AsrEngineName;
        assert_eq!(AsrBackend::from(&UtrEngine::RevAi), AsrBackend::RustRevAi);
        assert_eq!(
            AsrBackend::from(&UtrEngine::Whisper),
            AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2)
        );
        assert_eq!(
            AsrBackend::from(&UtrEngine::HkTencent),
            AsrBackend::Worker(AsrWorkerMode::HkTencentV2)
        );
        let backend = |engine: AsrEngineName| {
            AsrBackend::try_from_engine(&engine).expect("engine is implemented")
        };
        assert_eq!(backend(AsrEngineName::RevAi), AsrBackend::RustRevAi);
        assert_eq!(
            backend(AsrEngineName::HkTencent),
            AsrBackend::Worker(AsrWorkerMode::HkTencentV2)
        );
        assert_eq!(
            backend(AsrEngineName::HkAliyun),
            AsrBackend::Worker(AsrWorkerMode::HkAliyunV2)
        );
        assert_eq!(
            backend(AsrEngineName::HkFunaudio),
            AsrBackend::Worker(AsrWorkerMode::HkFunaudioV2)
        );
    }

    /// Provenance must name the engine that was SELECTED, for every engine the
    /// build can run.
    ///
    /// The case list used to include `("whisper_oai", "whisper")`, pinning the
    /// defect: `whisper_oai` fell through a catch-all to stock local Whisper
    /// and the transcript then claimed "whisper". Selection now refuses it, so
    /// the pair is unrepresentable and the table covers only real engines. The
    /// closing assertion makes the table exhaustive over `AsrEngineName::ALL`,
    /// so a new engine variant fails this test until its provenance is stated.
    #[test]
    fn asr_backend_provenance_names_preserve_every_engine_identity() {
        use crate::types::engines::{AsrEngineName, SelectableEngine};

        let cases = [
            (AsrEngineName::RevAi, "rev"),
            (AsrEngineName::WhisperRs, "whisper_rs"),
            (AsrEngineName::Whisper, "whisper"),
            (AsrEngineName::WhisperHub, "whisper_hub"),
            (AsrEngineName::HkTencent, "tencent"),
            (AsrEngineName::HkAliyun, "aliyun"),
            (AsrEngineName::HkFunaudio, "funaudio"),
            (AsrEngineName::HkQwen, "qwen"),
        ];

        for (engine, expected_provenance) in &cases {
            assert_eq!(
                AsrBackend::try_from_engine(engine)
                    .expect("every engine in this table is implemented")
                    .provenance_name()
                    .as_str(),
                *expected_provenance,
                "provenance must retain the selected ASR engine identity"
            );
        }

        for engine in AsrEngineName::ALL {
            let covered = cases.iter().any(|(listed, _)| listed == engine);
            let refused = AsrBackend::try_from_engine(engine).is_err();
            assert!(
                covered || refused,
                "{} is neither covered by this provenance table nor refused by \
                 selection; state which it is",
                engine.as_wire_name()
            );
        }
    }

    #[test]
    fn test_convert_asr_response_groups_by_speaker() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "hello".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.5)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "world".into(),
                    start_s: Some(DurationSeconds(0.5)),
                    end_s: Some(DurationSeconds(1.0)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "hi".into(),
                    start_s: Some(DurationSeconds(1.0)),
                    end_s: Some(DurationSeconds(1.5)),
                    speaker: Some("1".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        };

        let output = convert_asr_response(&response);
        assert_eq!(output.monologues.len(), 2);
        assert_eq!(output.monologues[0].speaker, SpeakerIndex(0));
        assert_eq!(output.monologues[0].elements.len(), 2);
        assert_eq!(output.monologues[1].speaker, SpeakerIndex(1));
        assert_eq!(output.monologues[1].elements.len(), 1);
    }

    #[test]
    fn test_convert_asr_response_handles_speaker_change_and_back() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "a".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.3)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "b".into(),
                    start_s: Some(DurationSeconds(0.3)),
                    end_s: Some(DurationSeconds(0.6)),
                    speaker: Some("1".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "c".into(),
                    start_s: Some(DurationSeconds(0.6)),
                    end_s: Some(DurationSeconds(0.9)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        };

        let output = convert_asr_response(&response);
        assert_eq!(output.monologues.len(), 3);
        assert_eq!(output.monologues[0].speaker, 0);
        assert_eq!(output.monologues[1].speaker, 1);
        assert_eq!(output.monologues[2].speaker, 0);
    }

    #[test]
    fn test_convert_asr_response_empty() {
        let response = AsrResponse {
            tokens: vec![],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        };
        let output = convert_asr_response(&response);
        assert!(output.monologues.is_empty());
    }

    #[test]
    fn test_convert_asr_response_no_speaker_defaults_to_zero() {
        let response = AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        };

        let output = convert_asr_response(&response);
        assert_eq!(output.monologues.len(), 1);
        assert_eq!(output.monologues[0].speaker, 0);
    }

    #[test]
    fn flat_asr_missing_timestamp_is_preserved() {
        let response = AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: None,
                end_s: Some(DurationSeconds(2.0)),
                speaker: Some("0".into()),
                confidence: None,
            }],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        };
        let output = convert_asr_response(&response);
        assert_eq!(
            output.monologues[0].elements[0].ts,
            asr_postprocess::AsrTimestampSecs::Absent
        );
        assert_eq!(output.monologues[0].elements[0].end_ts, 2.0);
    }

    /// Regression test for an operator's bug report (2026-03-18): bare
    /// `batchalign3 transcribe` with no `--diarization` flag must still
    /// produce multi-speaker output when the ASR engine (Rev.AI) returns
    /// speaker-labeled monologues.
    ///
    /// In batchalign2, `process_generation()` unconditionally reads
    /// `utterance["speaker"]` from Rev.AI monologues. The `--diarize` flag
    /// only controls whether a *separate* Pyannote stage runs. BA3 must
    /// match this: speaker labels from the ASR engine are always used.
    #[test]
    fn test_convert_asr_response_always_uses_speaker_labels() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "hello".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.5)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "world".into(),
                    start_s: Some(DurationSeconds(0.5)),
                    end_s: Some(DurationSeconds(1.0)),
                    speaker: Some("1".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        };

        // Speaker labels must be respected regardless of any diarization flag.
        // Previously this test asserted the opposite (1 monologue, speaker 0),
        // which enshrined the bug.
        let output = convert_asr_response(&response);
        assert_eq!(
            output.monologues.len(),
            2,
            "each speaker change must start a new monologue"
        );
        assert_eq!(output.monologues[0].speaker, 0);
        assert_eq!(output.monologues[0].elements.len(), 1);
        assert_eq!(output.monologues[1].speaker, 1);
        assert_eq!(output.monologues[1].elements.len(), 1);
    }

    /// Legacy token speakers are admitted as NUMBERS, and nothing else.
    ///
    /// This used to accept `SPEAKER_2` by taking the text after the last
    /// underscore, which is how two distinct provider labels (`A_0`, `B_0`)
    /// became one track, and how any label ending in a number acquired a
    /// speaker it never claimed. The live worker path no longer reaches here
    /// at all: it carries a typed attribution. What remains is replayed
    /// evidence and Rev's own projection, both numeric, and a zero must keep
    /// working.
    #[test]
    fn legacy_token_speakers_are_admitted_as_numbers_only() {
        assert_eq!(admit_token_speaker(Some("1"), "hello"), SpeakerIndex(1));
        // Replayed `_asr_response.json` evidence carries this.
        assert_eq!(admit_token_speaker(Some("0"), "hello"), SpeakerIndex(0));
        // No label at all: the single track of an undiarized recording.
        assert_eq!(admit_token_speaker(None, "hello"), SpeakerIndex(0));
        // Not a speaker number: reported, and it takes the first track rather
        // than being folded onto whichever speaker its suffix resembles.
        assert_eq!(
            admit_token_speaker(Some("SPEAKER_2"), "hello"),
            SpeakerIndex(0)
        );
        assert_eq!(
            admit_token_speaker(Some("not-a-speaker"), "hello"),
            SpeakerIndex(0)
        );
    }

    #[test]
    fn test_observed_participant_ids() {
        let utterances = vec![
            asr_postprocess::Utterance {
                speaker: SpeakerIndex(0),
                words: vec![],
                lang: None,
            },
            asr_postprocess::Utterance {
                speaker: SpeakerIndex(1),
                words: vec![],
                lang: None,
            },
        ];
        let transcript = build_chat::NamedAsrUtterances::numbered(&utterances)
            .into_transcript(&["eng".to_string()], None, false)
            .unwrap();
        let ids: Vec<_> = transcript
            .description
            .participants
            .iter()
            .map(|participant| participant.id.as_str())
            .collect();
        assert_eq!(ids, vec!["PAR0", "PAR1"]);
    }

    #[test]
    fn test_observed_participant_ids_sparse_speakers() {
        let utterances = vec![asr_postprocess::Utterance {
            speaker: SpeakerIndex(9),
            words: vec![],
            lang: None,
        }];
        let transcript = build_chat::NamedAsrUtterances::numbered(&utterances)
            .into_transcript(&["eng".to_string()], None, false)
            .unwrap();
        assert_eq!(transcript.description.participants.len(), 1);
        assert_eq!(transcript.description.participants[0].id, "PAR9");
    }

    // -----------------------------------------------------------------------
    // Canned-response integration tests
    //
    // Exercise the full conversion chain with realistic ASR payloads:
    //   AsrResponse → convert_asr_response() → process_raw_asr()
    //   → NamedAsrUtterances → into_transcript()
    //   → build_chat() → to_chat_string()
    //
    // These catch bugs that unit tests on individual stages miss, the same
    // class of bugs that echo-worker integration tests failed to expose.
    // -----------------------------------------------------------------------

    /// Build a realistic canned Rev.AI-style response: 2 speakers, ~20 tokens
    /// each, with timing and speaker labels. Simulates a short interview.
    fn canned_revai_two_speaker_response() -> AsrResponse {
        AsrResponse {
            tokens: vec![
                // Speaker 0: first turn
                AsrToken {
                    text: "so".into(),
                    start_s: Some(DurationSeconds(0.24)),
                    end_s: Some(DurationSeconds(0.42)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "tell".into(),
                    start_s: Some(DurationSeconds(0.42)),
                    end_s: Some(DurationSeconds(0.60)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "me".into(),
                    start_s: Some(DurationSeconds(0.60)),
                    end_s: Some(DurationSeconds(0.72)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "about".into(),
                    start_s: Some(DurationSeconds(0.72)),
                    end_s: Some(DurationSeconds(0.96)),
                    speaker: Some("0".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "your".into(),
                    start_s: Some(DurationSeconds(0.96)),
                    end_s: Some(DurationSeconds(1.14)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "experience".into(),
                    start_s: Some(DurationSeconds(1.14)),
                    end_s: Some(DurationSeconds(1.68)),
                    speaker: Some("0".into()),
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "with".into(),
                    start_s: Some(DurationSeconds(1.68)),
                    end_s: Some(DurationSeconds(1.86)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "the".into(),
                    start_s: Some(DurationSeconds(1.86)),
                    end_s: Some(DurationSeconds(1.98)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "program.".into(),
                    start_s: Some(DurationSeconds(1.98)),
                    end_s: Some(DurationSeconds(2.52)),
                    speaker: Some("0".into()),
                    confidence: Some(0.95),
                },
                // Speaker 1: response
                AsrToken {
                    text: "well".into(),
                    start_s: Some(DurationSeconds(3.00)),
                    end_s: Some(DurationSeconds(3.24)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "I".into(),
                    start_s: Some(DurationSeconds(3.24)),
                    end_s: Some(DurationSeconds(3.36)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "started".into(),
                    start_s: Some(DurationSeconds(3.36)),
                    end_s: Some(DurationSeconds(3.72)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "about".into(),
                    start_s: Some(DurationSeconds(3.72)),
                    end_s: Some(DurationSeconds(3.96)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "3".into(),
                    start_s: Some(DurationSeconds(3.96)),
                    end_s: Some(DurationSeconds(4.14)),
                    speaker: Some("1".into()),
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "years".into(),
                    start_s: Some(DurationSeconds(4.14)),
                    end_s: Some(DurationSeconds(4.38)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "ago.".into(),
                    start_s: Some(DurationSeconds(4.38)),
                    end_s: Some(DurationSeconds(4.68)),
                    speaker: Some("1".into()),
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "it".into(),
                    start_s: Some(DurationSeconds(4.80)),
                    end_s: Some(DurationSeconds(4.92)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "was".into(),
                    start_s: Some(DurationSeconds(4.92)),
                    end_s: Some(DurationSeconds(5.10)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "really".into(),
                    start_s: Some(DurationSeconds(5.10)),
                    end_s: Some(DurationSeconds(5.40)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "helpful".into(),
                    start_s: Some(DurationSeconds(5.40)),
                    end_s: Some(DurationSeconds(5.82)),
                    speaker: Some("1".into()),
                    confidence: Some(0.96),
                },
                // Speaker 0: follow-up
                AsrToken {
                    text: "that".into(),
                    start_s: Some(DurationSeconds(6.00)),
                    end_s: Some(DurationSeconds(6.18)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "sounds".into(),
                    start_s: Some(DurationSeconds(6.18)),
                    end_s: Some(DurationSeconds(6.48)),
                    speaker: Some("0".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "great".into(),
                    start_s: Some(DurationSeconds(6.48)),
                    end_s: Some(DurationSeconds(6.78)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                // Speaker 1: closing
                AsrToken {
                    text: "yeah".into(),
                    start_s: Some(DurationSeconds(7.00)),
                    end_s: Some(DurationSeconds(7.24)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "I".into(),
                    start_s: Some(DurationSeconds(7.24)),
                    end_s: Some(DurationSeconds(7.36)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "would".into(),
                    start_s: Some(DurationSeconds(7.36)),
                    end_s: Some(DurationSeconds(7.56)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "recommend".into(),
                    start_s: Some(DurationSeconds(7.56)),
                    end_s: Some(DurationSeconds(8.04)),
                    speaker: Some("1".into()),
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "it".into(),
                    start_s: Some(DurationSeconds(8.04)),
                    end_s: Some(DurationSeconds(8.16)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
            ],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        }
    }

    /// Build a canned Whisper-style response: no speaker labels, single
    /// contiguous stream of tokens with timing.
    fn canned_whisper_no_speaker_response() -> AsrResponse {
        AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "the".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.18)),
                    speaker: None,
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "quick".into(),
                    start_s: Some(DurationSeconds(0.18)),
                    end_s: Some(DurationSeconds(0.42)),
                    speaker: None,
                    confidence: Some(0.93),
                },
                AsrToken {
                    text: "brown".into(),
                    start_s: Some(DurationSeconds(0.42)),
                    end_s: Some(DurationSeconds(0.66)),
                    speaker: None,
                    confidence: Some(0.94),
                },
                AsrToken {
                    text: "fox".into(),
                    start_s: Some(DurationSeconds(0.66)),
                    end_s: Some(DurationSeconds(0.90)),
                    speaker: None,
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "jumps".into(),
                    start_s: Some(DurationSeconds(0.90)),
                    end_s: Some(DurationSeconds(1.20)),
                    speaker: None,
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "over".into(),
                    start_s: Some(DurationSeconds(1.20)),
                    end_s: Some(DurationSeconds(1.44)),
                    speaker: None,
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "the".into(),
                    start_s: Some(DurationSeconds(1.44)),
                    end_s: Some(DurationSeconds(1.56)),
                    speaker: None,
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "lazy".into(),
                    start_s: Some(DurationSeconds(1.56)),
                    end_s: Some(DurationSeconds(1.86)),
                    speaker: None,
                    confidence: Some(0.94),
                },
                AsrToken {
                    text: "dog.".into(),
                    start_s: Some(DurationSeconds(1.86)),
                    end_s: Some(DurationSeconds(2.22)),
                    speaker: None,
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "then".into(),
                    start_s: Some(DurationSeconds(2.40)),
                    end_s: Some(DurationSeconds(2.58)),
                    speaker: None,
                    confidence: Some(0.93),
                },
                AsrToken {
                    text: "it".into(),
                    start_s: Some(DurationSeconds(2.58)),
                    end_s: Some(DurationSeconds(2.70)),
                    speaker: None,
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "sat".into(),
                    start_s: Some(DurationSeconds(2.70)),
                    end_s: Some(DurationSeconds(2.94)),
                    speaker: None,
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "down".into(),
                    start_s: Some(DurationSeconds(2.94)),
                    end_s: Some(DurationSeconds(3.18)),
                    speaker: None,
                    confidence: Some(0.96),
                },
            ],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        }
    }

    /// Run the full canned-response conversion chain and return CHAT text.
    ///
    /// Mirrors the pipeline stages in `pipeline/transcribe.rs`:
    /// `convert_asr_response` → `process_raw_asr` → `NamedAsrUtterances`
    /// → `into_transcript` → `build_chat` → `to_chat_string`.
    fn run_canned_response_to_chat(response: &AsrResponse, media_name: Option<&str>) -> String {
        let asr_output = convert_asr_response(response);
        let utterances = asr_postprocess::process_raw_asr(&asr_output, response.lang.as_ref())
            .expect("test: ASR post-processing must not refuse this input");
        let desc = build_chat::NamedAsrUtterances::numbered(&utterances)
            .into_transcript(&[response.lang.to_string()], media_name, false)
            .expect("test: transcript_from_asr_utterances should succeed")
            .description;
        let chat_file = build_chat::build_chat(&desc).expect("build_chat must succeed");
        to_chat_string(&chat_file)
    }

    /// Full pipeline test: canned Rev.AI 2-speaker response produces valid
    /// multi-speaker CHAT with correct headers and timing.
    #[test]
    fn canned_revai_response_produces_multi_speaker_chat() {
        let response = canned_revai_two_speaker_response();
        let chat = run_canned_response_to_chat(&response, Some("interview.mp3"));

        // Must have 2 @Participants entries (PAR0 + PAR1, generic numbered codes)
        let participants_line = chat
            .lines()
            .find(|l| l.starts_with("@Participants:"))
            .expect("@Participants header missing");
        assert!(
            participants_line.contains("PAR0") && participants_line.contains("PAR1"),
            "expected PAR0 and PAR1 in @Participants, got: {participants_line}"
        );

        // Must have 2 @ID lines
        let id_count = chat.lines().filter(|l| l.starts_with("@ID:")).count();
        assert_eq!(id_count, 2, "expected 2 @ID lines, got {id_count}");

        // Must have utterances from both speakers
        let par0_count = chat.lines().filter(|l| l.starts_with("*PAR0:")).count();
        let par1_count = chat.lines().filter(|l| l.starts_with("*PAR1:")).count();
        assert!(
            par0_count >= 1,
            "expected at least 1 *PAR0 utterance, got {par0_count}"
        );
        assert!(
            par1_count >= 1,
            "expected at least 1 *PAR1 utterance, got {par1_count}"
        );

        // Timing bullets must be present (the \x15 delimiters)
        assert!(
            chat.contains('\x15'),
            "timing bullets missing from output CHAT"
        );

        // @Media header
        assert!(
            chat.contains("@Media:\tinterview, audio"),
            "expected @Media header with stripped extension"
        );

        // Must reparse cleanly
        let parser = batchalign_transform::parse::TreeSitterParser::new().unwrap();
        let (_parsed, errors) = batchalign_transform::parse::parse_lenient(&parser, &chat);
        assert!(
            errors.is_empty(),
            "generated CHAT must reparse cleanly: {errors:?}"
        );
    }

    /// Full pipeline test: canned Whisper response (no speaker labels) produces
    /// single-speaker CHAT with exactly 1 participant.
    #[test]
    fn canned_whisper_response_produces_single_speaker_chat() {
        let response = canned_whisper_no_speaker_response();
        let chat = run_canned_response_to_chat(&response, Some("recording.wav"));

        // Must have exactly 1 participant
        let id_count = chat.lines().filter(|l| l.starts_with("@ID:")).count();
        assert_eq!(
            id_count, 1,
            "expected 1 @ID line for single-speaker, got {id_count}"
        );

        // All utterances must be from PAR0 (speaker 0)
        let non_par0_utts: Vec<&str> = chat
            .lines()
            .filter(|l| l.starts_with('*') && !l.starts_with("*PAR0:"))
            .collect();
        assert!(
            non_par0_utts.is_empty(),
            "all utterances should be *PAR0 for single-speaker, found: {non_par0_utts:?}"
        );

        // Must have at least 1 utterance
        let par0_count = chat.lines().filter(|l| l.starts_with("*PAR0:")).count();
        assert!(
            par0_count >= 1,
            "expected at least 1 *PAR0 utterance, got {par0_count}"
        );

        // Timing bullets must be present
        assert!(
            chat.contains('\x15'),
            "timing bullets missing from single-speaker output"
        );

        // Must reparse cleanly
        let parser = batchalign_transform::parse::TreeSitterParser::new().unwrap();
        let (_parsed, errors) = batchalign_transform::parse::parse_lenient(&parser, &chat);
        assert!(
            errors.is_empty(),
            "generated CHAT must reparse cleanly: {errors:?}"
        );
    }

    /// Regression test: Rev.AI response with speaker labels must produce
    /// multi-speaker output regardless of the diarization flag.
    ///
    /// This is the end-to-end version of the
    /// `test_convert_asr_response_always_uses_speaker_labels` unit test.
    /// It exercises the full chain through CHAT serialization to catch
    /// any stage that might collapse speakers.
    #[test]
    fn canned_revai_speaker_labels_produce_multi_speaker_regardless_of_diarize_flag() {
        let response = canned_revai_two_speaker_response();

        // The pipeline does not consult opts.diarize during
        // convert_asr_response → process_raw_asr → build_chat. Verify this
        // by running the same canned data through the conversion chain.
        let chat = run_canned_response_to_chat(&response, Some("test.mp3"));

        // Count distinct speaker codes in utterance lines
        let speaker_codes: std::collections::BTreeSet<&str> = chat
            .lines()
            .filter(|l| l.starts_with('*'))
            .filter_map(|l| l.split(':').next())
            .map(|code| code.trim_start_matches('*'))
            .collect();
        assert!(
            speaker_codes.len() >= 2,
            "Rev.AI response with speaker labels must produce at least 2 distinct speakers \
             in the output CHAT, but only found: {speaker_codes:?}. \
             This was an operator's bug report: speaker labels from ASR must always be used."
        );
    }

    /// Legacy unlabeled Whisper conversion produces one observed speaker.
    /// Requested counts no longer enter participant naming at all.
    #[test]
    fn canned_whisper_no_labels_stays_single_speaker_even_with_high_num_speakers() {
        let response = canned_whisper_no_speaker_response();
        // The conversion's existing absence policy maps these tokens to zero;
        // participant naming must not add speakers beyond that observation.
        let chat = run_canned_response_to_chat(&response, None);

        let speaker_codes: std::collections::BTreeSet<&str> = chat
            .lines()
            .filter(|l| l.starts_with('*'))
            .filter_map(|l| l.split(':').next())
            .map(|code| code.trim_start_matches('*'))
            .collect();
        assert_eq!(
            speaker_codes.len(),
            1,
            "Whisper response without speaker labels should produce exactly 1 speaker, got: {speaker_codes:?}"
        );
        assert!(
            speaker_codes.contains("PAR0"),
            "sole speaker should be PAR0"
        );
    }

    /// Verify that number expansion works end-to-end in the canned Rev.AI
    /// response (the token "3" should become "three" in the output).
    #[test]
    fn canned_revai_response_expands_numbers() {
        let response = canned_revai_two_speaker_response();
        let chat = run_canned_response_to_chat(&response, None);

        assert!(
            chat.contains("three"),
            "number '3' in canned response should be expanded to 'three' in CHAT output"
        );
        // The raw digit should not appear as a standalone word
        let has_raw_digit = chat
            .lines()
            .any(|l| l.starts_with('*') && l.split_whitespace().any(|w| w == "3"));
        assert!(
            !has_raw_digit,
            "raw digit '3' should not appear as a standalone word in utterance lines"
        );
    }

    /// Verify that embedded sentence-ending punctuation in canned responses
    /// (e.g. "program." or "ago.") splits correctly into utterance boundaries.
    #[test]
    fn canned_revai_response_splits_on_embedded_periods() {
        let response = canned_revai_two_speaker_response();
        let asr_output = convert_asr_response(&response);
        let utterances = asr_postprocess::process_raw_asr(&asr_output, response.lang.as_ref())
            .expect("test: ASR post-processing must not refuse this input");

        // "program." and "ago." should create utterance boundaries, so we
        // expect more than 2 utterances from the 4-turn conversation.
        assert!(
            utterances.len() >= 3,
            "expected at least 3 utterances from embedded-period splitting, got {}",
            utterances.len()
        );

        // Every utterance must end with a terminator
        for (i, utt) in utterances.iter().enumerate() {
            let last = utt.words.last().expect("utterance should have words");
            assert!(
                matches!(last.text.as_str(), "." | "?" | "!"),
                "utterance {i} should end with a terminator, got: {:?}",
                last.text
            );
        }
    }
}
