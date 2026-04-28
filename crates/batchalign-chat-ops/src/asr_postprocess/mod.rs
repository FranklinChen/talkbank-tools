//! Compatibility re-export for the canonical ASR post-processing pipeline now
//! defined in `talkbank-transform`.

pub use talkbank_transform::asr_postprocess::*;

#[cfg(test)]
mod integration_tests {
    use super::{
        AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs,
        SpeakerIndex, process_raw_asr,
    };

    #[test]
    fn pipeline_output_still_roundtrips_through_build_chat() {
        let parser = talkbank_parser::TreeSitterParser::new().unwrap();
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![AsrElement {
                    value: AsrRawText::new(
                        "這麼搞笑?我還清了啊!我還覺得奇怪為什麼在一個三次頭的電話打工呢?",
                    ),
                    ts: AsrTimestampSecs(0.0),
                    end_ts: AsrTimestampSecs(0.0),
                    kind: AsrElementKind::Text,
                }],
            }],
        };

        let utterances = process_raw_asr(&output, "yue");
        let desc = crate::build_chat::transcript_from_asr_utterances(
            &utterances,
            &["PAR".to_string()],
            &["yue".to_string()],
            Some("05b_clip"),
            true,
        )
        .expect("test: transcript_from_asr_utterances should succeed");
        let chat = crate::build_chat::build_chat(&desc).expect("build chat");
        let serialized = crate::serialize::to_chat_string(&chat);
        let (_parsed, errors) = crate::parse::parse_lenient(&parser, &serialized);
        assert!(
            errors.is_empty(),
            "generated CHAT should reparse cleanly: {errors:?}"
        );
    }
}
