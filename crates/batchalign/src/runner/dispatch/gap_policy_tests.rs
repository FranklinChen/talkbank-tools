//! CLI and persisted policy through real dispatch, cache identity and final CHAT.
//! Synthetic boundaries verify policy, not acoustic model accuracy.

use super::extract_fa_dispatch_params;
use crate::chat_ops::WordIdx;
use crate::chat_ops::fa::{
    AudioIdentity, BulletRepairPolicy, FaGroup, FaProjection, FaWord, MainBulletAuthority,
    TimeSpan, WhisperTextMode, WordGapHealing, WordTiming, apply_fa_results_with_projection_policy,
    cache_key,
};
use crate::cli::args::{Cli, build_typed_options};
use crate::options::CommandOptions;
use crate::types::params::CachePolicy;
use crate::types::worker_v2::FaTextModeV2;
use clap::Parser;
use talkbank_model::UtteranceIdx;
use talkbank_model::model::WriteChat;
use talkbank_parser::TreeSitterParser;

fn submitted(flags: &[&str]) -> CommandOptions {
    let mut argv = vec!["batchalign3", "align", "input.cha"];
    argv.extend_from_slice(flags);
    let cli = Cli::try_parse_from(argv).expect("valid CLI policy");
    let options = build_typed_options(&cli.command, &cli.global)
        .expect("admission")
        .expect("align options");
    serde_json::from_value(serde_json::to_value(options).expect("persist options"))
        .expect("read submitted options")
}

#[test]
fn measured_end_policy_reaches_final_wor_without_changing_model_inputs() {
    let controls: &[(&[&str], u64, WhisperTextMode)] = &[
        (&[], 400, WhisperTextMode::SpaceJoined),
        (&["--pauses"], 400, WhisperTextMode::CharSpaced),
        (&["--word-gaps", "heal"], 600, WhisperTextMode::SpaceJoined),
        (
            &["--pauses", "--word-gaps", "heal"],
            600,
            WhisperTextMode::CharSpaced,
        ),
    ];
    let mut keys = Vec::new();
    for &(flags, expected_end, expected_text) in controls {
        let dispatch = extract_fa_dispatch_params(&submitted(flags), CachePolicy::UseCache)
            .expect("align dispatch");
        let params = dispatch.fa_params;
        assert_eq!(params.whisper_text, expected_text);
        let words = vec!["hello".to_owned(), "world".to_owned()];
        keys.push(cache_key(
            &words,
            &AudioIdentity::from_metadata("sample.wav", 1, 2),
            0,
            2000,
            params.whisper_text,
            params.engine,
        ));
        let parser = TreeSitterParser::new().expect("parser");
        let mut chat = parser
            .parse_chat_file(
                "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n\
             @ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n\
             *SPK:\thello world . \u{15}0_2000\u{15}\n@End\n",
            )
            .expect_built();
        let groups = [FaGroup::test_fixture(
            TimeSpan::new(0, 2000),
            words
                .into_iter()
                .enumerate()
                .map(|(slot, text)| FaWord {
                    utterance_index: UtteranceIdx::new(0),
                    utterance_word_index: WordIdx::new(slot),
                    text,
                })
                .collect(),
            vec![UtteranceIdx::new(0)],
        )];
        let responses = [vec![
            WordTiming::fixture(100, 400),
            WordTiming::fixture(600, 900),
        ]];
        let authority =
            MainBulletAuthority::bind(params.main_bullets, &chat).expect("main bullets");
        let _finalized = apply_fa_results_with_projection_policy(
            &mut chat,
            &groups,
            &responses,
            FaProjection::new(params.projection_policy(), authority),
            params.wor_tier.should_write(),
        )
        .expect("projection")
        .then_finalize(&mut chat, BulletRepairPolicy::Disabled)
        .expect("valid final output");
        let output = chat.to_chat_string();
        assert!(
            output.contains(&format!("hello \u{15}100_{expected_end}\u{15}")),
            "{flags:?}: {output}"
        );
        assert!(output.contains("world \u{15}600_900\u{15}"), "{output}");
    }
    assert!(
        keys.windows(2).all(|pair| pair[0] == pair[1]),
        "neither gap policy nor Whisper-only shaping changes interval-model input"
    );
}

#[test]
fn whisper_shaping_and_legacy_cache_keys_are_independent_of_gap_healing() {
    use crate::worker::request_builder_v2::{fa_backend_for_engine, text_mode_for};
    for (pauses, expected_mode, historical_key_field) in [
        (false, FaTextModeV2::SpaceJoined, "heal_gaps"),
        (true, FaTextModeV2::CharSpaced, "preserve_measured"),
    ] {
        for healing in ["preserve-measured", "heal"] {
            let mut flags = vec!["--fa-engine", "whisper", "--word-gaps", healing];
            if pauses {
                flags.push("--pauses");
            }
            let params = extract_fa_dispatch_params(&submitted(&flags), CachePolicy::UseCache)
                .expect("dispatch")
                .fa_params;
            assert_eq!(
                text_mode_for(fa_backend_for_engine(params.engine), params.whisper_text),
                expected_mode
            );
            assert!(params.word_end_policy().ends_are_derived());
            let key = cache_key(
                &["hello".to_owned()],
                &AudioIdentity::from_metadata("sample.wav", 1, 2),
                0,
                2000,
                params.whisper_text,
                params.engine,
            );
            assert_eq!(
                key,
                crate::chat_ops::CacheKey::from_content(&format!(
                    "sample.wav|1|2|0|2000|hello|{historical_key_field}|whisper_fa"
                ))
            );
        }
    }
}

#[test]
fn omitted_wire_policy_preserves_measured_ends_and_unknown_policy_is_refused() {
    let default = extract_fa_dispatch_params(&submitted(&[]), CachePolicy::UseCache)
        .expect("default")
        .fa_params;
    let mut value = serde_json::to_value(submitted(&[])).expect("wire");
    value.as_object_mut().expect("object").remove("word_gaps");
    for pauses in [false, true] {
        value["pauses"] = serde_json::json!(pauses);
        let options: CommandOptions = serde_json::from_value(value.clone()).expect("old request");
        let actual = extract_fa_dispatch_params(&options, CachePolicy::UseCache).expect("dispatch");
        assert_eq!(
            actual.fa_params.gap_healing,
            WordGapHealing::PreserveMeasured
        );
        assert_eq!(actual.fa_params.gap_healing, default.gap_healing);
    }
    value["word_gaps"] = serde_json::json!("guess");
    assert!(serde_json::from_value::<CommandOptions>(value).is_err());
    assert!(
        Cli::try_parse_from(["batchalign3", "align", "--word-gaps", "guess", "input.cha"]).is_err()
    );
}
