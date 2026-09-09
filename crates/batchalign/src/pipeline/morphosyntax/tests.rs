//! Tests for per-file morphotag pass-through decisions. The full pipeline
//! is exercised by integration tests against a worker pool; these unit
//! tests cover the local predicate and pass-through serialization logic.
use super::states::{ParsedFile, RunOptions};
use super::{run_morphosyntax_pipeline, unsupported_primary_language_error};

/// Params for the pipeline tests: the settings the removed positional
/// arguments used to carry, and no progress port (no job, no reporter).
fn test_params<'a>(
    lang: &'a crate::api::LanguageCode3,
    mwt: &'a MwtDict,
) -> crate::params::MorphosyntaxParams<'a> {
    crate::params::MorphosyntaxParams {
        lang,
        tokenization_mode: TokenizationMode::StanzaRetokenize,
        multilingual_policy: MultilingualPolicy::from_skip_flag(false),
        mwt,
        policy: crate::params::MorphotagExecutionPolicy {
            l2: crate::params::L2MorphotagPolicy::Analyze,
            pos_hints: crate::params::PosHintPolicy::Ignore,
            ca_policy: crate::options::CaMorphotagPolicy::Honor,
        },
        review_level: crate::chat_ops::fa::ReviewLevel::None,
        progress: None,
        cancellation: crate::infer_retry::Cancellation::NotWired {
            reason: "unit test fixture, no job",
        },
    }
}
use crate::api::EngineVersion;
use crate::cache::UtteranceCache;
use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, MwtDict, TokenizationMode};
use crate::pipeline::PipelineServices;
use crate::worker::pool::{PoolConfig, WorkerPool};
use batchalign_transform::parse::parse_lenient;
use batchalign_transform::serialize::to_chat_string;

fn parse(text: &str) -> talkbank_model::model::ChatFile {
    let parser = crate::chat_parser();
    let (chat_file, _) = parse_lenient(&parser, text);
    chat_file
}

#[test]
fn unsupported_primary_language_returns_actionable_error_message() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\tsrp\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\tsrp|test|CHI||female|||Target_Child|||\n\
                *CHI:\tnešto .\n\
                @End\n";
    let chat_file = parse(chat);
    let msg = unsupported_primary_language_error(&chat_file)
        .expect("unsupported primary language must produce an error message");
    assert!(
        msg.contains("srp"),
        "error must name the unsupported lang: {msg}"
    );
    assert!(
        msg.contains("not supported by Stanza"),
        "error must explicitly call out unsupported-by-Stanza so the \
         operator sees the cause in the dashboard: {msg}"
    );
    assert!(
        msg.contains("Fix the @Languages header"),
        "error must be actionable: tell the operator what to do: {msg}"
    );
}

#[test]
fn supported_primary_language_passes() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\tcan't go$v .\n\
                @End\n";
    let chat_file = parse(chat);
    assert!(
        unsupported_primary_language_error(&chat_file).is_none(),
        "eng must pass the Stanza-supported gate",
    );
}

#[test]
fn empty_languages_header_passes_for_ba2_compat() {
    // BA2 defaulted to ["eng"] when no @Languages was present and proceeded.
    // The gate intentionally allows this, files lacking @Languages are not
    // hard-errored; they fall through to the dispatch's default-lang path.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                @End\n";
    let chat_file = parse(chat);
    assert!(
        unsupported_primary_language_error(&chat_file).is_none(),
        "missing @Languages must NOT hard-error (BA2 parity: defaults to eng)",
    );
}

#[test]
fn resolve_per_file_lang_uses_primary_languages_header() {
    // Regression test for the 2026-05-03 morning incident: the morphotag
    // pipeline took its lang from the job-level CommandProfile sentinel
    // ("eng") instead of the file's @Languages header, so every Czech /
    // Spanish / Polish / etc. file got tagged with English Stanza and a
    // falsified `lang=eng` provenance comment.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\tces\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\tces|test|CHI||female|||Target_Child|||\n\
                *CHI:\tahoj .\n\
                @End\n";
    let chat_file = parse(chat);
    let resolved =
        super::resolve_per_file_lang(&chat_file).expect("Czech header must resolve cleanly");
    assert_eq!(
        resolved.as_ref(),
        "ces",
        "Czech file must resolve to ces, not the job-level sentinel",
    );
}

#[test]
fn resolve_per_file_lang_errors_when_languages_absent() {
    // No silent eng fallback, a CHAT file with no `@Languages:` header
    // is a real provenance failure. Surface a typed error so the
    // operator fixes the header and re-runs.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                @End\n";
    let chat_file = parse(chat);
    let err = super::resolve_per_file_lang(&chat_file)
        .expect_err("missing @Languages must error, not silently default to eng");
    assert!(
        err.to_string().contains("`@Languages:`"),
        "error must point at the missing header: {err}"
    );
}

#[test]
fn resolve_per_file_lang_uses_primary_only_when_bilingual() {
    // Bilingual file: primary lang wins. Secondary is consumed by the
    // multilingual policy / per-utterance routing, not by the pipeline's
    // top-level lang choice for inference + provenance.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\tspa, eng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\tspa|test|CHI||female|||Target_Child|||\n\
                *CHI:\thola .\n\
                @End\n";
    let chat_file = parse(chat);
    let resolved = super::resolve_per_file_lang(&chat_file)
        .expect("primary lang must resolve cleanly for bilingual file");
    assert_eq!(
        resolved.as_ref(),
        "spa",
        "primary lang wins; secondary is for per-utterance routing only",
    );
}

#[test]
fn supported_language_with_unsupported_secondary_passes() {
    // A bilingual file where primary is supported (eng) and secondary is
    // not Stanza-supported (e.g., a non-Stanza tongue). The gate looks at
    // primary only; multilingual policy handles per-utterance routing.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng, srp\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                @End\n";
    let chat_file = parse(chat);
    assert!(
        unsupported_primary_language_error(&chat_file).is_none(),
        "primary=eng with secondary=srp must pass (gate is on primary only)",
    );
}

#[tokio::test]
async fn unsupported_primary_language_returns_typed_validation_error() {
    // 2026-05-10 inversion: unsupported primary language is no longer
    // a silent pass-through. The pipeline returns a typed
    // `ServerError::Validation` so the per-file dispatch surfaces the
    // failure to the operator via the dashboard. The OLD behavior
    // (round-trip unchanged with no provenance) was dishonest UX
    // operators got their input back with no signal that nothing
    // happened. See `unsupported_primary_language_error` doc comment
    // for the full rationale.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\tsrp\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\tsrp|test|CHI||female|||Target_Child|||\n\
                *CHI:\tnešto .\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let engine_version = EngineVersion::from("test-morphotag");
    let services = PipelineServices::new(&pool, &cache, &engine_version);

    let lang = crate::api::LanguageCode3::eng();
    let mwt = MwtDict::default();
    let params = test_params(&lang, &mwt);
    let err = run_morphosyntax_pipeline(chat, services, &params)
        .await
        .expect_err("unsupported primary language must surface as Err");

    let msg = err.to_string();
    assert!(
        msg.contains("srp"),
        "error must name the unsupported lang: {msg}"
    );
    assert!(
        msg.contains("not supported by Stanza"),
        "error must call out unsupported-by-Stanza: {msg}"
    );
}

#[tokio::test]
async fn noalign_files_get_morphotagged_with_provenance() {
    // Pin the post-2026-05-07 inversion: NoAlign no longer skips
    // morphotag. The CA disposition producer never consults NoAlign.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                @Options:\tNoAlign\n\
                *CHI:\thello .\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let engine_version = EngineVersion::from("test-morphotag");
    let services = PipelineServices::new(&pool, &cache, &engine_version);

    let lang = crate::api::LanguageCode3::eng();
    let mwt = MwtDict::default();
    let params = test_params(&lang, &mwt);
    let output = run_morphosyntax_pipeline(chat, services, &params)
        .await
        .expect("NoAlign file should be processed (no longer skipped)")
        .into_text();

    assert!(
        output.contains("[ba3 morphotag |"),
        "NoAlign file must receive morphotag provenance, the \
         pipeline is no longer pass-through for NoAlign. Output: {output}"
    );
    // The @Options: NoAlign line itself is preserved (we don't
    // strip it; the directive remains for the `align` command
    // which is what it was always for).
    assert!(
        output.contains("@Options:\tNoAlign"),
        "NoAlign directive must be preserved verbatim",
    );
}

#[tokio::test]
async fn ca_pass_through_strips_legacy_decision_tiers() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                @Options:\tCA\n\
                *CHI:\thello .\n\
                %xalign:\tlegacy\n\
                %xrev:\t[?]\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let engine_version = EngineVersion::from("test-morphotag");
    let services = PipelineServices::new(&pool, &cache, &engine_version);

    let lang = crate::api::LanguageCode3::eng();
    let mwt = MwtDict::default();
    let params = test_params(&lang, &mwt);
    let output = run_morphosyntax_pipeline(chat, services, &params)
        .await
        .expect("CA input should remain a successful pass-through")
        .into_text();

    assert!(!output.contains("%xalign:"), "output: {output}");
    assert!(!output.contains("%xrev:"), "output: {output}");
}

#[tokio::test]
async fn explicit_ca_analyze_policy_injects_result_and_clears_stale_morphology() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                @Options:\tCA\n\
                *CHI:\thello .\n\
                %mor:\tnoun|wrong .\n\
                %gra:\t1|0|ROOT 2|1|PUNCT\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let engine_version = EngineVersion::from("test-morphotag");
    let services = PipelineServices::new(&pool, &cache, &engine_version);

    let lang = crate::api::LanguageCode3::eng();
    let mwt = MwtDict::default();
    let mut params = test_params(&lang, &mwt);
    params.policy.ca_policy = crate::options::CaMorphotagPolicy::Analyze;
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed
        .admit()
        .expect("prevalidate stage")
        .clear()
        .collect(&options);
    let ParsedFile::Analyze(reparsed) = ParsedFile::parse(chat, params.policy.ca_policy)
        .expect("parse again for missing-response boundary")
    else {
        panic!("expected analysis");
    };
    let missing_response = reparsed
        .admit()
        .expect("admit")
        .clear()
        .collect(&options)
        .with_responses(Vec::new());
    assert!(
        missing_response.is_err(),
        "missing worker output cannot admit injection"
    );
    let responses = vec![
        serde_json::from_str(
            r#"{"sentences":[{"words":[
            {"id":1,"text":"hello","lemma":"hello","upos":"NOUN","head":0,"deprel":"root"},
            {"id":2,"text":".","lemma":".","upos":"PUNCT","head":1,"deprel":"punct"}
        ]}]}"#,
        )
        .expect("synthetic Stanza response"),
    ];
    let applied = collected
        .with_responses(responses)
        .expect("response cardinality")
        .apply(&options)
        .await
        .expect("result application stage");
    let parsed = applied.into_chat();
    let serialized = to_chat_string(&parsed);
    assert!(serialized.contains("@Options:\tCA"), "CHAT: {serialized}");
    assert!(!serialized.contains("noun|wrong"), "CHAT: {serialized}");
    assert!(serialized.contains("noun|hello"), "CHAT: {serialized}");
}

#[tokio::test]
async fn pos_hint_evidence_survives_retokenization() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\tcan't go$v .\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let engine_version = EngineVersion::from("test-morphotag");
    let services = PipelineServices::new(&pool, &cache, &engine_version);

    let lang = crate::api::LanguageCode3::eng();
    let mwt = MwtDict::default();
    let mut params = test_params(&lang, &mwt);
    params.policy.pos_hints = crate::params::PosHintPolicy::Honor;
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed
        .admit()
        .expect("prevalidate stage")
        .clear()
        .collect(&options);
    let responses = vec![
        serde_json::from_str(
            r#"{"sentences":[{"words":[
            {"id":1,"text":"ca","lemma":"can","upos":"AUX","head":3,"deprel":"aux"},
            {"id":2,"text":"n't","lemma":"not","upos":"PART","head":3,"deprel":"advmod"},
            {"id":3,"text":"go","lemma":"go","upos":"NOUN","head":0,"deprel":"root"},
            {"id":4,"text":".","lemma":".","upos":"PUNCT","head":3,"deprel":"punct"}
        ]}]}"#,
        )
        .expect("synthetic Stanza response"),
    ];
    let applied = collected
        .with_responses(responses)
        .expect("response cardinality")
        .apply(&options)
        .await
        .expect("result application stage");
    let parsed = applied.into_chat();
    let serialized = to_chat_string(&parsed);
    assert!(serialized.contains("verb|go"), "CHAT: {serialized}");
    assert!(!serialized.contains("verb|not"), "CHAT: {serialized}");
}
