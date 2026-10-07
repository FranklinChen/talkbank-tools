//! Tests for per-file morphotag pass-through decisions. The full pipeline
//! is exercised by integration tests against a worker pool; these unit
//! tests cover the local predicate and pass-through serialization logic.
use super::run_morphosyntax_pipeline;
use super::states::{ParsedFile, RunOptions};

/// Params for the pipeline tests: the settings the removed positional
/// arguments used to carry, and no progress port (no job, no reporter).
fn test_params(mwt: &MwtDict) -> crate::params::MorphosyntaxParams<'_> {
    crate::params::MorphosyntaxParams {
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
use crate::cache::UtteranceCache;
use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, MwtDict, TokenizationMode};
use crate::morphosyntax::identity::AdmittedMorphosyntaxResponse;
use crate::pipeline::PipelineServices;
use crate::worker::pool::{PoolConfig, WorkerPool};
use batchalign_transform::parse::parse_lenient;
use batchalign_transform::serialize::to_chat_string;

const ADMISSION_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI||female|||Target_Child|||\n\
*CHI:\thello world .\n@End\n";

#[test]
fn a_contradicted_replacement_plan_is_a_system_failure_not_invalid_chat() {
    let error = crate::error::ServerError::ReplacementPlanContradicted {
        planned: Some(talkbank_parser::ReplacementTiers::Morphosyntax),
        admitted: Some(talkbank_parser::ReplacementTiers::WordTiming),
    };
    assert_eq!(
        crate::runner::util::classify_server_error(&error),
        crate::scheduling::FailureCategory::System
    );
    #[cfg(feature = "server")]
    assert_eq!(
        axum::response::IntoResponse::into_response(error).status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[test]
fn regeneration_admission_distinguishes_replaced_and_retained_faults() {
    use crate::options::CaMorphotagPolicy::{Analyze, Honor};
    let broken_mor = ADMISSION_CHAT.replace("@End", "%mor:\tnoun|hello\n@End");
    assert!(matches!(
        ParsedFile::parse(&broken_mor, Honor),
        Ok(ParsedFile::Analyze(_))
    ));
    let ca_broken_mor = broken_mor.replace("@ID:", "@Options:\tCA\n@ID:");
    assert!(matches!(
        ParsedFile::parse(&ca_broken_mor, Honor),
        Err(crate::error::ServerError::ChatReplacementAdmission(_))
    ));
    assert!(matches!(
        ParsedFile::parse(&ca_broken_mor, Analyze),
        Ok(ParsedFile::Analyze(_))
    ));

    // Word timing is a sidecar: count drift is legal, unlike phonology alignment.
    let drifted_wor = ADMISSION_CHAT.replace("@End", "%wor:\thello .\n@End");
    for policy in [Honor, Analyze] {
        assert!(matches!(
            ParsedFile::parse(&drifted_wor, policy),
            Ok(ParsedFile::Analyze(_))
        ));
    }
    for input in [
        ADMISSION_CHAT.replace("@End", "%pho:\thəloʊ\n@End"),
        ADMISSION_CHAT.replace("*CHI:", "@Languages:\teng\n*CHI:"),
        ADMISSION_CHAT.replace("*CHI:", "@Options:\tCA\n*CHI:"),
        ADMISSION_CHAT.replace("hello world .", "<hello [/] world ."),
    ] {
        for policy in [Honor, Analyze] {
            assert!(
                matches!(
                    ParsedFile::parse(&input, policy),
                    Err(crate::error::ServerError::ChatReplacementAdmission(_))
                ),
                "retained fault must refuse before collection: {input}"
            );
        }
    }
}

#[tokio::test]
async fn admitted_ca_pass_through_preserves_ordinary_comment_tiers() {
    let chat = ADMISSION_CHAT
        .replace("@ID:", "@Options:\tCA\n@ID:")
        .replace("@End", "%com:\tordinary contributor content\n@End");
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let mwt = MwtDict::default();
    let output = run_morphosyntax_pipeline(&chat, services, &test_params(&mwt))
        .await
        .expect("valid CA requires no worker")
        .into_text();
    assert!(output.contains("%com:\tordinary contributor content"));
    assert!(!output.contains("%mor:"));
    insta::assert_snapshot!(output, @"
    @UTF8
    @Begin
    @Languages:\teng
    @Participants:\tCHI Target_Child
    @Options:\tCA
    @ID:\teng|test|CHI||female|||Target_Child|||
    *CHI:\thello world .
    %com:\tordinary contributor content
    @End
    ");
}

#[tokio::test]
async fn incremental_morphology_refuses_invalid_prior_tiers_before_reuse() {
    let before = ADMISSION_CHAT.replace("@End", "%mor:\tnoun|hello\n@End");
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let mwt = MwtDict::default();
    let result = crate::morphosyntax::process_morphosyntax_incremental(
        &before,
        ADMISSION_CHAT,
        services,
        &test_params(&mwt),
    )
    .await;
    assert!(matches!(
        result,
        Err(crate::error::ServerError::ChatAdmission(_))
    ));
}

fn parse(text: &str) -> talkbank_model::model::ChatFile {
    let parser = crate::chat_parser();
    let (chat_file, _) = parse_lenient(&parser, text);
    chat_file
}

fn primary_capability(
    chat: &talkbank_model::model::ChatFile,
) -> Result<(), crate::error::ServerError> {
    let language = super::resolve_per_file_lang(chat)?;
    crate::morphosyntax::AnalysisUnavailable::admit_primary(&language)?;
    Ok(())
}

#[tokio::test]
async fn incremental_morphology_copies_only_admitted_prior_analysis_without_inference() {
    let before = ADMISSION_CHAT.replace(
        "@End",
        "%mor:\tintj|hello noun|world .\n\
%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n@End",
    );
    let after = ADMISSION_CHAT.replace("@End", "%mor:\tnoun|wrong\n@End");
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let output = crate::morphosyntax::process_morphosyntax_incremental(
        &before,
        &after,
        PipelineServices::new(&pool, &cache),
        &test_params(&MwtDict::default()),
    )
    .await
    .expect("valid prior analysis needs no inference")
    .into_text();
    assert!(output.contains("%mor:\tintj|hello noun|world ."));
    assert!(output.contains("%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT"));
    assert!(!output.contains("noun|wrong"));
    insta::assert_snapshot!(output, @"
    @UTF8
    @Begin
    @Languages:\teng
    @Participants:\tCHI Target_Child
    @ID:\teng|test|CHI||female|||Target_Child|||
    *CHI:\thello world .
    %mor:\tintj|hello noun|world .
    %gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT
    @End
    ");
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
    let msg = primary_capability(&chat_file)
        .expect_err("unsupported primary language must refuse analysis")
        .to_string();
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
        msg.contains("Keep truthful language declarations") && !msg.contains("Fix the @Languages"),
        "model availability must never advise falsifying CHAT: {msg}"
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
        primary_capability(&chat_file).is_ok(),
        "eng must pass the Stanza-supported gate",
    );
}

#[test]
fn a_language_support_predicate_is_not_complete_input_admission() {
    // No unsupported code is present, but missing required headers must still
    // refuse complete admission. This predicate cannot establish validity.
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                @End\n";
    let english = crate::api::LanguageCode3::eng();
    assert!(crate::morphosyntax::AnalysisUnavailable::admit_primary(&english).is_ok());
    assert!(matches!(
        ParsedFile::parse(chat, crate::options::CaMorphotagPolicy::Honor),
        Err(crate::error::ServerError::ChatReplacementAdmission(_))
    ));
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
    // Declaring an unsupported secondary is not itself a request to analyze
    // an utterance in that language. Per-word L2 retains its own policy.
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
        primary_capability(&chat_file).is_ok(),
        "primary=eng with secondary=srp must pass (gate is on primary only)",
    );
}

#[tokio::test]
async fn unsupported_primary_language_returns_typed_analysis_unavailable() {
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
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let err = run_morphosyntax_pipeline(chat, services, &params)
        .await
        .expect_err("unsupported primary language must surface as Err");

    let msg = err.to_string();
    assert!(matches!(
        err,
        crate::error::ServerError::AnalysisUnavailable(_)
    ));
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
async fn effective_precode_capability_refuses_without_inference_or_partial_output() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let caps = std::collections::BTreeMap::from([(
        "eng".to_string(),
        crate::types::worker::StanzaLanguageProcessors {
            alpha2: "en".to_string(),
            processors: ["tokenize", "pos", "lemma", "depparse"]
                .into_iter()
                .map(String::from)
                .collect(),
        },
    )]);
    let pool = WorkerPool::with_test_stanza_registry(
        crate::stanza_registry::StanzaRegistry::from_capabilities(&caps),
    );
    // Supported English registry but no workers: refusal happens at admission,
    // not a worker timeout or missing response. Mixed files refuse as a whole.
    for main in [
        "[- que] hello world .",
        "hello world .\n*CHI:\t[- que] hello world .",
    ] {
        let chat = ADMISSION_CHAT
            .replace("@Languages:\teng", "@Languages:\teng, que")
            .replace("hello world .", main);
        let error = run_morphosyntax_pipeline(
            &chat,
            PipelineServices::new(&pool, &cache),
            &test_params(&MwtDict::default()),
        )
        .await
        .expect_err("unsupported effective work must refuse");
        let crate::error::ServerError::AnalysisUnavailable(ref unavailable) = error else {
            panic!("expected analysis unavailability before inference");
        };
        assert_eq!(unavailable.language().as_ref(), "que");
        assert_eq!(
            unavailable.requirement(),
            crate::morphosyntax::AnalysisLanguageRequirement::EffectiveUtterance
        );
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::AnalysisUnavailable
        );
    }
}

#[tokio::test]
async fn incremental_complete_unsupported_precode_needs_no_model_capability() {
    let chat = ADMISSION_CHAT
        .replace("@Languages:\teng", "@Languages:\teng, que")
        .replace("hello world .", "[- que] hello world .");
    let before = chat.replace(
        "@End",
        "%mor:\tintj|hello noun|world .\n\
%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n@End",
    );
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let output = crate::morphosyntax::process_morphosyntax_incremental(
        &before,
        &chat,
        PipelineServices::new(&pool, &cache),
        &test_params(&MwtDict::default()),
    )
    .await
    .expect("source-admitted compatible complete analysis needs no model")
    .into_text();
    assert!(output.contains("[- que] hello world ."));
    assert!(output.contains("%mor:\tintj|hello noun|world ."));
    assert!(output.contains("%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT"));
    let incomplete = chat.replace("@End", "%mor:\tintj|hello noun|world .\n@End");
    let error = crate::morphosyntax::process_morphosyntax_incremental(
        &incomplete,
        &chat,
        PipelineServices::new(&pool, &cache),
        &test_params(&MwtDict::default()),
    )
    .await
    .expect_err("incomplete prior pair cannot discharge unavailable analysis");
    assert!(matches!(
        error,
        crate::error::ServerError::AnalysisUnavailable(_)
    ));
}

#[test]
fn unsupported_primary_nonlexical_policy_and_ca_disposition_remain_distinct() {
    let chat = ADMISSION_CHAT
        .replace("eng", "que")
        .replace("hello world", "xxx");
    let error = ParsedFile::parse(&chat, crate::options::CaMorphotagPolicy::Honor)
        .err()
        .expect("primary support policy also applies to nonlexical files");
    assert!(matches!(
        error,
        crate::error::ServerError::AnalysisUnavailable(_)
    ));
    let ca = chat.replace("@ID:", "@Options:\tCA\n@ID:");
    assert!(matches!(
        ParsedFile::parse(&ca, crate::options::CaMorphotagPolicy::Honor),
        Ok(ParsedFile::PassThrough(_))
    ));
    assert!(matches!(
        ParsedFile::parse(&ca, crate::options::CaMorphotagPolicy::Analyze),
        Err(crate::error::ServerError::AnalysisUnavailable(_))
    ));
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
                @Options:\tNoAlign\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy).expect("parse")
    else {
        panic!("NoAlign must not bypass morphology");
    };
    let response = serde_json::from_str(
        r#"{"sentences":[{"words":[
        {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"},
        {"id":2,"text":".","lemma":".","upos":"PUNCT","head":1,"deprel":"punct"}
    ]}]}"#,
    )
    .expect("worker response fixture");
    let output = parsed
        .collect(&options)
        .with_responses(vec![AdmittedMorphosyntaxResponse::for_test(
            response, "test", "eng",
        )])
        .expect("matched response")
        .apply(&options)
        .await
        .expect("complete injection")
        .postcheck(&options)
        .expect("postcheck")
        .serialize()
        .into_text();
    assert!(
        output.contains("%mor:"),
        "NoAlign must receive actual morphology: {output}"
    );
    assert!(
        output.contains("%gra:"),
        "NoAlign must receive actual dependencies: {output}"
    );

    assert!(
        output.contains("[fc-ba3 morphotag |"),
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

/// The morphotag stamp names the model the worker reported for the responses
/// the file applied. Nothing on the pipeline's services can reach it: they
/// carry no engine version, which is how `transcribe` (which runs this same
/// pipeline through `process_morphosyntax`) used to stamp its ASR engine here.
#[tokio::test]
async fn morphotag_provenance_names_the_worker_reported_model() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed.collect(&options);
    let responses: Vec<crate::chat_ops::nlp::UdResponse> = vec![
        serde_json::from_str(
            r#"{"sentences":[{"words":[
            {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"},
            {"id":2,"text":".","lemma":".","upos":"PUNCT","head":1,"deprel":"punct"}
        ]}]}"#,
        )
        .expect("synthetic Stanza response"),
    ];
    let output = collected
        .with_responses(
            responses
                .into_iter()
                .map(|response| AdmittedMorphosyntaxResponse::for_test(response, "9.9.9", "eng"))
                .collect(),
        )
        .expect("response cardinality")
        .apply(&options)
        .await
        .expect("result application stage")
        .postcheck(&options)
        .expect("post-validation stage")
        .serialize()
        .into_text();
    assert!(
        output.contains(&format!(
            "{}engine=stanza-9.9.9:eng:standard ; lang=eng ; retokenize=true | ",
            crate::provenance::written_stamp_opening("morphotag")
        )),
        "{output}"
    );
}

/// Worker cardinality is an analysis fault, not invalid source CHAT.
#[test]
fn worker_response_count_failure_cannot_admit_inference_or_blame_valid_chat() {
    let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
    batchalign_transform::parse_and_validate_with_parser(
        &parser,
        ADMISSION_CHAT,
        talkbank_model::ParseValidateOptions::default().with_alignment(),
    )
    .expect("fully valid source control");
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::default());
    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(PipelineServices::new(&pool, &cache), &params);
    for count in [0, 2] {
        let ParsedFile::Analyze(parsed) =
            ParsedFile::parse(ADMISSION_CHAT, params.policy.ca_policy)
                .expect("valid source admission")
        else {
            panic!("analysis expected")
        };
        let responses = (0..count)
            .map(|_| {
                AdmittedMorphosyntaxResponse::for_test(
                    crate::chat_ops::nlp::UdResponse {
                        sentences: Vec::new(),
                    },
                    "test",
                    "eng",
                )
            })
            .collect();
        let error = match parsed.collect(&options).with_responses(responses) {
            Ok(_) => panic!("unpaired responses must not mint Analysis<Inferred>"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            crate::error::ServerError::MorphosyntaxInjection(_)
        ));
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::System
        );
    }
}

/// Correct batch counts still cannot certify lexical completion.
#[tokio::test]
async fn incomplete_model_analysis_cannot_advance_to_applied() {
    let chat = "@UTF8\n@Begin\n@Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI|||||Target_Child|||\n\
                *CHI:\thello good world .\n@End\n";
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let mwt = MwtDict::default();
    let cases = [
        ("no sentences", r#"{"sentences":[]}"#, "nlp_no_sentences"),
        (
            "missing words",
            r#"{"sentences":[{"words":[
                {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"}
            ]}]}"#,
            "misalignment_bug",
        ),
        (
            "malformed dependency",
            r#"{"sentences":[{"words":[
                {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":9,"deprel":"dep"},
                {"id":2,"text":"good","lemma":"good","upos":"ADJ","head":3,"deprel":"amod"},
                {"id":3,"text":"world","lemma":"world","upos":"NOUN","head":0,"deprel":"root"}
            ]}]}"#,
            "mapping_failed",
        ),
    ];
    // Retokenization can intentionally change the number of main-tier words,
    // so lexical cardinality is tested in Preserve mode, not by imposing a
    // Preserve invariant on that distinct policy.
    for mode in [
        TokenizationMode::Preserve,
        TokenizationMode::StanzaRetokenize,
    ] {
        for (label, wire, strategy) in cases {
            if label == "missing words" && mode == TokenizationMode::StanzaRetokenize {
                continue;
            }
            let mut params = test_params(&mwt);
            params.tokenization_mode = mode;
            let options = RunOptions::new(services, &params);
            let ParsedFile::Analyze(parsed) =
                ParsedFile::parse(chat, params.policy.ca_policy).expect("parse")
            else {
                panic!("lexical CHAT requires analysis");
            };
            let response = serde_json::from_str(wire).expect("worker wire fixture");
            let inferred = parsed
                .collect(&options)
                .with_responses(vec![AdmittedMorphosyntaxResponse::for_test(
                    response, "test", "eng",
                )])
                .expect("one response for one utterance");
            let error = match inferred.apply(&options).await {
                Ok(_) => panic!("{label} in {mode:?} admitted incomplete output"),
                Err(error) => error,
            };
            assert!(
                matches!(error, crate::error::ServerError::MorphosyntaxInjection(_)),
                "{error}"
            );
            assert!(error.to_string().contains(strategy), "{label}: {error}");
        }
    }
}

#[tokio::test]
async fn ca_pass_through_strips_legacy_decision_tiers() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @Options:\tCA\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                %xalign:\tlegacy\n\
                %xrev:\t[?]\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let params = test_params(&mwt);
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
                @Options:\tCA\n\
                @ID:\teng|test|CHI||female|||Target_Child|||\n\
                *CHI:\thello .\n\
                %mor:\tnoun|wrong .\n\
                %gra:\t1|0|ROOT 2|1|PUNCT\n\
                @End\n";
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let mut params = test_params(&mwt);
    params.policy.ca_policy = crate::options::CaMorphotagPolicy::Analyze;
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed.collect(&options);
    let ParsedFile::Analyze(reparsed) = ParsedFile::parse(chat, params.policy.ca_policy)
        .expect("parse again for missing-response boundary")
    else {
        panic!("expected analysis");
    };
    let missing_response = reparsed.collect(&options).with_responses(Vec::new());
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
        .with_responses(
            responses
                .into_iter()
                .map(|response| AdmittedMorphosyntaxResponse::for_test(response, "9.9.9", "eng"))
                .collect(),
        )
        .expect("response cardinality")
        .apply(&options)
        .await
        .expect("result application stage");
    let parsed = applied.into_chat();
    let serialized = to_chat_string(&parsed);
    assert!(serialized.contains("@Options:\tCA"), "CHAT: {serialized}");
    assert!(!serialized.contains("noun|wrong"), "CHAT: {serialized}");
    // The injected response went through the production invariant chain:
    // an isolated `hello` the worker read as a noun is a communicator
    // (Defect 11, `discourse_marker.rs`), so the noun reading does not
    // survive, and neither does the stale one.
    assert!(serialized.contains("intj|hello"), "CHAT: {serialized}");
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
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let mut params = test_params(&mwt);
    params.policy.pos_hints = crate::params::PosHintPolicy::Honor;
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed.collect(&options);
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
        .with_responses(
            responses
                .into_iter()
                .map(|response| AdmittedMorphosyntaxResponse::for_test(response, "9.9.9", "eng"))
                .collect(),
        )
        .expect("response cardinality")
        .apply(&options)
        .await
        .expect("result application stage");
    let parsed = applied.into_chat();
    let serialized = to_chat_string(&parsed);
    assert!(serialized.contains("verb|go"), "CHAT: {serialized}");
    assert!(!serialized.contains("verb|not"), "CHAT: {serialized}");
}

#[tokio::test]
async fn morphosyntax_pos_hints_follow_admitted_mor_positions() {
    use talkbank_model::model::{MorStem, PosCategory};
    let ordinary = r#"{"sentences":[{"words":[
        {"id":1,"text":"I","lemma":"I","upos":"PRON","head":2,"deprel":"nsubj"},
        {"id":2,"text":"run","lemma":"run","upos":"VERB","head":0,"deprel":"root"},
        {"id":3,"text":"home","lemma":"home","upos":"ADV","head":2,"deprel":"advmod"}
    ]}]}"#;
    let comma = r#"{"sentences":[{"words":[
        {"id":1,"text":"I","lemma":"I","upos":"PRON","head":3,"deprel":"nsubj"},
        {"id":2,"text":",","lemma":",","upos":"PUNCT","head":3,"deprel":"punct"},
        {"id":3,"text":"run","lemma":"run","upos":"VERB","head":0,"deprel":"root"},
        {"id":4,"text":"home","lemma":"home","upos":"ADV","head":3,"deprel":"advmod"}
    ]}]}"#;
    let mains = [
        ("I run$n home .", ordinary),
        ("I , run$n home .", comma),
        ("&-um I run$n home .", ordinary),
        ("&+un I run$n home .", ordinary),
        ("xxx I run$n home .", ordinary),
        ("I goed [: run$n] home .", ordinary),
        ("nam [: I run$n] home .", ordinary),
        ("I run$v [: run$n] home .", ordinary),
        ("I run$v [/] run$n home .", ordinary),
    ];
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let services = PipelineServices::new(&pool, &cache);
    let mwt = MwtDict::default();
    for (main, model) in mains {
        let chat = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n\
            *CHI:\t{main}\n@End\n"
        );
        for mode in [
            TokenizationMode::Preserve,
            TokenizationMode::StanzaRetokenize,
        ] {
            for policy in [
                crate::params::PosHintPolicy::Honor,
                crate::params::PosHintPolicy::Ignore,
            ] {
                let mut params = test_params(&mwt);
                params.tokenization_mode = mode;
                params.policy.pos_hints = policy;
                let options = RunOptions::new(services, &params);
                let ParsedFile::Analyze(parsed) = ParsedFile::parse(&chat, params.policy.ca_policy)
                    .expect("fully admitted source")
                else {
                    panic!("expected analysis");
                };
                let collected = parsed.collect(&options);
                let response = serde_json::from_str(model).expect("model double");
                let applied = collected
                    .with_responses(vec![AdmittedMorphosyntaxResponse::for_test(
                        response, "9.9.9", "eng",
                    )])
                    .expect("response admission")
                    .apply(&options)
                    .await
                    .expect("application");
                let mut output = applied.into_chat();
                talkbank_model::validate_chat_file_with_options(
                    &mut output,
                    &talkbank_model::ParseValidateOptions::default().with_alignment(),
                )
                .expect("fully valid output");
                let utterance = output.utterances().next().expect("utterance");
                let mor = utterance.mor_tier().expect("complete morphology");
                let run_pos = match policy {
                    crate::params::PosHintPolicy::Honor => "noun",
                    crate::params::PosHintPolicy::Ignore => "verb",
                };
                let run = mor
                    .items()
                    .iter()
                    .find(|item| item.main.lemma == MorStem::new("run"))
                    .expect("run has a morphology item");
                let home = mor
                    .items()
                    .iter()
                    .find(|item| item.main.lemma == MorStem::new("home"))
                    .expect("home has a morphology item");
                assert_eq!(run.main.pos, PosCategory::new(run_pos), "{main}");
                assert_eq!(home.main.pos, PosCategory::new("adv"), "{main}");
                assert!(utterance.gra_tier().is_some(), "{main}");
            }
        }
    }
}

/// A generated transcript diagnosed for one invalid word ("b2", in its second
/// utterance) built from invented utterances, localized to that utterance.
fn transcript_diagnosed_in_its_second_utterance()
-> crate::pipeline::post_validate::LocalizedDiagnosis {
    use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};
    const GENERATED: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tPAR0 Participant\n@ID:\teng|test|PAR0|||||Participant|||\n\
*PAR0:\thello .\n\
*PAR0:\twe took it .\n\
*PAR0:\tgoodbye .\n@End\n";
    let (mut file, errors) = parse_lenient(&crate::chat_parser(), GENERATED);
    assert!(errors.is_empty(), "{errors:?}");
    let utterance = (&mut file.lines)
        .into_iter()
        .filter_map(|line| match line {
            Line::Utterance(utterance) => Some(utterance),
            _ => None,
        })
        .nth(1)
        .expect("three utterances");
    utterance.main.content.content = TierContentItems::new(
        ["we", "took", "b2"]
            .into_iter()
            .map(|word| UtteranceContent::Word(Box::new(Word::simple(word))))
            .collect(),
    );
    let crate::pipeline::post_validate::ProducedOutput::Diagnosed(diagnosed) =
        crate::pipeline::post_validate::PostValidated::produced(
            file,
            crate::api::ReleasedCommand::Transcribe,
        )
    else {
        panic!("the invalid word must diagnose the transcript");
    };
    let localized = diagnosed
        .localize()
        .expect("the finding is the second utterance's");
    assert_eq!(localized.held_out().ordinals().collect::<Vec<_>>(), [1]);
    localized
}

/// One single-word utterance's analysis, as a worker returns it.
fn one_word_analysis(word: &str) -> AdmittedMorphosyntaxResponse {
    let response = serde_json::from_str(&format!(
        r#"{{"sentences":[{{"words":[
        {{"id":1,"text":"{word}","lemma":"{word}","upos":"INTJ","head":0,"deprel":"root"}},
        {{"id":2,"text":".","lemma":".","upos":"PUNCT","head":1,"deprel":"punct"}}
    ]}}]}}"#
    ))
    .expect("worker response fixture");
    AdmittedMorphosyntaxResponse::for_test(response, "test", "eng")
}

/// Transcribe's morphosyntax on a transcript diagnosed for one utterance tags
/// every other utterance. The faulty one is never collected for a worker
/// (two responses pair with exactly the two other utterances), keeps its
/// form with no `%mor`, and the result is judged afresh: still diagnosed,
/// for that word alone. Before, the whole file went untagged.
#[tokio::test]
async fn a_diagnosed_transcript_is_tagged_outside_its_faulty_utterance() {
    use super::states::LocalizedParse;
    let tempdir = tempfile::tempdir().expect("tempdir");
    let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
        .await
        .expect("cache");
    let pool = WorkerPool::new(PoolConfig::default());
    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(PipelineServices::new(&pool, &cache), &params);
    let LocalizedParse::Analyze(parsed) = ParsedFile::from_localized(
        transcript_diagnosed_in_its_second_utterance(),
        params.policy.ca_policy,
    )
    .expect("an English transcript is analyzable") else {
        panic!("a transcript without @Options: CA is analyzed");
    };
    let produced = parsed
        .collect(&options)
        .with_responses(vec![
            one_word_analysis("hello"),
            one_word_analysis("goodbye"),
        ])
        .expect("exactly the two utterances outside the held-out one were collected")
        .apply(&options)
        .await
        .expect("injection")
        .postcheck(&options)
        .expect("the analysis added no finding of its own")
        .serialize();
    let crate::pipeline::post_validate::ProducedOutput::Diagnosed(still) = &produced else {
        panic!("the invalid word is still there, so the output is still diagnosed");
    };
    assert!(
        still
            .findings()
            .errors()
            .all(|finding| finding.message.contains("b2")),
        "{still:?}"
    );
    let text = produced.as_str();
    assert!(text.contains("*PAR0:\thello .\n%mor:"), "{text}");
    assert!(text.contains("*PAR0:\twe took b2 .\n*PAR0:"), "{text}");
    assert!(text.contains("*PAR0:\tgoodbye .\n%mor:"), "{text}");
}

/// An analysis that breaks an utterance it was given is refused, as the
/// strict gate refuses an admitted document's stage output, even though the
/// document was diagnosed before the stage: its findings must stay the
/// held-out utterances' own.
#[test]
fn an_analysis_that_adds_a_finding_outside_the_held_out_utterances_is_refused() {
    use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};
    let (mut file, held_out) = transcript_diagnosed_in_its_second_utterance().into_model();
    assert!(matches!(
        crate::pipeline::post_validate::PostValidated::produced_outside(
            file.clone(),
            &held_out.in_place(),
            crate::api::ReleasedCommand::Morphotag,
        ),
        Ok(crate::pipeline::post_validate::ProducedOutput::Diagnosed(_))
    ));
    // The stage "breaks" the first utterance, which it was given.
    if let Some(Line::Utterance(first)) = (&mut file.lines)
        .into_iter()
        .find(|line| matches!(line, Line::Utterance(_)))
    {
        first.main.content.content =
            TierContentItems::new(vec![UtteranceContent::Word(Box::new(Word::simple("c3")))]);
    }
    let Err(failure) = crate::pipeline::post_validate::PostValidated::produced_outside(
        file,
        &held_out.in_place(),
        crate::api::ReleasedCommand::Morphotag,
    ) else {
        panic!("a finding outside the held-out utterances is the stage's own");
    };
    let rendered = failure.to_string();
    assert!(
        rendered.contains("c3") && !rendered.contains("b2"),
        "{rendered}"
    );
}
