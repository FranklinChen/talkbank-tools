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

/// Guards: Secondary extraction failure preserves the utterance without inventing a word position.
/// Kind: transcript
/// Not a type: Actual source annotations can fail extraction before a concrete span is established.
/// State: Admitted multilingual CHAT and a synthetic response exercise diagnosis with inert services.
/// Concurrency: The model and report are fixture-owned.
#[tokio::test]
async fn secondary_extraction_failure_reaches_the_output_report_without_guessed_positions() {
    // Keep a primary-language word: tagging every lexical word @s would be
    // invalid CHAT (a whole-turn switch requires a precode). Both words have
    // explicit synthesis, so malformed model alignment can preserve them.
    let chat = ADMISSION_CHAT.replace("hello world", "hi@i camino@s:spa");
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let mwt = MwtDict::default();
    let mut params = test_params(&mwt);
    params.tokenization_mode = TokenizationMode::Preserve;
    let options = RunOptions::new(PipelineServices::new(&pool, &cache), &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(&chat, params.policy.ca_policy, &test_registry()).unwrap()
    else {
        panic!("analysis");
    };
    let response = serde_json::from_str(
        r#"{"sentences":[{"words":[
        {"id":1,"text":"hi","lemma":"hi","upos":"INTJ","head":0,"deprel":"root"},
        {"id":2,"text":"cami","lemma":"cami","upos":"VERB","head":1,"deprel":"dep"},
        {"id":3,"text":"no","lemma":"no","upos":"ADV","head":2,"deprel":"advmod"}
    ]}]}"#,
    )
    .unwrap();
    let output = parsed
        .collect(&options)
        .expect("admitted fixture language context")
        .with_responses(vec![AdmittedMorphosyntaxResponse::for_test(
            response, "9.9.9", "eng",
        )])
        .unwrap()
        .apply(&options)
        .await
        .unwrap()
        .postcheck(&options)
        .unwrap()
        .serialize();
    let (proof, report) = output.into_parts();
    assert!(proof.as_str().contains("L2|xxx"));
    let (shortfalls, exclusions) = report.into_records(proof.document()).unwrap();
    assert!(exclusions.is_empty());
    assert!(
        matches!(&shortfalls[..], [crate::api::OutputShortfallRecord::SecondaryMorphologyUnplaced {
        utterances: 1, words: 1, first_utterances,
    }] if first_utterances[0].location.utterance() == 1 && first_utterances[0].location.main_tier().is_some() && first_utterances[0].cause == crate::api::SecondaryExtractionFailure::Alignment)
    );
    let wire = serde_json::to_value(shortfalls).unwrap();
    assert!(wire[0]["first_utterances"][0].get("first_word").is_none());
}

/// Exercises primary injection, real secondary dispatch admission, final CHAT
/// judgement and the wire report together, without loading a model.
///
/// The secondary language is admitted by the registry that admitted the
/// file, so a language it does not serve is an exclusion. "No registry" was
/// a second outcome here (a `registry_unavailable` failure), reachable only
/// by a test: a file is analyzed only once its primary language was
/// admitted, by a registry its proof now carries.
/// Guards: Unsupported secondary analysis is excluded with span evidence while primary analysis is retained.
/// Kind: policy
/// Not a type: A valid primary model response can coexist with a secondary language that is not supported.
/// State: Multilingual CHAT, a registry double and scripted primary result exercise the exclusion.
/// Concurrency: All input and output facts are fixture-owned.
#[tokio::test]
async fn an_unsupported_secondary_language_is_excluded() {
    use crate::api::OutputExclusionRecord;
    use crate::pipeline::post_validate::{OutputReport, ProducedOutput};
    let chat = ADMISSION_CHAT.replace("hello world", "hello mundo@s:spa");
    let cache = UtteranceCache::noop();
    // The pool holds no registry; the file's proof does.
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let mwt = MwtDict::default();
    let mut params = test_params(&mwt);
    params.tokenization_mode = TokenizationMode::Preserve;
    let options = RunOptions::new(PipelineServices::new(&pool, &cache), &params);
    let english_only = Ok(crate::stanza_registry::StanzaRegistry::admitting(&["eng"]).shared());
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(&chat, params.policy.ca_policy, &english_only).unwrap()
    else {
        panic!("ordinary CHAT must be analyzed");
    };
    let response = serde_json::from_str(
        r#"{"sentences":[{"words":[
        {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"},
        {"id":2,"text":"xbxxx","lemma":"xbxxx","upos":"NOUN","head":1,"deprel":"obj"},
        {"id":3,"text":".","lemma":".","upos":"PUNCT","head":1,"deprel":"punct"}
    ]}]}"#,
    )
    .unwrap();
    let output = parsed
        .collect(&options)
        .expect("admitted fixture language context")
        .with_responses(vec![AdmittedMorphosyntaxResponse::for_test(
            response, "9.9.9", "eng",
        )])
        .unwrap()
        .apply(&options)
        .await
        .unwrap()
        .postcheck(&options)
        .unwrap()
        .serialize();
    let (proof, report) = output.into_parts();
    assert!(
        proof.as_str().contains("intj|hello L2|xxx ."),
        "{}",
        proof.as_str()
    );
    assert!(
        proof
            .as_str()
            .contains("L2 morphology unsupported for spa: 1 word(s) in 1 span(s)")
    );
    assert_eq!(
        proof
            .as_str()
            .matches("[batchalign morphology availability]")
            .count(),
        1
    );
    let (shortfalls, exclusions) = report.into_records(proof.document()).unwrap();
    let produced = ProducedOutput::Admitted(proof);
    assert!(shortfalls.is_empty());
    assert!(matches!(
        OutputReport::of(&produced, &shortfalls),
        OutputReport::Clean
    ));
    assert!(
        matches!(&exclusions[..], [OutputExclusionRecord::UnsupportedSecondaryMorphology { spans: 1, words: 1, first_spans }]
        if first_spans[0].location.utterance() == 1 && first_spans[0].location.main_tier().is_some() && first_spans[0].first_word == 2)
    );
}

/// Guards: A contradicted regeneration plan is classified as a system failure and keeps the cause.
/// Kind: policy
/// Not a type: Invalid user input and a system plan contradiction call for different operator actions.
/// State: Typed refusals exercise classification and rendering directly.
/// Concurrency: No worker, cache or filesystem starts.
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

/// Guards: Regeneration admits damaged replaced tiers and still refuses defects in retained source regions.
/// Kind: transcript
/// Not a type: Real parse findings can land on opposite sides of the admitted replacement boundary.
/// State: Representative malformed CHAT sources enter the actual replacement and retained edges.
/// Concurrency: Each source model is fixture-owned.
#[test]
fn regeneration_admission_distinguishes_replaced_and_retained_faults() {
    use crate::options::CaMorphotagPolicy::{Analyze, Honor};
    let broken_mor = ADMISSION_CHAT.replace("@End", "%mor:\tnoun|hello\n@End");
    assert!(matches!(
        ParsedFile::parse(&broken_mor, Honor, &test_registry()),
        Ok(ParsedFile::Analyze(_))
    ));
    let ca_broken_mor = broken_mor.replace("@ID:", "@Options:\tCA\n@ID:");
    assert!(matches!(
        ParsedFile::parse(&ca_broken_mor, Honor, &test_registry()),
        Err(crate::error::ServerError::ChatReplacementAdmission(_))
    ));
    assert!(matches!(
        ParsedFile::parse(&ca_broken_mor, Analyze, &test_registry()),
        Ok(ParsedFile::Analyze(_))
    ));

    // Word timing is a sidecar: count drift is legal, unlike phonology alignment.
    let drifted_wor = ADMISSION_CHAT.replace("@End", "%wor:\thello .\n@End");
    for policy in [Honor, Analyze] {
        assert!(matches!(
            ParsedFile::parse(&drifted_wor, policy, &test_registry()),
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
                    ParsedFile::parse(&input, policy, &test_registry()),
                    Err(crate::error::ServerError::ChatReplacementAdmission(_))
                ),
                "retained fault must refuse before collection: {input}"
            );
        }
    }
}

/// Guards: CA decline preserves contributor comments and drops legacy decision tiers.
/// Kind: transcript
/// Not a type: Actual CA input can combine authored comments and obsolete generated annotations.
/// State: One CA source, inert services and an exact output snapshot.
/// Concurrency: The command returns before output inspection.
#[tokio::test]
async fn admitted_ca_pass_through_preserves_ordinary_comment_tiers() {
    let chat = ADMISSION_CHAT
        .replace("@ID:", "@Options:\tCA\n@ID:")
        .replace(
            "@End",
            "%com:\tordinary contributor content\n%xalign:\tlegacy\n%xrev:\t[?]\n@End",
        );
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);
    let mwt = MwtDict::default();
    let output = run_morphosyntax_pipeline(&chat, services, &test_params(&mwt))
        .await
        .expect("valid CA requires no worker")
        .into_text();
    assert!(output.contains("%com:\tordinary contributor content"));
    for removed in ["%mor:", "%xalign:", "%xrev:"] {
        assert!(!output.contains(removed), "{output}");
    }
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

/// Guards: An invalid prior is discarded and analysis uses only the admitted current source.
/// Kind: transcript
/// Not a type: Saved prior annotations can be invalid even when the current version admits.
/// State: Before and after CHAT and an inert cache/pool exercise the real incremental command.
/// Concurrency: The fixture owns both source versions.
#[tokio::test]
async fn incremental_morphology_discards_invalid_prior_and_analyzes_current() {
    let before = ADMISSION_CHAT.replace("@End", "%mor:\tnoun|hello\n@End");
    let cache = UtteranceCache::noop();
    let pool = registry_pool();
    let services = PipelineServices::new(&pool, &cache);
    let mwt = MwtDict::default();
    // Fresh analysis of this valid nonlexical current file needs no worker.
    // The invalid prior must contribute neither its words nor its partial MOR.
    let current = ADMISSION_CHAT.replace("hello world", "xxx");
    let result = crate::morphosyntax::process_morphosyntax_incremental(
        &before,
        &current,
        services,
        &test_params(&mwt),
    )
    .await
    .expect("invalid reuse input must not reject valid current CHAT")
    .into_text();
    assert!(result.contains("*CHI:\txxx ."));
    assert!(!result.contains("hello"));
    assert!(!result.contains("noun|"));

    // Fallback does not bypass admission of the current document.
    let invalid_current = current.replace("@Languages:\teng\n", "");
    let refusal = crate::morphosyntax::process_morphosyntax_incremental(
        &before,
        &invalid_current,
        PipelineServices::new(&pool, &cache),
        &test_params(&mwt),
    )
    .await;
    assert!(matches!(
        refusal,
        Err(crate::error::ServerError::ChatReplacementAdmission(_))
    ));
}

fn parse(text: &str) -> talkbank_model::model::ChatFile {
    let parser = crate::chat_parser();
    let (chat_file, _) = parse_lenient(&parser, text);
    chat_file
}

/// The registry a worker reporting these tests' languages would produce:
/// English, Spanish and Czech with every morphotag processor, and nothing
/// else (`que` is the language no catalog serves).
fn test_catalog() -> crate::stanza_registry::StanzaRegistry {
    crate::stanza_registry::StanzaRegistry::admitting(&["eng", "spa", "ces"])
}

/// [`test_catalog`] as an admitted pool's lookup finds it.
fn test_registry() -> crate::stanza_registry::StanzaRegistryLookup {
    Ok(test_catalog().shared())
}

/// A pool whose admitted registry is [`test_catalog`]; it spawns no worker.
fn registry_pool() -> WorkerPool {
    WorkerPool::with_test_stanza_registry(test_catalog())
}

fn primary_capability(
    chat: &talkbank_model::model::ChatFile,
) -> Result<(), crate::error::ServerError> {
    let language = super::resolve_per_file_lang(chat)?;
    crate::morphosyntax::AnalysisUnavailable::admit(
        crate::morphosyntax::AnalysisLanguageRequirement::PrimaryHeader,
        &language,
        &test_registry(),
    )?;
    Ok(())
}

/// Guards: Unchanged input copies only the admitted prior morphology and grammar pair.
/// Kind: transcript
/// Not a type: Two revisions can disagree in generated tiers while retaining the same analysis content.
/// State: A reusable prior pair, stale current tiers and exact output snapshot.
/// Concurrency: No worker is started during reuse.
#[tokio::test]
async fn incremental_morphology_copies_only_admitted_prior_analysis_without_inference() {
    let before = ADMISSION_CHAT.replace(
        "@End",
        "%mor:\tintj|hello noun|world .\n\
%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n@End",
    );
    let before = before.replace(
        "*CHI:",
        "@Comment:\t[ba3 morphotag | engine=stanza-1.10.0 | 2026-03-28T10:00:00-04:00]\n*CHI:",
    );
    let after = ADMISSION_CHAT.replace("@End", "%mor:\tnoun|wrong\n@End");
    let cache = UtteranceCache::noop();
    let pool = registry_pool();
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
    @Comment:\t[ba3 morphotag | engine=stanza-1.10.0 | 2026-03-28T10:00:00-04:00]
    *CHI:\thello world .
    %mor:\tintj|hello noun|world .
    %gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT
    @End
    ");
}

/// Guards: Reused morphology retains its producer stamp or explicitly reports missing provenance.
/// Kind: transcript
/// Not a type: A prior transcript can have missing or malformed metadata independently of reusable morphology.
/// State: Prior and current revision matrices and an isolated registry double.
/// Concurrency: Every transformation owns its source models.
#[tokio::test]
async fn reused_morphology_carries_its_own_stamps_without_requiring_readable_metadata() {
    let cache = UtteranceCache::noop();
    let pool = registry_pool();
    let morphology = "%mor:\tintj|hello noun|world .\n%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT\n";
    let other_stamp = "[ba3 transcribe | asr=whisper | 2026-03-28T10:00:00-04:00]";
    for prior_stamp in [
        Some("[fc-ba3 morphotag | engine=stanza-1.10.0 ; lang=eng | 2026-03-28T10:00:00-04:00]"),
        Some("[ba3 morphotag | engine=stanza-1.10.0 ; lang=eng | 2026-03-28T10:00:00-04:00]"),
        Some(
            "[fc-ba3 morphotag | engine=stanza-1.10.0 ; ud_repairs=damaged | 2026-03-28T10:00:00-04:00]",
        ),
        None,
    ] {
        let before = ADMISSION_CHAT
            .replace(
                "*CHI:",
                &format!(
                    "@Comment:\tprior contributor note\n{}*CHI:",
                    prior_stamp.map_or(String::new(), |stamp| format!("@Comment:\t{stamp}\n"))
                ),
            )
            .replace("@End", &format!("{morphology}@End"));
        for current_stamp in [
            "",
            "@Comment:\t[fc-ba3 morphotag | engine=wrong-current-model | old]\n",
        ] {
            let after = ADMISSION_CHAT.replace("*CHI:", &format!(
                "@Comment:\tcurrent contributor note\n@Comment:\t{other_stamp}\n{current_stamp}*CHI:"));
            let output = crate::morphosyntax::process_morphosyntax_incremental(
                &before,
                &after,
                PipelineServices::new(&pool, &cache),
                &test_params(&MwtDict::default()),
            )
            .await
            .expect("optional provenance cannot make reused morphology fail");
            let (document, report) = output.into_parts();
            let (shortfalls, exclusions) = report.into_records(document.document()).unwrap();
            let incomplete = prior_stamp.is_none_or(|stamp| stamp.contains("damaged"));
            assert_eq!(shortfalls.len(), usize::from(incomplete));
            assert!(exclusions.is_empty());
            let output = document.into_text();
            assert_eq!(output.contains("provenance_incomplete=true"), incomplete);
            assert!(output.contains(morphology), "{output}");
            assert!(output.contains("current contributor note"), "{output}");
            assert!(output.contains(other_stamp), "{output}");
            assert!(!output.contains("prior contributor note"), "{output}");
            assert!(!output.contains("wrong-current-model"), "{output}");
            match prior_stamp {
                Some(stamp) => assert!(output.contains(stamp), "{output}"),
                None => assert!(!output.contains("engine="), "{output}"),
            }
        }
    }
}

/// Guards: Unsupported primary language keeps the catalog gap and actionable advice.
/// Kind: policy
/// Not a type: An actual registry may lack an advertised language and advice must preserve truthful CHAT declarations.
/// State: A Quechua source exercises a limited registry and diagnostic renderer.
/// Concurrency: Pure lookup and rendering over local fixtures.
#[test]
fn unsupported_primary_language_returns_actionable_error_message() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\tque\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\tque|test|CHI||female|||Target_Child|||\n\
                *CHI:\twasi .\n\
                @End\n";
    let chat_file = parse(chat);
    let error = primary_capability(&chat_file)
        .expect_err("a language the registry does not serve must refuse analysis");
    let crate::error::ServerError::AnalysisUnavailable(ref unavailable) = error else {
        panic!("expected typed analysis unavailability, got {error}");
    };
    let crate::stanza_registry::StanzaRefusal::Gap(gap) = unavailable.reason() else {
        panic!("a served registry refuses with a gap");
    };
    assert_eq!(gap.kind(), &crate::stanza_registry::GapKind::NotInCatalog);
    let msg = error.to_string();
    assert!(
        msg.contains("que"),
        "error must name the unsupported lang: {msg}"
    );
    assert!(
        msg.contains("Stanza catalog has no model"),
        "error must call out the missing Stanza model so the operator sees \
         the cause in the dashboard: {msg}"
    );
    assert!(
        msg.contains("Keep truthful language declarations") && !msg.contains("Fix the @Languages"),
        "model availability must never advise falsifying CHAT: {msg}"
    );
}

/// Guards: A supported language alone cannot admit a transcript lacking required headers.
/// Kind: transcript
/// Not a type: Actual source bytes may fail structural checks even when a matching registry serves that language.
/// State: A positive English registry control and missing-header CHAT exercise distinct edges.
/// Concurrency: One immutable registry double is shared locally.
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
    assert!(
        crate::morphosyntax::AnalysisUnavailable::admit(
            crate::morphosyntax::AnalysisLanguageRequirement::PrimaryHeader,
            &english,
            &test_registry(),
        )
        .is_ok()
    );
    assert!(matches!(
        ParsedFile::parse(
            chat,
            crate::options::CaMorphotagPolicy::Honor,
            &test_registry()
        ),
        Err(crate::error::ServerError::ChatReplacementAdmission(_))
    ));
}

/// Guards: Single and bilingual headers select the primary language and missing headers fail.
/// Kind: transcript
/// Not a type: Source bytes can name unexpected primary languages or omit the declaration altogether.
/// State: Czech, Spanish-primary bilingual and absent header cases enter shared resolution.
/// Concurrency: A local parser owns each source observation.
#[test]
fn resolve_per_file_lang_uses_primary_languages_header() {
    for (languages, primary) in [("ces", "ces"), ("spa, eng", "spa")] {
        let source = ADMISSION_CHAT
            .replace("@Languages:\teng", &format!("@Languages:\t{languages}"))
            .replace("@ID:\teng|", &format!("@ID:\t{primary}|"));
        let resolved = super::resolve_per_file_lang(&parse(&source)).unwrap();
        assert_eq!(resolved.as_ref(), primary);
    }
    let missing = ADMISSION_CHAT.replace("@Languages:\teng\n", "");
    let refusal = super::resolve_per_file_lang(&parse(&missing)).unwrap_err();
    assert!(refusal.to_string().contains("`@Languages:`"));
}

/// Guards: English primary morphology admits with or without an unsupported secondary declaration.
/// Kind: policy
/// Not a type: A real header can name a served primary and an unsupported secondary without requesting it.
/// State: Two English source variants enter the real registry check.
/// Concurrency: Only immutable local fixtures are used.
#[test]
fn supported_language_with_unsupported_secondary_passes() {
    for languages in ["eng", "eng, srp"] {
        let chat = ADMISSION_CHAT.replace("@Languages:\teng", &format!("@Languages:\t{languages}"));
        assert!(primary_capability(&parse(&chat)).is_ok(), "{languages}");
    }
}

/// Guards: An unsupported primary returns typed analysis unavailability before output.
/// Kind: contract
/// Not a type: A transcript's admitted language can lack an engine at actual command entry.
/// State: A Quechua CHAT fixture and a registry-backed inert pool reach the command path.
/// Concurrency: One source and effect environment belong to this test.
#[tokio::test]
async fn unsupported_primary_language_returns_typed_analysis_unavailable() {
    let chat = "@UTF8\n\
                @PID:\t11312/c-test\n\
                @Begin\n\
                @Languages:\tque\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\tque|test|CHI||female|||Target_Child|||\n\
                *CHI:\twasi .\n\
                @End\n";
    let cache = UtteranceCache::noop();
    let pool = registry_pool();
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
        msg.contains("que"),
        "error must name the unsupported lang: {msg}"
    );
    assert!(
        msg.contains("Stanza catalog has no model"),
        "error must call out the missing Stanza model: {msg}"
    );
}

/// Guards: Unsupported effective precode stops analysis without partial output and names the gap.
/// Kind: contract
/// Not a type: An available primary can coexist with an unsupported language requested by actual utterance syntax.
/// State: Code-switched CHAT with inert services and a limited registry.
/// Concurrency: One source owns all command effects.
#[tokio::test]
async fn effective_precode_capability_refuses_without_inference_or_partial_output() {
    let cache = UtteranceCache::noop();
    let pool = registry_pool();
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

/// Guards: Complete prior analysis reuses unsupported precode while fresh analysis still requires capability.
/// Kind: transcript
/// Not a type: Previously valid tiers may survive unchanged even when the current registry cannot serve their language.
/// State: Prior/current CHAT and a deliberately limited registry exercise both routes.
/// Concurrency: Input model pairs are independently owned.
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
    let cache = UtteranceCache::noop();
    let pool = registry_pool();
    let output = crate::morphosyntax::process_morphosyntax_incremental(
        &before,
        &chat,
        PipelineServices::new(&pool, &cache),
        &test_params(&MwtDict::default()),
    )
    .await
    .expect("source-admitted compatible complete analysis needs no model");
    let (document, report) = output.into_parts();
    assert!(matches!(
        &report.into_records(document.document()).unwrap().0[..],
        [crate::api::OutputShortfallRecord::MorphologyProvenanceIncomplete { .. }]
    ));
    let output = document.into_text();
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

/// Guards: No primary capability, nonlexical input and CA decline produce distinct verdicts.
/// Kind: policy
/// Not a type: A catalog gap, actual nonlexical words and a CA option are independently observable.
/// State: Three source shapes enter the real parse and policy boundary.
/// Concurrency: One local immutable catalog supplies observations.
#[test]
fn unsupported_primary_nonlexical_policy_and_ca_disposition_remain_distinct() {
    let chat = ADMISSION_CHAT
        .replace("eng", "que")
        .replace("hello world", "xxx");
    let error = ParsedFile::parse(
        &chat,
        crate::options::CaMorphotagPolicy::Honor,
        &test_registry(),
    )
    .err()
    .expect("primary support policy also applies to nonlexical files");
    assert!(matches!(
        error,
        crate::error::ServerError::AnalysisUnavailable(_)
    ));
    let ca = chat.replace("@ID:", "@Options:\tCA\n@ID:");
    assert!(matches!(
        ParsedFile::parse(
            &ca,
            crate::options::CaMorphotagPolicy::Honor,
            &test_registry()
        ),
        Ok(ParsedFile::PassThrough(_))
    ));
    assert!(matches!(
        ParsedFile::parse(
            &ca,
            crate::options::CaMorphotagPolicy::Analyze,
            &test_registry()
        ),
        Err(crate::error::ServerError::AnalysisUnavailable(_))
    ));
}

/// Guards: `NoAlign` still receives morphology and its actual provenance.
/// Kind: transcript
/// Not a type: Existing `NoAlign` source syntax can bypass alignment while still qualifying for morphology.
/// State: `NoAlign` CHAT and synthetic worker data exercise output and stamp.
/// Concurrency: All resources are fixture-owned and inert.
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
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy, &test_registry()).expect("parse")
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
        .expect("admitted fixture language context")
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
/// Guards: Morphology provenance names the worker's reported model and language.
/// Kind: wire
/// Not a type: Actual worker model identifiers can differ from configured engine aliases or version names.
/// State: A synthetic provider response drives output and its stamp.
/// Concurrency: The test uses inert services and local evidence.
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
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy, &test_registry()).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed
        .collect(&options)
        .expect("admitted fixture language context");
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
/// Guards: Too few or too many worker replies refuse morphology as protocol failure.
/// Kind: wire
/// Not a type: A worker can return cardinality that disagrees with the exact admitted item population.
/// State: A checked CHAT fixture and zero or two typed responses exercise admission.
/// Concurrency: Each case owns its prepared population.
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
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(PipelineServices::new(&pool, &cache), &params);
    for count in [0, 2] {
        let ParsedFile::Analyze(parsed) =
            ParsedFile::parse(ADMISSION_CHAT, params.policy.ca_policy, &test_registry())
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
        let error = match parsed
            .collect(&options)
            .expect("admitted fixture language context")
            .with_responses(responses)
        {
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

/// Correct batch counts still cannot certify lexical completion: an
/// utterance the model's analysis does not cover is declined, left without
/// `%mor`, and named as a shortfall of the written file. One utterance never
/// fails the file (it did until 2026-11-12).
/// Guards: Incomplete worker analysis declines the affected utterance with the exact shortfall cause.
/// Kind: transcript
/// Not a type: An otherwise valid response may be empty or fail realignment after request admission.
/// State: Scripted result cases exercise full and incremental completion and report text.
/// Concurrency: Each case owns its analysis observations.
#[tokio::test]
async fn incomplete_model_analysis_is_declined_and_reported_not_certified() {
    let chat = "@UTF8\n@Begin\n@Languages:\teng\n\
                @Participants:\tCHI Target_Child\n\
                @ID:\teng|test|CHI|||||Target_Child|||\n\
                *CHI:\thello good world .\n@End\n";
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);
    let mwt = MwtDict::default();
    use crate::api::PrimaryMorphologyDeclineCause as Cause;
    /// What the worker answers for the one utterance.
    #[derive(Clone, Copy)]
    enum Answer {
        Wire(&'static str),
        Failed(&'static str),
    }
    use Answer::{Failed, Wire};
    let cases = [
        (
            "no sentences",
            Wire(r#"{"sentences":[]}"#),
            Cause::NoAnalysis,
        ),
        // A worker failure for this utterance alone declines it with the
        // worker's reason; it used to refuse the whole file.
        (
            "worker failed",
            Failed("dependency projection failed"),
            Cause::WorkerFailed,
        ),
        (
            "missing words",
            Wire(
                r#"{"sentences":[{"words":[
                {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"}
            ]}]}"#,
            ),
            Cause::Misaligned,
        ),
        (
            "malformed dependency",
            Wire(
                r#"{"sentences":[{"words":[
                {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":9,"deprel":"dep"},
                {"id":2,"text":"good","lemma":"good","upos":"ADJ","head":3,"deprel":"amod"},
                {"id":3,"text":"world","lemma":"world","upos":"NOUN","head":0,"deprel":"root"}
            ]}]}"#,
            ),
            Cause::MappingFailed,
        ),
    ];
    // Retokenization can intentionally change the number of main-tier words,
    // so lexical cardinality is tested in Preserve mode, not by imposing a
    // Preserve invariant on that distinct policy.
    for mode in [
        TokenizationMode::Preserve,
        TokenizationMode::StanzaRetokenize,
    ] {
        for (label, answer, cause) in cases {
            if label == "missing words" && mode == TokenizationMode::StanzaRetokenize {
                continue;
            }
            let mut params = test_params(&mwt);
            params.tokenization_mode = mode;
            let options = RunOptions::new(services, &params);
            let ParsedFile::Analyze(parsed) =
                ParsedFile::parse(chat, params.policy.ca_policy, &test_registry()).expect("parse")
            else {
                panic!("lexical CHAT requires analysis");
            };
            let admitted = match answer {
                Wire(wire) => AdmittedMorphosyntaxResponse::for_test(
                    serde_json::from_str(wire).expect("worker wire fixture"),
                    "test",
                    "eng",
                ),
                Failed(reason) => AdmittedMorphosyntaxResponse::worker_failed(reason.into()),
            };
            let inferred = parsed
                .collect(&options)
                .expect("admitted fixture language context")
                .with_responses(vec![admitted])
                .expect("one response for one utterance");
            let finished = inferred
                .apply(&options)
                .await
                .unwrap_or_else(|error| panic!("{label} in {mode:?} failed the file: {error}"))
                .postcheck(&options)
                .expect("a declined utterance is valid CHAT")
                .serialize()
                .finish()
                .expect("the report locates its utterance");
            let [
                crate::api::OutputShortfallRecord::PrimaryMorphologyIncomplete {
                    utterances: 1,
                    first_utterances,
                },
            ] = finished.shortfalls.as_slice()
            else {
                panic!("{label} in {mode:?}: {:?}", finished.shortfalls);
            };
            assert_eq!(first_utterances[0].cause, cause, "{label} in {mode:?}");
            let utterance = finished
                .document
                .document()
                .utterances()
                .next()
                .expect("utterance");
            assert!(
                utterance.mor_tier().is_none(),
                "{label}: no %mor is written"
            );
            // The same decline is marked in the CHAT, on that utterance.
            let notes: Vec<String> = utterance
                .dependent_tiers
                .iter()
                .filter(|entry| entry.kind() == batchalign_transform::notes::NOTE_TIER)
                .map(ToString::to_string)
                .collect();
            assert_eq!(notes.len(), 1, "{label}: one note: {notes:?}");
            assert!(
                notes[0].starts_with("%xbatchalign:\tmorphotag: not analysed; ")
                    && notes[0].contains(cause.describe()),
                "{label}: {notes:?}"
            );
        }
    }
}

/// Guards: Explicit CA analysis clears stale morphology and installs the observed response.
/// Kind: transcript
/// Not a type: CA source can contain old generated analysis even when policy authorizes fresh tagging.
/// State: Admitted CA source with stale tiers and a synthetic worker result.
/// Concurrency: The model and proof are fixture-owned.
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
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let mut params = test_params(&mwt);
    params.policy.ca_policy = crate::options::CaMorphotagPolicy::Analyze;
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy, &test_registry()).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed
        .collect(&options)
        .expect("admitted fixture language context");
    let ParsedFile::Analyze(reparsed) =
        ParsedFile::parse(chat, params.policy.ca_policy, &test_registry())
            .expect("parse again for missing-response boundary")
    else {
        panic!("expected analysis");
    };
    let missing_response = reparsed
        .collect(&options)
        .expect("admitted fixture language context")
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

/// Guards: POS evidence reaches retokenizing analysis and applies only to the intended word.
/// Kind: transcript
/// Not a type: Source hint positions can diverge from actual worker positions after retokenization.
/// State: A contraction with a POS hint and synthetic response enter real retokenization.
/// Concurrency: All model observations are local.
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
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let services = PipelineServices::new(&pool, &cache);

    let mwt = MwtDict::default();
    let mut params = test_params(&mwt);
    params.policy.pos_hints = crate::params::PosHintPolicy::Honor;
    let options = RunOptions::new(services, &params);
    let ParsedFile::Analyze(parsed) =
        ParsedFile::parse(chat, params.policy.ca_policy, &test_registry()).expect("parse stage")
    else {
        panic!("expected analysis");
    };
    let collected = parsed
        .collect(&options)
        .expect("admitted fixture language context");
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

/// Guards: POS hints select the right admitted morphology positions around repeated words and punctuation.
/// Kind: transcript
/// Not a type: Actual punctuation and repeated words can move intended hints relative to worker ordinals.
/// State: A table of source utterances and typed analysis replies exercises the real projection.
/// Concurrency: Each control owns its morphology response.
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
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
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
                let ParsedFile::Analyze(parsed) =
                    ParsedFile::parse(&chat, params.policy.ca_policy, &test_registry())
                        .expect("fully admitted source")
                else {
                    panic!("expected analysis");
                };
                let collected = parsed
                    .collect(&options)
                    .expect("admitted fixture language context");
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
            .map(|word| {
                UtteranceContent::Word(Box::new(Word::new(
                    talkbank_model::Location::Unlocated,
                    talkbank_model::WordText::new(word).expect("nonempty fixture word"),
                )))
            })
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
/// Guards: Localized analysis tags sound utterances around a held-out faulty utterance.
/// Kind: transcript
/// Not a type: A diagnosed document may have unaffected neighboring utterances that still require actual work.
/// State: A source with one invalid word, neighboring utterances and scripted worker data.
/// Concurrency: No live worker or shared cache is started.
#[tokio::test]
async fn a_diagnosed_transcript_is_tagged_outside_its_faulty_utterance() {
    use super::states::LocalizedParse;
    let cache = UtteranceCache::noop();
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let mwt = MwtDict::default();
    let params = test_params(&mwt);
    let options = RunOptions::new(PipelineServices::new(&pool, &cache), &params);
    let LocalizedParse::Analyze(parsed) = ParsedFile::from_localized(
        transcript_diagnosed_in_its_second_utterance(),
        params.policy.ca_policy,
        &test_registry(),
    )
    .expect("an English transcript is analyzable") else {
        panic!("a transcript without @Options: CA is analyzed");
    };
    let produced = parsed
        .collect(&options)
        .expect("admitted fixture language context")
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
        .serialize()
        .without_reports()
        .expect("fixture has no secondary work");
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
/// Guards: Localized analysis refuses a new finding outside the held-out original region.
/// Kind: transcript
/// Not a type: A real output edit can add a defect outside the source's original diagnosis.
/// State: A bound diagnosis and deliberately changed neighbor enter actual output judgment.
/// Concurrency: All document and finding effects stay in memory.
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
            TierContentItems::new(vec![UtteranceContent::Word(Box::new(Word::new(
                talkbank_model::Location::Unlocated,
                talkbank_model::WordText::new("c3").expect("nonempty fixture word"),
            )))]);
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
