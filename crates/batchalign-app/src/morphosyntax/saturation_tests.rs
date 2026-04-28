//! End-to-end corruption-regression tests for `run_morphosyntax_batch_impl`.
//!
//! Drives the orchestrator with a `FakeDispatcher` that returns canned
//! per-language Ok/Err, so failure routing can be verified without
//! spawning Python workers. The unit tests in `outcomes.rs` cover the
//! pure aggregator and per-file classifier; this module proves the
//! orchestrator wires them together correctly and never ships a CHAT
//! document with stripped `%mor`/`%gra` tiers when a language group
//! fails.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use batchalign_chat_ops::morphosyntax::{
    BatchItemWithPosition, MultilingualPolicy, MwtDict, TokenizationMode,
};
use batchalign_chat_ops::nlp::UdResponse;

use crate::api::{EngineVersion, LanguageCode3};
use crate::cache::UtteranceCache;
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::text_batch::{TextBatchFileInput, TextBatchFileResults};
use crate::types::worker_v2::ProgressEventV2;
use crate::worker::pool::{PoolConfig, WorkerPool};

use super::dispatcher::LanguageGroupDispatcher;
use super::run_morphosyntax_batch_impl;

// ── Fake dispatcher ──────────────────────────────────────────────────

/// Returns a configured failure message per language as a
/// `ServerError::Validation`, or an Ok-with-empty responses fallback.
/// Empty UD is enough to drive the orchestrator's routing paths; the
/// NLP content is irrelevant to these tests.
///
/// Message stored as `String` rather than `ServerError` because the
/// latter does not implement `Clone` (some variants wrap non-clone
/// error sources like `sqlx::Error`). `ServerError::Validation` is
/// built per-call from the stored message.
struct FakeDispatcher {
    failures: HashMap<LanguageCode3, String>,
}

impl FakeDispatcher {
    fn new() -> Self {
        Self {
            failures: HashMap::new(),
        }
    }

    fn fail(mut self, lang: LanguageCode3, message: impl Into<String>) -> Self {
        self.failures.insert(lang, message.into());
        self
    }
}

impl LanguageGroupDispatcher for FakeDispatcher {
    fn dispatch<'a>(
        &'a self,
        lang: &'a LanguageCode3,
        items: &'a [BatchItemWithPosition],
        _mwt: &'a MwtDict,
        _retokenize: bool,
        _progress_tx: Option<&'a tokio::sync::mpsc::Sender<ProgressEventV2>>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<UdResponse>, ServerError>> + Send + 'a>> {
        let result: Result<Vec<UdResponse>, ServerError> = match self.failures.get(lang) {
            Some(message) => Err(ServerError::Validation(message.clone())),
            None => Ok((0..items.len())
                .map(|_| UdResponse {
                    sentences: Vec::new(),
                })
                .collect()),
        };
        Box::pin(async move { result })
    }
}

// ── Harness ──────────────────────────────────────────────────────────

/// Default pool + noop-cache + engine-version triple for tests. The
/// pool spawns no Python workers; its presence satisfies capacity
/// queries (`stanza_registry`, `max_workers_per_key`). The noop cache
/// exists only to satisfy `PipelineServices::new`'s signature — text
/// tasks don't cache and audio tasks aren't exercised here.
fn make_services_fixtures() -> (WorkerPool, UtteranceCache, EngineVersion) {
    (
        WorkerPool::new(PoolConfig::default()),
        UtteranceCache::noop(),
        EngineVersion::from("test-morphotag"),
    )
}

fn default_params<'a>(lang: &'a LanguageCode3, mwt: &'a MwtDict) -> MorphosyntaxParams<'a> {
    MorphosyntaxParams {
        lang,
        tokenization_mode: TokenizationMode::Preserve,
        multilingual_policy: MultilingualPolicy::ProcessAll,
        mwt,
        l2_morphotag: false,
        respect_pos_hints: false,
    }
}

async fn run_batch(
    files: &[TextBatchFileInput],
    dispatcher: &dyn LanguageGroupDispatcher,
) -> TextBatchFileResults {
    let (pool, cache, engine_version) = make_services_fixtures();
    let services = PipelineServices::new(&pool, &cache, &engine_version);
    let primary_lang = LanguageCode3::eng();
    let mwt: MwtDict = Default::default();
    let params = default_params(&primary_lang, &mwt);
    run_morphosyntax_batch_impl(
        files,
        services,
        dispatcher,
        &params,
        None,
        std::time::Duration::from_secs(5),
    )
    .await
}

/// Multi-language CHAT with pre-populated `%mor`/`%gra` on every
/// utterance. If the orchestrator silently soft-failed the `deu` group
/// the serialization would have fewer `%mor` lines than the input —
/// the corruption signature the integration tests guard against.
fn multi_language_chat_with_populated_mor() -> String {
    "\
@UTF8
@Begin
@Languages:\teng, deu
@Participants:\tCHI Child
@ID:\teng|test|CHI|||||Child|||
*CHI:\tI see a cat .
%mor:\tpro:sub|I v|see det|a n|cat .
%gra:\t1|2|SUBJ 2|0|ROOT 3|4|DET 4|2|OBJ 5|2|PUNCT
*CHI:\t[- deu] ich sehe eine Katze .
%mor:\tpro:per|ich v|sehen det|eine n|Katze .
%gra:\t1|2|SUBJ 2|0|ROOT 3|4|DET 4|2|OBJ 5|2|PUNCT
@End
"
    .to_string()
}

fn count_mor_tiers(chat: &str) -> usize {
    chat.lines().filter(|l| l.starts_with("%mor:")).count()
}

fn saturation_message(lang: &str) -> String {
    format!(
        "worker error: global worker cap reached and no workers exist for \
         Profile(Stanza)/{lang} — cannot wait (would deadlock)"
    )
}

// ── Tests ────────────────────────────────────────────────────────────

/// Happy path: no failures configured → every file returns Ok. Guards
/// against an Ok→Err routing regression passing off as a corruption fix.
#[tokio::test]
async fn all_languages_succeed_returns_ok_result_for_every_file() {
    let dispatcher = FakeDispatcher::new();
    let files = vec![TextBatchFileInput::new(
        "multi_lang.cha",
        multi_language_chat_with_populated_mor(),
    )];

    let results = run_batch(&files, &dispatcher).await;

    assert_eq!(results.len(), 1);
    assert!(
        results[0].result.is_ok(),
        "all groups succeeded, file must be Ok: got {:?}",
        results[0].result
    );
}

/// Corruption-regression guard: a failed language group must produce
/// per-file errors, not a serialized CHAT with stripped tiers.
/// RED-verified by reverting the orchestrator's skip branch to always
/// inject — the test then gets an Ok CHAT output with every `%mor`
/// and `%gra` tier stripped, exactly reproducing the silent-corruption
/// pattern the audit tooling had to catch after the fact.
#[tokio::test]
async fn failed_language_group_produces_per_file_error_not_stripped_chat() {
    let dispatcher = FakeDispatcher::new().fail(LanguageCode3::deu(), saturation_message("deu"));

    let input_chat = multi_language_chat_with_populated_mor();
    let input_mor_count = count_mor_tiers(&input_chat);
    assert_eq!(
        input_mor_count, 2,
        "fixture must start with populated %mor on both utterances"
    );

    let files = vec![TextBatchFileInput::new("multi_lang.cha", input_chat)];

    let results = run_batch(&files, &dispatcher).await;

    assert_eq!(results.len(), 1);
    let result = &results[0].result;
    assert!(
        result.is_err(),
        "file with failed deu utterances must be marked err, not serialized \
         with stripped tiers. Got: {result:?}",
    );
    let err_msg = result.as_ref().unwrap_err().to_string();
    assert!(
        err_msg.contains("deu"),
        "per-file error should name the failed language; got: {err_msg}"
    );

    if let Ok(output) = result {
        let output_mor_count = count_mor_tiers(output);
        panic!(
            "orchestrator produced an Ok CHAT output when deu failed; input \
             %mor count = {input_mor_count}, output = {output_mor_count} — \
             this is the silent-corruption regression."
        );
    }
}

/// Per-file discrimination: a failure affecting only some files must
/// not cascade. English-only file still injects; multi-language file
/// with deu utterances is marked err.
#[tokio::test]
async fn successful_files_inject_while_affected_files_are_marked_err() {
    let dispatcher = FakeDispatcher::new().fail(LanguageCode3::deu(), saturation_message("deu"));

    let eng_only_chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Child
@ID:\teng|test|CHI|||||Child|||
*CHI:\tI see a cat .
%mor:\tpro:sub|I v|see det|a n|cat .
%gra:\t1|2|SUBJ 2|0|ROOT 3|4|DET 4|2|OBJ 5|2|PUNCT
@End
"
    .to_string();

    let files = vec![
        TextBatchFileInput::new("eng_only.cha", eng_only_chat),
        TextBatchFileInput::new("multi_lang.cha", multi_language_chat_with_populated_mor()),
    ];

    let results = run_batch(&files, &dispatcher).await;

    assert_eq!(results.len(), 2);
    let eng_result = results
        .iter()
        .find(|r| r.filename.as_ref() == "eng_only.cha")
        .expect("eng_only.cha in results");
    assert!(
        eng_result.result.is_ok(),
        "eng-only file's language group succeeded — must inject. Got: {:?}",
        eng_result.result
    );

    let multi_result = results
        .iter()
        .find(|r| r.filename.as_ref() == "multi_lang.cha")
        .expect("multi_lang.cha in results");
    assert!(
        multi_result.result.is_err(),
        "multi-lang file has deu utterances — deu failed, file must be err. Got: {:?}",
        multi_result.result
    );
}

/// Failure isolation must not be special-cased to the primary language. When
/// one secondary-language group fails, unrelated successful groups in other
/// files must still inject normally.
#[tokio::test]
async fn failed_language_group_does_not_poison_other_successful_non_primary_files() {
    let dispatcher = FakeDispatcher::new().fail(LanguageCode3::deu(), saturation_message("deu"));

    let eng_only_chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Child
@ID:\teng|test|CHI|||||Child|||
*CHI:\tI see a cat .
%mor:\tpro:sub|I v|see det|a n|cat .
%gra:\t1|2|SUBJ 2|0|ROOT 3|4|DET 4|2|OBJ 5|2|PUNCT
@End
"
    .to_string();

    let spa_only_chat = "\
@UTF8
@Begin
@Languages:\tspa
@Participants:\tCHI Child
@ID:\tspa|test|CHI|||||Child|||
*CHI:\thola mundo .
%mor:\tintj|hola n|mundo .
%gra:\t1|0|ROOT 2|1|OBJ 3|1|PUNCT
@End
"
    .to_string();

    let files = vec![
        TextBatchFileInput::new("eng_only.cha", eng_only_chat),
        TextBatchFileInput::new("spa_only.cha", spa_only_chat),
        TextBatchFileInput::new("multi_lang.cha", multi_language_chat_with_populated_mor()),
    ];

    let results = run_batch(&files, &dispatcher).await;

    assert_eq!(results.len(), 3);

    let eng_result = results
        .iter()
        .find(|r| r.filename.as_ref() == "eng_only.cha")
        .expect("eng_only.cha in results");
    assert!(
        eng_result.result.is_ok(),
        "eng-only file should still inject successfully. Got: {:?}",
        eng_result.result
    );

    let spa_result = results
        .iter()
        .find(|r| r.filename.as_ref() == "spa_only.cha")
        .expect("spa_only.cha in results");
    assert!(
        spa_result.result.is_ok(),
        "successful non-primary file should not be poisoned by deu failure. Got: {:?}",
        spa_result.result
    );

    let multi_result = results
        .iter()
        .find(|r| r.filename.as_ref() == "multi_lang.cha")
        .expect("multi_lang.cha in results");
    assert!(
        multi_result.result.is_err(),
        "file containing failed deu utterances must still be marked err. Got: {:?}",
        multi_result.result
    );
}
