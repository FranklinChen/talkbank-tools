use std::sync::Arc;

use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, MwtDict, TokenizationMode};
use async_trait::async_trait;

use crate::api::LanguageCode3;
use crate::cache::UtteranceCache;
use crate::error::ServerError;
use crate::infer_retry::Cancellation;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::pipeline::post_validate::PostValidated;
use crate::text_batch::{TextBatchFileInput, TextBatchFileResults};
use crate::types::engines::TranslateEngineName;
use crate::worker::pool::WorkerPool;

/// Runtime morphotag options resolved from command options for execution.
///
/// Owned (no borrowed `MwtDict`) so the value can move freely across
/// `tokio::spawn` boundaries used by per-file fanout in
/// `dispatch_morphotag_job`.
#[derive(Clone)]
pub(crate) struct MorphotagRuntimeOptions {
    pub(crate) tokenization_mode: TokenizationMode,
    pub(crate) multilingual_policy: MultilingualPolicy,
    pub(crate) mwt: Arc<MwtDict>,
    pub(crate) l2_policy: crate::params::L2MorphotagPolicy,
    pub(crate) pos_hint_policy: crate::params::PosHintPolicy,
    pub(crate) ca_policy: crate::options::CaMorphotagPolicy,
    pub(crate) should_merge_abbrev: bool,
    /// Review-tier verbosity for the incremental morphotag path
    /// Legacy review-level request retained for stored-job compatibility.
    /// No value emits CHAT decision tiers.
    ///
    /// [`MorphotagOptions`]: crate::options::MorphotagOptions
    pub(crate) review_level: crate::chat_ops::fa::ReviewLevel,
}

/// Worker-system seam consumed by the new execution kernel.
#[async_trait]
pub(crate) trait WorkerGateway: Send + Sync {
    /// Run the compare command's morphosyntax stage on one CHAT input.
    ///
    /// Returns the PROOF, for the same reason [`Self::morphotag_single`] does,
    /// and then one further reason. Comparison does not write these bytes: it
    /// continues from this document, and taking the proof is what lets it
    /// continue from the very model the gate judged. This returned a `String`
    /// until 2026-09-16, and the document was parsed back out of it by a
    /// LENIENT parser that admits what the validating one refuses, with the
    /// parse errors warned about and dropped.
    async fn morphotag_for_compare(
        &self,
        chat_text: &str,
        lang: &LanguageCode3,
        mwt: &MwtDict,
        cancellation: Cancellation<'_>,
    ) -> Result<PostValidated, ServerError>;

    /// Run morphotag on one CHAT file.
    ///
    /// `progress` is this file's port into the job's batch-progress reporter, or
    /// `None` where no reporter exists (the CLI's direct path, tests). It is a
    /// separate argument rather than a field on `MorphotagRuntimeOptions`
    /// because it is an output port, not an option: options say what to compute,
    /// this says where to narrate it.
    ///
    /// Returns the gate-proven output rather than a bare string, so a caller
    /// cannot report a morphotag file as succeeded without the proof that its
    /// output passed post-validation.
    async fn morphotag_single(
        &self,
        chat_text: &str,
        before_text: Option<&str>,
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions,
        progress: Option<&crate::execution::morphotag::progress::BackendProgressPort>,
        cancellation: Cancellation<'_>,
    ) -> Result<PostValidated, ServerError>;

    /// Run utterance segmentation over one cross-file batch of CHAT inputs.
    ///
    /// `allow_stanza_fallback` propagates the
    /// `--utseg-fallback-stanza` operator opt-in: when `true`, the
    /// worker engages the legacy Stanza constituency-parser segmenter
    /// for languages without a TalkBank BERT utseg model. When
    /// `false` (default), the worker raises `UtsegModelNotFoundError`
    /// rather than silently substituting one model for another.
    async fn utseg_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        allow_stanza_fallback: bool,
        cancellation: Cancellation<'_>,
    ) -> TextBatchFileResults;

    /// Translate one CHAT input. Each result names the engine that
    /// translated it.
    ///
    /// One file, not a batch: translate stops a file at its first failed
    /// utterance, a verdict that is only per-file when the file is sent
    /// alone. `engine` is the job's selection: it keys the worker that serves
    /// the requests and sets the pacing and retry policy they are sent under.
    async fn translate_file(
        &self,
        file: &TextBatchFileInput,
        lang: &LanguageCode3,
        engine: &TranslateEngineName,
        cancellation: Cancellation<'_>,
    ) -> TextBatchFileResults;

    /// Run coreference resolution over one cross-file batch of CHAT inputs.
    /// Each result names the engine that resolved it.
    ///
    /// Takes no language. Coref has no `--lang`, is English-only, and reads
    /// per-file English-ness from each file's `@Languages:` header; the
    /// inference language is the constant `eng` the command owns. The
    /// parameter that used to sit here was passed the job's language, which
    /// for a per-file command is never a resolved code, and was then discarded
    /// unread by the implementation.
    async fn coref_batch(
        &self,
        files: &[TextBatchFileInput],
        cancellation: Cancellation<'_>,
    ) -> TextBatchFileResults;
}

/// Worker gateway backed by the existing worker pool and cache.
///
/// Carries no engine identity: every text stage names its engines from the
/// results it applies.
#[derive(Clone)]
pub(crate) struct PooledWorkerGateway {
    pool: Arc<WorkerPool>,
    cache: Arc<UtteranceCache>,
}

impl PooledWorkerGateway {
    /// Build a pool-backed worker gateway for one execution attempt.
    pub(crate) fn new(pool: Arc<WorkerPool>, cache: Arc<UtteranceCache>) -> Self {
        Self { pool, cache }
    }
}

#[async_trait]
impl WorkerGateway for PooledWorkerGateway {
    async fn morphotag_for_compare(
        &self,
        chat_text: &str,
        lang: &LanguageCode3,
        mwt: &MwtDict,
        cancellation: Cancellation<'_>,
    ) -> Result<PostValidated, ServerError> {
        let params = MorphosyntaxParams {
            lang,
            tokenization_mode: TokenizationMode::Preserve,
            multilingual_policy: MultilingualPolicy::ProcessAll,
            mwt,
            policy: crate::params::MorphotagExecutionPolicy {
                l2: crate::params::L2MorphotagPolicy::Placeholder,
                pos_hints: crate::params::PosHintPolicy::Ignore,
                ca_policy: crate::options::CaMorphotagPolicy::Honor,
            },
            // Compare's internal morphotag never surfaces review tiers.
            review_level: crate::chat_ops::fa::ReviewLevel::None,
            // Compare runs morphotag on its own inputs, not on the job's files,
            // so there is no file row to report utterance counts against.
            progress: None,
            cancellation,
        };
        // Compare consumes the DOCUMENT, not the bytes: its output is a
        // comparison artifact, never a written CHAT file, and the artifact is
        // built in the AST. The proof travels intact to
        // `compare::process_compare_morphotagged_main`, which is the only
        // consumer and takes nothing else.
        crate::morphosyntax::process_morphosyntax(
            chat_text,
            PipelineServices::new(&self.pool, &self.cache),
            &params,
        )
        .await
    }

    async fn morphotag_single(
        &self,
        chat_text: &str,
        before_text: Option<&str>,
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions,
        progress: Option<&crate::execution::morphotag::progress::BackendProgressPort>,
        cancellation: Cancellation<'_>,
    ) -> Result<PostValidated, ServerError> {
        let params = MorphosyntaxParams {
            lang,
            tokenization_mode: options.tokenization_mode,
            multilingual_policy: options.multilingual_policy,
            mwt: &options.mwt,
            policy: crate::params::MorphotagExecutionPolicy {
                l2: options.l2_policy,
                pos_hints: options.pos_hint_policy,
                ca_policy: options.ca_policy,
            },
            review_level: options.review_level,
            progress,
            cancellation,
        };
        let services = PipelineServices::new(&self.pool, &self.cache);
        if let Some(before) = before_text {
            crate::morphosyntax::process_morphosyntax_incremental(
                before, chat_text, services, &params,
            )
            .await
        } else {
            crate::morphosyntax::process_morphosyntax(chat_text, services, &params).await
        }
    }

    async fn utseg_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        allow_stanza_fallback: bool,
        cancellation: Cancellation<'_>,
    ) -> TextBatchFileResults {
        crate::utseg::process_utseg_batch(
            files,
            lang,
            &self.pool,
            allow_stanza_fallback,
            cancellation,
        )
        .await
    }

    async fn translate_file(
        &self,
        file: &TextBatchFileInput,
        lang: &LanguageCode3,
        engine: &TranslateEngineName,
        cancellation: Cancellation<'_>,
    ) -> TextBatchFileResults {
        crate::translate::process_translate_file(file, lang, engine, &self.pool, cancellation).await
    }

    async fn coref_batch(
        &self,
        files: &[TextBatchFileInput],
        cancellation: Cancellation<'_>,
    ) -> TextBatchFileResults {
        crate::coref::process_coref_batch(files, &self.pool, cancellation).await
    }
}
