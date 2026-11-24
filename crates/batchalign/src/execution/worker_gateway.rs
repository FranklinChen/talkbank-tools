use std::sync::Arc;

use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, MwtDict, TokenizationMode};
use async_trait::async_trait;

use crate::cache::UtteranceCache;
use crate::error::ServerError;
use crate::execution::kernel::CompareMorphologySubmission;
use crate::execution::text_io::{
    CorefSubmission, MorphotagSource, MorphotagSubmission, TranslateSubmission, UtsegSubmission,
};
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::pipeline::post_validate::PostValidated;
use crate::text_batch::TextBatchFileResults;
use crate::worker::pool::WorkerPool;

/// Runtime morphotag options resolved from command options for execution.
///
/// Owned (no borrowed `MwtDict`) so the value can move freely across
/// `tokio::spawn` boundaries used by per-file fanout in
/// `dispatch_morphotag_job`.
pub struct MorphotagRuntimeOptions {
    pub(crate) tokenization_mode: TokenizationMode,
    pub(crate) multilingual_policy: MultilingualPolicy,
    pub(crate) mwt: Arc<MwtDict>,
    pub(crate) l2_policy: crate::params::L2MorphotagPolicy,
    pub(crate) pos_hint_policy: crate::params::PosHintPolicy,
    pub(crate) ca_policy: crate::options::CaMorphotagPolicy,
    /// Review-tier verbosity for the incremental morphotag path
    /// Legacy review-level request retained for stored-job compatibility.
    /// No value emits CHAT decision tiers.
    ///
    /// [`MorphotagOptions`]: crate::options::MorphotagOptions
    pub(crate) review_level: crate::chat_ops::fa::ReviewLevel,
}

/// The only worker effect available to an admitted comparison.
#[async_trait]
pub trait CompareMorphology: Send + Sync {
    /// Run the compare command's morphosyntax stage on one CHAT input.
    ///
    /// Returns the PROOF, for the same reason [`MorphotagGateway::morphotag_single`] does,
    /// and then one further reason. Comparison does not write these bytes: it
    /// continues from this document, and taking the proof is what lets it
    /// continue from the very model the gate judged. This returned a `String`
    /// until 2026-09-16, and the document was parsed back out of it by a
    /// LENIENT parser that admits what the validating one refuses, with the
    /// parse errors warned about and dropped.
    ///
    /// No language: the transcript's own `@Languages:` header decides, inside
    /// the morphosyntax pipeline, as for `morphotag`. A job-level code used to
    /// be passed here and discarded.
    async fn morphotag_for_compare(
        &self,
        submission: CompareMorphologySubmission<'_>,
    ) -> Result<PostValidated, ServerError>;
}

/// Morphology effects available to a morphotag file task.
#[async_trait]
pub trait MorphotagGateway: Send + Sync {
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
        submission: MorphotagSubmission<'_>,
    ) -> Result<crate::morphosyntax::report::MorphosyntaxOutput, ServerError>;
}

/// Segmentation of one source on its admitted route.
#[async_trait]
pub trait UtsegGateway: Send + Sync {
    /// Segment one CHAT source while retaining the admitted route.
    ///
    /// `route` is the admitted route: its language and the one segmenter the
    /// worker request names, so the worker never chooses a segmenter itself.
    async fn utseg_file(&self, submission: UtsegSubmission<'_>) -> TextBatchFileResults;
}

/// Translation of one source on its admitted route.
#[async_trait]
pub trait TranslateGateway: Send + Sync {
    /// Translate one CHAT input. Each result names the engine that
    /// translated it.
    ///
    /// One file, not a batch: translate stops a file at its first failed
    /// utterance, a verdict that is only per-file when the file is sent
    /// alone. The sealed route holds the file's source, job's target and engine;
    /// its engine keys the worker that serves
    /// the requests and sets the pacing and retry policy they are sent under.
    async fn translate_file(&self, submission: TranslateSubmission<'_>) -> TextBatchFileResults;
}

/// Coreference effects available to the batch dispatcher.
#[async_trait]
pub trait CorefGateway: Send + Sync {
    /// Run coreference resolution over one cross-file batch of CHAT inputs.
    /// Each result names the engine that resolved it.
    ///
    /// Takes no language. Coref has no `--lang`, is English-only, and reads
    /// per-file English-ness from each file's `@Languages:` header; the
    /// inference language is the constant `eng` the command owns. The
    /// parameter that used to sit here was passed the job's language, which
    /// for a per-file command is never a resolved code, and was then discarded
    /// unread by the implementation.
    async fn coref_batch(&self, submission: CorefSubmission<'_>) -> TextBatchFileResults;
}

/// Worker gateway backed by the existing worker pool and cache.
///
/// Carries no engine identity: every text stage names its engines from the
/// results it applies.
#[derive(Clone)]
pub struct PooledWorkerGateway {
    pool: Arc<WorkerPool>,
    cache: Arc<UtteranceCache>,
}

impl PooledWorkerGateway {
    /// Build a pool-backed worker gateway for one execution attempt.
    pub(crate) const fn new(pool: Arc<WorkerPool>, cache: Arc<UtteranceCache>) -> Self {
        Self { pool, cache }
    }
}

#[async_trait]
impl CompareMorphology for PooledWorkerGateway {
    async fn morphotag_for_compare(
        &self,
        submission: CompareMorphologySubmission<'_>,
    ) -> Result<PostValidated, ServerError> {
        let (chat_text, mwt, cancellation) = submission.into_parts();
        let params = MorphosyntaxParams {
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
            &chat_text,
            PipelineServices::new(&self.pool, &self.cache),
            &params,
        )
        .await?
        .without_reports()
    }
}

#[async_trait]
impl MorphotagGateway for PooledWorkerGateway {
    async fn morphotag_single(
        &self,
        submission: MorphotagSubmission<'_>,
    ) -> Result<crate::morphosyntax::report::MorphosyntaxOutput, ServerError> {
        let (source, options, progress, cancellation) = submission.into_parts();
        let params = MorphosyntaxParams {
            tokenization_mode: options.tokenization_mode,
            multilingual_policy: options.multilingual_policy,
            mwt: &options.mwt,
            policy: crate::params::MorphotagExecutionPolicy {
                l2: options.l2_policy,
                pos_hints: options.pos_hint_policy,
                ca_policy: options.ca_policy,
            },
            review_level: options.review_level,
            progress: progress.as_ref(),
            cancellation,
        };
        let services = PipelineServices::new(&self.pool, &self.cache);
        match source {
            MorphotagSource::Full(text) => {
                crate::morphosyntax::process_morphosyntax(&text, services, &params).await
            }
            MorphotagSource::Incremental { before, after } => {
                crate::morphosyntax::process_morphosyntax_incremental(
                    &before, &after, services, &params,
                )
                .await
            }
        }
    }
}

#[async_trait]
impl UtsegGateway for PooledWorkerGateway {
    async fn utseg_file(&self, submission: UtsegSubmission<'_>) -> TextBatchFileResults {
        crate::utseg::process_utseg_file(submission, &self.pool).await
    }
}

#[async_trait]
impl TranslateGateway for PooledWorkerGateway {
    async fn translate_file(&self, submission: TranslateSubmission<'_>) -> TextBatchFileResults {
        crate::translate::process_translate_file(submission, &self.pool).await
    }
}

#[async_trait]
impl CorefGateway for PooledWorkerGateway {
    async fn coref_batch(&self, submission: CorefSubmission<'_>) -> TextBatchFileResults {
        crate::coref::process_coref_batch(submission, &self.pool).await
    }
}
