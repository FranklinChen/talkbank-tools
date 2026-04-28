use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use batchalign_chat_ops::morphosyntax::{MultilingualPolicy, MwtDict, TokenizationMode};

use crate::api::{EngineVersion, LanguageCode3, ReleasedCommand, WorkerLanguage};
use crate::cache::UtteranceCache;
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::text_batch::{TextBatchFileInput, TextBatchFileResults};
use crate::types::worker_v2::ProgressEventV2;
use crate::worker::pool::WorkerPool;

/// Runtime morphotag options resolved from command options for execution.
#[derive(Clone, Copy)]
pub(crate) struct MorphotagRuntimeOptions<'a> {
    pub(crate) tokenization_mode: TokenizationMode,
    pub(crate) multilingual_policy: MultilingualPolicy,
    pub(crate) mwt: &'a MwtDict,
    pub(crate) l2_morphotag: bool,
    pub(crate) respect_pos_hints: bool,
    pub(crate) should_merge_abbrev: bool,
}

/// Worker-system seam consumed by the new execution kernel.
#[async_trait]
pub(crate) trait WorkerGateway: Send + Sync {
    /// Ensure the worker system can execute the requested released command.
    async fn ensure_command_capabilities(
        &self,
        command: ReleasedCommand,
        lang: WorkerLanguage,
        engine_overrides: &str,
    ) -> Result<crate::capability::WorkerCapabilitySnapshot, String>;

    /// Run the compare command's morphosyntax stage on one CHAT input.
    async fn morphotag_for_compare(
        &self,
        chat_text: &str,
        lang: &LanguageCode3,
        mwt: &MwtDict,
    ) -> Result<String, ServerError>;

    /// Run batch morphotag over one window of CHAT files.
    async fn morphotag_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions<'_>,
        progress_tx: Option<tokio::sync::mpsc::Sender<ProgressEventV2>>,
        group_timeout: Duration,
    ) -> TextBatchFileResults;

    /// Run morphotag on one CHAT file.
    async fn morphotag_single(
        &self,
        chat_text: &str,
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions<'_>,
    ) -> Result<String, ServerError>;

    /// Run incremental morphotag using a before/after pair.
    async fn morphotag_incremental(
        &self,
        before_text: &str,
        after_text: &str,
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions<'_>,
    ) -> Result<String, ServerError>;

    /// Run utterance segmentation over one cross-file batch of CHAT inputs.
    async fn utseg_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
    ) -> TextBatchFileResults;

    /// Run translation over one cross-file batch of CHAT inputs.
    async fn translate_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
    ) -> TextBatchFileResults;

    /// Run coreference resolution over one cross-file batch of CHAT inputs.
    async fn coref_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
    ) -> TextBatchFileResults;
}

/// Worker gateway backed by the existing worker pool and cache.
#[derive(Clone)]
pub(crate) struct PooledWorkerGateway {
    pool: Arc<WorkerPool>,
    cache: Arc<UtteranceCache>,
    engine_version: EngineVersion,
}

impl PooledWorkerGateway {
    /// Build a pool-backed worker gateway for one execution attempt.
    pub(crate) fn new(
        pool: Arc<WorkerPool>,
        cache: Arc<UtteranceCache>,
        engine_version: EngineVersion,
    ) -> Self {
        Self {
            pool,
            cache,
            engine_version,
        }
    }
}

#[async_trait]
impl WorkerGateway for PooledWorkerGateway {
    async fn ensure_command_capabilities(
        &self,
        command: ReleasedCommand,
        lang: WorkerLanguage,
        engine_overrides: &str,
    ) -> Result<crate::capability::WorkerCapabilitySnapshot, String> {
        self.pool
            .ensure_command_capabilities_with_overrides(command, lang, engine_overrides)
            .await
            .map_err(|error| error.to_string())?;
        let detected = self.pool.detected_capabilities().ok_or_else(|| {
            "worker capability probe completed without detected capabilities".to_string()
        })?;
        Ok(crate::capability::WorkerCapabilitySnapshot {
            capabilities: detected.commands.clone(),
            infer_tasks: detected.infer_tasks.clone(),
            engine_versions: detected.engine_versions.clone(),
        })
    }

    async fn morphotag_for_compare(
        &self,
        chat_text: &str,
        lang: &LanguageCode3,
        mwt: &MwtDict,
    ) -> Result<String, ServerError> {
        let params = MorphosyntaxParams {
            lang,
            tokenization_mode: TokenizationMode::Preserve,
            multilingual_policy: MultilingualPolicy::ProcessAll,
            mwt,
            l2_morphotag: false,
            respect_pos_hints: false,
        };
        crate::morphosyntax::process_morphosyntax(
            chat_text,
            PipelineServices::new(&self.pool, &self.cache, &self.engine_version),
            &params,
        )
        .await
    }

    async fn morphotag_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions<'_>,
        progress_tx: Option<tokio::sync::mpsc::Sender<ProgressEventV2>>,
        group_timeout: Duration,
    ) -> TextBatchFileResults {
        let params = MorphosyntaxParams {
            lang,
            tokenization_mode: options.tokenization_mode,
            multilingual_policy: options.multilingual_policy,
            mwt: options.mwt,
            l2_morphotag: options.l2_morphotag,
            respect_pos_hints: options.respect_pos_hints,
        };
        let dispatcher = crate::morphosyntax::PoolDispatcher::new(&self.pool);
        crate::morphosyntax::run_morphosyntax_batch_impl(
            files,
            PipelineServices::new(&self.pool, &self.cache, &self.engine_version),
            &dispatcher,
            &params,
            progress_tx,
            group_timeout,
        )
        .await
    }

    async fn morphotag_single(
        &self,
        chat_text: &str,
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions<'_>,
    ) -> Result<String, ServerError> {
        let params = MorphosyntaxParams {
            lang,
            tokenization_mode: options.tokenization_mode,
            multilingual_policy: options.multilingual_policy,
            mwt: options.mwt,
            l2_morphotag: options.l2_morphotag,
            respect_pos_hints: options.respect_pos_hints,
        };
        crate::morphosyntax::process_morphosyntax(
            chat_text,
            PipelineServices::new(&self.pool, &self.cache, &self.engine_version),
            &params,
        )
        .await
    }

    async fn morphotag_incremental(
        &self,
        before_text: &str,
        after_text: &str,
        lang: &LanguageCode3,
        options: MorphotagRuntimeOptions<'_>,
    ) -> Result<String, ServerError> {
        let params = MorphosyntaxParams {
            lang,
            tokenization_mode: options.tokenization_mode,
            multilingual_policy: options.multilingual_policy,
            mwt: options.mwt,
            l2_morphotag: options.l2_morphotag,
            respect_pos_hints: options.respect_pos_hints,
        };
        crate::morphosyntax::process_morphosyntax_incremental(
            before_text,
            after_text,
            PipelineServices::new(&self.pool, &self.cache, &self.engine_version),
            &params,
        )
        .await
    }

    async fn utseg_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
    ) -> TextBatchFileResults {
        crate::utseg::process_utseg_batch(
            files,
            lang,
            &self.pool,
            &self.cache,
            &self.engine_version,
        )
        .await
    }

    async fn translate_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
    ) -> TextBatchFileResults {
        crate::translate::process_translate_batch(
            files,
            lang,
            &self.pool,
            &self.cache,
            &self.engine_version,
        )
        .await
    }

    async fn coref_batch(
        &self,
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
    ) -> TextBatchFileResults {
        crate::coref::process_coref_batch(files, lang, &self.pool).await
    }
}
