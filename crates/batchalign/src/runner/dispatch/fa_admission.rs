//! Per-language worker admission for `align`.
//!
//! Align is a per-file command: it has no `--lang`, and each file's language is
//! read from its own `@Languages:` header once the file is parsed. The FA
//! worker a file's requests reach is keyed by that language (the Qwen3 and
//! Cantonese aligners fix their language at bootstrap), so there is no
//! job-level worker to probe before the files are read, and the job carries
//! `LanguageSpec::PerFile` rather than a placeholder code.
//!
//! The router used to probe one worker for the whole job, keyed by the job's
//! language, and read two facts off it: that the worker supports forced
//! alignment at all, and the FA engine it loaded, which is the namespace every
//! FA cache row and evidence envelope is written under. Both facts are now
//! read per language, here, the first time a file of that language is reached:
//! [`FaWorkerAdmission::admit`] probes the worker keyed by that language (the
//! one the file's requests will reach, so the namespace is the engine that
//! will actually align the file) through the same
//! [`super::probe_command_worker`] the router uses for a job-level command,
//! pre-scales it the way the router pre-scales a job-level worker, and
//! remembers the outcome, so a job of five hundred English files probes once
//! and a job whose Cantonese model cannot load fails every Cantonese file by
//! the same name while its English files proceed.
//!
//! The memo is a `tokio::sync::Mutex` held across the probe, and only the
//! probe. That is deliberate: it makes "one probe per language per job" true
//! by construction (two files of the same language reaching the gate together
//! wait for one probe instead of racing two). Files of OTHER languages wait
//! too while a probe runs; a probe is one worker's `ensure_task`, bounded by
//! the pool's timeout. The pre-scale that follows a successful probe runs
//! after the lock is released, by the one task that probed, so the memo is
//! served while the extra workers come up.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::api::{LanguageCode3, NumWorkers, ReleasedCommand};
use crate::engine_reports::FaCacheNamespace;
use crate::options::CommandOptions;
use crate::scheduling::FailureCategory;
use crate::worker::pool::{GpuModelPlan, WorkerPool};

use super::{WorkerProbeRefusal, probe_command_worker};

/// Why a file's language could not be admitted to a worker: the language, and
/// what the probe of its worker came back with.
///
/// The file is failed with this text, so it names the language: the operator
/// needs to know which model the host could not serve.
#[derive(Debug, thiserror::Error)]
#[error("cannot run 'align' for language '{lang}': {cause}")]
pub(crate) struct FaWorkerRefusal {
    /// The file's declared primary language.
    lang: LanguageCode3,
    /// What the probe of the worker keyed by that language refused on.
    #[source]
    cause: WorkerProbeRefusal,
}

impl FaWorkerRefusal {
    /// How the file this refusal fails is classified, and so whether it is
    /// retried; the cause decides.
    pub(crate) fn category(&self) -> FailureCategory {
        self.cause.category()
    }
}

/// One align job's gate from a file's declared language to the FA engine that
/// will align it.
///
/// Built once per job in `dispatch_fa_infer` and shared by every file task.
/// The only way to a [`FaCacheNamespace`] inside the FA pipeline is
/// [`Self::admit`], so no file can be aligned under a namespace that was not
/// read off the worker keyed by its own language.
pub(crate) struct FaWorkerAdmission {
    /// The pool the probe and the file's requests both reach.
    pool: Arc<WorkerPool>,
    /// The job's typed options; the worker key is derived from them.
    options: CommandOptions,
    /// The job's GPU model plan; the probe spawns the worker those engines
    /// select, which is the worker the job's FA requests are routed to.
    gpu: Arc<GpuModelPlan>,
    /// The job's worker budget, which bounds pre-scaling.
    num_workers: NumWorkers,
    /// Outcome per language, successes and refusals alike, so each language
    /// is probed once however many files declare it.
    probed:
        tokio::sync::Mutex<BTreeMap<LanguageCode3, Result<FaCacheNamespace, Arc<FaWorkerRefusal>>>>,
}

impl FaWorkerAdmission {
    /// The gate for one job. Nothing is probed until a file asks.
    pub(crate) fn new(
        pool: Arc<WorkerPool>,
        options: CommandOptions,
        gpu: Arc<GpuModelPlan>,
        num_workers: NumWorkers,
    ) -> Self {
        Self {
            pool,
            options,
            gpu,
            num_workers,
            probed: tokio::sync::Mutex::new(BTreeMap::new()),
        }
    }

    /// The FA engine the worker keyed by `lang` reports, probing that worker
    /// the first time this job reaches the language and answering from the
    /// memo after.
    ///
    /// A refusal is remembered too: the second file of a language whose model
    /// cannot load is failed by the same name without spawning again.
    pub(crate) async fn admit(
        &self,
        lang: &LanguageCode3,
    ) -> Result<FaCacheNamespace, Arc<FaWorkerRefusal>> {
        let command = ReleasedCommand::Align;
        let outcome = {
            let mut probed = self.probed.lock().await;
            if let Some(outcome) = probed.get(lang) {
                return outcome.clone();
            }
            let outcome =
                probe_command_worker(&self.pool, command, lang.clone(), &self.options, &self.gpu)
                    .await
                    .and_then(|loaded| {
                        FaCacheNamespace::from_loaded(&loaded)
                            .map_err(WorkerProbeRefusal::Unavailable)
                    })
                    .map_err(|cause| {
                        Arc::new(FaWorkerRefusal {
                            lang: lang.clone(),
                            cause,
                        })
                    });
            probed.insert(lang.clone(), outcome.clone());
            outcome
        };
        // The lock is released: the memo serves the other files while the
        // probing task alone warms this language's workers, the same
        // pre-scale a job-level command gets after its probe, on the key this
        // language's files dispatch to.
        if outcome.is_ok() {
            super::pre_scale_workers(
                &self.pool,
                command,
                lang.clone(),
                self.num_workers,
                &self.options,
                &self.gpu,
            )
            .await;
        }
        outcome
    }
}
