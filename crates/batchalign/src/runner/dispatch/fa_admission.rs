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
//! The memo is one `tokio::sync::OnceCell` per language, in a map whose lock
//! is held only to find or insert a cell, never across a probe. The cell
//! makes "one probe per language per job" true by construction: two files of
//! the same language reaching the gate together wait on one initialization
//! instead of racing two probes. Files of OTHER languages do not wait: their
//! cells are their own. (The memo used to be one map-wide `tokio::sync::Mutex`
//! held across the probe, so every language queued behind one language's
//! probe, and a first probe can include a model download measured in
//! minutes.) The pre-scale that follows a successful probe is started in the
//! background by the one task that probed, under the job's cancellation, so
//! the memo is served, and that file proceeds, while the extra workers come
//! up.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tokio::sync::OnceCell;

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
    probed: ProbeMemo<LanguageCode3, AdmissionOutcome>,
}

/// What one language's admission came to.
type AdmissionOutcome = Result<FaCacheNamespace, Arc<FaWorkerRefusal>>;

/// Whether this caller ran the probe, or was answered by another's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Probed {
    /// This caller's probe produced the value.
    Here,
    /// Another caller's probe did, before or while this one waited.
    Elsewhere,
}

/// One value per key, computed once.
///
/// Concurrent askers of one key wait on one computation (a
/// `tokio::sync::OnceCell` per key); askers of other keys never wait, because
/// the map's lock is held only to find or insert a key's cell, never across a
/// computation. A computation whose task is dropped leaves its cell empty, so
/// the next asker computes.
struct ProbeMemo<K, V> {
    cells: Mutex<BTreeMap<K, Arc<OnceCell<V>>>>,
}

impl<K: Ord + Clone, V: Clone> ProbeMemo<K, V> {
    fn new() -> Self {
        Self {
            cells: Mutex::new(BTreeMap::new()),
        }
    }

    /// `key`'s value, computed by `probe` if no caller has yet, and who
    /// computed it.
    async fn get_or_probe<F, Fut>(&self, key: &K, probe: F) -> (V, Probed)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = V>,
    {
        let cell = {
            let mut cells = self
                .cells
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // The key is cloned only for the first file of a language.
            match cells.get(key) {
                Some(cell) => Arc::clone(cell),
                None => Arc::clone(cells.entry(key.clone()).or_default()),
            }
        };
        let mut probed = Probed::Elsewhere;
        let value = cell
            .get_or_init(|| async {
                probed = Probed::Here;
                probe().await
            })
            .await
            .clone();
        (value, probed)
    }
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
            probed: ProbeMemo::new(),
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
        let (outcome, probed) = self
            .probed
            .get_or_probe(lang, || async {
                probe_command_worker(
                    &self.pool,
                    command,
                    lang.clone(),
                    &self.options,
                    &self.gpu,
                    // Admission runs inside a supervised file task, which is
                    // dropped the moment the job is cancelled (`job_scope`).
                    crate::infer_retry::Cancellation::NotWired {
                        reason: "align admission runs in a supervised file task, dropped on cancel",
                    },
                )
                .await
                .and_then(|loaded| {
                    FaCacheNamespace::from_loaded(&loaded).map_err(WorkerProbeRefusal::Unavailable)
                })
                .map_err(|cause| {
                    Arc::new(FaWorkerRefusal {
                        lang: lang.clone(),
                        cause,
                    })
                })
            })
            .await;
        // The memo is set and serves the other files. The task whose probe
        // ran starts warming this language's extra workers in the background
        // (the pre-scale a job-level command gets after its probe, on the key
        // this language's files dispatch to) and goes on to its own file: the
        // pre-scale starts workers one at a time, minutes on a cold host.
        match (probed, &outcome) {
            (Probed::Here, Ok(_)) => {
                let pool = Arc::clone(&self.pool);
                let options = self.options.clone();
                let gpu = Arc::clone(&self.gpu);
                let num_workers = self.num_workers;
                let lang = lang.clone();
                crate::runner::job_scope::spawn_background(async move {
                    super::pre_scale_workers(&pool, command, lang, num_workers, &options, &gpu)
                        .await;
                });
            }
            (Probed::Here, Err(_)) | (Probed::Elsewhere, _) => {}
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::{ProbeMemo, Probed};

    /// Files of one language reaching the gate together wait on one probe.
    #[tokio::test]
    async fn concurrent_askers_of_one_key_share_one_probe() {
        let memo = ProbeMemo::<&str, u32>::new();
        let runs = AtomicUsize::new(0);
        let ask = || {
            memo.get_or_probe(&"eng", || async {
                runs.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                7
            })
        };
        let (a, b, c) = tokio::join!(ask(), ask(), ask());
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!([a.0, b.0, c.0], [7, 7, 7]);
        let here = [a.1, b.1, c.1]
            .iter()
            .filter(|probed| **probed == Probed::Here)
            .count();
        assert_eq!(here, 1, "exactly one asker ran the probe");
    }

    /// A slow probe of one language does not hold up another language.
    #[tokio::test]
    async fn another_key_does_not_wait_for_a_running_probe() {
        let memo = ProbeMemo::<&str, u32>::new();
        let (release, held) = tokio::sync::oneshot::channel::<()>();
        let slow = memo.get_or_probe(&"yue", || async {
            held.await.expect("the test releases the slow probe");
            1
        });
        let fast = async {
            let answer = memo.get_or_probe(&"eng", || async { 2 }).await;
            // Only once the other language is answered is the slow one let go.
            release.send(()).expect("the slow probe is still waiting");
            answer
        };
        let (slow, fast) =
            tokio::time::timeout(Duration::from_secs(5), async { tokio::join!(slow, fast) })
                .await
                .expect("a probe of one key never blocks another key");
        assert_eq!((slow.0, fast.0), (1, 2));
    }
}
