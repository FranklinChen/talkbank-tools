//! Per-task engine identities a worker reported, admitted once where the
//! worker pool records a capability report.
//!
//! A worker's capability response carries two parallel wire fields: the infer
//! tasks it supports, and an entry per task. The entry for forced alignment is
//! the FA engine's name (a validated [`ReportedEngineName`]), or `null` before
//! an FA model has loaded; the entry for every other task is `null`, because
//! those stages name their engines on the results they return. So a report
//! carries task support plus the FA identity, and nothing else. The wire type
//! is frozen, so the pair stays; this module is where it stops being a pair.
//! [`WorkerEngineReports::admit`] reads both into ONE map keyed by
//! [`InferTask`], refusing a task with no entry, an entry for a task nobody
//! advertised, or a name for any task but forced alignment. The pool calls it
//! at the one place it records a report (`WorkerPool::record_capabilities`)
//! and keeps the admitted value per worker key; everything downstream reads
//! that value, never the raw report.
//!
//! # Why a report is not an engine identity
//!
//! An unreported engine is a legal fact (the worker can run the task but has
//! not said with what), so it is admitted. What it can never do is name
//! anything durable: a cache namespace or a provenance field built from it
//! would be a fabricated identity. The only route from a report to such a name
//! is [`FaCacheNamespace::from_loaded`], which requires a named engine and
//! returns a typed refusal otherwise.
//!
//! # Supporting a task is not knowing its engine
//!
//! A lazily loading worker supports forced alignment before it has loaded an
//! FA model, and names the FA engine only after. Those are two facts, read at
//! two moments. Support is read from any admitted report, and is what decides
//! whether a command is advertised. An identity is read only from a
//! [`LoadedCapabilities`], the report the pool takes AFTER it has loaded the
//! command's task on the selected worker, so `null` there means "still unnamed
//! after loading", and that alone is a refusal. The loaded report records
//! which task was loaded, so an FA identity is never read from a report taken
//! after some other task loaded.
//!
//! # Which identity is read from a report at all
//!
//! Only forced alignment's. It is the one stage that must name its engine
//! BEFORE dispatch, because every FA cache row is read under that namespace
//! before any worker runs. Every other stage takes its provenance from the
//! results it applies (the worker names the engine on each result), never from
//! a capability report, which is why admission refuses a report that names
//! one: a name nothing reads would only invite a later reader to treat it as
//! an identity.
//!
//! ```text
//! worker capabilities (wire)
//!   -> WorkerEngineReports::admit        (once, where the pool records it)
//!   -> LoadedCapabilities                 (after `ensure_task` loaded the task)
//!   -> FaCacheNamespace::from_loaded      (the one pre-dispatch identity)
//!   -> FA cache rows / FA provenance field
//! ```

use std::collections::BTreeMap;

use crate::api::ReportedEngineName;
use crate::worker::InferTask;
use crate::worker::pool::LoadedCapabilities;
use crate::worker::target::task_name;

/// What one worker said about the engine behind one infer task it supports.
///
/// Private: a consumer asks [`WorkerEngineReports`] whether a task is
/// supported, and asks [`FaCacheNamespace::from_loaded`] for a name. Neither
/// needs to match on the report itself.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EngineReport {
    /// The worker named the engine (for example `wave2vec-fa-v1`).
    Reported(ReportedEngineName),
    /// The worker supports the task but did not name the engine.
    Unreported,
}

/// Every infer task one worker advertised, with its engine report.
///
/// The advertised task set is this map's keys, so there is no second list to
/// disagree with. Built only by [`Self::admit`] from a worker's report: there
/// is no "assumed" constructor, because a view taken before any worker has
/// answered names no engine and is modeled by the capability snapshot, not by
/// a report nobody made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerEngineReports(BTreeMap<InferTask, EngineReport>);

/// Why a worker's capability report was refused at admission.
///
/// A blank or separator-bearing name and an unknown task key cannot reach this
/// point: [`ReportedEngineName`] and [`InferTask`] refuse them while the wire
/// report is deserialized.
///
/// Also a wire type: `/health` returns it for a worker key whose latest report
/// was refused, so an operator sees why that worker is not used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EngineReportAdmissionError {
    /// A task was advertised with no engine entry at all.
    #[error(
        "worker capability report refused: infer task '{}' is advertised but \
         engine_versions['{}'] is missing",
        task_name(*.task),
        task_name(*.task)
    )]
    MissingEngineReport {
        /// The advertised task.
        #[cfg_attr(feature = "server", schema(value_type = String))]
        task: InferTask,
    },
    /// An engine entry names a task the worker did not advertise.
    #[error(
        "worker capability report refused: engine_versions['{}'] names a task the \
         worker does not advertise",
        task_name(*.task)
    )]
    UnadvertisedEngineReport {
        /// The unmatched task.
        #[cfg_attr(feature = "server", schema(value_type = String))]
        task: InferTask,
    },
    /// An entry names an engine for a task other than forced alignment. Only
    /// FA's engine is read from a capability report; every other stage names
    /// its engine on the results it returns, so its entry is `null`.
    #[error(
        "worker capability report refused: engine_versions['{}'] names an engine, but a \
         capability report names only the forced-alignment engine",
        task_name(*.task)
    )]
    EngineNamedForNonFaTask {
        /// The task whose entry named an engine.
        #[cfg_attr(feature = "server", schema(value_type = String))]
        task: InferTask,
    },
}

/// Why a command cannot run against a worker: the task is not supported, the
/// engine a command must name before dispatch is still unnamed after the task
/// was loaded, or the loaded report is for another task.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EngineIdentityUnavailable {
    /// The worker does not support the task at all.
    #[error("the selected worker does not support infer task '{}'", task_name(*.task))]
    NotSupported {
        /// The task.
        task: InferTask,
    },
    /// The worker loaded the task and still did not name its engine.
    #[error(
        "the selected worker loaded infer task '{}' but still did not report which \
         engine runs it, so no cache namespace or provenance field can name it",
        task_name(*.task)
    )]
    UnreportedAfterLoad {
        /// The task.
        task: InferTask,
    },
    /// The report was taken after a different task loaded, so it says nothing
    /// about the engine behind the task that was asked for.
    #[error(
        "the selected worker's post-load report is for infer task '{}', not '{}'",
        task_name(*.loaded),
        task_name(*.wanted)
    )]
    LoadedAnotherTask {
        /// The task that was loaded before the report was taken.
        loaded: InferTask,
        /// The task whose engine was asked for.
        wanted: InferTask,
    },
}

impl WorkerEngineReports {
    /// Admit one worker's capability report.
    ///
    /// Every advertised task must have an entry, only forced alignment's may
    /// be a name (every other entry is null), and no entry may name an
    /// unadvertised task. The engine map is taken by value so the FA name
    /// moves into the admitted report instead of being copied; it is kept byte
    /// for byte, because it is a cache namespace earlier runs already wrote
    /// under.
    pub(crate) fn admit(
        infer_tasks: &[InferTask],
        mut engine_versions: BTreeMap<InferTask, Option<ReportedEngineName>>,
    ) -> Result<Self, EngineReportAdmissionError> {
        let mut reports = BTreeMap::new();
        for &task in infer_tasks {
            // A task listed twice was already admitted by its first entry; its
            // engine entry has been moved out, so it is not "missing".
            if reports.contains_key(&task) {
                continue;
            }
            let report = match engine_versions.remove(&task) {
                None => return Err(EngineReportAdmissionError::MissingEngineReport { task }),
                Some(None) => EngineReport::Unreported,
                // Exhaustive over tasks, so a new task must decide whether its
                // engine is read from a report.
                Some(Some(name)) => match task {
                    InferTask::Fa => EngineReport::Reported(name),
                    InferTask::Morphosyntax
                    | InferTask::Utseg
                    | InferTask::Translate
                    | InferTask::Coref
                    | InferTask::Asr
                    | InferTask::Opensmile
                    | InferTask::Avqi
                    | InferTask::Speaker => {
                        return Err(EngineReportAdmissionError::EngineNamedForNonFaTask { task });
                    }
                },
            };
            reports.insert(task, report);
        }
        // Whatever is left was never advertised.
        if let Some(task) = engine_versions.into_keys().next() {
            return Err(EngineReportAdmissionError::UnadvertisedEngineReport { task });
        }
        Ok(Self(reports))
    }

    /// Whether the worker advertised `task`.
    pub fn supports(&self, task: InferTask) -> bool {
        self.0.contains_key(&task)
    }

    /// The advertised tasks, in `InferTask` order.
    pub fn tasks(&self) -> impl Iterator<Item = InferTask> + '_ {
        self.0.keys().copied()
    }

    /// The engine named for `task` in a post-load report, or why there is
    /// none. Private: [`FaCacheNamespace::from_loaded`] is the only caller, so
    /// an engine is never read from a report taken before the task loaded.
    fn reported_after_load(
        &self,
        task: InferTask,
    ) -> Result<&ReportedEngineName, EngineIdentityUnavailable> {
        match self.0.get(&task) {
            None => Err(EngineIdentityUnavailable::NotSupported { task }),
            Some(EngineReport::Unreported) => {
                Err(EngineIdentityUnavailable::UnreportedAfterLoad { task })
            }
            Some(EngineReport::Reported(name)) => Ok(name),
        }
    }
}

/// The forced-alignment engine the selected worker reported. Its name is also
/// the namespace every FA cache row and FA evidence envelope is written and
/// admitted under, so it is exactly the string the worker sent.
///
/// The field is private and the only production constructor is
/// [`Self::from_loaded`], so a value of this type is proof the identity came
/// from a worker's admitted report, taken after FA loaded, and not from a
/// literal or a fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaCacheNamespace(ReportedEngineName);

impl FaCacheNamespace {
    /// Read the FA engine out of a worker's report taken after FA was loaded
    /// on it. A report taken after another task loaded is refused rather than
    /// read, because it says nothing about the FA engine.
    pub(crate) fn from_loaded(
        loaded: &LoadedCapabilities,
    ) -> Result<Self, EngineIdentityUnavailable> {
        match loaded.task() {
            InferTask::Fa => loaded
                .reports()
                .reported_after_load(InferTask::Fa)
                .map(|name| Self(name.clone())),
            loaded_task => Err(EngineIdentityUnavailable::LoadedAnotherTask {
                loaded: loaded_task,
                wanted: InferTask::Fa,
            }),
        }
    }

    /// The reported engine name, byte for byte. The cache reads it through
    /// `CacheNamespace`; evidence envelopes and provenance record it.
    pub fn name(&self) -> &ReportedEngineName {
        &self.0
    }

    /// A reported identity for tests that exercise a consumer without a worker.
    #[cfg(test)]
    pub(crate) fn for_test(name: &str) -> Self {
        Self(ReportedEngineName::try_from(name).expect("test engine name is valid"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn versions(
        entries: &[(InferTask, Option<&str>)],
    ) -> BTreeMap<InferTask, Option<ReportedEngineName>> {
        entries
            .iter()
            .map(|(task, name)| {
                (
                    *task,
                    name.map(|name| ReportedEngineName::try_from(name).expect("valid name")),
                )
            })
            .collect()
    }

    /// The report as dispatch sees it, after FA was loaded.
    fn loaded_fa(reports: WorkerEngineReports) -> LoadedCapabilities {
        LoadedCapabilities::for_test(InferTask::Fa, reports)
    }

    #[test]
    fn a_named_engine_is_admitted_and_names_the_fa_namespace() {
        let reports = WorkerEngineReports::admit(
            &[InferTask::Fa, InferTask::Coref],
            versions(&[
                (InferTask::Fa, Some("wave2vec-fa-v1")),
                (InferTask::Coref, None),
            ]),
        )
        .expect("complete report");
        assert!(reports.supports(InferTask::Coref));
        assert_eq!(
            FaCacheNamespace::from_loaded(&loaded_fa(reports))
                .expect("reported")
                .name()
                .as_str(),
            "wave2vec-fa-v1"
        );
    }

    /// Null is admitted (the task is still supported) but can never name an
    /// engine: a post-load report that still says null is a typed refusal,
    /// never an "unknown".
    #[test]
    fn a_null_report_is_admitted_as_supported_and_names_nothing() {
        let reports = WorkerEngineReports::admit(
            &[InferTask::Asr, InferTask::Fa],
            versions(&[(InferTask::Asr, None), (InferTask::Fa, None)]),
        )
        .expect("null is a legal report");
        assert!(reports.supports(InferTask::Asr));
        assert!(reports.supports(InferTask::Fa));
        assert_eq!(
            FaCacheNamespace::from_loaded(&loaded_fa(reports)),
            Err(EngineIdentityUnavailable::UnreportedAfterLoad {
                task: InferTask::Fa
            })
        );
    }

    #[test]
    fn a_task_the_worker_does_not_advertise_names_nothing() {
        let reports = WorkerEngineReports::admit(
            &[InferTask::Morphosyntax],
            versions(&[(InferTask::Morphosyntax, None)]),
        )
        .expect("complete report");
        assert!(!reports.supports(InferTask::Fa));
        assert_eq!(
            FaCacheNamespace::from_loaded(&loaded_fa(reports)),
            Err(EngineIdentityUnavailable::NotSupported {
                task: InferTask::Fa
            })
        );
    }

    /// A report taken after another task loaded names an FA engine only by
    /// accident of what that worker had loaded before, so it is refused even
    /// when it happens to carry one.
    #[test]
    fn an_fa_identity_is_not_read_from_a_report_taken_after_another_task_loaded() {
        let reports = WorkerEngineReports::admit(
            &[InferTask::Fa, InferTask::Asr],
            versions(&[
                (InferTask::Fa, Some("wave2vec-fa-v1")),
                (InferTask::Asr, None),
            ]),
        )
        .expect("complete report");
        assert_eq!(
            FaCacheNamespace::from_loaded(&LoadedCapabilities::for_test(InferTask::Asr, reports)),
            Err(EngineIdentityUnavailable::LoadedAnotherTask {
                loaded: InferTask::Asr,
                wanted: InferTask::Fa,
            })
        );
    }

    #[test]
    fn admission_refuses_a_missing_or_unadvertised_report() {
        let missing = WorkerEngineReports::admit(&[InferTask::Morphosyntax], BTreeMap::new());
        assert_eq!(
            missing,
            Err(EngineReportAdmissionError::MissingEngineReport {
                task: InferTask::Morphosyntax
            })
        );
        assert!(
            missing
                .expect_err("refused")
                .to_string()
                .contains("engine_versions['morphosyntax'] is missing")
        );
        assert_eq!(
            WorkerEngineReports::admit(
                &[InferTask::Fa],
                versions(&[
                    (InferTask::Fa, Some("wave2vec-fa-v1")),
                    (InferTask::Utseg, None)
                ])
            ),
            Err(EngineReportAdmissionError::UnadvertisedEngineReport {
                task: InferTask::Utseg
            })
        );
    }

    /// A report names only forced alignment's engine. A name for any other
    /// task is refused, so a report never carries a name nothing reads.
    #[test]
    fn admission_refuses_an_engine_named_for_any_task_but_fa() {
        for &task in InferTask::ALL.iter().filter(|task| **task != InferTask::Fa) {
            assert_eq!(
                WorkerEngineReports::admit(&[task], versions(&[(task, Some("stanza-1.11.1"))])),
                Err(EngineReportAdmissionError::EngineNamedForNonFaTask { task })
            );
            assert!(
                WorkerEngineReports::admit(&[task], versions(&[(task, None)]))
                    .expect("null is the report for a non-FA task")
                    .supports(task)
            );
        }
    }

    /// WIRE FORMAT: `/health` returns a refusal, so its tagged JSON shape is
    /// pinned and reads back.
    #[test]
    fn a_refusal_serializes_as_a_tagged_reason_naming_the_task() {
        let refusal = EngineReportAdmissionError::MissingEngineReport {
            task: InferTask::Fa,
        };
        let json = serde_json::to_value(&refusal).expect("serializes");
        assert_eq!(
            json,
            serde_json::json!({"kind": "missing_engine_report", "task": "fa"})
        );
        assert_eq!(
            serde_json::from_value::<EngineReportAdmissionError>(json).expect("reads back"),
            refusal
        );
    }

    /// A task listed twice is one advertisement, not a missing second report.
    #[test]
    fn a_task_listed_twice_is_admitted_once() {
        let reports = WorkerEngineReports::admit(
            &[InferTask::Fa, InferTask::Fa],
            versions(&[(InferTask::Fa, Some("wave2vec-fa-v1"))]),
        )
        .expect("duplicate advertisement");
        assert_eq!(reports.tasks().collect::<Vec<_>>(), vec![InferTask::Fa]);
    }

    /// FA cache rows and FA evidence envelopes were written under the exact
    /// string the worker reported, so the namespace must be that string byte
    /// for byte: no trimming, prefixing or normalizing.
    #[test]
    fn the_fa_cache_namespace_is_the_reported_string_byte_for_byte() {
        for reported in ["wave2vec-fa-v1", "whisper-fa-v1", "wav2vec-canto-v2"] {
            let reports = WorkerEngineReports::admit(
                &[InferTask::Fa],
                versions(&[(InferTask::Fa, Some(reported))]),
            )
            .expect("complete report");
            let namespace = FaCacheNamespace::from_loaded(&loaded_fa(reports)).expect("reported");
            assert_eq!(namespace.name().as_str(), reported);
        }
    }
}
