//! Runner-facing projections over the job store.

use crate::api::JobId;

use super::super::JobStore;
use super::super::job::RunnerJobSnapshot;

impl JobStore {
    /// Return the immutable runner-facing snapshot for one job.
    pub async fn runner_snapshot(&self, job_id: &JobId) -> Option<RunnerJobSnapshot> {
        self.registry.runner_snapshot(job_id).await
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::broadcast;

    use crate::ws::BROADCAST_CAPACITY;

    use super::super::tests::{make_job, test_config};
    use super::*;

    /// Runner snapshots include only unfinished files and clone stable config.
    /// Guards: Runner projection keeps stable dispatch metadata and excludes the completed file.
    /// Kind: contract
    /// Not a type: A claimed run cannot establish which files the actor snapshot projects after a later file-completion event.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect. File completion uses the actual admitted binding before snapshot inspection.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn runner_snapshot_filters_terminal_files() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into(), "b.cha".into()],
        );
        store.submit(job).await.unwrap();
        let id = JobId::from("j1");
        let claim = crate::store::queries::tests::claim_registry_runner(&store, &id).await;
        store
            .mark_file_done(
                claim.binding(),
                "a.cha",
                store.event_time(),
                crate::store::FileCompletion::Clean(crate::store::CompletedFileOutput {
                    exclusions: Vec::new(),
                    filename: "a.cha".into(),
                    content_type: crate::api::ContentType::Chat,
                    stamp: crate::api::FileStampOutcome::Unrecorded,
                }),
            )
            .await;

        let snapshot = store
            .runner_snapshot(&JobId::from("j1"))
            .await
            .expect("runner snapshot");

        assert_eq!(snapshot.dispatch.command.as_ref(), "morphotag");
        assert_eq!(
            snapshot
                .pending_files
                .iter()
                .map(|file| file.filename().as_ref())
                .collect::<Vec<_>>(),
            ["b.cha"]
        );
        store.release_run(claim).await;
    }
}
