//! Consumer fixtures enter through the production preparation edge.

use std::sync::Arc;

use super::{CoreferenceText, PreparedTextJob};
use crate::api::{NumWorkers, ReleasedCommand};
use crate::execution::MorphotagRuntimeOptions;
use crate::options::{CommandOptions, CommonOptions, CorefOptions};
use crate::runner::DispatchHostContext;
use crate::runner::util::test_sink::RecordingSink;
use crate::store::test_support::{
    PathsFixture, paths_inputs, queued_morphotag_job_fixture, with_pending,
};
use crate::store::{ContentInput, InputKind, RunnerJobSnapshot};
use crate::text_batch::TextBatchFileResult;

const CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";

/// The selected producer case owns the command and option-family association.
enum SourceCase {
    Incremental(PriorObservation),
    Coreference,
}

enum PriorObservation {
    Absent,
    Readable,
    Missing,
    InvalidUtf8,
}

struct Sources {
    root: PathsFixture,
    job: RunnerJobSnapshot,
    sink: Arc<RecordingSink>,
}

impl Sources {
    fn observe(case: SourceCase) -> Self {
        let root = PathsFixture::create();
        let mut job = queued_morphotag_job_fixture(ContentInput {
            display: "a.cha".into(),
            kind: InputKind::Chat,
        })
        .to_runner_snapshot()
        .expect("admitted source fixture");
        let mut first = root.input("a.cha", InputKind::Chat);
        std::fs::write(root.path().join("a.cha"), CHAT).unwrap();
        let inputs = match case {
            SourceCase::Incremental(prior) => {
                match prior {
                    PriorObservation::Absent => {}
                    PriorObservation::Readable => {
                        std::fs::write(root.path().join("prior.cha"), CHAT).unwrap();
                        first.before = Some(root.source("prior.cha"));
                    }
                    PriorObservation::Missing => first.before = Some(root.source("prior.cha")),
                    PriorObservation::InvalidUtf8 => {
                        std::fs::write(root.path().join("prior.cha"), [0xff]).unwrap();
                        first.before = Some(root.source("prior.cha"));
                    }
                }
                vec![first]
            }
            SourceCase::Coreference => {
                job.dispatch.command = ReleasedCommand::Coref;
                job.dispatch.options = CommandOptions::Coref(CorefOptions {
                    common: CommonOptions::default(),
                });
                std::fs::write(root.path().join("b.cha"), CHAT).unwrap();
                vec![first, root.input("b.cha", InputKind::Chat)]
            }
        };
        job.filesystem.inputs = paths_inputs(inputs);
        job.filesystem.staging_dir = batchalign_types::paths::ServerPath::new(root.path());
        Self {
            root,
            job: with_pending(job),
            sink: Arc::new(RecordingSink::default()),
        }
    }
}

/// Guards: An absent prior permits fresh analysis, readable prior bytes survive preparation, and unreadable declared prior bytes refuse that file.
/// Kind: boundary
/// Not a type: The morphotag preparation state restricts prior reading and consuming ownership excludes re-preparation. Existence and UTF-8 validity of external files can change after path admission and require a real read observation.
/// State: A private directory holds current CHAT and the selected prior bytes. The shared recording sink observes the exact failure and absence of completion. No registry, worker or ledger lease is needed.
/// Concurrency: Each row owns its admitted root, files and event sink. File creation completes before the consuming read begins.
#[tokio::test]
async fn prior_observations_survive_preparation() {
    for prior in [
        PriorObservation::Absent,
        PriorObservation::Readable,
        PriorObservation::Missing,
        PriorObservation::InvalidUtf8,
    ] {
        let expected = match prior {
            PriorObservation::Absent => Ok(None),
            PriorObservation::Readable => Ok(Some(CHAT)),
            PriorObservation::Missing | PriorObservation::InvalidUtf8 => Err(()),
        };
        let sources = Sources::observe(SourceCase::Incremental(prior));
        let host = DispatchHostContext::for_recorded_effects(Arc::clone(&sources.sink));
        let mut files =
            PreparedTextJob::<MorphotagRuntimeOptions>::for_test(sources.job, host, NumWorkers(1));
        let read = files.next_file().await;
        match expected {
            Ok(before) => {
                let file = read.expect("read admitted input");
                assert_eq!(file.input().chat_text, CHAT);
                assert_eq!(file.before(), before);
                assert!(sources.sink.errors().is_empty());
            }
            Err(()) => {
                assert!(read.is_none(), "declared prior failure must stop this file");
                let errors = sources.sink.errors();
                assert_eq!(errors.len(), 1);
                assert_eq!(errors[0].filename, "a.cha");
                assert_eq!(
                    errors[0].category,
                    crate::scheduling::FailureCategory::InputMissing
                );
                assert!(
                    errors[0]
                        .error
                        .starts_with("Failed to read declared prior input:")
                );
            }
        }
        assert!(files.next_file().await.is_none());
        assert!(sources.sink.completed_files().is_empty());
        assert!(!sources.root.path().join("output/a.cha").exists());
    }
}

/// Guards: Gateway results must correspond exactly to prepared sources before publication, including reordered batches and refused missing, repeated or foreign names.
/// Kind: boundary
/// Not a type: Private publication owners prevent duplicate writes or sink substitution. The gateway still returns externally computed name-tagged results whose actual cardinality and correspondence must be admitted once.
/// State: Two temporary CHAT files exercise real reading and atomic publication. The shared recording sink observes completion or source-bound errors. Fixture outputs carry actual post-validation proofs without starting models.
/// Concurrency: Every row owns its directory and recorder. All correspondence admission completes before the first output effect, with no shared paths or external processes.
#[tokio::test]
async fn result_correspondence_precedes_publication() {
    enum Response {
        ReorderedBatch,
        MissingBatch,
        RepeatedBatch,
        ForeignBatch,
        MissingSingle,
        RepeatedSingle,
        ForeignSingle,
    }
    for response in [
        Response::ReorderedBatch,
        Response::MissingBatch,
        Response::RepeatedBatch,
        Response::ForeignBatch,
        Response::MissingSingle,
        Response::RepeatedSingle,
        Response::ForeignSingle,
    ] {
        let sources = Sources::observe(SourceCase::Coreference);
        let host = DispatchHostContext::for_recorded_effects(Arc::clone(&sources.sink));
        let mut files =
            PreparedTextJob::<CoreferenceText>::for_test(sources.job, host, NumWorkers(1));
        let output = |name: &str| {
            TextBatchFileResult::ok(
                name,
                crate::pipeline::post_validate::PostValidated::for_test(
                    CHAT,
                    ReleasedCommand::Coref,
                ),
            )
        };
        let success = matches!(response, Response::ReorderedBatch);
        match response {
            Response::ReorderedBatch => {
                files
                    .read_batch()
                    .await
                    .complete(vec![output("b.cha"), output("a.cha")])
                    .await
            }
            Response::MissingBatch => {
                files
                    .read_batch()
                    .await
                    .complete(vec![output("a.cha")])
                    .await
            }
            Response::RepeatedBatch => {
                files
                    .read_batch()
                    .await
                    .complete(vec![output("a.cha"), output("a.cha")])
                    .await
            }
            Response::ForeignBatch => {
                files
                    .read_batch()
                    .await
                    .complete(vec![output("a.cha"), output("other.cha")])
                    .await
            }
            Response::MissingSingle => files.next_file().await.unwrap().complete(vec![]).await,
            Response::RepeatedSingle => {
                files
                    .next_file()
                    .await
                    .unwrap()
                    .complete(vec![output("a.cha"), output("a.cha")])
                    .await
            }
            Response::ForeignSingle => {
                files
                    .next_file()
                    .await
                    .unwrap()
                    .complete(vec![output("other.cha")])
                    .await
            }
        }
        if success {
            assert!(sources.sink.errors().is_empty());
            let mut completed = sources.sink.completed_files();
            completed.sort();
            assert_eq!(completed, ["a.cha", "b.cha"]);
            for name in ["a.cha", "b.cha"] {
                assert_eq!(
                    std::fs::read_to_string(sources.root.path().join("output").join(name)).unwrap(),
                    CHAT
                );
                assert_eq!(
                    std::fs::read_to_string(sources.root.path().join(name)).unwrap(),
                    CHAT
                );
            }
        } else {
            assert!(sources.sink.completed_files().is_empty());
            assert!(!sources.sink.errors().is_empty());
            assert!(!sources.root.path().join("output/a.cha").exists());
            assert!(!sources.root.path().join("output/b.cha").exists());
        }
    }
}
