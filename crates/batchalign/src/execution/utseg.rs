//! Per-file segmentation uses the runner's admitted worker capacity.

use super::text_io::{PreparedTextJob, SegmentationText};
use super::worker_gateway::UtsegGateway;
use crate::runner::util::{FileRunTracker, FileStage};
use crate::scheduling::WorkUnitKind;
use crate::utseg_route::ServableUtsegRoute;
use std::sync::Arc;

pub(crate) async fn dispatch_utseg_job(
    files: PreparedTextJob<SegmentationText>,
    gateway: Arc<dyn UtsegGateway>,
    route: &ServableUtsegRoute,
) -> Result<(), crate::error::ServerError> {
    let route = route.clone();
    files
        .run_files(move |file, _mode| {
            let gateway = Arc::clone(&gateway);
            let route = route.clone();
            async move {
                FileRunTracker::new(file.host().sink().as_ref(), file.input().filename.as_ref())
                    .begin_first_attempt(WorkUnitKind::BatchInfer, FileStage::Segmenting)
                    .await;
                let results = crate::runner::util::observing_file_waits(
                    file.host().sink(),
                    [file.input().filename.to_string()],
                    gateway.utseg_file(file.input(), &route, file.cancellation()),
                )
                .await;
                file.complete(results).await;
            }
        })
        .await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{
        CorrelationId, DisplayPath, JobId, LanguageCode3, LanguageSpec, NumSpeakers,
        ReleasedCommand,
    };
    use crate::options::{CommandOptions, CommonOptions, UtsegOptions};
    use crate::runner::DispatchHostContext;
    use crate::store::RunnerJobSnapshot;
    use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};
    use crate::utseg_route::UtsegRoute;

    struct FakeUtsegGateway {
        events: Mutex<Vec<SegmentationEvent>>,
        started: tokio::sync::mpsc::UnboundedSender<()>,
        release: tokio::sync::Semaphore,
    }

    enum SegmentationEvent {
        Started(ObservedSegmentation),
        Finished,
    }

    #[derive(Debug, PartialEq)]
    struct ObservedSegmentation {
        source: DisplayPath,
        route: ServableUtsegRoute,
    }

    #[async_trait]
    impl UtsegGateway for FakeUtsegGateway {
        async fn utseg_file(
            &self,
            file: &TextBatchFileInput,
            route: &ServableUtsegRoute,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> TextBatchFileResults {
            self.events
                .lock()
                .unwrap()
                .push(SegmentationEvent::Started(ObservedSegmentation {
                    source: file.filename.clone(),
                    route: route.clone(),
                }));
            self.started.send(()).unwrap();
            self.release.acquire().await.unwrap().forget();
            self.events
                .lock()
                .unwrap()
                .push(SegmentationEvent::Finished);
            vec![TextBatchFileResult::ok(
                file.filename.clone(),
                crate::pipeline::post_validate::PostValidated::for_test(
                    file.chat_text.clone(),
                    crate::api::ReleasedCommand::Utseg,
                ),
            )]
        }
    }

    const SOURCES: [&str; 6] = ["a.cha", "b.cha", "c.cha", "d.cha", "e.cha", "f.cha"];

    fn utseg_snapshot(staging_dir: &std::path::Path, language: LanguageCode3) -> RunnerJobSnapshot {
        let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
@ID:\teng|test|PAR|||||Participant|||\n*PAR:\tF B I .\n@End\n";
        let input_dir = staging_dir.join("input");
        std::fs::create_dir_all(&input_dir).unwrap();
        for name in SOURCES {
            std::fs::write(input_dir.join(name), text).unwrap();
        }
        crate::store::test_support::with_pending(RunnerJobSnapshot {
            identity: crate::store::RunnerJobIdentity {
                job_id: JobId::from("job-utseg"),
                correlation_id: CorrelationId::from("corr-utseg"),
            },
            dispatch: crate::store::RunnerDispatchConfig {
                command: ReleasedCommand::Utseg,
                lang: LanguageSpec::Resolved(language),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Utseg(UtsegOptions {
                    common: CommonOptions::default(),
                    utseg_fallback: false.into(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: crate::store::RunnerFilesystemConfig {
                inputs: crate::store::JobInputs::Content(
                    SOURCES
                        .into_iter()
                        .map(|name| crate::store::ContentInput {
                            display: name.into(),
                            kind: crate::store::InputKind::Chat,
                        })
                        .collect(),
                ),
                staging_dir: batchalign_types::paths::ServerPath::new(
                    staging_dir.display().to_string(),
                ),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new(
                    staging_dir.display().to_string(),
                ),
            },
            cancel_token: CancellationToken::new(),
            pending_files: Vec::new(),
        })
    }

    fn host() -> DispatchHostContext {
        DispatchHostContext::for_test_effects()
    }

    /// A registry serving English and Spanish with every processor the
    /// Stanza route loads.
    fn registry() -> crate::stanza_registry::StanzaRegistry {
        use crate::stanza_registry::StanzaProcessor::{Constituency, Pos, Tokenize};
        crate::stanza_registry::StanzaRegistry::serving(&[
            ("eng", &[Tokenize, Pos, Constituency]),
            ("spa", &[Tokenize, Pos, Constituency]),
        ])
    }

    enum FallbackCase {
        English,
        Spanish,
    }

    impl FallbackCase {
        fn language(&self) -> LanguageCode3 {
            match self {
                Self::English => LanguageCode3::eng(),
                Self::Spanish => LanguageCode3::spa(),
            }
        }
    }

    /// Guards: Each source reaches segmentation once with its admitted route, and concurrent work follows grants of one and six files.
    /// Kind: contract
    /// Not a type: Consuming file owners exclude repeated preparation and completion. Actual gateway entry, source propagation, adaptive scheduling and persisted CHAT bytes remain asynchronous effects.
    /// State: Six temporary CHAT files exercise real preparation and publication. A recording gateway holds inference at explicit events. The host snapshot is the standard declared laptop, with no memory lease or model spawn in this dispatcher contract.
    /// Concurrency: Each case owns its directory, event mutex, channel and release semaphore. The controller waits for the granted initial wave before releasing it. A timeout is a deadlock watchdog.
    #[tokio::test]
    async fn utseg_admitted_fallback_reaches_each_gateway_call() {
        for (case, grant) in [(FallbackCase::English, 1), (FallbackCase::Spanish, 6)] {
            let temp = tempfile::tempdir().unwrap();
            let host = host();
            let (started, mut arrivals) = tokio::sync::mpsc::unbounded_channel();
            let gateway = Arc::new(FakeUtsegGateway {
                events: Mutex::new(Vec::new()),
                started,
                release: tokio::sync::Semaphore::new(0),
            });
            let language = case.language();
            let job = utseg_snapshot(temp.path(), language.clone());
            let route =
                UtsegRoute::resolve(&language, crate::params::UtsegFallbackPolicy::AllowStanza)
                    .expect("model or authorized fallback")
                    .admit(&Ok(registry().shared()))
                    .expect("registry serves the route");
            let task_gateway = Arc::clone(&gateway);
            let task_route = route.clone();
            let task = tokio::spawn(async move {
                dispatch_utseg_job(
                    PreparedTextJob::for_test(job, host, crate::api::NumWorkers(grant)),
                    task_gateway,
                    &task_route,
                )
                .await
            });
            for _ in 0..grant {
                tokio::time::timeout(std::time::Duration::from_secs(5), arrivals.recv())
                    .await
                    .expect("granted initial wave reaches the gateway")
                    .expect("arrival");
            }
            gateway.release.add_permits(SOURCES.len());
            task.await
                .expect("joined file dispatch")
                .expect("admitted dispatch");
            let events = gateway.events.lock().unwrap();
            let mut active = 0;
            let mut peak = 0;
            let mut requests = Vec::new();
            for event in &*events {
                match event {
                    SegmentationEvent::Started(request) => {
                        active += 1;
                        peak = peak.max(active);
                        requests.push(request);
                    }
                    SegmentationEvent::Finished => active -= 1,
                }
            }
            assert_eq!(active, 0);
            assert_eq!(peak, grant);
            requests.sort_by(|left, right| left.source.cmp(&right.source));
            assert_eq!(requests.len(), SOURCES.len());
            for (request, name) in requests.into_iter().zip(SOURCES) {
                assert_eq!(request.source.as_ref(), name);
                assert_eq!(request.route, route);
                assert_eq!(
                    std::fs::read_to_string(temp.path().join("output").join(name)).unwrap(),
                    std::fs::read_to_string(temp.path().join("input").join(name)).unwrap()
                );
            }
        }
    }
}
