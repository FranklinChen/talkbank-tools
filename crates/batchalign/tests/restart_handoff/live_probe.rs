//! HTTP cancellation while a restarted capability load keeps advancing.
//! The peer cannot infer or download; the real pool, probe and runner execute.

use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::Duration;

use batchalign::api::{JobInfo, JobStatus, LanguageCode3, LanguageSpec};
use batchalign::config::{RuntimeLayout, ServerConfig};
use batchalign::options::AsrEngineName;
use batchalign::worker::handle::WorkerRuntimeConfig;
use batchalign::worker::pool::PoolConfig;

/// Own the isolated server, its workers and every file the peer can touch.
struct ProbeServer {
    root: tempfile::TempDir,
    state: Arc<batchalign::state::AppState>,
    server: tokio::task::JoinHandle<()>,
    client: reqwest::Client,
    url: String,
}

/// Restart is offered only after the first run has actually failed.
struct FailedProbe {
    server: ProbeServer,
    job_id: String,
}

/// Cancellation is tested only after the restarted worker advances its load.
struct LoadingProbe {
    server: ProbeServer,
    job_id: String,
}

/// A completed observation owns cleanup; it cannot restart or dispatch again.
struct CancelledProbe {
    server: ProbeServer,
    outcome: Option<JobInfo>,
    spawns_before: String,
    spawns_after: String,
}

/// Cancellation boundaries before file dispatch; none runs inference.
#[derive(Clone, Copy, Debug)]
enum LoadBoundary {
    Capability,
    WorkerWarmup,
    JobSlot,
}

impl LoadBoundary {
    fn peer_mode(self) -> &'static str {
        match self {
            Self::Capability | Self::JobSlot => "live",
            Self::WorkerWarmup => "warmup",
        }
    }
}

impl ProbeServer {
    async fn start() -> Self {
        let root = tempfile::tempdir().expect("probe root");
        let state_dir = root.path().join("state");
        std::fs::create_dir(&state_dir).expect("state dir");
        std::fs::write(state_dir.join("mode"), "refuse").expect("initial load policy");
        let executable = root.path().join("protocol-peer");
        // Use the suite's admitted interpreter, without a new environment
        // contract. The peer itself imports only the standard library.
        let python = crate::prerequisite::PythonWorker::require().interpreter();
        std::fs::write(
            &executable,
            format!("#!{python}\n{}", include_str!("live_probe_worker.py")),
        )
        .expect("write peer");
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))
            .expect("executable peer");
        // The planner and the worker must describe the same CPU host. A GPU
        // planner would cap this job at one and never exercise pre-scaling.
        let config = ServerConfig {
            force_cpu: Some(true),
            max_concurrent_jobs: Some(1),
            max_workers_per_job: Some(2),
            // Exercise the supported tier override on the same declared
            // 16GB host. Small/Medium's one-worker policy cannot enter warmup.
            // Physical readings, leases and all admission gates stay active.
            memory_tier: Some(batchalign::types::runtime::MemoryTierKind::Large),
            ..ServerConfig::default()
        };
        let host = Arc::new(batchalign::host_facts::HostSnapshot::for_tests_in(
            &config,
            state_dir.join("host-memory.json"),
        ));
        let pool = PoolConfig {
            python_path: executable.display().to_string(),
            ensure_task_stall: batchalign::worker::LoadingStall::new(
                batchalign::api::PositiveSeconds::literal::<1>(),
            ),
            startup_stall: batchalign::worker::LoadingStall::new(
                batchalign::api::PositiveSeconds::literal::<1>(),
            ),
            health_check_interval_s: batchalign::api::PositiveSeconds::literal::<600>(),
            worker_registry_path: Some(state_dir.join("workers.json")),
            runtime: WorkerRuntimeConfig {
                force_cpu: true,
                host: host.runtime_host(),
                state_dir: Some(state_dir.clone()),
                ..WorkerRuntimeConfig::for_tests()
            },
            ..PoolConfig::for_tests()
        };
        // test_echo stays false: echo routing bypasses the capability probe.
        let workers = batchalign::worker_setup::prepare_workers(
            pool,
            batchalign::worker_setup::RegistryDiscovery::Ignore,
        )
        .await
        .expect("model-free worker pool");
        let (router, state) = batchalign::create_app_with_prepared_workers(
            config,
            host,
            RuntimeLayout::from_state_dir(state_dir),
            batchalign::AppStorageOverrides {
                jobs_dir: Some(root.path().join("jobs").display().to_string()),
                db_dir: Some(root.path().join("db")),
                cache_dir: Some(root.path().join("cache")),
            },
            Some("live-probe-test".into()),
            workers,
            Arc::new(batchalign::clock::SystemClock),
        )
        .await
        .expect("probe app");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let url = format!("http://{}", listener.local_addr().expect("address"));
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .expect("serve probe app");
        });
        Self {
            root,
            state,
            server,
            client: reqwest::Client::new(),
            url,
        }
    }

    fn control(&self, name: &str) -> std::path::PathBuf {
        self.root.path().join("state").join(name)
    }

    async fn info(&self, id: &str) -> JobInfo {
        super::get_job(&self.client, &self.url, id).await
    }

    async fn fail_first_run(self) -> FailedProbe {
        let media = crate::common::prepare_audio_fixtures(self.root.path());
        let second = self.root.path().join("second.mp3");
        std::fs::copy(&media.audio, &second).expect("second synthetic input");
        let mut request = super::echo_submission();
        request.files.clear();
        request.lang = LanguageSpec::Resolved(LanguageCode3::zho());
        request.paths_mode = true;
        request.source_paths = vec![
            media.audio.display().to_string().as_str().into(),
            second.display().to_string().as_str().into(),
        ];
        request.output_paths = ["out.cha", "second.cha"]
            .map(|p| {
                self.root
                    .path()
                    .join(p)
                    .display()
                    .to_string()
                    .as_str()
                    .into()
            })
            .into();
        if let batchalign::options::CommandOptions::Transcribe(options) = &mut request.options {
            options.asr_engine = AsrEngineName::HkFunaudio;
            options.common.engine_overrides = serde_json::from_value(serde_json::json!({
                "funaudio_model": "paraformer-zh"
            }))
            .expect("the admitted Paraformer option");
        } else {
            unreachable!("transcribe fixture");
        }
        let info: JobInfo = self
            .client
            .post(format!("{}/jobs", self.url))
            .json(&request)
            .send()
            .await
            .expect("submit")
            .error_for_status()
            .expect("admitted")
            .json()
            .await
            .expect("job");
        let id = info.job_id.to_string();
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let info = self.info(&id).await;
                if info.status == JobStatus::Failed {
                    break;
                }
                assert!(!info.status.is_terminal(), "unexpected first run: {info:?}");
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("first run must fail its load");
        FailedProbe {
            server: self,
            job_id: id,
        }
    }
}

impl FailedProbe {
    async fn restart_into_live_load(self, boundary: LoadBoundary) -> LoadingProbe {
        std::fs::write(self.server.control("mode"), boundary.peer_mode())
            .expect("live load policy");
        self.server
            .client
            .post(format!("{}/jobs/{}/restart", self.server.url, self.job_id))
            .send()
            .await
            .expect("restart")
            .error_for_status()
            .expect("restart accepted");
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let ticks = std::fs::read_to_string(self.server.control("load-ticks"));
                // At least 1.2s of progress: beyond this fixture's 1s stall
                // window, so the load is live rather than a silent timeout.
                if ticks.is_ok_and(|t| t.parse::<u32>().is_ok_and(|n| n >= 25)) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("restarted probe must keep advancing");
        let info = self.server.info(&self.job_id).await;
        assert!(
            info.file_statuses
                .iter()
                .all(|f| f.status == batchalign::api::FileStatusKind::Queued),
            "the target must be before file dispatch, not a file's own load: {boundary:?}: {info:?}"
        );
        let job_id = match boundary {
            LoadBoundary::Capability | LoadBoundary::WorkerWarmup => self.job_id,
            LoadBoundary::JobSlot => {
                // The first job owns the only slot and stays in its live
                // capability load. A distinct content-mode job must wait
                // before it can prepare any worker or dispatch any file.
                let waiting: JobInfo = self
                    .server
                    .client
                    .post(format!("{}/jobs", self.server.url))
                    .json(&super::echo_submission())
                    .send()
                    .await
                    .expect("submit waiting job")
                    .error_for_status()
                    .expect("waiting job admitted")
                    .json()
                    .await
                    .expect("waiting job");
                let id = waiting.job_id.to_string();
                tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        let info = self.server.info(&id).await;
                        if info.active_lease.is_some() {
                            assert_eq!(info.status, JobStatus::Queued);
                            assert!(info.num_workers.is_none());
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                })
                .await
                .expect("waiting runner must own its claim");
                id
            }
        };
        LoadingProbe {
            server: self.server,
            job_id,
        }
    }
}

impl LoadingProbe {
    async fn cancel(self) -> CancelledProbe {
        let spawns_before =
            std::fs::read_to_string(self.server.control("spawns")).expect("spawn audit");
        self.server
            .client
            .post(format!("{}/jobs/{}/cancel", self.server.url, self.job_id))
            .send()
            .await
            .expect("cancel")
            .error_for_status()
            .expect("cancel accepted");
        let outcome = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let info = self.server.info(&self.job_id).await;
                if info.status == JobStatus::Cancelled
                    && info.file_statuses.iter().all(|f| f.status.is_terminal())
                {
                    break info;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .ok();
        tokio::time::sleep(Duration::from_millis(250)).await;
        let spawns_after =
            std::fs::read_to_string(self.server.control("spawns")).expect("spawn audit");
        CancelledProbe {
            server: self.server,
            outcome,
            spawns_before,
            spawns_after,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_a_restarted_live_capability_probe_stops_without_respawn() {
    check_cancellation(LoadBoundary::Capability).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_a_restarted_worker_warmup_stops_without_respawn() {
    check_cancellation(LoadBoundary::WorkerWarmup).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_while_waiting_for_a_job_slot_closes_files_without_capacity() {
    check_cancellation(LoadBoundary::JobSlot).await;
}

async fn check_cancellation(boundary: LoadBoundary) {
    let stopped = ProbeServer::start()
        .await
        .fail_first_run()
        .await
        .restart_into_live_load(boundary)
        .await
        .cancel()
        .await;
    let CancelledProbe {
        server,
        outcome,
        spawns_before,
        spawns_after,
    } = stopped;
    crate::common::cleanup_active_session(server.state, server.root, server.server, "live probe")
        .await;
    assert!(
        outcome.is_some(),
        "cancelled runner left files open during {boundary:?}"
    );
    assert_eq!(
        spawns_before, spawns_after,
        "a cancelled probe started another worker"
    );
}
