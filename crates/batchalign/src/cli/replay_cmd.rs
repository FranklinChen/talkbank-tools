//! `batchalign3 replay`: replay a captured failed IPC request.
//!
//! Takes a `failed_ipc_*.json` dump from `~/.batchalign3/debug/` and sends
//! the exact request to a fresh worker, reporting the response. The worker is
//! the one the dump records (`worker_launch`: profile or task, language,
//! engine overrides, device flags, bootstrap mode), spawned through the
//! pool's own handle and command builder, and asked to load the request's
//! task first when it loads on demand, as the pool does before it dispatches.

use crate::cli::args::ReplayArgs;
use crate::cli::error::CliError;
use crate::cli::python::resolve_python_executable;
use crate::types::worker_v2::ExecuteRequestV2;
use crate::worker::handle::{WorkerHandle, WorkerLaunch};

/// Parsed dump file structure.
#[derive(Debug, serde::Deserialize)]
struct FailedIpcDump {
    worker_label: String,
    /// The worker the failing request ran on. A dump written before the
    /// record existed has none and is refused: it cannot say which engines
    /// its worker ran, and a worker booted for the request's task with
    /// default engines is not a replay of it.
    worker_launch: Option<WorkerLaunch>,
    request: String,
    error_message: String,
}

/// The `execute_v2` request line the dump captured, read as the protocol's
/// own request type.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum CapturedRequest {
    ExecuteV2 { request: ExecuteRequestV2 },
}

/// The worker a dump's request is replayed on: the recorded launch, under
/// the operator's `--lang` when given.
fn replay_launch(dump: &FailedIpcDump, lang: Option<&str>) -> Result<WorkerLaunch, CliError> {
    let Some(launch) = dump.worker_launch.clone() else {
        return Err(CliError::InvalidArgument(format!(
            "the dump records no worker launch (worker label {:?}): it was written by a \
             build that did not record which engines and device its worker ran, so no \
             worker can be said to replay it",
            dump.worker_label
        )));
    };
    match lang {
        None => Ok(launch),
        // Parsed as the worker's own language type, not just present: a
        // misspelled code would boot a worker for a language that is not the
        // failing one's.
        Some(lang) => crate::api::WorkerLanguage::parse_untrusted(lang)
            .map(|lang| launch.with_lang(lang))
            .map_err(|error| CliError::InvalidArgument(format!("replay language: {error}"))),
    }
}

/// Read the captured request.
fn captured_request(dump: &FailedIpcDump) -> Result<ExecuteRequestV2, CliError> {
    let CapturedRequest::ExecuteV2 { request } = serde_json::from_str(&dump.request)
        .map_err(|e| CliError::InvalidArgument(format!("Cannot parse request from dump: {e}")))?;
    Ok(request)
}

/// Run the replay command.
pub async fn run(args: &ReplayArgs) -> Result<(), CliError> {
    let dump_text = std::fs::read_to_string(&args.dump_file).map_err(|e| {
        CliError::InvalidArgument(format!(
            "Cannot read dump file {}: {e}",
            args.dump_file.display()
        ))
    })?;
    let dump: FailedIpcDump = serde_json::from_str(&dump_text)
        .map_err(|e| CliError::InvalidArgument(format!("Cannot parse dump file: {e}")))?;

    eprintln!(
        "Replaying failed IPC request from {}",
        args.dump_file.display()
    );
    eprintln!("  Worker label: {}", dump.worker_label);
    eprintln!("  Original error: {}", dump.error_message);

    let launch = replay_launch(&dump, args.lang.as_deref())?;
    let request = captured_request(&dump)?;
    let python = args
        .python
        .clone()
        .unwrap_or_else(resolve_python_executable);
    eprintln!("  Lang: {}", launch.lang());
    eprintln!("  Python: {python}");
    eprintln!();

    let loads_on_demand = launch.loads_on_demand();
    // The dump records the launch, not the host; size it as a server on
    // this host would, from the operator's config.
    let server_config =
        crate::config::load_config_from_layout(&crate::config::RuntimeLayout::from_env(), None)?;
    let memory_tier = crate::host_facts::HostSnapshot::read(&server_config)?.tier();
    let config = launch.into_config(python, memory_tier);
    let ensure_task_timeout = config.ready_timeout_s;
    let mut worker = WorkerHandle::spawn(config)
        .await
        .map_err(|e| CliError::InvalidArgument(format!("Cannot start the worker: {e}")))?;
    if loads_on_demand {
        let (task, overrides) = crate::worker::pool::ensure_task_params(&request)
            .map_err(|e| CliError::InvalidArgument(format!("Cannot load the task: {e}")))?;
        worker
            .ensure_task(task, overrides.as_ref(), ensure_task_timeout)
            .await
            .map_err(|e| CliError::InvalidArgument(format!("Cannot load the task: {e}")))?;
    }

    eprintln!("Sending original request...");
    let outcome = worker.execute_v2(&request).await;
    if let Err(error) = worker.shutdown().await {
        eprintln!("(worker shutdown: {error})");
    }
    let response =
        outcome.map_err(|e| CliError::InvalidArgument(format!("Worker returned error: {e}")))?;
    let rendered = serde_json::to_string_pretty(&response)
        .map_err(|e| CliError::InvalidArgument(format!("Cannot render the response: {e}")))?;
    println!("{rendered}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FailedIpcDump, captured_request, replay_launch};
    use crate::worker::handle::{WorkerConfig, WorkerLaunch, WorkerRuntimeConfig};

    /// A dump as the pool writes one, for a Cantonese FA worker running a
    /// non-default engine on CPU.
    fn dump_of(config: &WorkerConfig) -> FailedIpcDump {
        let json = serde_json::json!({
            "worker_label": config.bootstrap_label(),
            "worker_launch": WorkerLaunch::of(config),
            "request": "{\"op\":\"execute_v2\",\"request\":{\"request_id\":\"r\",\"task\":\"utseg\",\"payload\":{\"kind\":\"utseg\",\"lang\":\"yue\",\"payload_ref_id\":\"a\",\"item_count\":0,\"segmenter\":\"stanza_constituency\"},\"attachments\":[]}}",
            "error_message": "boom",
        });
        serde_json::from_value(json).expect("a dump the pool writes is readable")
    }

    /// RED FIRST: replay booted `--task <request task> --lang <label>` and
    /// nothing else, so a dump from a worker with engine overrides (Qwen3
    /// ASR, Cantonese FA), a profile or device flags replayed on a
    /// default-engine worker. The dump now records the launch, and the
    /// replay rebuilds the very config the pool spawned.
    #[test]
    fn a_replay_rebuilds_the_worker_the_dump_records() {
        let mut runtime = WorkerRuntimeConfig::local(crate::worker::handle::test_memory_tier())
            .with_bootstrap_mode(crate::worker::WorkerBootstrapMode::LazyProfile);
        runtime.force_cpu = true;
        let original = WorkerConfig {
            profile: crate::worker::WorkerProfile::Gpu,
            task: None,
            lang: crate::api::WorkerLanguage::parse_untrusted("yue").expect("yue"),
            engine_overrides: r#"{"fa":"wav2vec_canto"}"#.to_owned(),
            runtime,
            ..WorkerConfig::for_tests()
        };
        let dump = dump_of(&original);

        let replayed = replay_launch(&dump, None)
            .expect("the dump records its launch")
            .into_config(
                "python3".to_owned(),
                crate::worker::handle::test_memory_tier(),
            );
        assert_eq!(WorkerLaunch::of(&replayed), WorkerLaunch::of(&original));
        assert!(WorkerLaunch::of(&original).loads_on_demand());
        assert_eq!(
            captured_request(&dump)
                .expect("the captured line")
                .request_id
                .to_string(),
            "r"
        );

        let french = replay_launch(&dump, Some("fra")).expect("operator names fra");
        assert_eq!(french.lang().as_worker_arg(), "fra");
        assert!(replay_launch(&dump, Some("french")).is_err());
    }

    /// A dump written before the launch record cannot say which worker ran
    /// it, and is refused rather than replayed on an invented one.
    #[test]
    fn a_dump_without_a_launch_record_is_refused() {
        let dump: FailedIpcDump = serde_json::from_value(serde_json::json!({
            "worker_label": "profile:gpu:spa",
            "request": "{}",
            "error_message": "boom",
        }))
        .expect("an older dump still parses");
        assert!(replay_launch(&dump, Some("spa")).is_err());
    }
}
