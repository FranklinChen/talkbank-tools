//! Native launch bytes and sequential control admission.

use super::spawn::build_worker_command;
use super::{WorkerConfig, WorkerRuntimeConfig};
use crate::api::{LanguageCode3, WorkerLanguage};
use crate::worker::provider_credentials::HkAsrCredentialSources;
use crate::worker::{InferTask, WorkerProfile};
use std::collections::BTreeMap;

fn command_args(config: &WorkerConfig) -> Vec<String> {
    build_worker_command(config)
        .get_args()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect()
}
fn command_envs(config: &WorkerConfig) -> BTreeMap<String, String> {
    build_worker_command(config)
        .get_envs()
        .filter_map(|(key, value)| {
            value.map(|value| {
                (
                    key.to_string_lossy().into_owned(),
                    value.to_string_lossy().into_owned(),
                )
            })
        })
        .collect()
}
fn argument<'a>(arguments: &'a [String], flag: &str) -> &'a str {
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .expect("serialized launch flag")[1]
        .as_str()
}

/// Guards: the actual native command emits the selected target, supervisor,
/// device hints and serving/thread contract for sequential and concurrent workers.
/// Kind: wire
/// Not a type: Python receives actual argument spellings and values, and their
/// serialization must agree with the already typed target and serving choices.
/// State: local Command values inspect only the producer's argv without spawning.
/// Concurrency: none
#[test]
fn worker_launch_bytes_preserve_target_device_and_serving_policy() {
    enum Device {
        Automatic,
        Cpu,
        Mps,
    }
    for device in [Device::Automatic, Device::Cpu, Device::Mps] {
        enum Bootstrap {
            GpuProfile,
            MorphologyTask,
        }
        for bootstrap in [Bootstrap::GpuProfile, Bootstrap::MorphologyTask] {
            let task = match bootstrap {
                Bootstrap::GpuProfile => None,
                Bootstrap::MorphologyTask => Some(InferTask::Morphosyntax),
            };
            let runtime = match device {
                Device::Automatic => WorkerRuntimeConfig {
                    gpu_thread_pool_size: 7,
                    ..WorkerRuntimeConfig::for_tests()
                },
                Device::Cpu => WorkerRuntimeConfig {
                    force_cpu: true,
                    gpu_thread_pool_size: 7,
                    ..WorkerRuntimeConfig::for_tests()
                },
                Device::Mps => WorkerRuntimeConfig {
                    allow_mps: true,
                    gpu_thread_pool_size: 7,
                    ..WorkerRuntimeConfig::for_tests()
                },
            };
            let config = WorkerConfig {
                python_path: "python3".into(),
                profile: WorkerProfile::Gpu,
                task,
                runtime,
                ..WorkerConfig::for_tests()
            };
            let arguments = command_args(&config);
            assert_eq!(
                argument(&arguments, "--supervisor-pid"),
                std::process::id().to_string()
            );
            match bootstrap {
                Bootstrap::GpuProfile => {
                    assert_eq!(argument(&arguments, "--profile"), "gpu");
                    assert!(!arguments.iter().any(|arg| arg == "--task"));
                }
                Bootstrap::MorphologyTask => {
                    assert_eq!(argument(&arguments, "--task"), "morphosyntax");
                    assert!(!arguments.iter().any(|arg| arg == "--profile"));
                }
            }
            assert_eq!(
                arguments.iter().any(|arg| arg == "--force-cpu"),
                matches!(device, Device::Cpu)
            );
            assert_eq!(
                arguments.iter().any(|arg| arg == "--allow-mps"),
                matches!(device, Device::Mps)
            );
            let shared = matches!(device, Device::Automatic | Device::Mps) && task.is_none();
            assert_eq!(
                argument(&arguments, "--serving"),
                if shared { "concurrent" } else { "sequential" }
            );
            assert_eq!(
                argument(&arguments, "--gpu-thread-pool-size"),
                if shared { "7" } else { "1" }
            );
        }
    }
}

/// Guards: the command emits the typed Rev key and only the selected backend's
/// Tencent credential fields from the explicit account snapshot.
/// Kind: wire
/// Not a type: environment key spelling and selected-backend injection are wire
/// choices connecting independently implemented Rust and Python processes.
/// State: synthetic accounts and local Commands require no real credentials or IO.
/// Concurrency: no process environment is mutated or inherited for these observations.
#[test]
fn worker_environment_bytes_follow_selected_backend_and_typed_accounts() {
    let config = WorkerConfig {
        python_path: "python3".into(),
        profile: WorkerProfile::Gpu,
        runtime: WorkerRuntimeConfig {
            provider_accounts:
                crate::worker::provider_credentials::ProviderAccounts::with_revai_api_key(
                    crate::revai::RevAiApiKey::parse("  injected-key  "),
                ),
            ..WorkerRuntimeConfig::for_tests()
        },
        ..WorkerConfig::for_tests()
    };
    assert_eq!(
        command_envs(&config)
            .get("BATCHALIGN_REV_API_KEY")
            .map(String::as_str),
        Some("injected-key")
    );
    let accounts = crate::worker::provider_credentials::ProviderAccounts::from_explicit_hk_sources(
        &HkAsrCredentialSources::from_sources(
            Some("id"),
            Some("key"),
            Some("ap-guangzhou"),
            Some("bucket"),
            None,
            None,
            None,
            Some("/tmp/unused-home"),
        ),
    );
    enum Backend {
        Tencent,
        Qwen,
        Default,
    }
    for backend in [Backend::Tencent, Backend::Qwen, Backend::Default] {
        let selected = match backend {
            Backend::Tencent => Some("tencent"),
            Backend::Qwen => Some("qwen"),
            Backend::Default => None,
        };
        let engine_overrides = match selected {
            Some(engine) => serde_json::json!({"asr":engine}).to_string(),
            None => String::new(),
        };
        let config = WorkerConfig {
            python_path: "python3".into(),
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::yue()),
            engine_overrides,
            runtime: WorkerRuntimeConfig {
                provider_accounts: accounts.clone(),
                ..WorkerRuntimeConfig::for_tests()
            },
            ..WorkerConfig::for_tests()
        };
        let env = command_envs(&config);
        match backend {
            Backend::Tencent => {
                assert_eq!(
                    env.get("BATCHALIGN_TENCENT_ID").map(String::as_str),
                    Some("id")
                );
                assert_eq!(
                    env.get("BATCHALIGN_TENCENT_BUCKET").map(String::as_str),
                    Some("bucket")
                );
            }
            Backend::Qwen | Backend::Default => assert!(
                !env.keys()
                    .any(|name| name.starts_with("BATCHALIGN_TENCENT_"))
            ),
        }
    }
}

/// Guards: an unrelated control reply cannot authorize worker reuse, while a
/// correctly reported operation failure remains a completed exchange.
/// Kind: contract
/// Not a type: a subprocess can emit a well-formed foreign terminal line, and
/// reuse eligibility depends on admitting its actual operation correspondence.
/// State: private shell stubs exercise real ready and control protocol pipes.
/// Concurrency: each exchange owns its subprocess and all captured descriptors.
#[tokio::test]
async fn only_an_admitted_control_terminal_permits_sequential_reuse() {
    use super::{RequestFlight, WorkerHandle};
    use crate::worker::error::WorkerError;
    for (terminal, expected_flight) in [
        (
            r#"{"op":"health","response":{"status":"ok","command":"infer:morphosyntax","lang":"eng","pid":4242,"uptime_s":0.0}}"#,
            RequestFlight::InFlight,
        ),
        (
            r#"{"op":"error","kind":"runtime","error":"declared failure"}"#,
            RequestFlight::Idle,
        ),
    ] {
        let ready = super::test_support::ready_line();
        let script = format!(
            "printf '%s\\n' '{}'; read request; printf '%s\\n' '{}'",
            ready.trim(),
            terminal
        );
        let mut worker = WorkerHandle::spawn_stub_for_test(&script).await;
        let error = worker
            .capabilities()
            .await
            .expect_err("the worker provided no capabilities");
        match expected_flight {
            RequestFlight::InFlight => assert!(matches!(error, WorkerError::Protocol(_))),
            RequestFlight::Idle => assert!(matches!(error, WorkerError::WorkerResponse(_))),
        }
        assert_eq!(worker.request_flight, expected_flight);
    }
}
