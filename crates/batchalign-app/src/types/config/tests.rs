//! Tests for the config module family (layout, server, resolve, load).

use std::path::{Path, PathBuf};

use crate::api::MemoryMb;

use super::*;

#[test]
fn default_config() {
    let cfg = ServerConfig::default();
    assert_eq!(cfg.port, 8000);
    assert_eq!(cfg.host, "0.0.0.0");
    assert_eq!(cfg.default_lang, "eng"); // PartialEq<&str>
    assert_eq!(cfg.job_ttl_days, 7);
    assert!(cfg.auto_daemon);
    assert_eq!(cfg.temporal_server_url, "");
    assert_eq!(cfg.temporal_namespace, "default");
    // Default task queue is per-host (`batchalign3-{hostname}`); see
    // `default_task_queue_is_per_hostname` below for the invariant check.
    assert!(cfg.temporal_task_queue.starts_with("batchalign3-"));
    assert_eq!(cfg.temporal_heartbeat_s, 10);
    assert_eq!(cfg.temporal_activity_timeout_s, 86_400);
    assert_eq!(cfg.worker_idle_timeout_s, 600);
    assert_eq!(cfg.worker_health_interval_s, 30);
    // C2.7: `memory_gate_mb` defaults to `None`. The tier-aware
    // headroom value is delivered by `resolved_memory_gate_mb()`,
    // which falls through to `resolved_memory_tier().headroom_mb`
    // when no override is set.
    assert_eq!(cfg.memory_gate_mb, None);
    let tier = crate::types::runtime::MemoryTier::detect();
    assert_eq!(cfg.resolved_memory_gate_mb(), tier.headroom_mb);
    assert_eq!(cfg.max_concurrent_worker_startups, 1);
    assert_eq!(cfg.max_workers_per_key, None);
    assert_eq!(cfg.worker_ready_timeout_s, 300);
    assert_eq!(cfg.max_body_bytes_mb, MemoryMb(512));
    assert_eq!(cfg.memory_gate_timeout_s, 120);
    assert_eq!(cfg.memory_gate_poll_s, 5);
    assert_eq!(cfg.memory_warning_mb, MemoryMb(4096));
    // `gpu_thread_pool_size` defaults to `None` post-host-facts-migration.
    // Production builders read from `EffectiveConfig` (which yields 4 on
    // CUDA-functional hosts and 1 elsewhere); tests that need an explicit
    // value set `Some(...)` directly. See `host_facts/effective.rs`.
    assert_eq!(cfg.gpu_thread_pool_size, None);
    assert_eq!(cfg.local_lease_ttl_s, 300);
    assert_eq!(cfg.audio_task_timeout_s, 0);
    assert_eq!(cfg.analysis_task_timeout_s, 0);
    assert!(
        !cfg.use_temporal(),
        "empty temporal_server_url means local backend"
    );
}

#[test]
fn use_temporal_recognizes_sentinel_values() {
    let mut cfg = ServerConfig::default();
    assert!(!cfg.use_temporal());

    cfg.temporal_server_url = "http://net:7233".to_string();
    assert!(cfg.use_temporal());

    for sentinel in ["", "none", "local", "disabled", "  ", " none "] {
        cfg.temporal_server_url = sentinel.to_string();
        assert!(
            !cfg.use_temporal(),
            "'{sentinel}' should mean local backend"
        );
    }
}

#[test]
fn deserialize_yaml() {
    let yaml = r#"
media_roots:
  - /data/media
  - /data/media2
media_mappings:
  childes-data: /nfs/childes
default_lang: spa
port: 9000
backend: temporal
max_concurrent_jobs: 4
warmup_commands:
  - morphotag
  - align
auto_daemon: true
"#;
    let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(cfg.media_roots.len(), 2);
    assert_eq!(cfg.media_roots[0].as_str(), "/data/media");
    assert_eq!(
        cfg.media_mappings[&batchalign_types::paths::MediaMappingKey::new("childes-data")].as_str(),
        "/nfs/childes"
    );
    assert_eq!(cfg.default_lang, "spa");
    assert_eq!(cfg.port, 9000);
    assert_eq!(cfg.max_concurrent_jobs, Some(4));
    assert_eq!(cfg.warmup_commands, vec!["morphotag", "align"]);
    assert!(cfg.auto_daemon);
}

#[test]
fn deserialize_empty_yaml() {
    let yaml = "{}";
    let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(cfg, ServerConfig::default());
}

#[test]
fn deserialize_rejects_unknown_fields() {
    let yaml = r#"
port: 9000
warmup: false
"#;
    let error = serde_yaml::from_str::<ServerConfig>(yaml).unwrap_err();
    assert!(
        error.to_string().contains("unknown field `warmup`"),
        "unexpected error: {error}"
    );
}

#[test]
fn validate_fixes_bad_values() {
    // Several legacy validator clamps have moved into the type system
    // via the host-facts `Option<u32>` migrations: `gpu_thread_pool_size`
    // (C2.1) and `max_concurrent_jobs` (C2.4) no longer carry sentinel
    // values that need clamping — `0` deserializes to `None` and
    // negative values are no longer expressible. The remaining
    // clamps below are for fields that have not yet migrated.
    let mut cfg = ServerConfig {
        job_ttl_days: 0,
        memory_gate_poll_s: 0,
        max_concurrent_worker_startups: 0,
        ..Default::default()
    };
    let warnings = cfg.validate();
    assert_eq!(cfg.job_ttl_days, 1);
    assert_eq!(cfg.memory_gate_poll_s, 1);
    assert_eq!(cfg.max_concurrent_worker_startups, 1);
    assert_eq!(warnings.len(), 3);
}

#[test]
fn load_missing_file_returns_defaults() {
    let cfg = load_config(Some(Path::new("/nonexistent/server.yaml"))).unwrap();
    assert_eq!(cfg, ServerConfig::default());
}

#[test]
fn load_validated_config_clamps_bad_values() {
    let dir = tempfile::tempdir().unwrap();
    let state_dir = dir.path().join("state");
    std::fs::create_dir_all(&state_dir).unwrap();
    let config_path = state_dir.join("server.yaml");
    std::fs::write(
        &config_path,
        "port: 0\njob_ttl_days: 0\nmemory_gate_poll_s: 0\nmax_concurrent_worker_startups: 0\ngpu_thread_pool_size: 0\n",
    )
    .unwrap();

    let layout = RuntimeLayout::from_state_dir(state_dir);
    let (cfg, warnings) = load_validated_config_from_layout(&layout, None).unwrap();
    assert_eq!(cfg.port, 8000);
    assert_eq!(cfg.job_ttl_days, 1);
    assert_eq!(cfg.memory_gate_poll_s, 1);
    assert_eq!(cfg.max_concurrent_worker_startups, 1);
    // Legacy `gpu_thread_pool_size: 0` in YAML now collapses to `None`
    // via the `zero_as_none` serde shim — no validator warning fires.
    assert_eq!(cfg.gpu_thread_pool_size, None);
    assert_eq!(warnings.len(), 4);
}

#[test]
fn runtime_layout_prefers_explicit_state_dir() {
    let layout = RuntimeLayout::from_sources(Some("/tmp/batchalign-state"), Some("/Users/test"));
    assert_eq!(layout.state_dir(), Path::new("/tmp/batchalign-state"));
    assert_eq!(
        layout.config_path(),
        Path::new("/tmp/batchalign-state/server.yaml")
    );
}

#[test]
fn runtime_layout_falls_back_to_home_dir() {
    let layout = RuntimeLayout::from_sources(None, Some("/Users/test"));
    assert_eq!(layout.state_dir(), Path::new("/Users/test/.batchalign3"));
    assert_eq!(
        layout.config_path(),
        Path::new("/Users/test/.batchalign3/server.yaml")
    );
}

#[test]
fn runtime_layout_derives_owned_subpaths() {
    let layout = RuntimeLayout::from_state_dir(PathBuf::from("/tmp/batchalign-state"));
    assert_eq!(
        layout.jobs_dir(),
        PathBuf::from("/tmp/batchalign-state/jobs")
    );
    assert_eq!(
        layout.logs_dir(),
        PathBuf::from("/tmp/batchalign-state/logs")
    );
    assert_eq!(
        layout.bug_reports_dir(),
        PathBuf::from("/tmp/batchalign-state/bug-reports")
    );
    assert_eq!(
        layout.dashboard_dir(),
        PathBuf::from("/tmp/batchalign-state/dashboard")
    );
    assert_eq!(
        layout.server_pid_path(),
        PathBuf::from("/tmp/batchalign-state/server.pid")
    );
    assert_eq!(
        layout.server_log_path(),
        PathBuf::from("/tmp/batchalign-state/server.log")
    );
}

#[test]
fn runtime_layout_load_config_uses_layout_config_path() {
    let dir = tempfile::tempdir().unwrap();
    let layout = RuntimeLayout::from_state_dir(dir.path().join("state"));
    std::fs::create_dir_all(layout.state_dir()).unwrap();
    std::fs::write(layout.config_path(), "port: 9123\n").unwrap();

    let cfg = load_config_from_layout(&layout, None).unwrap();
    assert_eq!(cfg.port, 9123);
}

// ---------------------------------------------------------------------
// `force_cpu` migration to `Option<bool>` — Phase C2.2.
//
// The CLI `--force-cpu` switch stays a presence-only bool; the
// YAML side of the field migrates to `Option<bool>` so server.yaml
// can express the tri-state ("None = let host-facts decide").
// Production builders convert the CLI override at the boundary
// (if `cli.force_cpu` then `cfg.force_cpu = Some(true)`).
// Unlike the integer migrations there is no `zero_as_none` shim:
// `false` is not a sentinel for "auto"; it's a meaningful explicit
// value (operator asserts "use GPU even where the recommendation
// would force CPU"). Default uses standard serde Option behavior:
// field absent -> None, field present -> Some(value).
// ---------------------------------------------------------------------

#[test]
fn force_cpu_field_absent_yields_none() {
    let cfg: ServerConfig = serde_yaml::from_str("{}").unwrap();
    assert_eq!(cfg.force_cpu, None);
}

#[test]
fn force_cpu_explicit_true_preserves_some_true() {
    let cfg: ServerConfig = serde_yaml::from_str("force_cpu: true\n").unwrap();
    assert_eq!(cfg.force_cpu, Some(true));
}

#[test]
fn force_cpu_explicit_false_preserves_some_false() {
    let cfg: ServerConfig = serde_yaml::from_str("force_cpu: false\n").unwrap();
    assert_eq!(cfg.force_cpu, Some(false));
}

#[test]
fn force_cpu_round_trip_canonical_form() {
    for v in [None, Some(true), Some(false)] {
        let original = ServerConfig {
            force_cpu: v,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&original).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.force_cpu, v, "round-trip lost data for {v:?}");
        match v {
            None => assert!(
                !yaml.contains("force_cpu"),
                "None must omit the field; got: {yaml}"
            ),
            Some(_) => assert!(
                yaml.contains("force_cpu:"),
                "Some(_) must emit the field; got: {yaml}"
            ),
        }
    }
}

#[test]
fn force_cpu_propagates_to_config_overrides() {
    for v in [None, Some(true), Some(false)] {
        let cfg = ServerConfig {
            force_cpu: v,
            ..Default::default()
        };
        let overrides = crate::host_facts::ConfigOverrides::from(&cfg);
        assert_eq!(overrides.force_cpu, v);
    }
}

// ---------------------------------------------------------------------
// `max_workers_per_key` migration to `Option<u32>` — Phase C2.6.
//
// Same shape as the integer migrations. The host-facts model has a
// per-profile shape (`PerProfile<u32>` in `EffectiveConfig`); the
// legacy ServerConfig is a single knob. When `Some(n)`, the
// `ConfigOverrides::from(&ServerConfig)` impl populates all three
// profile overrides with `Some(n)` — uniform-across-profiles, the
// same semantics the legacy single knob has today.
// ---------------------------------------------------------------------

#[test]
fn max_workers_per_key_legacy_zero_deserializes_to_none() {
    let cfg: ServerConfig = serde_yaml::from_str("max_workers_per_key: 0\n").unwrap();
    assert_eq!(cfg.max_workers_per_key, None);
}

#[test]
fn max_workers_per_key_explicit_value_preserves_some() {
    let cfg: ServerConfig = serde_yaml::from_str("max_workers_per_key: 6\n").unwrap();
    assert_eq!(cfg.max_workers_per_key, Some(6));
}

#[test]
fn max_workers_per_key_field_absent_yields_none() {
    let cfg: ServerConfig = serde_yaml::from_str("{}").unwrap();
    assert_eq!(cfg.max_workers_per_key, None);
}

#[test]
fn max_workers_per_key_round_trip_canonical_form() {
    for v in [None, Some(1u32), Some(4), Some(8)] {
        let original = ServerConfig {
            max_workers_per_key: v,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&original).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            parsed.max_workers_per_key, v,
            "round-trip lost data for {v:?}"
        );
        match v {
            None => assert!(
                !yaml.contains("max_workers_per_key"),
                "None must omit the field; got: {yaml}"
            ),
            Some(_) => assert!(
                yaml.contains("max_workers_per_key:"),
                "Some(_) must emit the field; got: {yaml}"
            ),
        }
    }
}

/// `Some(n)` on the single legacy knob populates *every* per-profile
/// override in `ConfigOverrides`. The mapping mirrors the legacy
/// "single value applies uniformly across profiles" semantics; per-
/// profile differentiation is future work that will require a
/// `ServerConfig` shape change.
#[test]
fn max_workers_per_key_some_propagates_to_every_profile_override() {
    let cfg = ServerConfig {
        max_workers_per_key: Some(5),
        ..Default::default()
    };
    let overrides = crate::host_facts::ConfigOverrides::from(&cfg);
    assert_eq!(overrides.max_workers_per_key_by_profile.gpu, Some(5));
    assert_eq!(overrides.max_workers_per_key_by_profile.stanza, Some(5));
    assert_eq!(overrides.max_workers_per_key_by_profile.io, Some(5));
}

#[test]
fn max_workers_per_key_none_leaves_every_profile_override_none() {
    let cfg = ServerConfig {
        max_workers_per_key: None,
        ..Default::default()
    };
    let overrides = crate::host_facts::ConfigOverrides::from(&cfg);
    assert_eq!(overrides.max_workers_per_key_by_profile.gpu, None);
    assert_eq!(overrides.max_workers_per_key_by_profile.stanza, None);
    assert_eq!(overrides.max_workers_per_key_by_profile.io, None);
}

// ---------------------------------------------------------------------
// `memory_gate_mb` migration to `Option<MemoryMb>` — Phase C2.7.
//
// Same shape as the integer migrations; the underlying type is the
// newtype `MemoryMb(u64)`, which now implements `IsZero` so the
// existing `zero_as_none` shim works. The previous default
// `default_memory_gate_mb()` (which polled the live tier) is gone:
// `None` falls through to `resolved_memory_gate_mb()`, which still
// honors the `memory_tier` override.
// ---------------------------------------------------------------------

#[test]
fn memory_gate_mb_legacy_zero_deserializes_to_none() {
    let cfg: ServerConfig = serde_yaml::from_str("memory_gate_mb: 0\n").unwrap();
    assert_eq!(cfg.memory_gate_mb, None);
}

#[test]
fn memory_gate_mb_explicit_value_preserves_some() {
    let cfg: ServerConfig = serde_yaml::from_str("memory_gate_mb: 8192\n").unwrap();
    assert_eq!(cfg.memory_gate_mb, Some(MemoryMb(8192)));
}

#[test]
fn memory_gate_mb_field_absent_yields_none() {
    let cfg: ServerConfig = serde_yaml::from_str("{}").unwrap();
    assert_eq!(cfg.memory_gate_mb, None);
}

#[test]
fn memory_gate_mb_round_trip_canonical_form() {
    for v in [None, Some(MemoryMb(2048)), Some(MemoryMb(8192))] {
        let original = ServerConfig {
            memory_gate_mb: v,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&original).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.memory_gate_mb, v, "round-trip lost data for {v:?}");
        match v {
            None => assert!(
                !yaml.contains("memory_gate_mb"),
                "None must omit the field; got: {yaml}"
            ),
            Some(_) => assert!(
                yaml.contains("memory_gate_mb:"),
                "Some(_) must emit the field; got: {yaml}"
            ),
        }
    }
}

// ---------------------------------------------------------------------
// `max_concurrent_jobs` migration to `Option<u32>` — Phase C2.4.
//
// Same shape as C2.1 / C2.3 / C2.5. The single production consumer is
// `JobStore::new`, which currently dispatches on `> 0`; after the
// migration it pattern-matches on `Some(n) | None` and falls through
// to `HostExecutionPolicy::auto_max_concurrent_jobs()` (a host-aware
// auto-tune that Phase B4 will subsume into
// `EffectiveConfig::max_concurrent_jobs`).
// ---------------------------------------------------------------------

#[test]
fn max_concurrent_jobs_legacy_zero_deserializes_to_none() {
    let cfg: ServerConfig = serde_yaml::from_str("max_concurrent_jobs: 0\n").unwrap();
    assert_eq!(cfg.max_concurrent_jobs, None);
}

#[test]
fn max_concurrent_jobs_explicit_value_preserves_some() {
    let cfg: ServerConfig = serde_yaml::from_str("max_concurrent_jobs: 4\n").unwrap();
    assert_eq!(cfg.max_concurrent_jobs, Some(4));
}

#[test]
fn max_concurrent_jobs_field_absent_yields_none() {
    let cfg: ServerConfig = serde_yaml::from_str("{}").unwrap();
    assert_eq!(cfg.max_concurrent_jobs, None);
}

#[test]
fn max_concurrent_jobs_round_trip_canonical_form() {
    for v in [None, Some(1u32), Some(4), Some(16)] {
        let original = ServerConfig {
            max_concurrent_jobs: v,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&original).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            parsed.max_concurrent_jobs, v,
            "round-trip lost data for {v:?}"
        );
        match v {
            None => assert!(
                !yaml.contains("max_concurrent_jobs"),
                "None must omit the field; got: {yaml}"
            ),
            Some(_) => assert!(
                yaml.contains("max_concurrent_jobs:"),
                "Some(_) must emit the field; got: {yaml}"
            ),
        }
    }
}

// ---------------------------------------------------------------------
// `max_workers_per_job` migration to `Option<u32>` — Phase C2.5.
//
// Same shape as C2.1 / C2.3. The runtime authority for the resolved
// per-command value is `EffectiveConfig::max_workers_per_job(command)`;
// the legacy single-knob `ServerConfig.max_workers_per_job` is now an
// operator override (uniform across commands) carried through the
// `ConfigOverrides::max_workers_per_job` field.
// ---------------------------------------------------------------------

#[test]
fn max_workers_per_job_legacy_zero_deserializes_to_none() {
    let cfg: ServerConfig = serde_yaml::from_str("max_workers_per_job: 0\n").unwrap();
    assert_eq!(cfg.max_workers_per_job, None);
}

#[test]
fn max_workers_per_job_explicit_value_preserves_some() {
    let cfg: ServerConfig = serde_yaml::from_str("max_workers_per_job: 3\n").unwrap();
    assert_eq!(cfg.max_workers_per_job, Some(3));
}

#[test]
fn max_workers_per_job_field_absent_yields_none() {
    let cfg: ServerConfig = serde_yaml::from_str("{}").unwrap();
    assert_eq!(cfg.max_workers_per_job, None);
}

#[test]
fn max_workers_per_job_round_trip_canonical_form() {
    for v in [None, Some(1u32), Some(4), Some(16)] {
        let original = ServerConfig {
            max_workers_per_job: v,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&original).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            parsed.max_workers_per_job, v,
            "round-trip lost data for {v:?}"
        );
        match v {
            None => assert!(
                !yaml.contains("max_workers_per_job"),
                "None must omit the field; got: {yaml}"
            ),
            Some(_) => assert!(
                yaml.contains("max_workers_per_job:"),
                "Some(_) must emit the field; got: {yaml}"
            ),
        }
    }
}

// ---------------------------------------------------------------------
// `max_total_workers` migration to `Option<u32>` — Phase C2.3.
//
// Same shape as the C2.1 `gpu_thread_pool_size` migration: legacy
// `i32` with sentinel `0 = auto` becomes `Option<u32>` where the
// canonical "no override" form is field-absent. The `zero_as_none`
// shim collapses pre-migration `max_total_workers: 0` (the deployed
// fleet idiom) to `None` so existing `server.yaml` files keep
// working until Phase F re-renders them.
// ---------------------------------------------------------------------

#[test]
fn max_total_workers_legacy_zero_deserializes_to_none() {
    let cfg: ServerConfig = serde_yaml::from_str("max_total_workers: 0\n").unwrap();
    assert_eq!(cfg.max_total_workers, None);
}

#[test]
fn max_total_workers_explicit_value_preserves_some() {
    let cfg: ServerConfig = serde_yaml::from_str("max_total_workers: 12\n").unwrap();
    assert_eq!(cfg.max_total_workers, Some(12));
}

#[test]
fn max_total_workers_field_absent_yields_none() {
    let cfg: ServerConfig = serde_yaml::from_str("{}").unwrap();
    assert_eq!(cfg.max_total_workers, None);
}

#[test]
fn max_total_workers_round_trip_canonical_form() {
    for v in [None, Some(2u32), Some(8), Some(32)] {
        let original = ServerConfig {
            max_total_workers: v,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&original).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            parsed.max_total_workers, v,
            "round-trip lost data for {v:?}"
        );
        match v {
            None => assert!(
                !yaml.contains("max_total_workers"),
                "None must omit the field; got: {yaml}"
            ),
            Some(_) => assert!(
                yaml.contains("max_total_workers:"),
                "Some(_) must emit the field; got: {yaml}"
            ),
        }
    }
}

// ---------------------------------------------------------------------
// `gpu_thread_pool_size` migration to `Option<u32>` — Phase C2.1.
//
// These tests pin the contract that lets pre-migration `server.yaml`
// files (with literal `gpu_thread_pool_size: 0`) keep working while
// the canonical post-migration form is "field absent". Owned by the
// host-facts migration; remove alongside the `zero_as_none` shim in
// Phase G2 once every fleet `server.yaml` has been re-rendered.
// ---------------------------------------------------------------------

#[test]
fn gpu_thread_pool_size_legacy_zero_deserializes_to_none() {
    let cfg: ServerConfig = serde_yaml::from_str("gpu_thread_pool_size: 0\n").unwrap();
    assert_eq!(cfg.gpu_thread_pool_size, None);
}

#[test]
fn gpu_thread_pool_size_explicit_value_preserves_some() {
    let cfg: ServerConfig = serde_yaml::from_str("gpu_thread_pool_size: 7\n").unwrap();
    assert_eq!(cfg.gpu_thread_pool_size, Some(7));
}

#[test]
fn gpu_thread_pool_size_field_absent_yields_none() {
    let cfg: ServerConfig = serde_yaml::from_str("{}").unwrap();
    assert_eq!(cfg.gpu_thread_pool_size, None);
}

/// `Some(n)` round-trips as `gpu_thread_pool_size: n`, `None`
/// round-trips as a missing field (canonical post-migration form).
#[test]
fn gpu_thread_pool_size_round_trip_canonical_form() {
    for v in [None, Some(1u32), Some(4), Some(8)] {
        let original = ServerConfig {
            gpu_thread_pool_size: v,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&original).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            parsed.gpu_thread_pool_size, v,
            "round-trip lost data for {v:?}"
        );
        match v {
            None => assert!(
                !yaml.contains("gpu_thread_pool_size"),
                "None must omit the field; got: {yaml}"
            ),
            Some(_) => assert!(
                yaml.contains("gpu_thread_pool_size:"),
                "Some(_) must emit the field; got: {yaml}"
            ),
        }
    }
}

#[test]
fn roundtrip_json() {
    let cfg = ServerConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let back: ServerConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, back);
}

// ---- Memory tier overrides ----

#[test]
fn resolved_tier_defaults_to_detect() {
    let cfg = ServerConfig::default();
    let tier = cfg.resolved_memory_tier();
    let detected = crate::types::runtime::MemoryTier::detect();
    assert_eq!(tier.kind, detected.kind);
}

#[test]
fn resolved_tier_override_small() {
    let cfg = ServerConfig {
        memory_tier: Some(crate::types::runtime::MemoryTierKind::Small),
        ..Default::default()
    };
    let tier = cfg.resolved_memory_tier();
    assert_eq!(tier.kind, crate::types::runtime::MemoryTierKind::Small);
    assert_eq!(tier.headroom_mb.0, 2_000);
    assert_eq!(tier.stanza_startup_mb.0, 3_000);
}

#[test]
fn resolved_tier_individual_startup_overrides() {
    let cfg = ServerConfig {
        stanza_startup_mb: 5_000,
        gpu_startup_mb: 8_000,
        ..Default::default()
    };
    let tier = cfg.resolved_memory_tier();
    assert_eq!(tier.stanza_startup_mb.0, 5_000);
    assert_eq!(tier.gpu_startup_mb.0, 8_000);
    // IO should be unchanged (0 = use tier default)
    assert!(tier.io_startup_mb.0 > 0);
}

#[test]
fn resolved_tier_yaml_with_overrides() {
    let yaml = r#"
memory_tier: small
stanza_startup_mb: 4000
"#;
    let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
    let tier = cfg.resolved_memory_tier();
    assert_eq!(tier.kind, crate::types::runtime::MemoryTierKind::Small);
    // Stanza overridden from small default (3000) to 4000
    assert_eq!(tier.stanza_startup_mb.0, 4_000);
    // GPU stays at small default
    assert_eq!(tier.gpu_startup_mb.0, 6_000);
}

#[test]
fn resolved_memory_gate_uses_tier_override_when_headroom_is_default() {
    let cfg = ServerConfig {
        memory_tier: Some(crate::types::runtime::MemoryTierKind::Small),
        ..Default::default()
    };
    assert_eq!(cfg.resolved_memory_gate_mb().0, 2_000);
}

#[test]
fn resolved_worker_idle_timeout_uses_tier_override_when_default_matches_detected() {
    let cfg = ServerConfig {
        memory_tier: Some(crate::types::runtime::MemoryTierKind::Small),
        ..Default::default()
    };
    assert_eq!(cfg.resolved_worker_idle_timeout_s(), 60);
}

#[test]
fn explicit_headroom_and_idle_timeout_override_memory_tier_defaults() {
    let cfg = ServerConfig {
        memory_tier: Some(crate::types::runtime::MemoryTierKind::Small),
        memory_gate_mb: Some(crate::api::MemoryMb(9_999)),
        worker_idle_timeout_s: 777,
        ..Default::default()
    };
    assert_eq!(cfg.resolved_memory_gate_mb().0, 9_999);
    assert_eq!(cfg.resolved_worker_idle_timeout_s(), 777);
}

/// The default Temporal task queue must encode the system hostname so
/// each server polls a unique queue. Shared literals (`batchalign3-fleet`,
/// `batchalign3-server`) must never be the default.
#[test]
fn default_task_queue_is_per_hostname() {
    let queue = crate::types::config::server::default_temporal_task_queue();
    let expected_hostname = sysinfo::System::host_name().expect("hostname must resolve in tests");
    assert!(queue.starts_with("batchalign3-"), "got {queue:?}");
    assert!(queue.ends_with(&expected_hostname), "got {queue:?}");
    assert_ne!(queue.as_ref() as &str, "batchalign3-fleet");
    assert_ne!(queue.as_ref() as &str, "batchalign3-server");
}
