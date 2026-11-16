//! Background tasks: health checking, idle timeout, worker spawning helpers.

use std::sync::Arc;

use tracing::{error, info, warn};

use crate::api::NumSpeakers;
use crate::worker::error::WorkerError;
use crate::worker::handle::WorkerConfig;

use super::capacity::{ClaimRejected, PermitRejected, ReservedWorkerSlot};
use super::cpu_gate;
use super::idle_eviction;
use super::memory_gate;
use super::rss_observer;
use super::spawn_runtime::PoolSpawnRuntime;
use super::{GroupsMap, WorkerGroup, WorkerKey, WorkerPool, WorkerRejectionKind};

/// Reason a `try_claim_spawn_slot` call was rejected. Distinguishes
/// the admission stages so callers retain the actual cause while selecting
/// the own-group availability, pool-wide return signal or host-pressure retry.
#[derive(Debug)]
pub(super) enum AdmissionRejection {
    /// Global permit pool exhausted. The acquisition increments
    /// `permit_rejections_total`.
    GlobalCap,
    /// Per-key cap (`max_workers_per_key.get(group.profile)`)
    /// reached. The CAS loop returns this without holding the
    /// global permit (the guard auto-drops). Increments
    /// `spawn_rejections_total`: currently the per-key counter.
    PerKeyCap,
    /// 1-minute CPU load average is at or above the host's logical
    /// CPU count. Adding another worker would oversubscribe; the
    /// admission seam refuses until live load drops. Increments
    /// `cpu_saturation_rejections_total`. Unlike `GlobalCap` and
    /// `PerKeyCap`, this rejection isn't unblocked by a worker
    /// returning, only by the host itself becoming less busy.
    CpuSaturated,
    /// Host's currently-available memory is at or below the
    /// hardcoded minimum-free floor (`memory_gate::MIN_FREE_MEMORY_MB`).
    /// Adding another worker risks an OOM-class outcome that no
    /// permit accounting can prevent; the admission seam refuses
    /// until live free memory rises. Increments
    /// `memory_constrained_rejections_total`. Like `CpuSaturated`,
    /// this rejection isn't unblocked by a worker returning, only
    /// by the host itself freeing memory (other processes exiting,
    /// caches reclaiming, etc.).
    MemoryConstrained,
    /// The host would not report its memory. Not counted as a capacity
    /// rejection and not waited on: the spawn fails with this error.
    HostUnreadable(crate::host_memory::HostUnreadable),
}

/// A completed publication or a retained reason to await another admission.
pub(super) enum SpawnOutcome {
    Published,
    Deferred(SpawnDeferral),
}

#[derive(Debug, Clone, Copy)]
pub(super) enum SpawnDeferral {
    GlobalCap,
    PerKeyCap,
    CpuSaturated,
    MemoryConstrained,
}

impl SpawnDeferral {
    /// A speculative refund changes global admission. It cannot unblock a
    /// per-key or host-pressure refusal, so those waits retain their own event.
    pub(super) async fn global_change(self, budget: &super::capacity::GlobalCapacity) {
        match self {
            Self::GlobalCap => budget.refund_event().await,
            Self::PerKeyCap | Self::CpuSaturated | Self::MemoryConstrained => {
                std::future::pending().await
            }
        }
    }
}

impl From<PermitRejected> for AdmissionRejection {
    fn from(_: PermitRejected) -> Self {
        AdmissionRejection::GlobalCap
    }
}

/// Logarithmic gate for the "global worker cap reached" WARN: emit
/// only when the rejection count is a power of two (1, 2, 4, 8, 16, …).
/// Sustained saturation produces O(log N) WARN events instead of N.
/// The monotonic `spawn_rejections_total` counter on the pool is the
/// authoritative observability surface; the log is a coarse signal.
pub(super) fn should_log_saturation(rejection_count: u64) -> bool {
    rejection_count.is_power_of_two()
}

impl WorkerPool {
    /// Start background tasks for health checking and pressure-driven
    /// eviction.
    ///
    /// Returns a `JoinHandle` that completes when the pool is shut down, or
    /// `None` for a pool built outside a Tokio runtime: such a pool can own no
    /// workers, so there is nothing to health-check. See `spawn_runtime.rs`.
    pub fn start_background_tasks(&self) -> Option<tokio::task::JoinHandle<()>> {
        let groups = self.groups.clone();
        let cancel = self.cancel.clone();
        let health_interval = self.config.health_check_interval_s.duration();
        let pool_config = self.config.clone();
        let observed_worker_runtimes = self.observed_worker_runtimes.clone();
        // The loop restarts dead workers, so it needs the pool's runtime for
        // the same reason every other spawn site does. Cloned into the task
        // rather than reached through `self`, which the task outlives.
        let spawn_runtime = self.spawn_runtime.clone();

        self.spawn_runtime.spawn_background(async move {
            let mut interval = tokio::time::interval(health_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        info!("Worker pool background tasks cancelled");
                        break;
                    }
                    _ = interval.tick() => {
                        run_health_check(
                            &groups,
                            &pool_config,
                            &observed_worker_runtimes,
                            &spawn_runtime,
                        )
                        .await;
                        // Reap orphaned workers left behind by previous server
                        // crashes (SIGKILL, OOM). This is cheap (reads a small
                        // directory) and catches orphans that would otherwise
                        // hold 2-15 GB each until the next server restart.
                        let reaped = super::reaper::reap_orphaned_workers();
                        if reaped > 0 {
                            info!(reaped, "Periodic orphan reaper cleaned up workers");
                        }
                    }
                }
            }
        })
    }

    /// Build a `WorkerConfig` at the Rust/Python boundary for one typed key.
    ///
    /// Fallible since W3: the spawn argv carries the pinned ASR models, and a
    /// plan that cannot be resolved (a Tencent worker with no resolved
    /// language, say) must refuse the spawn by name rather than start a worker
    /// that will ask a provider for a model it does not define.
    pub(super) fn worker_config(&self, key: &WorkerKey) -> Result<WorkerConfig, WorkerError> {
        worker_config_for_key(&self.config, key)
    }

    /// Get or create the `WorkerGroup` for a key.
    pub(super) fn get_or_create_group(&self, key: &WorkerKey) -> Arc<WorkerGroup> {
        let profile = key.target.profile_kind();
        let mut groups = super::lock_recovered(&self.groups);
        groups
            .entry(key.clone())
            .or_insert_with(|| {
                Arc::new(WorkerGroup::new(
                    self.spawn_permits.clone(),
                    profile,
                    key.engine_selection.clone(),
                ))
            })
            .clone()
    }

    /// Try to atomically claim a spawn slot in a group via compare_exchange.
    ///
    /// Checks two limits:
    /// 1. Per-key cap: `max_workers_per_key.get(group.profile)`
    /// 2. Global cap: `max_total_workers` (prevents aggregate OOM).
    ///
    /// Returns `Ok(claimed_total)` if a slot was claimed, `Err(current)` if
    /// at capacity.
    pub(super) fn try_claim_spawn_slot(
        &self,
        group: &Arc<WorkerGroup>,
    ) -> Result<ReservedWorkerSlot, AdmissionRejection> {
        let max = self.config.max_workers_per_key.get(group.profile);

        // Pressure gates (CPU loadavg, memory headroom) implement
        // BACK-PRESSURE: they slow new spawns when existing workers
        // are pressuring the host. Cold-start (no workers in this
        // group) has nothing to back-pressure against; refusing here
        // leaves the pool dead-on-arrival with no recovery path.
        // Both gates honor the same cold-start bypass via their
        // `_with_state` variants. Capacity gates (Layer 1 permit,
        // Layer 2 per-key cap) below run regardless: those enforce
        // budget, not pressure.
        //
        // Safety against actual at-spawn OOM is one layer down:
        // `memory_guard` per-worker RSS observation + the OS OOM
        // killer. The pressure gates are an optimization for healthy
        // hosts, not load-bearing for correctness.
        let pool_state = if group.is_empty() {
            memory_gate::PoolGateState::ColdStart
        } else {
            memory_gate::PoolGateState::Warm
        };

        // Layer 0: CPU-loadavg gate. Refusing here before any permit
        // work means a saturated host doesn't churn the global-permit
        // semaphore. Threshold is the host's logical CPU count, polled
        // fresh each call. The load comes from the runtime's host
        // readings: live in production, the declared (idle) host in test
        // pools, so their spawn counts never depend on the machine's load.
        let readings = self.config.runtime.host.readings();
        let saturation = cpu_gate::check_cpu_saturation_with_state(pool_state, readings);
        if let Err(saturated) = saturation {
            let count = self
                .rejection_counters
                .record(WorkerRejectionKind::CpuSaturation);
            if should_log_saturation(count) {
                warn!(
                    loadavg_1m = saturated.loadavg_1m,
                    threshold = saturated.threshold,
                    rejection_count = count,
                    "Host CPU saturated, rejecting worker spawn"
                );
            }
            return Err(AdmissionRejection::CpuSaturated);
        }

        // Layer 0.5: live available-memory gate, with forward-looking
        // projection (Mode A) and observed-peer RSS substitution
        // (Mode B). Sibling to the CPU gate. Refuses spawns at
        // admission rather than reactively inside
        // `WorkerHandle::spawn`'s memory_guard, so a memory-constrained
        // host doesn't burn through the global-permit semaphore on
        // doomed spawns.
        //
        // The predicate is `available_mb - estimate_mb > floor`.
        // The floor is the hardcoded `memory_gate::MIN_FREE_MEMORY_MB`
        // (= 2 GB OS-protection headroom). The estimate is the new
        // worker's projected memory cost; we prefer the observed
        // average RSS of same-profile idle peers (Mode B), falling
        // back to the static `startup_reservation_mb_for_tier` value
        // (Mode A) when no peers are available. The fallback ensures
        // that pre-spawn admission decisions match Mode A behavior
        // exactly: Mode B is strictly an improvement, never worse.
        //
        // No env var, no PoolConfig field, no operator override.
        // Tier-scaled floor: the 2048 MB hardcode the rearch shipped
        // was right for Medium workstations and wrong on every other
        // tier. See `host_min_free_mb_threshold_for_tier`.
        let mem_threshold =
            memory_gate::host_min_free_mb_threshold_for_tier(&self.config.runtime.host.tier());
        // Engine-aware: lifts the reservation above the profile baseline
        // when the engine selection carries a heavy local model.
        let reservation_mb = group.startup_reservation.mb();
        let (estimate_mb, estimate_source) =
            match rss_observer::observed_avg_rss_mb_for_profile(&self.groups, group.profile) {
                Some(observed) => (observed, rss_observer::EstimateSource::ObservedAvgIdle),
                None => (reservation_mb, rss_observer::EstimateSource::Reservation),
            };
        let memory = memory_gate::check_memory_saturation_with_state(
            pool_state,
            readings,
            mem_threshold,
            estimate_mb,
        );
        match memory {
            Ok(()) => {}
            Err(memory_gate::GateRefusal::Unreadable(error)) => {
                return Err(AdmissionRejection::HostUnreadable(error));
            }
            Err(memory_gate::GateRefusal::Constrained(constrained)) => {
                let count = self
                    .rejection_counters
                    .record(WorkerRejectionKind::MemoryConstrained);
                if should_log_saturation(count) {
                    let estimate_source_label = match estimate_source {
                        rss_observer::EstimateSource::Reservation => "reservation",
                        rss_observer::EstimateSource::ObservedAvgIdle => "observed_avg_idle",
                    };
                    warn!(
                        available_mb = constrained.available_mb,
                        estimate_mb = constrained.reservation_mb,
                        estimate_source = estimate_source_label,
                        threshold_mb = constrained.threshold_mb,
                        projected_after_spawn_mb = constrained
                            .available_mb
                            .saturating_sub(constrained.reservation_mb),
                        rejection_count = count,
                        "Projected post-spawn memory below floor, rejecting worker spawn"
                    );
                }
                return Err(AdmissionRejection::MemoryConstrained);
            }
        }

        // Layer 1: acquire a global-cap permit. Failure here means
        // every worker slot allowed by `max_total_workers` is in use
        // (or speculatively claimed by another concurrent admission)
        // and the caller retains GlobalCap as its deferral cause when
        // selecting a capacity-refund event or worker-return event.
        ReservedWorkerSlot::try_claim(group, max).map_err(|rejection| match rejection {
            ClaimRejected::Global => {
                let count = self
                    .rejection_counters
                    .record(WorkerRejectionKind::GlobalPermit);
                if should_log_saturation(count) {
                    warn!(
                        permits_available = self.spawn_permits.available_permits(),
                        max_total = self.config.effective_max_total_workers(),
                        rejection_count = count,
                        "Global permit pool exhausted, rejecting spawn"
                    );
                }
                AdmissionRejection::GlobalCap
            }
            ClaimRejected::PerKey => {
                self.rejection_counters
                    .record(WorkerRejectionKind::PerKeyCap);
                AdmissionRejection::PerKeyCap
            }
        })
    }

    /// Spawn a worker into a group, using `try_claim_spawn_slot` for the
    /// atomic slot reservation.
    ///
    /// On success, the worker is pushed into the idle queue with a permit.
    /// On spawn failure, the slot is released.
    pub(super) async fn try_spawn_into_group(
        &self,
        group: &Arc<WorkerGroup>,
        key: &WorkerKey,
    ) -> Result<SpawnOutcome, WorkerError> {
        let guard = match self.try_claim_spawn_slot(group) {
            Ok(reservation) => reservation,
            Err(AdmissionRejection::HostUnreadable(error)) => return Err(error.into()),
            Err(AdmissionRejection::GlobalCap) => {
                return Ok(SpawnOutcome::Deferred(SpawnDeferral::GlobalCap));
            }
            Err(AdmissionRejection::PerKeyCap) => {
                return Ok(SpawnOutcome::Deferred(SpawnDeferral::PerKeyCap));
            }
            Err(AdmissionRejection::CpuSaturated) => {
                return Ok(SpawnOutcome::Deferred(SpawnDeferral::CpuSaturated));
            }
            Err(AdmissionRejection::MemoryConstrained) => {
                return Ok(SpawnOutcome::Deferred(SpawnDeferral::MemoryConstrained));
            }
        };
        // Reservation owns group/global counts and queue absence through every
        // await and fallible preparation edge. Publication consumes it.
        let _bootstrap_guard = group.bootstrap.lock().await;

        match self
            .spawn_runtime
            .spawn_reserved_worker(
                self.worker_config(key)?,
                idle_eviction::IdleWorkers(self.groups.clone()),
                guard,
            )
            .await
        {
            Ok(mut handle) => {
                self.admit_worker_runtime(&handle)?;
                // Lazily detect capabilities from the first spawned worker.
                // This is a single IPC round-trip on an already-running worker.
                if self.first_admitted_capabilities.get().is_none() {
                    match self.detect_capabilities_from_worker(key, &mut handle).await {
                        Ok(()) => {}
                        // A refused report drops the actual reservation and
                        // handle, releasing capacity and terminating the process.
                        Err(super::CapabilityProbeFailure::Refused(refusal)) => {
                            return Err(refusal.into());
                        }
                        Err(super::CapabilityProbeFailure::Unanswered(error)) => {
                            tracing::warn!(error = %error, "Failed to detect capabilities from first worker (continuing)");
                        }
                    }
                }
                handle.publish();
                Ok(SpawnOutcome::Published)
            }
            Err(e) => {
                // The original reservation drops on this failure.
                if matches!(e, WorkerError::MemoryGuard(_)) {
                    self.rejection_counters
                        .record(WorkerRejectionKind::MemoryGuard);
                }
                Err(e)
            }
        }
    }
}

fn worker_config_for_key(
    pool_config: &super::PoolConfig,
    key: &WorkerKey,
) -> Result<WorkerConfig, WorkerError> {
    Ok(WorkerConfig {
        python_path: pool_config.python_path.clone(),
        profile: key.target.profile_kind(),
        task: key.target.task(),
        lang: key.language.clone(),
        num_speakers: NumSpeakers(1),
        engine_overrides: spawn_engine_overrides(
            key,
            pool_config.runtime.provider_accounts.tencent_engine(),
        )?,
        test_echo: pool_config.test_echo,
        startup_stall: pool_config.startup_stall,
        verbose: pool_config.verbose,
        runtime: pool_config.runtime.clone(),
        task_timeouts: pool_config.task_timeouts,
        test_delay_ms: pool_config.test_delay_ms,
    })
}

/// The `--engine-overrides` JSON one worker is spawned with, carrying the
/// pinned models this worker will load: the ASR composition when it will load
/// an ASR engine, and the utterance-boundary model when its language has one.
///
/// Each pin is injected HERE, at the one place a spawn argv is built, and
/// deliberately NOT into the worker key. Both are pure functions of what the
/// key already carries (the ASR backend, the language and the extras) and of
/// the account's Tencent engine, which the runtime reads once and holds fixed
/// for the server's lifetime, so two equal keys in one server always imply the
/// same pins and the key needs no new component.
/// Putting them in the key instead would force the capability-probe path and
/// the execute path to derive them identically, which would make
/// `capability_key == execute_key` a coincidence that holds only while two code
/// paths agree, rather than a consequence of the key's own contents. A worker
/// spawned for a different pin is a different worker because its language or
/// its extras already differ.
///
/// A worker whose language has a boundary model carries that pin even when it
/// will never run utseg. That is deliberate and costs nothing: the key is
/// unchanged, so no worker group is fragmented, and the worker reads the pin
/// only if it loads the utseg task. The alternative, deciding here which tasks
/// a target will load, would put a second copy of the target's task set in this
/// function.
pub(super) fn spawn_engine_overrides(
    key: &WorkerKey,
    tencent: &crate::model_manifest::TencentEngineSetting,
) -> Result<String, WorkerError> {
    let overrides = key.engine_selection.overrides();
    // Only a worker that will actually load an ASR engine needs an ASR pin.
    // `EngineSelection` has already dropped the engines with no Python loader,
    // so this is exactly the set of worker-hosted ASR engines.
    let asr_backend = overrides
        .asr
        .as_ref()
        .and_then(crate::model_manifest::worker_backend_for_engine);
    // The boundary model is chosen by LANGUAGE alone, exactly as the route is.
    let boundary_model = key
        .language
        .as_resolved()
        .and_then(crate::model_manifest::utseg_boundary_model)
        .transpose()
        .map_err(|error| WorkerError::SpawnFailed(error.to_string()))?;

    let whisper_language = stock_whisper_decode_language(key)?;

    if asr_backend.is_none() && boundary_model.is_none() && whisper_language.is_none() {
        return Ok(key.engine_selection.worker_config_json());
    }

    let mut map = overrides.dispatch_overrides();
    if let Some(backend) = asr_backend {
        let models = crate::model_manifest::resolve_asr_models(
            backend,
            key.language.as_resolved(),
            &overrides.extras,
            tencent,
        )
        .map_err(|error| WorkerError::SpawnFailed(error.to_string()))?;
        insert_spawn_json(&mut map, SpawnKey::PinnedAsrModels, models.models())?;
        // Qwen loads only when named, which is exactly this branch, so every
        // Qwen worker receives the recognition policy for its language from
        // the same table that routes segmentation. The worker never decides it.
        if backend == crate::types::worker_v2::AsrBackendV2::HkQwen {
            insert_spawn_json(
                &mut map,
                SpawnKey::QwenRecognition,
                &crate::transcribe::segmentation::qwen_recognition(key.language.as_resolved()),
            )?;
        }
    }
    if let Some(language) = whisper_language {
        insert_spawn_json(&mut map, SpawnKey::WhisperDecodeLanguage, &language)?;
    }
    if let Some(model) = boundary_model {
        insert_spawn_json(&mut map, SpawnKey::PinnedUtsegModel, &model)?;
    }
    serde_json::to_string(&map).map_err(|error| WorkerError::SpawnFailed(error.to_string()))
}

/// The decode language for a worker whose key names stock Whisper, decided
/// from the same token table admission asked. The worker never maps a
/// language code itself.
///
/// Admission already refused a language with no Whisper token, so a refusal
/// here is a broken invariant and fails the spawn. A worker whose key names
/// no ASR engine loads none: no ASR request can reach it (an ASR route needs
/// the plan's ASR engine, which is then in the key), and every load is the
/// task's `ensure_task`.
fn stock_whisper_decode_language(
    key: &WorkerKey,
) -> Result<Option<crate::types::worker_v2::WhisperDecodeLanguageV2>, WorkerError> {
    use crate::types::engines::{AsrLanguageAuthority, LanguageTable};
    use crate::types::worker_v2::WhisperDecodeLanguageV2;
    // "Is this stock Whisper" is the engine's own language authority, the
    // classification admission used, not a second list of backends.
    match key
        .engine_selection
        .overrides()
        .asr
        .as_ref()
        .map(crate::types::engines::AsrEngineName::language_authority)
    {
        Some(AsrLanguageAuthority::Table(LanguageTable::WhisperTokens)) => {
            WhisperDecodeLanguageV2::for_worker_language(&key.language)
                .map(Some)
                .map_err(|error| WorkerError::SpawnFailed(error.to_string()))
        }
        // Named engines that are not stock Whisper's decoder: a fine-tune's
        // checkpoint carries its own language, and the others take the
        // language with each request.
        Some(
            AsrLanguageAuthority::Table(LanguageTable::WhisperCpp | LanguageTable::RevAi)
            | AsrLanguageAuthority::EngineOwned,
        ) => Ok(None),
        None => Ok(None),
    }
}

/// A spawn override the control plane injects, derived from the worker key.
///
/// The closed set of them, so the one place that writes them
/// ([`insert_spawn_json`]) and the one place that must remove them again
/// (`EngineSelection::from_registry_json`, which reads a worker's registered
/// overrides back into a key) cannot list different keys. A derived key read
/// back as a user extra would file a rediscovered worker under a key no
/// request ever computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpawnKey {
    PinnedAsrModels,
    PinnedUtsegModel,
    QwenRecognition,
    WhisperDecodeLanguage,
}

impl SpawnKey {
    /// Every key, for the strip on read-back.
    pub(super) const ALL: [Self; 4] = [
        Self::PinnedAsrModels,
        Self::PinnedUtsegModel,
        Self::QwenRecognition,
        Self::WhisperDecodeLanguage,
    ];

    /// The wire spelling, owned by the module that defines each key.
    /// Exhaustive, so a new variant stops the build here: list it in
    /// [`Self::ALL`] at the same time.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::PinnedAsrModels => crate::model_manifest::PINNED_ASR_MODELS_KEY,
            Self::PinnedUtsegModel => crate::model_manifest::PINNED_UTSEG_MODEL_KEY,
            Self::QwenRecognition => crate::types::worker_v2::QWEN_RECOGNITION_KEY,
            Self::WhisperDecodeLanguage => crate::types::worker_v2::WHISPER_DECODE_LANGUAGE_KEY,
        }
    }
}

/// Put one typed value into the spawn overrides as JSON under `key`; the
/// worker parses it back at its boundary.
fn insert_spawn_json<T: serde::Serialize>(
    map: &mut std::collections::BTreeMap<String, String>,
    key: SpawnKey,
    value: &T,
) -> Result<(), WorkerError> {
    let json = serde_json::to_string(value)
        .map_err(|error| WorkerError::SpawnFailed(error.to_string()))?;
    map.insert(key.as_str().to_owned(), json);
    Ok(())
}

/// Run a single round of pressure-driven eviction + health checks.
///
/// Only examines idle workers (checked-out workers are in use; errors
/// during dispatch are handled by the caller). Eviction is fully
/// pressure-driven via [`pressure_evict_idle_workers_if_needed`];
/// dead workers are removed from the idle queue by the liveness
/// loop and `total` is decremented.
pub(super) async fn run_health_check(
    groups_ref: &GroupsMap,
    pool_config: &super::PoolConfig,
    observed_worker_runtimes: &crate::worker::runtime_identity::ObservedWorkerRuntimes,
    spawn_runtime: &PoolSpawnRuntime,
) {
    pressure_evict_idle_workers_if_needed(groups_ref, &pool_config.runtime.host).await;

    // Snapshot group Arcs so we don't hold the groups lock across awaits.
    let group_snapshot: Vec<(WorkerKey, Arc<WorkerGroup>)> = {
        let groups = super::lock_recovered(groups_ref);
        groups.iter().map(|(k, g)| (k.clone(), g.clone())).collect()
    };

    for (key, group) in &group_snapshot {
        let workers_to_check = group.drain_stdio();

        let mut to_return = Vec::new();
        let mut restart_count = 0usize;

        for mut worker in workers_to_check {
            // Check if process is alive
            if !worker.is_alive() {
                warn!(
                    target = %key.target.label(),
                    lang = %key.language,
                    engine_selection = %key.engine_selection,
                    pid = %worker.pid(),
                    "Worker process died, scheduling restart"
                );
                restart_count += 1;
                // worker dropped here (SIGTERM+SIGKILL via WorkerHandle::Drop)
                continue;
            }

            // Health check via worker IPC
            match worker.health_check().await {
                Ok(_) => {
                    to_return.push(worker);
                }
                Err(e) => {
                    warn!(
                        target = %key.target.label(),
                        lang = %key.language,
                        engine_selection = %key.engine_selection,
                        pid = %worker.pid(),
                        error = %e,
                        "Health check failed, scheduling restart"
                    );
                    restart_count += 1;
                }
            }
        }

        for worker in to_return {
            worker.return_idle();
        }

        // Restart failed workers
        for _ in 0..restart_count {
            // Try to claim a global-cap permit for the restart. If
            // every permit has been grabbed by concurrent admissions
            // since the reaper refunded them, skip, a future
            // admission will spawn this worker on demand.
            let restart_guard = match ReservedWorkerSlot::try_adopt(group) {
                Ok(reservation) => reservation,
                Err(_) => {
                    warn!(target = %key.target.label(), "Skipping worker restart: global cap reached");
                    continue;
                }
            };

            info!(
                target = %key.target.label(),
                lang = %key.language,
                engine_selection = %key.engine_selection,
                "Restarting worker"
            );

            let _bootstrap_guard = group.bootstrap.lock().await;

            // A restart cannot propagate an error out of this loop, so an
            // unresolvable model plan skips this worker loudly rather than
            // restarting it with no pin.
            let config = match worker_config_for_key(pool_config, key) {
                Ok(config) => config,
                Err(error) => {
                    tracing::error!(
                        %error,
                        target = %key.target.label(),
                        lang = %key.language,
                        "Not restarting worker: its ASR model plan cannot be resolved"
                    );
                    continue;
                }
            };

            match spawn_runtime
                .spawn_reserved_worker(
                    config,
                    idle_eviction::IdleWorkers(groups_ref.clone()),
                    restart_guard,
                )
                .await
            {
                Ok(handle) => {
                    if let Err(error) = observed_worker_runtimes.admit(handle.runtime_identity()) {
                        tracing::error!(
                            %error,
                            target = %key.target.label(),
                            lang = %key.language,
                            "Refusing health-check replacement from a different worker runtime"
                        );
                        continue;
                    }
                    let pid = handle.pid();
                    handle.publish();
                    info!(
                        target = %key.target.label(),
                        lang = %key.language,
                        engine_selection = %key.engine_selection,
                        pid = %pid,
                        "Worker restarted"
                    );
                }
                Err(e) => {
                    // The original replacement reservation drops here.
                    error!(
                        target = %key.target.label(),
                        lang = %key.language,
                        engine_selection = %key.engine_selection,
                        error = %e,
                        "Failed to restart worker"
                    );
                }
            }
        }
    }

    // Clean up empty groups
    {
        let mut groups = super::lock_recovered(groups_ref);
        groups.retain(|_, g| g.total.count() > 0);
    }
}

/// Memory-pressure-driven eviction pre-pass invoked by
/// [`run_health_check`]. Skips out cheaply when there's no pressure.
/// On pressure, samples idle workers' RSS via the same machinery
/// `rss_observer` uses for admission, picks victims via
/// [`idle_eviction::select_pressure_evictions`] (largest-RSS first),
/// and shuts them down with the same teardown sequence the
/// time-based eviction loop uses. The original worker slot releases capacity
/// when its owned handle is retired.
///
/// Workers that were idle at snapshot time but checked out before
/// the eviction commits are silently skipped, that's not a bug,
/// just a TOCTOU race where the worker is now busy serving a
/// request and the next eviction round will reconsider it if
/// pressure persists.
async fn pressure_evict_idle_workers_if_needed(
    groups_ref: &GroupsMap,
    host: &crate::host_facts::RuntimeHost,
) {
    // Pressure is measured, never assumed: an unreadable host evicts nothing.
    let available_mb = match host.readings().memory() {
        Ok(memory) => memory.available_mb.0,
        Err(error) => {
            tracing::warn!(%error, "host memory unreadable; skipping pressure eviction");
            return;
        }
    };
    if available_mb >= idle_eviction::pressure_threshold(host) {
        return;
    }
    let samples = idle_eviction::snapshot_idle_workers_with_rss(groups_ref);
    let to_evict = idle_eviction::select_pressure_evictions(
        samples,
        available_mb,
        idle_eviction::pressure_threshold(host),
    );
    if to_evict.is_empty() {
        return;
    }

    info!(
        available_mb,
        threshold_mb = idle_eviction::pressure_threshold(host),
        evict_count = to_evict.len(),
        "Memory pressure detected, evicting idle workers (largest-RSS first)"
    );

    for sample in to_evict {
        let pid = sample.item().pid();
        let rss_mb = sample.rss_mb();
        let key = sample.item().key.clone();
        if !idle_eviction::retire_sampled(sample).await {
            // Checked out since the snapshot: left for the next round if
            // pressure persists.
            continue;
        }
        info!(
            pid = %pid,
            rss_mb = rss_mb,
            target = %key.target.label(),
            lang = %key.language,
            "Evicted idle worker for memory pressure"
        );
    }
}

#[cfg(test)]
mod saturation_log_tests {
    /// Guards: saturation logging emits at powers of two across short and
    /// high-volume runs, bounding repeated rejection reports logarithmically.
    /// Kind: policy
    /// Not a type: selecting a logarithmic notice schedule is a semantic logging
    /// policy even when capacity and rejection ownership are statically bound.
    /// State: integer observations require no pool, timer, filesystem or process.
    /// Concurrency: none
    #[test]
    fn saturation_notices_follow_the_logarithmic_policy() {
        for (count, expected) in [
            (0, false),
            (1, true),
            (2, true),
            (3, false),
            (4, true),
            (5, false),
            (8, true),
            (15, false),
            (16, true),
            (999, false),
            (1_048_576, true),
            (664_937, false),
        ] {
            assert_eq!(super::should_log_saturation(count), expected);
        }
    }
}

#[cfg(test)]
mod bootstrap_cancellation_contract {
    use super::*;
    use std::future::Future;
    use std::task::{Context, Waker};

    /// Guards: Cancelling an admitted spawn while it waits for bootstrap restores the original global and group capacity with zero launched workers.
    /// Kind: contract
    /// Not a type: A private slot owner can prevent forged refunds and duplicate publication. Its actual future drop must still release the shared semaphore and atomic observations at this cancellation point.
    /// State: An isolated pool uses the declared test host and a held bootstrap mutex. The native future is polled once while spawning remains blocked, so this claim needs no process, model, socket or inference server.
    /// Concurrency: One current-thread runtime owns the pool and lock. A noop waker and explicit Pending observation establish the cut without a sleep or background task.
    #[tokio::test(flavor = "current_thread")]
    async fn cancellation_before_bootstrap_restores_both_capacity_owners() {
        let pool = WorkerPool::new(super::super::PoolConfig {
            max_total_workers: 4,
            max_workers_per_key: crate::host_facts::PerProfile::uniform(4),
            ..super::super::PoolConfig::for_tests()
        });
        let key = WorkerKey::without_engine_selection(
            crate::worker::WorkerTarget::infer_task(crate::worker::InferTask::Morphosyntax),
            crate::api::WorkerLanguage::from(crate::api::LanguageCode3::eng()),
        );
        let group = pool.get_or_create_group(&key);
        let _bootstrap = group.bootstrap.lock().await;
        let permits_before = pool.spawn_permits.available_permits();
        let mut spawning = Box::pin(pool.try_spawn_into_group(&group, &key));
        let mut context = Context::from_waker(Waker::noop());
        assert!(spawning.as_mut().poll(&mut context).is_pending());
        assert_eq!(group.total.count(), 1);
        assert_eq!(pool.spawn_permits.available_permits(), permits_before - 1);
        drop(spawning);
        assert_eq!(pool.spawn_permits.available_permits(), permits_before);
        assert_eq!(group.total.count(), 0);
        assert!(group.idle.view().is_empty());
    }
}

#[cfg(test)]
mod health_cancellation_contract {
    use super::*;
    use crate::worker::handle::WorkerHandle;
    use std::future::Future;
    use std::task::{Context, Waker};

    /// Guards: Health checking consumes the queued worker's availability, and cancelling the check releases its actual worker and capacity owner.
    /// Kind: contract
    /// Not a type: The slot owns counts through the health future. The actual queue/semaphore handoff and an unread real worker pipe must still agree at cancellation, so later checkout cannot observe a phantom availability permit.
    /// State: One isolated pool uses the declared test host and an owned shell worker with real ready bytes and no health reply. The real pipe establishes a pending health exchange. No model or inference server is used.
    /// Concurrency: The stub reports its real shell PID. One explicit poll establishes Pending without sleeps. Dropping the native future retires its worker. The child handle owns teardown.
    #[tokio::test(flavor = "current_thread")]
    async fn health_cancellation_releases_owned_worker_and_availability() {
        let pool = WorkerPool::new(super::super::PoolConfig::for_tests());
        let key = WorkerKey::without_engine_selection(
            crate::worker::WorkerTarget::infer_task(crate::worker::InferTask::Morphosyntax),
            crate::api::WorkerLanguage::from(crate::api::LanguageCode3::eng()),
        );
        let group = pool.get_or_create_group(&key);
        let permits_before = pool.spawn_permits.available_permits();
        let reservation = pool
            .try_claim_spawn_slot(&group)
            .expect("admit actual stub worker");
        let script =
            super::super::checkout::tests::never_responds_script().replace("424242", "'\"$$\"'");
        let mut handle = WorkerHandle::spawn_stub_for_test(&script).await;
        assert!(handle.is_alive());
        reservation.publish_stdio(handle);
        assert_eq!(group.idle.view().len(), 1);
        let mut checking = Box::pin(run_health_check(
            &pool.groups,
            &pool.config,
            &pool.observed_worker_runtimes,
            &pool.spawn_runtime,
        ));
        let mut context = Context::from_waker(Waker::noop());
        assert!(checking.as_mut().poll(&mut context).is_pending());
        assert!(group.idle.view().is_empty());
        drop(checking);
        assert_eq!(group.total.count(), 0);
        assert_eq!(pool.spawn_permits.available_permits(), permits_before);
    }
}
