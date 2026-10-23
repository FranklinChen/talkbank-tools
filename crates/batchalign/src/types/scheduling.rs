//! Scheduling, retry, and work-unit domain types.
//!
//! These types describe the control-plane concepts that sit above individual
//! worker IPC messages:
//!
//! - what kind of unit is being scheduled
//! - how failures are classified
//! - whether a failure is retryable, deferrable, or terminal
//! - what retry policy should apply
//! - what happened on a concrete attempt
//!
//! The goal is to keep these concepts explicit and shared before fleet mode is
//! reintroduced, so the single-node server and future multi-node control plane
//! speak the same language.

use serde::{Deserialize, Serialize};

use crate::api::{JobId, MachineTime, NodeId};
// `batchalign_types`, not `crate::worker`. This said `use crate::worker::
// WorkerPid` until 2026-07-30, which resolved through `worker/mod.rs`'s
// `pub use crate::types::worker::*`, i.e. `types` imported its own re-export
// back out of `worker`. A round trip, and one of the three references that made
// `types` and `worker` mutually dependent and so kept every module downstream
// of `types` out of the core crate.
pub use batchalign_types::scheduling::{AttemptId, WorkUnitId};
use batchalign_types::worker::WorkerPid;

// `DurationMs` is defined in domain.rs alongside the other core numeric
// newtypes and re-exported here so that `crate::scheduling::DurationMs`
// paths continue to resolve unchanged.
pub use super::domain::DurationMs;

/// A schedulable unit of work within the control plane.
///
/// Today most work is effectively per-file, but the runner already has
/// different execution styles:
///
/// - per-file `process` dispatch
/// - per-file infer dispatch
/// - per-file forced alignment orchestration
/// - batched text inference
///
/// This enum makes those distinctions explicit so retry, leasing, and fleet
/// routing can target a concrete unit instead of relying on control-flow shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum WorkUnitKind {
    /// A pre-dispatch file validation or normalization unit.
    ///
    /// This is used when the control plane can reject a file before it chooses
    /// the concrete execution path (`file_process`, `file_infer`, or FA).
    FileSetup,
    /// A full per-file `process` request handled by a Python worker.
    FileProcess,
    /// A per-file infer-path orchestration handled by the Rust server.
    FileInfer,
    /// Native media encoding without a Python infer request.
    NativeMedia,
    /// A per-file forced-alignment orchestration.
    FileForcedAlignment,
    /// A cross-file batched text inference unit.
    BatchInfer,
}

impl std::fmt::Display for WorkUnitKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileSetup => write!(f, "file_setup"),
            Self::FileProcess => write!(f, "file_process"),
            Self::FileInfer => write!(f, "file_infer"),
            Self::NativeMedia => write!(f, "native_media"),
            Self::FileForcedAlignment => write!(f, "file_forced_alignment"),
            Self::BatchInfer => write!(f, "batch_infer"),
        }
    }
}

impl std::str::FromStr for WorkUnitKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "file_setup" => Ok(Self::FileSetup),
            "file_process" => Ok(Self::FileProcess),
            "file_infer" => Ok(Self::FileInfer),
            "native_media" => Ok(Self::NativeMedia),
            "file_forced_alignment" => Ok(Self::FileForcedAlignment),
            "batch_infer" => Ok(Self::BatchInfer),
            other => Err(format!("unknown WorkUnitKind: {other}")),
        }
    }
}

/// Broad classification for failures seen by the control plane.
///
/// Retry behavior should be defined against these categories rather than raw
/// error strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    /// Input failed schema or command-level validation.
    Validation,
    /// CHAT parse or semantic validation failed.
    ParseError,
    /// Required input file or media could not be found or read.
    InputMissing,
    /// A cache-only request could not be satisfied from reusable evidence.
    EvidenceUnavailable,
    /// Valid input requests analysis unavailable from the configured backend.
    AnalysisUnavailable,
    /// Worker process died unexpectedly.
    WorkerCrash,
    /// Worker or provider exceeded an expected time budget.
    WorkerTimeout,
    /// Worker IPC or protocol framing failed.
    WorkerProtocol,
    /// Worker reported a deterministic bootstrap-class failure (a model that
    /// will not load, a missing language pack, a package import) that will
    /// recur identically across retries. Non-retryable; the user-facing
    /// message is the verbatim error from the worker (all actionable). A
    /// download that failed in transit is not this category: it is
    /// `ProviderTransient`, retried. The historical reason this category
    /// exists: deterministic bootstrap failures classified as transient
    /// crashes produced multi-GB log explosions when retried 3× with a full
    /// traceback per attempt.
    WorkerBootstrap,
    /// A local model reported an unclassified runtime failure. The cause is
    /// unknown, so retry within the existing budget without claiming memory
    /// pressure, a remote outage or a deterministic input defect.
    LocalRuntime,
    /// Provider/backend returned a failure that may succeed on retry.
    ProviderTransient,
    /// Provider/backend returned a failure that should not be retried.
    ProviderTerminal,
    /// The provider's account quota or resource pack is exhausted: no request
    /// succeeds until an account holder restores it. Not retried, and it
    /// withholds the same-engine UTR fallback.
    ProviderQuotaExhausted,
    /// The provider refuses the account itself: its key is not accepted, the
    /// account is not permitted, or it has no credit left. Not retried, and
    /// it withholds the same-engine UTR fallback, as an exhausted quota does.
    ProviderAccountRefused,
    /// Memory ran short: a worker spawn the memory guard refused, or a
    /// device that ran out of memory running a request. Retried after a
    /// backoff, since memory frees as concurrent work ends.
    MemoryPressure,
    /// Work was cancelled intentionally.
    Cancelled,
    /// Catch-all infrastructure or system error.
    System,
    /// A pinned Hugging Face Hub artifact refused this machine's request: a
    /// gated repository requiring accepted terms, a missing/invalid token,
    /// or no cached copy while offline. Deterministic across retries (the
    /// operator's credentials do not change mid-job) and actionable, so it
    /// gets its own category rather than folding into `WorkerBootstrap` or
    /// the generic `Validation`, either of which would report a
    /// configuration condition on the server's machine as if it were a
    /// batchalign defect or bad input.
    ModelAccessDenied,
}

impl FailureCategory {
    /// Every category, so the spelling table below is the only one: parsing
    /// reads it, and a test holds serde's spelling to it.
    pub const ALL: [Self; 18] = [
        Self::Validation,
        Self::ParseError,
        Self::InputMissing,
        Self::EvidenceUnavailable,
        Self::AnalysisUnavailable,
        Self::WorkerCrash,
        Self::WorkerTimeout,
        Self::WorkerProtocol,
        Self::WorkerBootstrap,
        Self::LocalRuntime,
        Self::ProviderTransient,
        Self::ProviderTerminal,
        Self::ProviderQuotaExhausted,
        Self::ProviderAccountRefused,
        Self::MemoryPressure,
        Self::Cancelled,
        Self::System,
        Self::ModelAccessDenied,
    ];

    /// Stable wire spelling, suitable for bounded structured provenance.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::ParseError => "parse_error",
            Self::InputMissing => "input_missing",
            Self::EvidenceUnavailable => "evidence_unavailable",
            Self::AnalysisUnavailable => "analysis_unavailable",
            Self::WorkerCrash => "worker_crash",
            Self::WorkerTimeout => "worker_timeout",
            Self::WorkerProtocol => "worker_protocol",
            Self::WorkerBootstrap => "worker_bootstrap",
            Self::LocalRuntime => "local_runtime",
            Self::ProviderTransient => "provider_transient",
            Self::ProviderTerminal => "provider_terminal",
            Self::ProviderQuotaExhausted => "provider_quota_exhausted",
            Self::ProviderAccountRefused => "provider_account_refused",
            Self::MemoryPressure => "memory_pressure",
            Self::Cancelled => "cancelled",
            Self::System => "system",
            Self::ModelAccessDenied => "model_access_denied",
        }
    }
}

impl std::fmt::Display for FailureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for FailureCategory {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|category| category.as_str() == s)
            .ok_or_else(|| format!("unknown FailureCategory: {s}"))
    }
}

/// Outcome of one concrete attempt to execute a work unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum AttemptOutcome {
    /// The work unit finished successfully.
    Succeeded,
    /// The work unit failed and the failure is terminal.
    Failed,
    /// The attempt failed but the work unit should be retried later.
    RetryableFailure,
    /// The work unit was deferred without being treated as a failure.
    Deferred,
    /// The work unit was cancelled.
    Cancelled,
}

impl std::fmt::Display for AttemptOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Succeeded => write!(f, "succeeded"),
            Self::Failed => write!(f, "failed"),
            Self::RetryableFailure => write!(f, "retryable_failure"),
            Self::Deferred => write!(f, "deferred"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::str::FromStr for AttemptOutcome {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "succeeded" => Ok(Self::Succeeded),
            "failed" => Ok(Self::Failed),
            "retryable_failure" => Ok(Self::RetryableFailure),
            "deferred" => Ok(Self::Deferred),
            "cancelled" => Ok(Self::Cancelled),
            other => Err(format!("unknown AttemptOutcome: {other}")),
        }
    }
}

/// Scheduler decision after an attempt completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum RetryDisposition {
    /// Mark the work as terminally successful.
    Succeed,
    /// Mark the work as terminally failed.
    TerminalFailure,
    /// Retry the work after backoff.
    Retry,
    /// Leave the work queued for later without incrementing terminal failure.
    Defer,
}

impl std::fmt::Display for RetryDisposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Succeed => write!(f, "succeed"),
            Self::TerminalFailure => write!(f, "terminal_failure"),
            Self::Retry => write!(f, "retry"),
            Self::Defer => write!(f, "defer"),
        }
    }
}

impl std::str::FromStr for RetryDisposition {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "succeed" => Ok(Self::Succeed),
            "terminal_failure" => Ok(Self::TerminalFailure),
            "retry" => Ok(Self::Retry),
            "defer" => Ok(Self::Defer),
            other => Err(format!("unknown RetryDisposition: {other}")),
        }
    }
}

/// Retry/backoff policy attached to a class of work.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct RetryPolicy {
    /// Maximum total attempts, including the first attempt.
    pub max_attempts: u32,
    /// Initial backoff in milliseconds.
    pub initial_backoff_ms: DurationMs,
    /// Maximum backoff in milliseconds.
    pub max_backoff_ms: DurationMs,
    /// Exponential multiplier applied after each retry.
    pub backoff_multiplier: u32,
}

impl RetryPolicy {
    /// Conservative default suitable for transient worker/runtime failures.
    pub const fn conservative() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: DurationMs(1_000),
            max_backoff_ms: DurationMs(60_000),
            backoff_multiplier: 2,
        }
    }

    /// Compute the backoff delay for a 1-based retry number.
    pub fn backoff_for_retry(&self, retry_number: u32) -> DurationMs {
        if retry_number <= 1 {
            return DurationMs(self.initial_backoff_ms.0.min(self.max_backoff_ms.0));
        }

        let mut backoff = self.initial_backoff_ms.0;
        for _ in 1..retry_number {
            backoff = backoff.saturating_mul(self.backoff_multiplier as u64);
            if backoff >= self.max_backoff_ms.0 {
                return self.max_backoff_ms;
            }
        }
        DurationMs(backoff.min(self.max_backoff_ms.0))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::conservative()
    }
}

/// Whether another attempt could succeed after a failure of a category: the
/// one statement of the retry set, matched exhaustively, so a new category
/// must decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retryability {
    /// Another attempt may succeed: a crash, a timeout, a transient provider
    /// or download failure, memory pressure (it clears as concurrent work
    /// releases memory).
    Retryable,
    /// The same attempt fails the same way again.
    Final,
}

impl FailureCategory {
    /// Whether another attempt could succeed after a failure of this
    /// category.
    pub const fn retryability(self) -> Retryability {
        match self {
            Self::WorkerCrash
            | Self::WorkerTimeout
            | Self::ProviderTransient
            | Self::LocalRuntime
            | Self::MemoryPressure => Retryability::Retryable,
            Self::Validation
            | Self::ParseError
            | Self::InputMissing
            | Self::EvidenceUnavailable
            | Self::AnalysisUnavailable
            | Self::WorkerProtocol
            | Self::WorkerBootstrap
            | Self::ProviderTerminal
            | Self::ProviderQuotaExhausted
            | Self::ProviderAccountRefused
            | Self::Cancelled
            | Self::System
            | Self::ModelAccessDenied => Retryability::Final,
        }
    }
}

/// What a retry policy decides after a failed attempt: the transition from a
/// classified failure to the next step, so a caller acts on the decision
/// rather than combining a retryable flag with a budget test of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextAttempt {
    /// Attempt again after this backoff.
    RetryAfter(DurationMs),
    /// Stop: the failure is final, or the retry budget is spent.
    Stop(StopReason),
}

/// Why a retry policy stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The failure's category is final.
    Final,
    /// The category is retryable and every attempt the policy allows is spent.
    BudgetSpent,
}

impl RetryPolicy {
    /// The decision after attempt `attempt_number` (1-based) failed with
    /// `category`.
    pub fn after_failure(&self, attempt_number: u32, category: FailureCategory) -> NextAttempt {
        match category.retryability() {
            Retryability::Final => NextAttempt::Stop(StopReason::Final),
            Retryability::Retryable if attempt_number < self.max_attempts => {
                NextAttempt::RetryAfter(self.backoff_for_retry(attempt_number))
            }
            Retryability::Retryable => NextAttempt::Stop(StopReason::BudgetSpent),
        }
    }
}

/// Durable record of one attempt to execute a work unit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct AttemptRecord {
    /// Stable attempt identifier.
    pub attempt_id: AttemptId,
    /// Parent job identifier.
    pub job_id: JobId,
    /// Opaque work-unit identifier within the job.
    pub work_unit_id: WorkUnitId,
    /// Kind of work unit that was executed.
    pub work_unit_kind: WorkUnitKind,
    /// 1-based attempt number for this work unit.
    pub attempt_number: u32,
    /// When the attempt started.
    pub started_at: MachineTime,
    /// When the attempt finished; `None` while it runs.
    pub finished_at: Option<MachineTime>,
    /// Final outcome of the attempt.
    pub outcome: AttemptOutcome,
    /// Broad failure classification when the attempt did not succeed.
    pub failure_category: Option<FailureCategory>,
    /// Scheduler decision for what should happen next.
    pub disposition: RetryDisposition,
    /// Identifier of the node that executed the attempt.
    pub worker_node_id: Option<NodeId>,
    /// Worker process identifier, when execution happened in a local worker.
    pub worker_pid: Option<WorkerPid>,
}

/// Lease metadata for a claimed schedulable unit.
///
/// In single-node mode this is coarse-grained and attached at the job level:
/// the local dispatcher claims a queued job for a node, records the lease, and
/// later clears it when the runner exits. Fleet mode will extend the same shape
/// with real cross-node renewal and expiry handling.
///
/// A lease always expires strictly after its heartbeat.
//
// (A plain comment, not documentation: the doc text above is also the
// OpenAPI description.) The fields are private and every route in keeps the
// order: `taken` and `renew` compute the expiry from a `LeaseTtl`, which is
// positive, and `new`, shared by the database read and deserialization,
// refuses an expiry that is not after the heartbeat. The fields were public
// until 2026-10-01, so `expires_at <= heartbeat_at` was constructible anywhere.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "LeaseRecordWire")]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct LeaseRecord {
    /// Identifier of the node that currently owns the lease.
    leased_by_node: NodeId,
    /// When the lease was taken or most recently renewed.
    heartbeat_at: MachineTime,
    /// When the lease expires if not renewed.
    expires_at: MachineTime,
}

/// The unvalidated shape a [`LeaseRecord`] is deserialized through. Private:
/// nothing outside this module can hold a lease whose order is unchecked.
#[derive(Deserialize)]
struct LeaseRecordWire {
    leased_by_node: NodeId,
    heartbeat_at: MachineTime,
    expires_at: MachineTime,
}

impl TryFrom<LeaseRecordWire> for LeaseRecord {
    type Error = LeaseExpiryNotAfterHeartbeat;

    fn try_from(wire: LeaseRecordWire) -> Result<Self, Self::Error> {
        Self::new(wire.leased_by_node, wire.heartbeat_at, wire.expires_at)
    }
}

/// A lease whose expiry is not after its heartbeat names no lease that could
/// ever have been held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a lease must expire after its heartbeat: heartbeat {heartbeat_at}, expiry {expires_at}")]
pub struct LeaseExpiryNotAfterHeartbeat {
    /// The heartbeat the lease named.
    pub heartbeat_at: MachineTime,
    /// The expiry the lease named.
    pub expires_at: MachineTime,
}

impl LeaseRecord {
    /// A lease read back from storage or the wire, refused unless it expires
    /// strictly after its heartbeat.
    pub fn new(
        leased_by_node: NodeId,
        heartbeat_at: MachineTime,
        expires_at: MachineTime,
    ) -> Result<Self, LeaseExpiryNotAfterHeartbeat> {
        if expires_at > heartbeat_at {
            Ok(Self {
                leased_by_node,
                heartbeat_at,
                expires_at,
            })
        } else {
            Err(LeaseExpiryNotAfterHeartbeat {
                heartbeat_at,
                expires_at,
            })
        }
    }

    /// The lease `node` takes at `now`: it expires `ttl` later. With
    /// [`Self::renew`], the one place a lease's expiry is computed.
    ///
    /// `ttl` is positive, so the expiry is after `now`; the only exception
    /// would be `MachineTime::plus` saturating at the last representable
    /// instant, which no clock reaches.
    pub(crate) fn taken(node: NodeId, now: MachineTime, ttl: crate::config::LeaseTtl) -> Self {
        Self {
            leased_by_node: node,
            heartbeat_at: now,
            expires_at: now.plus(ttl.get()),
        }
    }

    /// Renew this lease in place at `now`: same owner, new heartbeat, and an
    /// expiry `ttl` later.
    pub(crate) fn renew(&mut self, now: MachineTime, ttl: crate::config::LeaseTtl) {
        self.heartbeat_at = now;
        self.expires_at = now.plus(ttl.get());
    }

    /// The node that holds the lease.
    pub fn leased_by_node(&self) -> &NodeId {
        &self.leased_by_node
    }

    /// When the lease was taken or last renewed.
    pub fn heartbeat_at(&self) -> MachineTime {
        self.heartbeat_at
    }

    /// When the lease lapses unless renewed; always after the heartbeat.
    pub fn expires_at(&self) -> MachineTime {
        self.expires_at
    }

    /// Whether the lease is still held at `now`. The one lease-held
    /// predicate; today only the test-only local queue-claim path asks.
    #[cfg(test)]
    pub(crate) fn is_held_at(&self, now: MachineTime) -> bool {
        self.expires_at > now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The order is enforced by both routes a stored lease comes back by.
    #[test]
    fn a_lease_that_does_not_expire_after_its_heartbeat_is_refused() {
        let at = crate::unix_time(100.0);
        assert_eq!(
            LeaseRecord::new(NodeId::from("n"), at, at),
            Err(LeaseExpiryNotAfterHeartbeat {
                heartbeat_at: at,
                expires_at: at
            })
        );
        assert!(LeaseRecord::new(NodeId::from("n"), at, crate::unix_time(99.0)).is_err());
        let held = LeaseRecord::new(NodeId::from("n"), at, crate::unix_time(160.0))
            .expect("an ordered lease");

        let json = serde_json::to_string(&held).expect("serializes");
        assert_eq!(
            serde_json::from_str::<LeaseRecord>(&json).expect("round trips"),
            held
        );
        let inverted = json
            .replace("\"expires_at\"", "\"was_expires\"")
            .replace("\"heartbeat_at\"", "\"expires_at\"")
            .replace("\"was_expires\"", "\"heartbeat_at\"");
        assert!(serde_json::from_str::<LeaseRecord>(&inverted).is_err());
    }

    /// The transition from a classified failure to the next step: retry
    /// after the policy's backoff while budget remains, stop when the
    /// category is final or the budget is spent.
    #[test]
    fn a_retry_policy_decides_the_next_attempt() {
        let policy = RetryPolicy::conservative();
        assert_eq!(
            policy.after_failure(1, FailureCategory::WorkerCrash),
            NextAttempt::RetryAfter(policy.backoff_for_retry(1))
        );
        assert_eq!(
            policy.after_failure(2, FailureCategory::MemoryPressure),
            NextAttempt::RetryAfter(policy.backoff_for_retry(2))
        );
        assert_eq!(
            policy.after_failure(3, FailureCategory::ProviderTransient),
            NextAttempt::Stop(StopReason::BudgetSpent)
        );
        assert_eq!(
            policy.after_failure(1, FailureCategory::WorkerBootstrap),
            NextAttempt::Stop(StopReason::Final)
        );
    }

    #[test]
    fn failure_category_roundtrip() {
        for category in FailureCategory::ALL {
            let json = serde_json::to_string(&category).unwrap();
            let back: FailureCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(category, back);
            assert_eq!(category.to_string(), json.trim_matches('"'));
        }
    }

    #[test]
    fn scheduling_enums_display_and_parse() {
        for kind in [
            WorkUnitKind::FileSetup,
            WorkUnitKind::FileProcess,
            WorkUnitKind::FileInfer,
            WorkUnitKind::FileForcedAlignment,
            WorkUnitKind::BatchInfer,
        ] {
            let raw = kind.to_string();
            assert_eq!(raw.parse::<WorkUnitKind>().unwrap(), kind);
        }

        for outcome in [
            AttemptOutcome::Succeeded,
            AttemptOutcome::Failed,
            AttemptOutcome::RetryableFailure,
            AttemptOutcome::Deferred,
            AttemptOutcome::Cancelled,
        ] {
            let raw = outcome.to_string();
            assert_eq!(raw.parse::<AttemptOutcome>().unwrap(), outcome);
        }

        for disposition in [
            RetryDisposition::Succeed,
            RetryDisposition::TerminalFailure,
            RetryDisposition::Retry,
            RetryDisposition::Defer,
        ] {
            let raw = disposition.to_string();
            assert_eq!(raw.parse::<RetryDisposition>().unwrap(), disposition);
        }
    }

    #[test]
    fn retry_policy_default_is_conservative() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, 3);
        assert_eq!(policy.initial_backoff_ms, 1_000u64);
        assert_eq!(policy.max_backoff_ms, 60_000u64);
        assert_eq!(policy.backoff_multiplier, 2);
        assert_eq!(policy.backoff_for_retry(1), 1_000u64);
        assert_eq!(policy.backoff_for_retry(2), 2_000u64);
        assert_eq!(policy.backoff_for_retry(3), 4_000u64);
    }
}
