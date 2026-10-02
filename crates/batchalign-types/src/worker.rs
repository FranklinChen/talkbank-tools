//! Worker IPC types: FROZEN legacy V1 protocol.
//!
//! ⚠️ **This module is frozen and no longer the active protocol.**
//!
//! - **Status:** Legacy compatibility surface (JSON-lines over stdio).
//! - **Active alternative:** [`crate::worker_v2`] (typed MessagePack envelopes).
//! - **Deprecation:** V1 support will be removed after all production tasks are migrated
//!   to [`crate::worker_v2`]. Currently used only by `dispatch_batch_infer()` for
//!   backward-compatibility paths that do not route through the task orchestration layer.
//!
//! ## What Changed in V2
//!
//! The original worker protocol in this module defined the request/response payloads
//! exchanged for `infer`, `batch-infer`, `health`, and `capabilities` operations over
//! JSON-lines.
//!
//! V2 replaces this with:
//! - **Typed envelopes** (`ExecuteRequestV2`, `ExecuteResponseV2`, `ProgressEventV2`).
//! - **Explicit artifact references** (no large binary payloads embedded in JSON).
//! - **Task-discriminated union** (task kind is encoded in the envelope, enabling
//!   type-safe deserialization on both sides).
//! - **Backward-incompatible wire format** (different serialization, no V1→V2 bridge).
//!
//! ## Frozen Contract
//!
//! To preserve backward compatibility with any remaining direct consumers:
//! - Do NOT add new fields to existing structs in this module.
//! - Do NOT change the order of existing fields.
//! - Do NOT rename types or enum variants.
//! - If a new feature is needed, define it in [`crate::worker_v2`] instead.
//!
//! A change to the JSON serialization format moves the Rust types and the
//! Python models in `batchalign/worker/_types.py` together, since the server
//! and the worker ship as one runtime (the item outcome union, `ItemOutcome`,
//! is one such change).
//!
//! ## Migration Path
//!
//! 1. Identify remaining callers of `dispatch_batch_infer()`.
//! 2. Route them through `dispatch_execute_v2()` with V2 request types.
//! 3. Remove this module and all V1 paths once migration is complete.
//!
//! See also: [`crate::worker_v2`] for the source-of-truth protocol definition.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::api::{LanguageCode3, NonNegativeSeconds, ReportedEngineName, WorkerLanguage};

// ---------------------------------------------------------------------------
// Domain newtypes (worker-specific)
// ---------------------------------------------------------------------------

numeric_id!(
    /// OS process ID of a Python worker.
    pub WorkerPid(u32) [Eq]
);

/// Worker health status returned by the Python process health check.
///
/// The Python side sends `"ok"` as a string over the IPC channel.
/// Any unrecognized value deserializes to `Unknown` and triggers a
/// crash-restart cycle in the pool health loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkerHealthStatus {
    /// Worker is responsive and its loaded pipeline is functioning.
    Ok,
    /// Unrecognized status value: treat as unhealthy.
    #[serde(other)]
    Unknown,
}

impl WorkerHealthStatus {
    /// Return `true` when the worker reported healthy status.
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }
}

impl std::fmt::Display for WorkerHealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Response from worker health operation.
///
/// Returned when the server sends `{"op":"health"}` over the worker's
/// stdio channel.  Used by the pool's health loop to detect stuck or
/// crashed workers before they affect job dispatch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerHealthResponse {
    /// Worker health status: `Ok` when responsive, `Unknown` otherwise.
    pub status: WorkerHealthStatus,
    /// The logical bootstrap target this worker was spawned for (for example
    /// `"infer:morphosyntax"`). Workers are specialized at spawn time and cannot
    /// change target. This is a worker-internal label, not a released command.
    pub command: String,
    /// Worker-runtime language string this worker was spawned for.
    ///
    /// This is a routing/bootstrap value rather than a true domain language,
    /// so it may be `"auto"` or an empty string for runtime-only worker keys.
    pub lang: WorkerLanguage,
    /// OS process ID of the Python worker.  Used for crash diagnostics
    /// and force-kill during shutdown.
    pub pid: WorkerPid,
    /// Seconds since the worker process started (wall clock).  Useful for
    /// monitoring idle workers and debugging memory leaks over time.
    pub uptime_s: NonNegativeSeconds,
}

/// Response from worker capabilities operation.
///
/// Returned when the server sends `{"op":"capabilities"}` during startup
/// probing.  Determines which commands the server advertises in its health
/// endpoint and whether to use thread-based or process-based concurrency.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerCapabilities {
    /// Compatibility-only command list reported by the worker.
    ///
    /// Rust no longer trusts this field for released command availability and
    /// instead derives the command surface from `infer_tasks`: a command is
    /// advertised when every infer task its recipe needs is supported.
    /// Test-echo still uses it for direct worker CLI coverage.
    pub commands: Vec<String>,
    /// Whether the worker is running on free-threaded Python (3.14t+).
    /// When `true`, the server uses thread workers with shared models
    /// instead of process workers with model copies, dramatically reducing
    /// memory usage for CPU-bound commands.
    pub free_threaded: bool,
    /// Tasks supported by the `infer` op.
    pub infer_tasks: Vec<InferTask>,
    /// One entry per advertised task: forced alignment's engine name (e.g.
    /// `{"fa": "wave2vec-fa-v1"}`), or `null` before an FA model has loaded,
    /// and `null` for every other task, whose engine is named on the results
    /// it returns instead. A worker never sends a guessed name or `"unknown"`.
    ///
    /// Keyed by [`InferTask`] (its snake_case wire names) and valued by the
    /// validated [`ReportedEngineName`], so an unknown task key, a blank name
    /// or a name carrying a provenance separator fails at deserialization. The
    /// server's capability gate
    /// (`batchalign::engine_reports::WorkerEngineReports::admit`) then admits
    /// this map and `infer_tasks` into one typed per-task value, refusing a
    /// task advertised without an entry, an entry for an unadvertised task, or
    /// a name for any task but forced alignment.
    pub engine_versions: BTreeMap<InferTask, Option<ReportedEngineName>>,
    /// Per-language Stanza processor availability.
    ///
    /// Key is ISO-639-3 code (e.g. "eng", "nld"); value lists available
    /// processor names. Built from Stanza's `resources.json` at worker
    /// startup. Empty when the worker does not load Stanza (e.g. IO
    /// profile) or when the worker is an older version that does not
    /// report this field.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stanza_capabilities: BTreeMap<String, StanzaLanguageProcessors>,
}

/// Processor availability for one language in Stanza.
///
/// Reported by the Python worker from `resources.json`. Used by the
/// Rust server for submission validation and dispatch routing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StanzaLanguageProcessors {
    /// Stanza alpha-2 code this ISO-639-3 code maps to.
    pub alpha2: String,
    /// Available processor names (e.g. `["tokenize", "pos", "lemma", "depparse", "mwt"]`).
    pub processors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Pure inference protocol (CHAT-divorced)
// ---------------------------------------------------------------------------

/// Declare `InferTask` and its complete list from ONE source.
///
/// The variants and `ALL` used to be written separately, with a test holding
/// them equal. That test could never catch the case it existed for: a new
/// variant added to the enum and omitted from `ALL` left every arm accounted
/// for and every listed task present, so it passed. Generating both from this
/// invocation removes the possibility instead of checking for it, and removes
/// the test with it.
macro_rules! infer_tasks {
    ($( $(#[$doc:meta])* $variant:ident ),+ $(,)?) => {
        /// Supported inference tasks for the CHAT-divorced worker protocol.
        ///
        /// This enum is serialized as snake_case strings on the wire.
        #[derive(
            Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord,
        )]
        #[serde(rename_all = "snake_case")]
        pub enum InferTask {
            $( $(#[$doc])* $variant, )+
        }

        impl InferTask {
            /// Every task, in declaration order, so code deriving a per-task
            /// fact cannot silently miss one.
            pub const ALL: &'static [Self] = &[ $( Self::$variant, )+ ];
        }
    };
}

infer_tasks! {
    /// Stanza morphosyntax tagging (`morphotag` command path).
    Morphosyntax,
    /// Utterance segmentation.
    Utseg,
    /// Machine translation.
    Translate,
    /// Coreference annotation.
    Coref,
    /// Forced alignment.
    Fa,
    /// Automatic speech recognition.
    Asr,
    /// OpenSMILE feature extraction.
    Opensmile,
    /// AVQI (Acoustic Voice Quality Index).
    Avqi,
    /// Speaker diarization.
    Speaker,
}

/// Request for a single inference operation.
///
/// The server owns all CHAT operations (parse, cache, inject, validate,
/// serialize). Workers are stateless inference endpoints that receive
/// structured payloads and return results. CHAT text never crosses the
/// IPC boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferRequest {
    /// Inference task identifier.
    pub task: InferTask,
    /// 3-letter ISO language code.
    pub lang: LanguageCode3,
    /// Task-specific payload (structure depends on `task`).
    pub payload: serde_json::Value,
}

/// What one inference item produced: its result or its failure, one of the
/// two by type.
///
/// On the wire `{"kind": "produced", "result": ...}` or `{"kind": "failed",
/// "error": "..."}`. A response holding both, or neither, is malformed rather
/// than a case every reader must decide.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ItemOutcome {
    /// The item's result (structure depends on the task).
    Produced {
        /// The task's result payload, never `null`.
        result: ItemPayload,
    },
    /// The item failed, and this is why.
    Failed {
        /// The worker's diagnosis.
        error: String,
    },
}

impl ItemOutcome {
    /// The result payload, when the item produced one.
    pub fn result(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Produced { result } => Some(result.as_value()),
            Self::Failed { .. } => None,
        }
    }

    /// The failure, when the item failed.
    pub fn error(&self) -> Option<&str> {
        match self {
            Self::Failed { error } => Some(error),
            Self::Produced { .. } => None,
        }
    }
}

/// A produced item's result payload: any JSON value but `null`.
///
/// A produced item with a `null` result is an item with no result, which the
/// outcome union exists to rule out; it is refused when the item is read, not
/// discovered by each task's reader.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(try_from = "serde_json::Value")]
pub struct ItemPayload(serde_json::Value);

impl ItemPayload {
    /// The payload as JSON.
    pub fn as_value(&self) -> &serde_json::Value {
        &self.0
    }
}

impl TryFrom<serde_json::Value> for ItemPayload {
    type Error = &'static str;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match value {
            serde_json::Value::Null => Err("a produced item's result must not be null"),
            value @ (serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_)
            | serde_json::Value::Array(_)
            | serde_json::Value::Object(_)) => Ok(Self(value)),
        }
    }
}

/// Response from a single inference operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InferResponse {
    /// The item's result or its failure.
    pub outcome: ItemOutcome,
    /// How long this item's own work took, or that no work is attributable
    /// to it. Required: a response without the key is malformed; `null`
    /// says the item never executed.
    pub elapsed_s: ItemElapsed,
}

/// How long one inference item's own work took, or the fact that none is
/// attributable to it.
///
/// On the wire this is `elapsed_s`: a non-negative number of seconds, or
/// `null`. `null` belongs to an item no work ran for: its payload never
/// parsed, its provider was not loaded, it had nothing to analyze, or its
/// whole language group failed before per-item work began. A `0.0` there
/// would be a measurement nobody took, indistinguishable from an item that
/// genuinely finished instantly.
///
/// Serialized through `Option` (`null` for [`Self::NotExecuted`]); read by
/// hand so that only an explicit `null` means "not executed" and a response
/// without the key stays malformed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(into = "Option<NonNegativeSeconds>")]
pub enum ItemElapsed {
    /// The worker timed this item's own work.
    Measured(NonNegativeSeconds),
    /// No work ran for this item, so there is nothing to measure.
    NotExecuted,
}

impl<'de> Deserialize<'de> for ItemElapsed {
    /// `deserialize_any`, not `Option::deserialize`: serde hands a missing
    /// key to its missing-field deserializer, whose `deserialize_option`
    /// answers `None` (so a forgotten key would read as "not executed") and
    /// whose `deserialize_any` refuses with "missing field".
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ElapsedVisitor;

        impl<'de> serde::de::Visitor<'de> for ElapsedVisitor {
            type Value = ItemElapsed;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a non-negative number of seconds, or null")
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<ItemElapsed, E> {
                Ok(ItemElapsed::NotExecuted)
            }

            fn visit_none<E: serde::de::Error>(self) -> Result<ItemElapsed, E> {
                Ok(ItemElapsed::NotExecuted)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<ItemElapsed, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                NonNegativeSeconds::deserialize(deserializer).map(ItemElapsed::Measured)
            }

            fn visit_f64<E: serde::de::Error>(self, seconds: f64) -> Result<ItemElapsed, E> {
                NonNegativeSeconds::try_from(seconds)
                    .map(ItemElapsed::Measured)
                    .map_err(E::custom)
            }

            fn visit_u64<E: serde::de::Error>(self, seconds: u64) -> Result<ItemElapsed, E> {
                // A JSON integer such as `2` is a whole number of seconds.
                self.visit_f64(seconds as f64)
            }

            fn visit_i64<E: serde::de::Error>(self, seconds: i64) -> Result<ItemElapsed, E> {
                // A negative one is refused by `NonNegativeSeconds`.
                self.visit_f64(seconds as f64)
            }
        }

        deserializer.deserialize_any(ElapsedVisitor)
    }
}

impl From<ItemElapsed> for Option<NonNegativeSeconds> {
    fn from(elapsed: ItemElapsed) -> Self {
        match elapsed {
            ItemElapsed::Measured(seconds) => Some(seconds),
            ItemElapsed::NotExecuted => None,
        }
    }
}

/// Request for batched inference (multiple items, one model call).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchInferRequest {
    /// Inference task identifier.
    pub task: InferTask,
    /// 3-letter ISO language code.
    pub lang: LanguageCode3,
    /// Batch of payloads to process together.
    pub items: Vec<serde_json::Value>,
    /// Multi-word token lexicon: surface form → expansion tokens.
    /// Only used by the `Morphosyntax` task. Empty when no custom
    /// lexicon is supplied. Backward-compatible: absent in JSON when empty,
    /// defaults to empty on deserialization.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mwt: BTreeMap<String, Vec<String>>,
    /// Operator opt-in to the legacy Stanza constituency-parser
    /// fallback for utseg when no language-specific TalkBank BERT
    /// model is configured. Set by the `--utseg-fallback-stanza` CLI
    /// flag. Only consulted by the `Utseg` task; ignored by other
    /// inference tasks. Default `false` mirrors the
    /// `WhisperHubModelNotFoundError` "refuse silent substitution"
    /// pattern.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_stanza_fallback: bool,
}

/// Response from batched inference, one `InferResponse` per item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchInferResponse {
    /// Results in the same order as the request's `items` vec.
    pub results: Vec<InferResponse>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The outcome union admits neither "no result" nor a field it does not
    /// name: a produced item with a `null` result, and an unknown field on
    /// the outcome or the response, are refused when read.
    #[test]
    fn an_item_with_no_result_or_an_unknown_field_is_refused() {
        let reads = |line: &str| serde_json::from_str::<InferResponse>(line).is_ok();
        assert!(reads(
            r#"{"outcome": {"kind": "produced", "result": {"x": 1}}, "elapsed_s": 0.5}"#
        ));
        assert!(!reads(
            r#"{"outcome": {"kind": "produced", "result": null}, "elapsed_s": 0.5}"#
        ));
        assert!(!reads(
            r#"{"outcome": {"kind": "produced", "result": 1, "error": "x"}, "elapsed_s": 0.5}"#
        ));
        assert!(!reads(
            r#"{"outcome": {"kind": "failed", "error": "x"}, "elapsed_s": null, "extra": 1}"#
        ));
    }

    #[test]
    fn worker_health_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let health = WorkerHealthResponse {
            status: WorkerHealthStatus::Ok,
            command: "infer:morphosyntax".into(),
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            pid: WorkerPid(12345),
            uptime_s: NonNegativeSeconds::try_from(120.5)?,
        };
        let json = serde_json::to_string(&health)?;
        let back: WorkerHealthResponse = serde_json::from_str(&json)?;
        assert_eq!(health, back);
        Ok(())
    }

    #[test]
    fn worker_capabilities_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let caps = WorkerCapabilities {
            commands: vec!["morphotag".into(), "align".into(), "opensmile".into()],
            free_threaded: true,
            infer_tasks: vec![],
            engine_versions: BTreeMap::new(),
            stanza_capabilities: BTreeMap::new(),
        };
        let json = serde_json::to_string(&caps)?;
        let back: WorkerCapabilities = serde_json::from_str(&json)?;
        assert_eq!(caps, back);
        Ok(())
    }

    #[test]
    fn worker_capabilities_with_infer_fields() -> Result<(), Box<dyn std::error::Error>> {
        let caps = WorkerCapabilities {
            commands: vec!["morphotag".into()],
            free_threaded: false,
            infer_tasks: vec![InferTask::Morphosyntax, InferTask::Utseg],
            engine_versions: BTreeMap::from([
                (
                    InferTask::Morphosyntax,
                    Some(ReportedEngineName::try_from("stanza-1.9.2")?),
                ),
                // A supported task whose engine the worker cannot name.
                (InferTask::Utseg, None),
            ]),
            stanza_capabilities: BTreeMap::new(),
        };
        let json = serde_json::to_string(&caps)?;
        assert!(json.contains(r#""utseg":null"#), "{json}");
        let back: WorkerCapabilities = serde_json::from_str(&json)?;
        assert_eq!(caps, back);
        Ok(())
    }

    /// An unknown task key or a blank engine name is refused where the report
    /// is read, before any admission code sees it.
    #[test]
    fn worker_capabilities_refuse_unknown_task_keys_and_blank_names() {
        let unknown_key = r#"{"commands":[],"free_threaded":false,"infer_tasks":[],"engine_versions":{"parsing":"x"}}"#;
        assert!(serde_json::from_str::<WorkerCapabilities>(unknown_key).is_err());
        let blank = r#"{"commands":[],"free_threaded":false,"infer_tasks":["fa"],"engine_versions":{"fa":" "}}"#;
        assert!(serde_json::from_str::<WorkerCapabilities>(blank).is_err());
    }

    #[test]
    fn worker_capabilities_missing_infer_fields_is_rejected()
    -> Result<(), Box<dyn std::error::Error>> {
        let json = r#"{"commands":["morphotag"],"free_threaded":false}"#;
        // The infer_* fields are mandatory: deserialization must fail, and the
        // error must name the missing `infer_tasks` field. Matching the error
        // arm directly (rather than `unwrap_err()`) keeps the test panic-free;
        // an unexpected success is reported as a propagated error.
        match serde_json::from_str::<WorkerCapabilities>(json) {
            Ok(caps) => Err(format!("expected deserialization to fail, got {caps:?}").into()),
            Err(err) => {
                assert!(
                    err.to_string().contains("infer_tasks"),
                    "rejection must name the missing infer_tasks field, got: {err}"
                );
                Ok(())
            }
        }
    }

    #[test]
    fn infer_request_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let req = InferRequest {
            task: InferTask::Morphosyntax,
            lang: LanguageCode3::eng(),
            payload: serde_json::json!({
                "words": ["the", "dog", "runs"],
                "terminator": ".",
                "special_forms": []
            }),
        };
        let json = serde_json::to_string(&req)?;
        let back: InferRequest = serde_json::from_str(&json)?;
        assert_eq!(req, back);
        Ok(())
    }

    #[test]
    fn infer_response_success() -> Result<(), Box<dyn std::error::Error>> {
        let resp = InferResponse {
            outcome: ItemOutcome::Produced {
                result: ItemPayload::try_from(
                    serde_json::json!({"mor": "det|the n|dog v|run-3S", "gra": "1|2|DET 2|3|SUBJ 3|0|ROOT"}),
                )?,
            },
            elapsed_s: ItemElapsed::Measured(NonNegativeSeconds::try_from(0.042)?),
        };
        let json = serde_json::to_string(&resp)?;
        assert!(!json.contains("error"));
        let back: InferResponse = serde_json::from_str(&json)?;
        assert_eq!(resp, back);
        Ok(())
    }

    /// The elapsed key is required and its number must be a length: a
    /// response without one used to read as zero seconds through
    /// `#[serde(default)]`. `null` is the one way to say no work ran.
    #[test]
    fn infer_response_refuses_a_missing_or_negative_elapsed_time()
    -> Result<(), Box<dyn std::error::Error>> {
        let failed = r#""outcome":{"kind":"failed","error":"x"}"#;
        assert!(serde_json::from_str::<InferResponse>(&format!("{{{failed}}}")).is_err());
        assert!(
            serde_json::from_str::<InferResponse>(&format!("{{{failed},\"elapsed_s\":-0.5}}"))
                .is_err()
        );
        let unexecuted: InferResponse =
            serde_json::from_str(&format!("{{{failed},\"elapsed_s\":null}}"))?;
        assert_eq!(unexecuted.elapsed_s, ItemElapsed::NotExecuted);
        assert_eq!(
            serde_json::to_value(&unexecuted)?,
            serde_json::json!({"outcome": {"kind": "failed", "error": "x"}, "elapsed_s": null})
        );
        Ok(())
    }

    #[test]
    fn infer_response_error() -> Result<(), Box<dyn std::error::Error>> {
        let resp = InferResponse {
            outcome: ItemOutcome::Failed {
                error: "model not loaded".into(),
            },
            elapsed_s: ItemElapsed::Measured(NonNegativeSeconds::try_from(0.001)?),
        };
        let json = serde_json::to_string(&resp)?;
        let back: InferResponse = serde_json::from_str(&json)?;
        assert_eq!(resp, back);
        Ok(())
    }

    #[test]
    fn batch_infer_request_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let req = BatchInferRequest {
            task: InferTask::Morphosyntax,
            lang: LanguageCode3::eng(),
            items: vec![
                serde_json::json!({"words": ["hello"]}),
                serde_json::json!({"words": ["goodbye"]}),
            ],
            mwt: BTreeMap::new(),
            allow_stanza_fallback: false,
        };
        let json = serde_json::to_string(&req)?;
        // Empty mwt should be omitted from JSON
        assert!(!json.contains("\"mwt\""));
        let back: BatchInferRequest = serde_json::from_str(&json)?;
        assert_eq!(req, back);
        Ok(())
    }

    #[test]
    fn batch_infer_request_with_mwt_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let req = BatchInferRequest {
            task: InferTask::Morphosyntax,
            lang: LanguageCode3::eng(),
            items: vec![serde_json::json!({"words": ["gonna"]})],
            mwt: BTreeMap::from([("gonna".into(), vec!["going".into(), "to".into()])]),
            allow_stanza_fallback: false,
        };
        let json = serde_json::to_string(&req)?;
        assert!(json.contains("\"mwt\""));
        let back: BatchInferRequest = serde_json::from_str(&json)?;
        assert_eq!(req, back);
        Ok(())
    }

    #[test]
    fn batch_infer_request_backward_compat_no_mwt_field() -> Result<(), Box<dyn std::error::Error>>
    {
        // Old workers/servers that don't send "mwt" should still deserialize
        let json = r#"{"task":"morphosyntax","lang":"eng","items":[]}"#;
        let req: BatchInferRequest = serde_json::from_str(json)?;
        assert!(req.mwt.is_empty());
        Ok(())
    }

    #[test]
    fn infer_task_wire_format_is_snake_case_string() -> Result<(), Box<dyn std::error::Error>> {
        let req = BatchInferRequest {
            task: InferTask::Translate,
            lang: LanguageCode3::eng(),
            items: vec![],
            mwt: BTreeMap::new(),
            allow_stanza_fallback: false,
        };
        let json = serde_json::to_string(&req)?;
        assert!(json.contains("\"task\":\"translate\""));
        Ok(())
    }

    #[test]
    fn batch_infer_response_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let resp = BatchInferResponse {
            results: vec![
                InferResponse {
                    outcome: ItemOutcome::Produced {
                        result: ItemPayload::try_from(serde_json::json!({"mor": "co|hello"}))?,
                    },
                    elapsed_s: ItemElapsed::Measured(NonNegativeSeconds::try_from(0.01)?),
                },
                InferResponse {
                    outcome: ItemOutcome::Failed {
                        error: "empty input".into(),
                    },
                    elapsed_s: ItemElapsed::NotExecuted,
                },
            ],
        };
        let json = serde_json::to_string(&resp)?;
        let back: BatchInferResponse = serde_json::from_str(&json)?;
        assert_eq!(resp, back);
        Ok(())
    }
}
