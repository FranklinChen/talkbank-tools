//! Progress events a V2 worker writes while it loads models or does a
//! request's work.
//!
//! An event is one of two scopes, and the scope decides what its number can
//! be:
//!
//! - [`ProgressEventV2::Load`]: a model load moving (worker startup,
//!   `ensure_task`, or a download inside one). No request owns it. Its measure
//!   is a download's growth or a phase boundary ([`LoadMeasureV2`]), never
//!   items. The supervisor's stall window reads these.
//! - [`ProgressEventV2::Request`]: one request's work counted in items
//!   ([`UnitProgressV2`]), under that request's id. Request progress reads
//!   these; they never extend a model load.
//!
//! So units in a load event, growth in a request event and a request event
//! with no request are not values of the type. They used to be: the event was
//! one record with an empty `request_id` meaning "a load" and a bare
//! `completed`/`total` pair whose meaning depended on the stage.
//!
//! Admission happens at deserialization, as for every validated wire type in
//! this crate (`#[serde(try_from = ...)]`): a [`UnitProgressV2`] has
//! `completed <= total`, and a [`RequestProgressV2`] names a request. The
//! fields of both are private, so the validating constructors are the only
//! routes to either.

use serde::{Deserialize, Serialize};

use super::requests::WorkerRequestIdV2;

/// One progress event, by scope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum ProgressEventV2 {
    /// A model load's progress (worker startup, `ensure_task`, or a download
    /// inside one): no request owns it.
    Load(LoadProgressV2),
    /// One request's work, counted in items.
    Request(RequestProgressV2),
}

impl ProgressEventV2 {
    /// The stable stage name, whatever the scope (for a phase, without any
    /// edge suffix). For logs and display labels; no decision reads it in
    /// place of the scope.
    #[must_use]
    pub fn stage(&self) -> &str {
        match self {
            Self::Load(load) => &load.stage,
            Self::Request(request) => request.stage(),
        }
    }
}

/// A model load's progress: what moved, under which stage.
///
/// A record of its own (not two fields of the enum variant) so a channel that
/// carries only a load's progress, such as a shared GPU worker's waiting
/// load, is typed to carry nothing else.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct LoadProgressV2 {
    /// Stable stage name (for a phase, without any edge suffix).
    pub stage: String,
    /// What this event measures.
    pub measure: LoadMeasureV2,
}

/// What a model load's progress event measures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LoadMeasureV2 {
    /// A model download's measured growth so far, in KiB.
    DownloadGrowth {
        /// KiB the download's cache trees have grown since it started.
        grown_kib: u32,
    },
    /// A named phase boundary.
    Phase {
        /// Which end of the phase this event marks.
        edge: PhaseEdgeV2,
    },
}

/// Which end of a phase a [`LoadMeasureV2::Phase`] event marks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PhaseEdgeV2 {
    /// The phase began.
    Begin,
    /// The phase finished.
    Complete,
}

/// Why a progress record was refused at the wire.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProgressRefusalV2 {
    /// More items were reported done than the request holds.
    #[error("progress reports {completed} of {total} items done")]
    CompletedExceedsTotal {
        /// Items reported done.
        completed: u32,
        /// Items in the request.
        total: u32,
    },
    /// Request progress named no request.
    #[error("request progress carries an empty request id")]
    EmptyRequestId,
}

/// The wire form of [`UnitProgressV2`], before admission.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema)]
struct UnitProgressWireV2 {
    /// Items finished so far.
    completed: u32,
    /// Items the request holds in total; never fewer than `completed`.
    total: u32,
}

/// Request work counted in items: how many of a request's items are done.
///
/// Admitted only with `completed <= total`, by [`Self::new`] or at
/// deserialization, so a consumer never sees a count past its own total.
/// A record of its own, not two fields beside the request id: two adjacent
/// `u32` counts are where completed and total get swapped.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(try_from = "UnitProgressWireV2", into = "UnitProgressWireV2")]
pub struct UnitProgressV2 {
    completed: u32,
    total: u32,
}

impl UnitProgressV2 {
    /// `completed` of `total` items done; refused when `completed > total`.
    pub fn new(completed: u32, total: u32) -> Result<Self, ProgressRefusalV2> {
        if completed <= total {
            Ok(Self { completed, total })
        } else {
            Err(ProgressRefusalV2::CompletedExceedsTotal { completed, total })
        }
    }

    /// Items finished so far.
    #[must_use]
    pub const fn completed(self) -> u32 {
        self.completed
    }

    /// Items the request holds in total.
    #[must_use]
    pub const fn total(self) -> u32 {
        self.total
    }
}

impl TryFrom<UnitProgressWireV2> for UnitProgressV2 {
    type Error = ProgressRefusalV2;

    fn try_from(wire: UnitProgressWireV2) -> Result<Self, Self::Error> {
        Self::new(wire.completed, wire.total)
    }
}

impl From<UnitProgressV2> for UnitProgressWireV2 {
    fn from(units: UnitProgressV2) -> Self {
        Self {
            completed: units.completed,
            total: units.total,
        }
    }
}

/// The wire form of [`RequestProgressV2`], before admission.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
struct RequestProgressWireV2 {
    /// Correlation id of the request whose work this is; never empty.
    request_id: WorkerRequestIdV2,
    /// Stable stage name.
    stage: String,
    /// How many of the request's items are done.
    progress: UnitProgressV2,
}

/// One request's work, counted in items, under that request's id.
///
/// Admitted only with a non-empty request id, by [`Self::new`] or at
/// deserialization: request progress belongs to a request, and an empty id is
/// how a load's progress used to be told apart, which the
/// [`ProgressEventV2::Load`] scope now says instead.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(try_from = "RequestProgressWireV2", into = "RequestProgressWireV2")]
pub struct RequestProgressV2 {
    request_id: WorkerRequestIdV2,
    stage: String,
    progress: UnitProgressV2,
}

impl RequestProgressV2 {
    /// Progress of the request `request_id`; refused when the id is empty.
    pub fn new(
        request_id: WorkerRequestIdV2,
        stage: String,
        progress: UnitProgressV2,
    ) -> Result<Self, ProgressRefusalV2> {
        if request_id.is_empty() {
            return Err(ProgressRefusalV2::EmptyRequestId);
        }
        Ok(Self {
            request_id,
            stage,
            progress,
        })
    }

    /// The request whose work this is.
    #[must_use]
    pub fn request_id(&self) -> &WorkerRequestIdV2 {
        &self.request_id
    }

    /// Stable stage name.
    #[must_use]
    pub fn stage(&self) -> &str {
        &self.stage
    }

    /// How many of the request's items are done.
    #[must_use]
    pub fn progress(&self) -> UnitProgressV2 {
        self.progress
    }
}

impl TryFrom<RequestProgressWireV2> for RequestProgressV2 {
    type Error = ProgressRefusalV2;

    fn try_from(wire: RequestProgressWireV2) -> Result<Self, Self::Error> {
        Self::new(wire.request_id, wire.stage, wire.progress)
    }
}

impl From<RequestProgressV2> for RequestProgressWireV2 {
    fn from(request: RequestProgressV2) -> Self {
        Self {
            request_id: request.request_id,
            stage: request.stage,
            progress: request.progress,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixture parses and refusals are the assertions themselves
mod tests {
    use super::*;

    fn parse(event: serde_json::Value) -> Result<ProgressEventV2, serde_json::Error> {
        serde_json::from_value(event)
    }

    /// The wire spelling of each scope and measure.
    #[test]
    fn each_scope_and_measure_has_one_wire_spelling() {
        let growth = parse(serde_json::json!({
            "scope": "load",
            "stage": "downloading_x",
            "measure": {"kind": "download_growth", "grown_kib": 7},
        }))
        .expect("growth is load progress");
        assert_eq!(
            growth,
            ProgressEventV2::Load(LoadProgressV2 {
                stage: "downloading_x".to_owned(),
                measure: LoadMeasureV2::DownloadGrowth { grown_kib: 7 },
            })
        );
        let phase = parse(serde_json::json!({
            "scope": "load",
            "stage": "loading_qwen_asr",
            "measure": {"kind": "phase", "edge": "complete"},
        }))
        .expect("a phase is load progress");
        assert_eq!(
            phase,
            ProgressEventV2::Load(LoadProgressV2 {
                stage: "loading_qwen_asr".to_owned(),
                measure: LoadMeasureV2::Phase {
                    edge: PhaseEdgeV2::Complete
                },
            })
        );
        let wire = serde_json::json!({
            "scope": "request",
            "request_id": "req-7",
            "stage": "stanza_processing",
            "progress": {"completed": 1, "total": 3},
        });
        let request = parse(wire.clone()).expect("units are request progress");
        let expected = RequestProgressV2::new(
            WorkerRequestIdV2::from("req-7"),
            "stanza_processing".to_owned(),
            UnitProgressV2::new(1, 3).expect("1 of 3"),
        )
        .expect("a named request");
        assert_eq!(request, ProgressEventV2::Request(expected));
        assert_eq!(
            serde_json::to_value(&request).expect("serializes"),
            wire,
            "a request event writes back exactly as it was read"
        );
    }

    /// The states the scope makes unrepresentable are refused at the wire:
    /// units in a load, growth in a request, a request with no id, a count
    /// past its total, and the pre-scope shape.
    #[test]
    fn a_scope_and_its_measure_cannot_disagree() {
        let refused = [
            serde_json::json!({
                "scope": "load", "stage": "s",
                "measure": {"kind": "units", "completed": 1, "total": 3},
            }),
            serde_json::json!({
                "scope": "request", "request_id": "req-7", "stage": "s",
                "progress": {"kind": "download_growth", "grown_kib": 7},
            }),
            serde_json::json!({
                "scope": "request", "request_id": "", "stage": "s",
                "progress": {"completed": 1, "total": 3},
            }),
            serde_json::json!({
                "scope": "request", "request_id": "req-7", "stage": "s",
                "progress": {"completed": 4, "total": 3},
            }),
            serde_json::json!({
                "scope": "load", "stage": "s",
                "measure": {"kind": "phase", "edge": "middle"},
            }),
            serde_json::json!({
                "request_id": "", "stage": "s",
                "measure": {"kind": "phase", "edge": "begin"},
            }),
        ];
        for event in refused {
            assert!(parse(event.clone()).is_err(), "admitted {event}");
        }
    }

    /// The constructors refuse what deserialization refuses.
    #[test]
    fn the_constructors_refuse_what_the_wire_refuses() {
        assert_eq!(
            UnitProgressV2::new(4, 3),
            Err(ProgressRefusalV2::CompletedExceedsTotal {
                completed: 4,
                total: 3
            })
        );
        let done = UnitProgressV2::new(3, 3).expect("all done is progress");
        assert_eq!((done.completed(), done.total()), (3, 3));
        assert_eq!(
            RequestProgressV2::new(WorkerRequestIdV2::from(""), "s".to_owned(), done),
            Err(ProgressRefusalV2::EmptyRequestId)
        );
    }
}
