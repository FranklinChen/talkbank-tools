//! Shared debug-artifact model for inspectable job runs.
//!
//! Direct and server-backed execution should share the shape of the debugging
//! handles they expose even if they do not share the same persistence or
//! transport layer.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::api::JobId;
use crate::store::JobDetail;

/// Stable debug handles for one completed or inspectable job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobDebugArtifacts {
    /// Stable job identifier.
    pub job_id: JobId,
    /// Host-local staging directory containing input/output/debug artifacts.
    pub staging_dir: PathBuf,
    /// Persisted trace file when trace capture was enabled and exported.
    pub trace_file: Option<PathBuf>,
}

impl JobDebugArtifacts {
    /// Build one debug-artifact summary from a job detail snapshot.
    pub fn from_job_detail(job_id: JobId, detail: &JobDetail, trace_file: Option<PathBuf>) -> Self {
        Self {
            job_id,
            staging_dir: detail.staging_dir.as_path().to_owned(),
            trace_file,
        }
    }
}
