//! Row types: flat DB rows mapping 1:1 to SQLite table columns.

use crate::api::MachineTime;
use crate::options::CommandOptions;
use crate::scheduling::LeaseRecord;

/// Flat representation of a row from the `jobs` SQLite table.
///
/// This is a persistence-layer type, not an API type. It mirrors the DB schema
/// column-for-column and is only used for insert/load roundtrips between
/// [`JobStore`](crate::store::JobStore) and SQLite. JSON-encoded array/map
/// columns are deserialized into their Rust equivalents on load.
#[derive(Debug, Clone)]
pub struct JobRow {
    /// Primary key. Column `job_id TEXT PRIMARY KEY`.
    /// A UUID generated at submission time.
    pub job_id: String,

    /// Column `correlation_id TEXT NOT NULL DEFAULT ''`.
    /// Client-supplied idempotency key that groups related resubmissions.
    /// Defaults to `job_id` when the client omits it.
    pub correlation_id: String,

    /// Column `command TEXT NOT NULL`.
    /// The processing command to run, e.g. `"morphotag"`, `"align"`, `"transcribe"`.
    /// Must be one of the commands advertised in worker capabilities.
    pub command: String,

    /// Column `lang TEXT NOT NULL`.
    /// ISO 639-3 language code (e.g. `"eng"`, `"spa"`).
    pub lang: String,

    /// Column `num_speakers INTEGER NOT NULL DEFAULT 1`.
    /// Expected speaker count, passed to diarization. Must be >= 1.
    pub num_speakers: u32,

    /// Column `status TEXT NOT NULL DEFAULT 'queued'`.
    /// Job lifecycle state. One of: `"queued"`, `"running"`, `"completed"`,
    /// `"failed"`, `"cancelled"`, `"interrupted"`.
    pub status: String,

    /// Column `error TEXT` (nullable).
    /// Human-readable error message when `status` is `"failed"`.
    /// `None` for non-failed jobs.
    pub error: Option<String>,

    /// Column `staging_dir TEXT NOT NULL`.
    /// Absolute path to the job's staging directory under the jobs root,
    /// where uploaded files and results are stored.
    pub staging_dir: String,

    /// Column `filenames TEXT NOT NULL`.
    /// Stored as a JSON array of strings (e.g. `["a.cha","b.cha"]`).
    /// The ordered list of filenames to process. Each entry has a
    /// corresponding row in `file_statuses`.
    pub filenames: Vec<String>,

    /// Column `has_chat TEXT NOT NULL`.
    /// Stored as a JSON array of booleans, parallel to `filenames`.
    /// `true` if the file already contains CHAT content (as opposed to
    /// being a media-only file that needs transcription).
    pub has_chat: Vec<bool>,

    /// Column `options TEXT NOT NULL DEFAULT '{}'`.
    /// Stored as serialized [`CommandOptions`] JSON.
    pub options: CommandOptions,

    /// Column `media_mapping TEXT NOT NULL DEFAULT ''`.
    /// Key into `ServerConfig.media_mappings` that tells the server how to
    /// resolve media file paths. Empty string means no mapping.
    pub media_mapping: String,

    /// Column `media_subdir TEXT NOT NULL DEFAULT ''`.
    /// Subdirectory under the resolved media root where media files for
    /// this job live. Empty string means the media root itself.
    pub media_subdir: String,

    /// Column `source_dir TEXT NOT NULL DEFAULT ''`.
    /// Original source directory on the submitter's machine. Used for
    /// display purposes and result download path construction.
    pub source_dir: String,

    /// Column `submitted_by TEXT NOT NULL DEFAULT ''`.
    /// The submitting client's address, `''` when none was recorded. Raw
    /// here; `store::Submitter::from_columns` is the one reader that turns
    /// it, with the name, into an `Option<Submitter>`.
    pub submitted_by: String,

    /// Column `submitted_by_name TEXT NOT NULL DEFAULT ''`.
    /// The name the address resolved to, `''` when none was recorded. Read
    /// only through `store::Submitter::from_columns`.
    pub submitted_by_name: String,

    /// Column `submitted_at REAL NOT NULL`.
    /// Unix timestamp (seconds since epoch, with fractional part) when the
    /// job was submitted.
    pub submitted_at: MachineTime,

    /// Column `completed_at REAL` (nullable).
    /// Unix timestamp when the job reached a terminal state (`"completed"`,
    /// `"failed"`, `"cancelled"`, `"interrupted"`). `None` while still active.
    pub completed_at: Option<MachineTime>,

    /// Column `num_workers INTEGER` (nullable).
    /// Number of workers assigned to this job once it starts running.
    /// `None` while queued.
    pub num_workers: Option<i32>,

    /// Column `next_eligible_at REAL` (nullable).
    /// Earliest unix timestamp when a deferred job should be retried.
    pub next_eligible_at: Option<MachineTime>,
    /// Columns `leased_by_node TEXT`, `lease_expires_at REAL` and
    /// `lease_heartbeat_at REAL` (all nullable, written together): the lease,
    /// or `None` when all three are NULL. A row holding some but not all of
    /// them fails to load.
    pub lease: Option<LeaseRecord>,

    /// Column `last_cancelled_at REAL` (nullable). Most recent cancel
    /// attempt's timestamp; `None` until at least one cancel arrives.
    /// Denormalized projection of the `cancellations` audit table for
    /// recovery hydration without a JOIN.
    pub last_cancelled_at: Option<MachineTime>,
    /// Column `last_cancelled_source TEXT` (nullable). Wire-format source
    /// of the most recent cancel (`tui`, `api`, `signal`, ...).
    pub last_cancelled_source: Option<String>,
    /// Column `last_cancelled_host TEXT` (nullable). Caller-reported host.
    pub last_cancelled_host: Option<String>,
    /// Column `last_cancelled_reason TEXT` (nullable). Caller-reported
    /// free-form reason text.
    pub last_cancelled_reason: Option<String>,

    /// Column `paths_mode INTEGER NOT NULL DEFAULT 0`.
    /// Stored as 0/1 integer. When `true`, the server reads/writes files
    /// directly on the filesystem (local daemon mode) instead of using
    /// staging directory content transfer.
    pub paths_mode: bool,

    /// Column `source_paths TEXT NOT NULL DEFAULT '[]'`.
    /// Stored as a JSON array of absolute paths. Only populated when
    /// `paths_mode` is `true`: the original input file paths on disk.
    /// Parallel to `filenames`.
    pub source_paths: Vec<String>,

    /// Column `output_paths TEXT NOT NULL DEFAULT '[]'`.
    /// Stored as a JSON array of absolute paths. Only populated when
    /// `paths_mode` is `true`, where results should be written on disk.
    /// Parallel to `filenames`.
    pub output_paths: Vec<String>,

    /// Eagerly loaded from the `file_statuses` table (one row per filename).
    /// Not a column on the `jobs` table itself, populated by a second query
    /// in [`JobDB::load_all_jobs`].
    pub file_statuses: Vec<FileStatusRow>,
}

/// Flat representation of a row from the `file_statuses` SQLite table.
///
/// Each row tracks the processing state of one file within a job. The
/// composite primary key is `(job_id, filename)` with a foreign key
/// cascade to the parent `jobs` row.
#[derive(Debug, Clone)]
pub struct FileStatusRow {
    /// Part of composite PK `(job_id, filename)`.
    /// Column `filename TEXT NOT NULL`. The basename of the file being
    /// processed (e.g. `"recording.cha"`).
    pub filename: String,

    /// Column `status TEXT NOT NULL DEFAULT 'queued'`.
    /// Per-file lifecycle state. One of: `"queued"`, `"processing"`,
    /// `"done"`, `"diagnosed"`, `"error"`, `"interrupted"`.
    pub status: String,

    /// Column `error TEXT` (nullable).
    /// Human-readable error message when `status` is `"error"`.
    /// `None` for non-errored files.
    pub error: Option<String>,

    /// Column `error_category TEXT` (nullable).
    /// Machine-readable error classification (e.g. `"parse_error"`,
    /// `"worker_crash"`). Used by the dashboard to group failures.
    pub error_category: Option<String>,

    /// Column `diagnostics TEXT` (nullable).
    /// JSON object of a diagnosed file's admission findings
    /// (`api::FileOutputDiagnostics`). Raw here; decoded at the database
    /// boundary (`store::queries::recover_file_phase`). `None` for every
    /// phase but `"diagnosed"`.
    pub diagnostics: Option<String>,

    /// Column `content_type TEXT NOT NULL DEFAULT 'chat'`.
    /// MIME-like content descriptor for the result file. Typically
    /// `"chat"` for `.cha` output or `"csv"` for analysis results.
    pub content_type: String,

    /// Column `started_at REAL` (nullable).
    /// Unix timestamp (seconds since epoch) when processing of this file
    /// began. `None` while queued.
    pub started_at: Option<MachineTime>,

    /// Column `finished_at REAL` (nullable).
    /// Unix timestamp (seconds since epoch) when processing of this file
    /// finished (success or failure). `None` while still processing.
    pub finished_at: Option<MachineTime>,

    /// Column `next_eligible_at REAL` (nullable).
    /// Earliest unix timestamp when a deferred retry may start.
    pub next_eligible_at: Option<MachineTime>,
}

/// Flat representation of a row from the `attempts` SQLite table.
///
/// Each row records one concrete execution attempt for a schedulable work
/// unit. The row mirrors the durable control-plane model in
/// `batchalign_types::scheduling::AttemptRecord`.
#[derive(Debug, Clone)]
pub struct AttemptRow {
    /// Primary key. Column `attempt_id TEXT PRIMARY KEY`.
    pub attempt_id: String,
    /// Parent job identifier. Column `job_id TEXT NOT NULL`.
    pub job_id: String,
    /// Opaque work-unit identifier within the job. Column
    /// `work_unit_id TEXT NOT NULL`.
    pub work_unit_id: String,
    /// Work-unit kind string (snake_case). Column
    /// `work_unit_kind TEXT NOT NULL`.
    pub work_unit_kind: String,
    /// 1-based attempt number for this work unit. Column
    /// `attempt_number INTEGER NOT NULL`.
    pub attempt_number: i32,
    /// Start timestamp, unix seconds with fractional precision. Column
    /// `started_at REAL NOT NULL`.
    pub started_at: MachineTime,
    /// Finish timestamp, unix seconds with fractional precision. Column
    /// `finished_at REAL`.
    pub finished_at: Option<MachineTime>,
    /// Final outcome string (snake_case). Column `outcome TEXT NOT NULL`.
    pub outcome: String,
    /// Broad failure category string (snake_case). Column
    /// `failure_category TEXT`.
    pub failure_category: Option<String>,
    /// Scheduler disposition string (snake_case). Column
    /// `disposition TEXT NOT NULL`.
    pub disposition: String,
    /// Future fleet node identifier. Column `worker_node_id TEXT`.
    pub worker_node_id: Option<String>,
    /// Local worker process id when known. Column `worker_pid INTEGER`.
    pub worker_pid: Option<i32>,
}
