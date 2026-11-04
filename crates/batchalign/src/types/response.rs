//! REST API response structures, job info, results, health, listing.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

pub use super::result_content::{ArtifactDigest, BinaryResultDescriptor, ResultContent};
use serde::{Deserialize, Serialize};

use crate::options::CommandOptions;
use crate::scheduling::{FailureCategory, LeaseRecord};

use super::domain::{
    ContentType, DisplayPath, HealthStatus, JobId, LanguageSpec, MachineTime, MemoryMb, NodeId,
    NonNegativeSeconds, ReleasedCommand,
};
use super::status::{FileProgressStage, FileStatusKind, JobStatus};

// ---------------------------------------------------------------------------
// Response models
// ---------------------------------------------------------------------------

/// Result for a single processed file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FileResult {
    /// Display path for this file: a bare basename (`"sample.cha"`) or a
    /// relative forward-slash path (`"PWA/TYO_a1.cha"`) for directory input.
    /// Backslashes are normalized to forward slashes on construction.
    pub filename: DisplayPath,
    /// Inline text or a binary artifact descriptor. This is untrusted wire
    /// data, not a destination or write-admission capability.
    #[serde(default)]
    pub content: ResultContent,
    /// MIME-like content discriminator: `"chat"` for CHAT files (default),
    /// `"csv"` for tabular output (e.g. opensmile features).
    #[serde(default)]
    pub content_type: ContentType,
    /// Human-readable error message if this file failed processing.
    /// `None` for successfully processed files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Processing provenance read from the output file, as a typed state:
    /// the parsed stamps, a stamp of ours that did not parse, or nothing read
    /// (non-CHAT output, or a file that failed).
    pub provenance: FileProvenance,
}

/// The provenance stamps read from one result file.
///
/// A typed state rather than a list that an unreadable stamp would either
/// silently shorten or turn into a failure of the whole response: a stamp of
/// ours that does not parse is that file's state, and the rest of the job's
/// results are still served.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FileProvenance {
    /// Nothing was read: the output is not CHAT, or the file failed.
    NotRead,
    /// Every stamp of ours parsed. Each entry records one batchalign3
    /// command that was applied (command name, fields, timestamp); empty when
    /// the file carries no stamps.
    Parsed {
        /// The stamps, in file order.
        entries: Vec<crate::provenance::ProvenanceEntry>,
    },
    /// A comment opens as one of our stamps but does not parse.
    Unparseable {
        /// Which stamp and what is wrong with it.
        reason: String,
    },
}

/// What a command decided about stamping one file with provenance.
///
/// Recorded when the file completes and reported with its per-file status, so
/// "this file carries no stamp" is an answer with a reason rather than
/// something an operator has to infer from an absent comment. Not persisted:
/// a file whose status is rebuilt from the database after a restart reads
/// `Unrecorded`, which is the honest answer for a record that never held it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FileStampOutcome {
    /// No stamp decision was recorded for this file: a command that writes no
    /// per-file provenance stamp, or a status restored from the database.
    #[default]
    Unrecorded,
    /// The command stamped this file.
    Stamped {
        /// The command whose stamp was written.
        command: String,
    },
    /// The command wrote no stamp on this file, for this reason.
    NotStamped {
        /// The command that ran.
        command: String,
        /// Why it wrote no stamp (for example, no engine produced anything
        /// that was applied).
        reason: String,
    },
}

impl FileStampOutcome {
    /// Whether no stamp decision was recorded for this file.
    ///
    /// Serialization omits the field in that case, so a per-file record grows
    /// a `stamp` exactly when a command decided one and the existing wire
    /// shape is unchanged for every command that stamps nothing.
    #[must_use]
    pub fn is_unrecorded(&self) -> bool {
        matches!(self, Self::Unrecorded)
    }
}

/// What the server decided about one worker key's latest capability report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct WorkerCapabilityAdmission {
    /// Worker key label: `target:lang`, plus the engine selection when set.
    pub worker_key: String,
    /// Admitted or refused.
    pub outcome: CapabilityAdmissionOutcome,
}

/// The outcome of admitting one worker key's capability report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CapabilityAdmissionOutcome {
    /// Admitted; dispatch uses this worker for these infer tasks.
    Admitted {
        /// The infer tasks the worker supports.
        #[cfg_attr(feature = "server", schema(value_type = Vec<String>))]
        infer_tasks: Vec<crate::worker::InferTask>,
    },
    /// Refused; the worker is not used until it reports again and is admitted.
    Refused {
        /// Why the report was refused.
        reason: crate::engine_reports::EngineReportAdmissionError,
    },
}

/// A registry daemon the server found alive and refused to adopt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct RefusedRegistryWorker {
    /// The daemon's worker key as its registry entry names it:
    /// `profile:<profile>:<lang>`.
    pub worker_key: String,
    /// The daemon's process id, from its registry entry.
    pub pid: u32,
    /// Why it was refused.
    pub reason: RegistryWorkerRefusal,
}

/// Why a registry daemon was not adopted. Either way the remedy is to restart
/// the daemon with this server's build (`batchalign3 worker stop`, then
/// `batchalign3 worker start` for its profile and language).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RegistryWorkerRefusal {
    /// The registry entry names a different build than this server's.
    ForeignBuild {
        /// The build the entry names.
        reported_build: String,
        /// This server's build.
        server_build: String,
    },
    /// The registry entry names no build at all (written by a daemon from
    /// before build identity was recorded, or started without it).
    UnreportedBuild {
        /// This server's build.
        server_build: String,
    },
}

/// What a file written with diagnostics reports.
///
/// Carried by a file whose status is [`FileStatusKind::Diagnosed`]: its output
/// was written but is not certified complete, because its producer's admission
/// found something (transcription generated CHAT that Chatter did not admit),
/// because requested work did not apply (a shortfall), or both. Validation is
/// exactly as strict as for any other output; what differs is that the verdict
/// is reported beside the written file instead of replacing it.
///
/// Bounded: its findings carry the count, a count per error code and the
/// first [`Self::FIRST_FINDINGS`] findings, because the record is copied into
/// every file status entry on every poll. A longer list is written once, in
/// full, to a sidecar file ([`FullFindings::Sidecar`]).
///
/// The bar the output was judged against exists only with findings: a file
/// diagnosed for its shortfalls alone was admitted, and has no findings and no
/// bar to report. Records written before the bar was recorded (a flat
/// `finding_count` at the top level) are still read: every such record was
/// written for transcription's generated output, the only producer of
/// diagnosed output then, which is judged against complete construction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(try_from = "FileOutputDiagnosticsWire")]
pub struct FileOutputDiagnostics {
    /// What output admission found, with the bar it judged against. Absent
    /// when the document was admitted and only shortfalls diagnose the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub findings: Option<JudgedFindingsRecord>,
    /// Requested work the written document does not carry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shortfalls: Vec<OutputShortfallRecord>,
}

/// What one output judgement found, bounded, with the bar it was held to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JudgedFindingsRecord {
    /// The bar the output was judged against.
    pub bar: JudgementBar,
    /// How many findings the judgement made, in all; at least one.
    pub finding_count: u64,
    /// How many findings carried each error code, most frequent first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub findings_by_code: Vec<FindingCodeCount>,
    /// The first findings, in the order the judgement found them, at most
    /// [`FileOutputDiagnostics::FIRST_FINDINGS`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub first_findings: Vec<OutputFindingRecord>,
    /// Where the complete list is.
    pub full_findings: FullFindings,
}

/// Every stored shape of [`FileOutputDiagnostics`], read into the current
/// one. Each shape refuses fields it does not have, so neither can read the
/// other (or a misspelled record) as a different, emptier record; the flat
/// shape alone has `finding_count` and `full_findings` at the top level.
#[derive(Deserialize)]
#[serde(untagged)]
enum FileOutputDiagnosticsWire {
    Flat(FlatDiagnosticsWire),
    Judged(JudgedDiagnosticsWire),
}

/// The flat shape build 95b74761 stored.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FlatDiagnosticsWire {
    finding_count: u64,
    #[serde(default)]
    findings_by_code: Vec<FindingCodeCount>,
    #[serde(default)]
    first_findings: Vec<OutputFindingRecord>,
    full_findings: FullFindings,
    #[serde(default)]
    shortfalls: Vec<OutputShortfallRecord>,
}

/// The current shape.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JudgedDiagnosticsWire {
    #[serde(default)]
    findings: Option<JudgedFindingsRecord>,
    #[serde(default)]
    shortfalls: Vec<OutputShortfallRecord>,
}

/// A stored record that reports nothing: a diagnosed file always has a
/// finding or a shortfall.
#[derive(Debug, thiserror::Error)]
#[error("a diagnostics record must carry findings or shortfalls")]
pub struct EmptyDiagnosticsRecord;

impl TryFrom<FileOutputDiagnosticsWire> for FileOutputDiagnostics {
    type Error = EmptyDiagnosticsRecord;

    fn try_from(wire: FileOutputDiagnosticsWire) -> Result<Self, Self::Error> {
        let record = match wire {
            FileOutputDiagnosticsWire::Judged(JudgedDiagnosticsWire {
                findings,
                shortfalls,
            }) => Self {
                findings,
                shortfalls,
            },
            FileOutputDiagnosticsWire::Flat(FlatDiagnosticsWire {
                finding_count,
                findings_by_code,
                first_findings,
                full_findings,
                shortfalls,
            }) => Self {
                // Only transcription wrote flat records, judged against
                // complete construction; see the type's documentation.
                findings: (finding_count > 0).then_some(JudgedFindingsRecord {
                    bar: JudgementBar::Construction,
                    finding_count,
                    findings_by_code,
                    first_findings,
                    full_findings,
                }),
                shortfalls,
            },
        };
        match (&record.findings, record.shortfalls.as_slice()) {
            (None, []) => Err(EmptyDiagnosticsRecord),
            _ => Ok(record),
        }
    }
}

impl FileOutputDiagnostics {
    /// How many findings are kept on the record itself.
    pub const FIRST_FINDINGS: usize = 20;

    /// The `file_statuses.diagnostics` column text: this record's own JSON.
    pub fn to_column_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_value(self).map(|value| value.to_string())
    }

    /// How many findings the record reports, zero when only shortfalls
    /// diagnose the file.
    pub fn finding_count(&self) -> u64 {
        self.findings
            .as_ref()
            .map_or(0, |findings| findings.finding_count)
    }

    /// The first findings on the record.
    pub fn first_findings(&self) -> &[OutputFindingRecord] {
        self.findings
            .as_ref()
            .map_or(&[], |findings| findings.first_findings.as_slice())
    }

    /// A record of `findings` (all on the record, judged against complete
    /// construction) and `shortfalls`, for tests of the consumers. Production
    /// records are built by recording a diagnosed output's draft.
    #[cfg(test)]
    pub(crate) fn of_findings(
        findings: Vec<OutputFindingRecord>,
        shortfalls: Vec<OutputShortfallRecord>,
    ) -> Self {
        Self {
            findings: (!findings.is_empty()).then(|| JudgedFindingsRecord {
                bar: JudgementBar::Construction,
                finding_count: findings.len() as u64,
                findings_by_code: FindingCodeCount::tally(&findings),
                first_findings: findings,
                full_findings: FullFindings::Inline,
            }),
            shortfalls,
        }
    }

    /// One coded finding, for tests.
    #[cfg(test)]
    pub(crate) fn coded_finding(code: &str, message: &str) -> OutputFindingRecord {
        OutputFindingRecord {
            code: Some(code.to_owned()),
            level: FindingLevel::StructurallyComplete,
            message: message.to_owned(),
        }
    }

    /// The record as operator-facing lines: the bar and the first findings,
    /// how many more there are and where, then each shortfall.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(findings) = &self.findings {
            lines.push(format!("judged against: {}", findings.bar));
            lines.extend(findings.first_findings.iter().map(ToString::to_string));
            let shown = findings.first_findings.len() as u64;
            if findings.finding_count > shown {
                let more = findings.finding_count - shown;
                lines.push(match &findings.full_findings {
                    FullFindings::Sidecar { path } => format!(
                        "... and {more} more finding(s); the full list is in {path}"
                    ),
                    FullFindings::Unwritten { path, error } => format!(
                        "... and {more} more finding(s); the full list could not be written to {path}: {error}"
                    ),
                    FullFindings::Inline => format!("... and {more} more finding(s)"),
                });
            }
        }
        for shortfall in &self.shortfalls {
            lines.push(shortfall.to_string());
            lines.extend(
                shortfall
                    .detail_lines()
                    .into_iter()
                    .map(|line| format!("  {line}")),
            );
        }
        lines
    }
}

/// One admission finding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct OutputFindingRecord {
    /// The CHAT error code (for example `E220`), when the finding has one: a
    /// command's own completion checks name none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// The validity level the finding belongs to.
    pub level: FindingLevel,
    /// What was found, without the code.
    pub message: String,
}

impl std::fmt::Display for OutputFindingRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.code {
            Some(code) => write!(f, "{code} {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

/// Which check made a finding: a coarse gate level, or complete construction
/// admission.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum FindingLevel {
    /// Gate: the document does not parse.
    Parseable,
    /// Gate: the document is not structurally complete.
    StructurallyComplete,
    /// Gate: a main tier is not well formed.
    MainTierValid,
    /// Complete CHAT construction admission (the full validator, the
    /// authoritative write check), which the gate levels do not grade.
    Construction,
}

/// How many findings carried one error code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FindingCodeCount {
    /// The code; absent for findings that carry none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// How many findings carried it.
    pub count: u64,
}

impl FindingCodeCount {
    /// Count `findings` per code, most frequent first (ties by code). The
    /// one tally behind every bounded record of findings.
    pub fn tally<'a>(findings: impl IntoIterator<Item = &'a OutputFindingRecord>) -> Vec<Self> {
        let mut by_code = std::collections::BTreeMap::<Option<String>, u64>::new();
        for finding in findings {
            *by_code.entry(finding.code.clone()).or_default() += 1;
        }
        let mut tally: Vec<Self> = by_code
            .into_iter()
            .map(|(code, count)| Self { code, count })
            .collect();
        tally.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.code.cmp(&b.code)));
        tally
    }
}

/// The bar an output was judged against, as a refusal reports it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum JudgementBar {
    /// Complete checked CHAT construction.
    Construction,
    /// Complete construction plus the facts recorded from the input.
    Preservation,
}

impl std::fmt::Display for JudgementBar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Construction => write!(f, "complete CHAT construction required"),
            Self::Preservation => write!(f, "complete CHAT construction and preservation required"),
        }
    }
}

/// Why an optional stage's own output was not admitted, bounded as
/// [`FileOutputDiagnostics`] is: it travels in every poll's file status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageRefusalRecord {
    /// The stage's output was judged and failed.
    Judged {
        /// The bar it was judged against.
        bar: JudgementBar,
        /// How many findings the judgement made, in all.
        finding_count: u64,
        /// How many findings carried each error code, most frequent first.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        findings_by_code: Vec<FindingCodeCount>,
        /// The first findings, at most [`FileOutputDiagnostics::FIRST_FINDINGS`].
        first_findings: Vec<OutputFindingRecord>,
    },
    /// The stage could not establish an output to judge; one statement of
    /// why, as its producer gave it.
    Unestablished {
        /// The producer's statement.
        reason: String,
    },
}

impl StageRefusalRecord {
    /// The bounded record of a judgement's `findings` against `bar`. The
    /// findings are walked, not collected: only the first few are kept.
    pub fn judged<'a, I>(bar: JudgementBar, findings: I) -> Self
    where
        I: IntoIterator<Item = &'a OutputFindingRecord>,
        I::IntoIter: Clone,
    {
        let findings = findings.into_iter();
        Self::Judged {
            bar,
            finding_count: findings.clone().count() as u64,
            findings_by_code: FindingCodeCount::tally(findings.clone()),
            first_findings: findings
                .take(FileOutputDiagnostics::FIRST_FINDINGS)
                .cloned()
                .collect(),
        }
    }
}

impl std::fmt::Display for StageRefusalRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Judged {
                bar,
                finding_count,
                first_findings,
                ..
            } => {
                write!(f, "{bar}: {finding_count} finding(s)")?;
                if let Some(first) = first_findings.first() {
                    write!(f, ", first: {first}")?;
                }
                Ok(())
            }
            Self::Unestablished { reason } => f.write_str(reason),
        }
    }
}

/// Where the complete list of a file's findings is.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FullFindings {
    /// On the record: `first_findings` is the whole list.
    Inline,
    /// Too long for the record: written once, in full, as JSON to this
    /// server-side file in the job's staging directory.
    Sidecar {
        /// Path of the sidecar file on the server.
        path: String,
    },
    /// Too long for the record, and writing the sidecar failed: only the
    /// first findings, the count and the tally per code are available. The
    /// output itself was written; a diagnostics file that could not be
    /// written does not make it an error.
    Unwritten {
        /// Where the sidecar was to be written.
        path: String,
        /// Why writing it failed.
        error: String,
    },
}

/// An optional later stage of a generating producer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum OptionalStage {
    /// Post-CHAT utterance segmentation.
    UtteranceSegmentation,
    /// Morphosyntactic analysis.
    Morphosyntax,
}

impl OptionalStage {
    /// The stage's name in a report.
    pub fn name(self) -> &'static str {
        match self {
            Self::UtteranceSegmentation => "utterance segmentation",
            Self::Morphosyntax => "morphosyntax",
        }
    }
}

/// Which timing-recovery attempt a failure belongs to.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum UtrPass {
    /// Recovery before the first forced-alignment attempt.
    Initial,
    /// The single recovery opportunity before an FA worker retry.
    Fallback,
}

impl UtrPass {
    /// Stable name used by the wire record, provenance and presentation.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Fallback => "fallback",
        }
    }
}

/// Requested work a written document does not carry, and why.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputShortfallRecord {
    /// A particular timing-recovery attempt was unavailable. A later pass may
    /// have contributed timing; this record does not invalidate that evidence.
    UtrUnavailable {
        /// Selected backend, not an engine claimed to have produced timing.
        engine: String,
        /// Which attempt failed; initial and fallback failures remain distinct.
        pass: UtrPass,
        /// Typed control-plane cause, also recorded in the CHAT stamp.
        category: FailureCategory,
        /// Bounded explanatory detail. Raw provider text is never stamp syntax.
        detail: String,
    },
    /// Morphology was retained/written, but its complete provenance could not
    /// be reconstructed. Original records remain in CHAT as evidence.
    MorphologyProvenanceIncomplete {
        /// First metadata defect, bounded independently of retained evidence.
        detail: String,
    },
    /// Primary injection could not locate secondary words safely for dispatch.
    SecondaryMorphologyUnplaced {
        /// Number of affected utterances.
        utterances: u64,
        /// Secondary words not dispatched.
        words: u64,
        /// Bounded utterance locations; exact morphology positions are unknown.
        first_utterances: Vec<SecondaryMorphologyUnplacedUtterance>,
    },
    /// Secondary morphology retained placeholders for requested spans.
    SecondaryMorphologyIncomplete {
        /// Number of spans whose requested analysis was not installed.
        spans: u64,
        /// Number of morphology positions in those spans.
        words: u64,
        /// Transcript-ordered, bounded examples with producer causes.
        first_spans: Vec<SecondaryMorphologyFailure>,
    },
    /// The stage requires an admitted document, and the generated output was
    /// diagnosed, so the stage did not run.
    StageSkipped {
        /// The stage.
        stage: OptionalStage,
    },
    /// The stage ran and its own output could not be admitted; the admitted
    /// document from before it was written instead.
    StageNotApplied {
        /// The stage.
        stage: OptionalStage,
        /// Why, as the stage's own admission reported it, bounded.
        refusal: StageRefusalRecord,
    },
    /// A per-utterance stage ran on a diagnosed document and left out the
    /// utterances its findings are confined to: every other utterance has
    /// the stage's result, these keep their generated form.
    StageHeldOut {
        /// The stage.
        stage: OptionalStage,
        /// How many utterances it left out, at least one.
        held_out_utterances: u64,
        /// The first of them, at most
        /// [`FileOutputDiagnostics::FIRST_FINDINGS`], located in the document
        /// from before the stage. They keep their generated form, so the
        /// main tier finds them in the written file; the position counts the
        /// utterances before the stage, which a segmentation stage changes.
        #[cfg_attr(feature = "server", schema(value_type = Vec<StoredUtteranceLocation>))]
        first_held_out: Vec<StoredUtteranceLocation>,
    },
    /// The recognizer's decode windows disagreed about a repeated observation
    /// at a seam, and the reconciler kept one copy by rule (the earlier
    /// window's) rather than by evidence. The transcript carries every other
    /// word; each seam is marked with a `%com` on the utterance that keeps the
    /// copy, for review. Whole recordings were refused for this until
    /// 2026-11-03.
    WindowSeamAmbiguous {
        /// How many seams were reconciled by rule, at least one.
        seams: u64,
        /// The first of them, in recording order, at most
        /// [`FileOutputDiagnostics::FIRST_FINDINGS`].
        first_seams: Vec<WindowSeamRecord>,
    },
    /// Forced alignment left required words untimed. The measured timing
    /// was written; these words are in the transcript without bullets.
    TimingIncomplete {
        /// Required lexical words in the file.
        required_words: u64,
        /// How many of them have no positive interval, at least one.
        untimed_words: u64,
        /// How many utterances have untimed words, at least one.
        untimed_utterances: u64,
        /// The first of those utterances, in transcript order, at most
        /// [`FileOutputDiagnostics::FIRST_FINDINGS`].
        first_untimed: Vec<UntimedUtteranceRecord>,
    },
}

impl OutputShortfallRecord {
    /// The lines under this shortfall's own line that a reader acts on, in
    /// order; empty when its line says everything. Exhaustive, so a new
    /// shortfall decides here whether it names anything.
    ///
    /// A shortfall that carries utterances names each one the record holds,
    /// as written, then how many more it counted: a reader acts on the
    /// utterances, and the headline gives only the counts.
    pub fn detail_lines(&self) -> Vec<String> {
        match self {
            Self::TimingIncomplete {
                untimed_utterances,
                first_untimed,
                ..
            } => listed(
                first_untimed,
                *untimed_utterances,
                "utterance(s) with untimed words",
            ),
            Self::StageHeldOut {
                held_out_utterances,
                first_held_out,
                ..
            } => listed(
                first_held_out,
                *held_out_utterances,
                "utterance(s) held out",
            ),
            Self::SecondaryMorphologyUnplaced {
                utterances,
                first_utterances,
                ..
            } => listed(first_utterances, *utterances, "utterance(s) not dispatched"),
            Self::SecondaryMorphologyIncomplete {
                spans, first_spans, ..
            } => listed(first_spans, *spans, "span(s) with L2 placeholders"),
            Self::WindowSeamAmbiguous { seams, first_seams } => {
                listed(first_seams, *seams, "seam(s) reconciled by rule")
            }
            Self::UtrUnavailable { .. }
            | Self::MorphologyProvenanceIncomplete { .. }
            | Self::StageSkipped { .. }
            | Self::StageNotApplied { .. } => Vec::new(),
        }
    }
}

/// One line per recorded item, then how many of `total` the record does not
/// hold, named by `rest`.
fn listed<T: std::fmt::Display>(recorded: &[T], total: u64, rest: &str) -> Vec<String> {
    let mut lines: Vec<String> = recorded.iter().map(ToString::to_string).collect();
    let shown = recorded.len() as u64;
    if total > shown {
        lines.push(format!("... and {} more {rest}", total - shown));
    }
    lines
}

impl std::fmt::Display for SecondaryMorphologyUnplacedUtterance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}   ({} word(s): {})",
            self.location, self.words, self.detail
        )
    }
}

impl std::fmt::Display for SecondaryMorphologyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.span, self.detail)
    }
}

impl std::fmt::Display for SecondaryMorphologySpan {
    /// The utterance as written, then which of its words the span covers.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}   ({}, {} word(s) from word {})",
            self.location, self.language, self.words, self.first_word
        )
    }
}

/// A primary-injection failure before a secondary span could be planned.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct SecondaryMorphologyUnplacedUtterance {
    /// Where the utterance is.
    #[serde(flatten)]
    pub location: StoredUtteranceLocation,
    /// Secondary words left as placeholders.
    pub words: u64,
    /// Typed extraction boundary.
    pub cause: SecondaryExtractionFailure,
    /// Bounded producer explanation.
    pub detail: String,
}

/// Why secondary positions could not be established from primary injection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum SecondaryExtractionFailure {
    /// The language marker has no dispatchable resolution in this document.
    LanguageUnresolved,
    /// Primary analysis does not align with the words.
    Alignment,
    /// Text and analysis tokenization disagree.
    TokenizationDisagrees,
    /// A secondary word became multiple morphology items.
    SplitWord,
    /// An analysis word has no morphology-item address.
    UnmappedWord,
}

/// Address of a secondary-language span in the post-injection document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct SecondaryMorphologySpan {
    /// Where the span's utterance is.
    #[serde(flatten)]
    pub location: StoredUtteranceLocation,
    /// One-based morphology-item position within that utterance.
    pub first_word: u64,
    /// Morphology items in the span, not expanded clitic chunks.
    pub words: u64,
    /// Requested CHAT language code.
    pub language: String,
}

/// Which producer boundary could not supply secondary morphology.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum SecondaryMorphologyFailureKind {
    /// The requested code cannot be dispatched.
    InvalidLanguage,
    /// Inference failed after its normal retry handling.
    Worker,
    /// Responses could not be paired with the requested spans.
    ResponseCount,
    /// No usable sentence was returned.
    EmptyAnalysis,
    /// Secondary analysis could not be merged with the planned span.
    Merge,
    /// The final coordinated splice refused or rolled back.
    Splice,
}

/// Why a secondary span failed, as a record holds it: a cause this build
/// writes, or one only earlier builds wrote and stored records still carry.
///
/// Untagged over two string enums, so either reads by its own spelling. The
/// writer produces a [`SecondaryMorphologyFailureKind`], which has no
/// retired cause to produce.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(untagged)]
pub enum StoredSecondaryMorphologyFailureKind {
    /// A cause this build writes.
    Written(SecondaryMorphologyFailureKind),
    /// A cause only earlier builds wrote.
    Retired(RetiredSecondaryMorphologyFailureKind),
}

impl From<SecondaryMorphologyFailureKind> for StoredSecondaryMorphologyFailureKind {
    /// A written cause, stored as it is.
    fn from(cause: SecondaryMorphologyFailureKind) -> Self {
        Self::Written(cause)
    }
}

/// A secondary-morphology cause earlier builds wrote and this one only reads.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum RetiredSecondaryMorphologyFailureKind {
    /// The runtime had no admitted Stanza registry when the file's secondary
    /// languages were checked. Unreachable since a file's secondary
    /// languages are admitted by the registry that admitted its primary
    /// language; builds from 2026-10-13 wrote it.
    RegistryUnavailable,
}

/// One secondary span's failure, distinct from an unsupported-language exclusion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct SecondaryMorphologyFailure {
    /// Where the retained placeholders belong.
    pub span: SecondaryMorphologySpan,
    /// Typed producer boundary.
    pub cause: StoredSecondaryMorphologyFailureKind,
    /// Bounded explanation from that boundary.
    pub detail: String,
}

/// One utterance forced alignment left with untimed words.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct UntimedUtteranceRecord {
    /// Where the utterance is: its position and, in records this build
    /// writes, its main tier as written.
    #[serde(flatten)]
    pub location: StoredUtteranceLocation,
    /// Required lexical words in it.
    pub words: u64,
    /// How many of them are untimed, at least one.
    pub untimed_words: u64,
    /// Why.
    pub cause: StoredUntimedCause,
}

impl std::fmt::Display for UntimedUtteranceRecord {
    /// One line a reader can act on: the utterance as written (its position
    /// when an earlier build's record lacks the text), then how much of it is
    /// untimed and why.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}   ({} of {} words untimed: {})",
            self.location, self.untimed_words, self.words, self.cause
        )
    }
}

/// Where an utterance a record names is: its position among the document's
/// utterances, counting from 1, and its main tier as written.
///
/// Code builds one only from the document that holds the utterance
/// ([`Self::of`]), so the number and the text always name the same
/// utterance; one read back from a stored record is as that record wrote it. The text is what
/// a reader searches for: a position cannot be searched for in CLAN or an
/// editor, and it names a different utterance once the file is edited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtteranceLocator {
    utterance: u64,
    main_tier: MainTierExcerpt,
}

/// An utterance index that is not in the document it was resolved against:
/// the index came from another document, an internal fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("utterance {} is not in the document it was looked up in", .0.raw() + 1)]
pub struct UtteranceNotInDocument(talkbank_model::UtteranceIdx);

impl UtteranceLocator {
    /// The utterance at `ordinal` (counting from 0) among `chat`'s utterances.
    pub(crate) fn of(
        chat: &talkbank_model::ChatFile,
        ordinal: talkbank_model::UtteranceIdx,
    ) -> Result<Self, UtteranceNotInDocument> {
        chat.utterances()
            .nth(ordinal.raw())
            .map(|utterance| Self {
                utterance: ordinal.raw() as u64 + 1,
                main_tier: MainTierExcerpt::of(&utterance.main),
            })
            .ok_or(UtteranceNotInDocument(ordinal))
    }

    /// A locator from its parts, for tests of the records that carry one.
    #[cfg(test)]
    pub(crate) fn for_test(utterance: u64, main_tier: &str) -> Self {
        Self {
            utterance,
            main_tier: MainTierExcerpt(main_tier.to_owned()),
        }
    }
}

/// Where a recorded utterance is, as a stored record holds it: located, as
/// this build writes it, or numbered only, as records stored by builds before
/// 0.8.7 carry it.
///
/// Its state is private. Code makes one only from an [`UtteranceLocator`]
/// (`From`), so nothing this build writes can lack the main tier; a numbered
/// location comes only from reading an earlier record. A record's
/// `main_tier` that does not read is an error, never a numbered location
/// with the text dropped. On the wire it is `{utterance, main_tier}`,
/// flattened into the record that carries it; an earlier build's bare
/// position (a number, or an object without `main_tier`) still reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "StoredLocationWire", into = "UtteranceLocationWire")]
pub struct StoredUtteranceLocation(StoredLocation);

#[derive(Debug, Clone, PartialEq, Eq)]
enum StoredLocation {
    Located(UtteranceLocator),
    Numbered(u64),
}

/// What a stored location is read through: the object this build writes, or
/// the bare position some earlier records hold.
#[derive(Deserialize)]
#[serde(untagged)]
enum StoredLocationWire {
    Object(UtteranceLocationWire),
    Bare(u64),
}

/// The object form of a location on the wire, and the schema of what a
/// location writes.
#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
struct UtteranceLocationWire {
    /// The utterance's position among the file's utterances, counting from 1.
    utterance: u64,
    /// The utterance's main tier as written, without its bullet; absent only
    /// in records stored before 0.8.7.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "server", schema(value_type = Option<String>))]
    main_tier: Option<MainTierExcerpt>,
}

impl From<StoredLocationWire> for StoredUtteranceLocation {
    fn from(wire: StoredLocationWire) -> Self {
        Self(match wire {
            StoredLocationWire::Object(UtteranceLocationWire {
                utterance,
                main_tier: Some(main_tier),
            }) => StoredLocation::Located(UtteranceLocator {
                utterance,
                main_tier,
            }),
            StoredLocationWire::Object(UtteranceLocationWire {
                utterance,
                main_tier: None,
            })
            | StoredLocationWire::Bare(utterance) => StoredLocation::Numbered(utterance),
        })
    }
}

impl From<StoredUtteranceLocation> for UtteranceLocationWire {
    fn from(location: StoredUtteranceLocation) -> Self {
        match location.0 {
            StoredLocation::Located(UtteranceLocator {
                utterance,
                main_tier,
            }) => Self {
                utterance,
                main_tier: Some(main_tier),
            },
            StoredLocation::Numbered(utterance) => Self {
                utterance,
                main_tier: None,
            },
        }
    }
}

impl From<UtteranceLocator> for StoredUtteranceLocation {
    fn from(locator: UtteranceLocator) -> Self {
        Self(StoredLocation::Located(locator))
    }
}

impl StoredUtteranceLocation {
    /// The utterance's position among the file's utterances, counting from 1.
    pub fn utterance(&self) -> u64 {
        match &self.0 {
            StoredLocation::Located(locator) => locator.utterance,
            StoredLocation::Numbered(utterance) => *utterance,
        }
    }

    /// The utterance's main tier as written; absent only in a record an
    /// earlier build stored.
    pub fn main_tier(&self) -> Option<&MainTierExcerpt> {
        match &self.0 {
            StoredLocation::Located(locator) => Some(&locator.main_tier),
            StoredLocation::Numbered(_) => None,
        }
    }
}

impl std::fmt::Display for StoredUtteranceLocation {
    /// The main tier as written, which a reader can search for; the
    /// position only when an earlier build's record lacks the text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            StoredLocation::Located(locator) => write!(f, "{}", locator.main_tier),
            StoredLocation::Numbered(utterance) => write!(f, "utterance {utterance}"),
        }
    }
}

/// The schema of what a location writes: the wire object's, derived from
/// it, because the type's own state is private and read through that form.
#[cfg(feature = "server")]
impl utoipa::PartialSchema for StoredUtteranceLocation {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        <UtteranceLocationWire as utoipa::PartialSchema>::schema()
    }
}

#[cfg(feature = "server")]
impl utoipa::ToSchema for StoredUtteranceLocation {
    fn name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("StoredUtteranceLocation")
    }
}

/// A main tier as written, without its bullet: what a reader searches for to
/// find the utterance. Its first [`Self::MAX_CHARS`] characters, followed by
/// ` ...` when the tier is longer. Rendered by the CHAT model's own writer,
/// never assembled.
///
/// Bounded because every job poll copies the records that carry it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(transparent)]
pub struct MainTierExcerpt(String);

impl MainTierExcerpt {
    /// The most characters kept; a longer tier is cut and ends in ` ...`.
    pub const MAX_CHARS: usize = 160;

    /// The tier as the writer renders it, its bullet removed (a bullet is
    /// timing, not something a reader searches for), cut to the bound.
    ///
    /// The tier is cloned to clear the bullet, because the model's writer
    /// always writes it; admission calls this for at most
    /// [`FileOutputDiagnostics::FIRST_FINDINGS`] utterances per file.
    pub(crate) fn of(main: &talkbank_model::model::MainTier) -> Self {
        use talkbank_model::WriteChat;
        let mut unbulleted = main.clone();
        unbulleted.content.bullet = None;
        let mut written = unbulleted.to_chat_string();
        if let Some((cut, _)) = written.char_indices().nth(Self::MAX_CHARS) {
            written.truncate(cut);
            written.push_str(" ...");
        }
        Self(written)
    }

    /// The excerpt's text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MainTierExcerpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why an utterance's words have no timing, as a record holds it: a cause
/// this build writes, or one only earlier builds wrote and stored records
/// still carry.
///
/// Untagged over two tagged enums, so either reads by its own `kind` and the
/// wire shape is the cause's. The writer produces an [`UntimedCauseRecord`]
/// (`fa::completion`), which has no retired cause to produce.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(untagged)]
pub enum StoredUntimedCause {
    /// A cause this build writes.
    Written(UntimedCauseRecord),
    /// A cause only earlier builds wrote.
    Retired(RetiredUntimedCause),
}

impl From<UntimedCauseRecord> for StoredUntimedCause {
    /// A written cause, stored as it is.
    fn from(cause: UntimedCauseRecord) -> Self {
        Self::Written(cause)
    }
}

/// Why an utterance's words have no timing: the causes this build writes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UntimedCauseRecord {
    /// Its audio window was refused, so no alignment request was made.
    WindowRefused {
        /// The refused window and its cause.
        window: crate::types::traces::RefusedWindowTrace,
    },
    /// It was placed in no request: the audio left for a run of untimed
    /// utterances could not contain their words.
    NotPlaced,
    /// No supplied timing or source-bound UTR search evidence; no request.
    LocalizationUnavailable,
    /// Requests covered it and no positive interval resulted for these
    /// words. The counts are what those requests yielded, from the run's
    /// own group evidence. A word whose interval the run measured and then
    /// removed is never here: it is [`Self::Removed`].
    Requested {
        /// Requests that returned timing evidence (a worker's answer, the
        /// cache, a replayed response, or reused `%wor`).
        answered: u64,
        /// Requests left unaligned: no worker evidence was obtained.
        unaligned: u64,
        /// Word ranges left unrequested: an anchored piece whose window
        /// could not be sent.
        unrequested: u64,
    },
    /// The run measured intervals for words of this utterance and removed
    /// them itself while enforcing timing order (a start regression strip,
    /// a same-speaker end clamp, or a yield to a kept bullet), so the words
    /// are untimed because of the finalizer, not because no interval was
    /// measured. Each removal names the rule, the boundary it enforced and
    /// every interval it took away, recorded at the moment it did so.
    Removed {
        /// Every finalization effect that removed timing from this
        /// utterance, in the order the run applied them; at least one, each
        /// with at least one interval.
        removals: Vec<crate::types::traces::WordTimingRemovalTrace>,
    },
    /// The run refreshed the file's bullets from its complete `%wor` without
    /// opening the recording, and removed this utterance's retained timing
    /// while enforcing timing order (it was backward). Nothing in such a run
    /// could measure the words again, so they need a run with the recording:
    /// a `batchalign3 align` with the recording present re-times them, since
    /// the stripped `%wor` leaves the file partly timed.
    NeedsRecording {
        /// Every finalization effect that removed the retained timing, in
        /// the order the run applied them; at least one, each with at least
        /// one interval.
        removals: Vec<crate::types::traces::WordTimingRemovalTrace>,
    },
    /// No grouping decision names it and no request covered it: the run
    /// recorded nothing that explains its untimed words.
    Unrecorded,
}

/// A cause earlier builds wrote and this one only reads, from records they
/// stored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RetiredUntimedCause {
    /// Recorded whenever no refusal named the utterance, whether or not any
    /// request covered it, so it does not say which of
    /// [`UntimedCauseRecord::Requested`] and [`UntimedCauseRecord::Unrecorded`]
    /// was true.
    NoUsableTiming,
    /// The transcript marks it as not speech in the recording. Such an
    /// utterance is now an [`OutputExclusionRecord::NotInRecording`], which
    /// does not diagnose the file.
    NotInRecording {
        /// The postcode that marks it.
        postcode: OffRecordPostcode,
    },
}

/// Part of the input a command left out under its documented scope policy:
/// a transcript convention, or an explicitly unsupported secondary language.
///
/// Information, never a shortfall: the command stayed within that scope,
/// so an exclusion does not make the file diagnosed, and a file whose only
/// untimed utterances are exclusions is clean. Reported beside the written
/// output, on a `done` file as on a `diagnosed` one
/// ([`FileStatusEntry::exclusions`]), and bounded like
/// [`FileOutputDiagnostics`] because every poll copies it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputExclusionRecord {
    /// The runtime registry explicitly lacks morphology for these languages.
    UnsupportedSecondaryMorphology {
        /// Number of spans excluded.
        spans: u64,
        /// Number of morphology positions excluded.
        words: u64,
        /// Transcript-ordered examples, bounded like output diagnostics.
        first_spans: Vec<SecondaryMorphologySpan>,
    },
    /// Forced alignment did not look for these utterances in the recording,
    /// because the transcript marks them as not speech in it (a
    /// `[+ diary]` note). Their words are written untimed; a bullet the
    /// input gave one is kept as given under `--main-bullets keep` or
    /// `exact`, and written without under `derive`.
    NotInRecording {
        /// How many utterances were left out, at least one.
        excluded_utterances: u64,
        /// Their lexical words in all, none of which is owed timing.
        excluded_words: u64,
        /// The first of them, in transcript order, at most
        /// [`FileOutputDiagnostics::FIRST_FINDINGS`].
        first_excluded: Vec<ExcludedUtteranceRecord>,
    },
}

/// One utterance forced alignment left out because it is not in the
/// recording.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct ExcludedUtteranceRecord {
    /// Where the utterance is.
    #[serde(flatten)]
    pub location: StoredUtteranceLocation,
    /// Its lexical words.
    pub words: u64,
    /// The postcode that marks it.
    pub postcode: OffRecordPostcode,
}

impl OutputExclusionRecord {
    /// The record in a short phrase, for a one-line summary of what a file
    /// left out.
    pub fn summary_phrase(&self) -> String {
        match self {
            Self::NotInRecording {
                excluded_utterances,
                ..
            } => format!(
                "{excluded_utterances} utterance(s) not in the recording, left out of alignment by design"
            ),
            Self::UnsupportedSecondaryMorphology { spans, .. } => {
                format!("{spans} span(s) excluded from secondary morphology: unsupported languages")
            }
        }
    }

    /// The lines under this exclusion's own line: each utterance or span the
    /// record names, then how many more it counted. Exhaustive, like
    /// [`OutputShortfallRecord::detail_lines`].
    pub fn detail_lines(&self) -> Vec<String> {
        match self {
            Self::NotInRecording {
                excluded_utterances,
                first_excluded,
                ..
            } => listed(
                first_excluded,
                *excluded_utterances,
                "utterance(s) not in the recording",
            ),
            Self::UnsupportedSecondaryMorphology {
                spans, first_spans, ..
            } => listed(first_spans, *spans, "span(s) excluded"),
        }
    }
}

impl std::fmt::Display for ExcludedUtteranceRecord {
    /// The utterance as written, then its postcode, said even though the
    /// tier carries it: an excerpt of a long tier is cut before its end.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}   ({})", self.location, self.postcode)
    }
}

impl std::fmt::Display for OutputExclusionRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSecondaryMorphology { spans, words, .. } => write!(
                f,
                "secondary morphology excluded: {spans} span(s), {words} words in unsupported languages; L2 placeholders retained"
            ),
            // Counts only: `detail_lines` names the utterances.
            Self::NotInRecording {
                excluded_utterances,
                excluded_words,
                first_excluded: _,
            } => write!(
                f,
                "left out of alignment by design: {excluded_utterances} utterance(s) \
                 ({excluded_words} words) marked as not in the recording were not \
                 looked for in it"
            ),
        }
    }
}

/// A postcode that marks an utterance as not speech in the recording, so
/// alignment leaves it untimed.
///
/// A closed set, read from Chatter's typed postcodes (never from the line's
/// text). The CHAT manual defines no fixed postcode set ("postcodes can be
/// designed to fit the needs of your particular project"), so membership
/// here is a recorded ruling, one per postcode, never inferred from a name.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum OffRecordPostcode {
    /// `[+ diary]`: a written diary note set into the transcript. It is not
    /// speech in the recording (ruling of 2026-10-07).
    Diary,
}

impl OffRecordPostcode {
    /// Every member, for the one reader that matches postcode text.
    const ALL: [Self; 1] = [Self::Diary];

    /// The postcode's text, as written between `[+ ` and `]`.
    pub const fn text(self) -> &'static str {
        match self {
            Self::Diary => "diary",
        }
    }

    /// The member whose text this postcode's is, exactly.
    pub fn of_postcode_text(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|member| member.text() == text)
    }
}

impl std::fmt::Display for OffRecordPostcode {
    /// The postcode as CHAT writes it: `[+ diary]`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[+ {}]", self.text())
    }
}

impl std::fmt::Display for UntimedCauseRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::types::traces::RefusedWindowTrace;
        match self {
            Self::WindowRefused { window } => {
                let cause = match window {
                    RefusedWindowTrace::OverBudget { .. } => "longer than the alignment budget",
                    RefusedWindowTrace::Empty { .. } => "empty",
                    RefusedWindowTrace::Inverted { .. } => "inverted",
                    RefusedWindowTrace::PastRecording { .. } => "past the end of the recording",
                    RefusedWindowTrace::AnchorGap { .. } => {
                        "over budget, with a gap between recovered anchors longer than the budget"
                    }
                    RefusedWindowTrace::AnchorsUnusable { .. } => {
                        "over budget, with no usable recovered anchor to split at"
                    }
                };
                write!(
                    f,
                    "no alignment request, its audio window was refused ({cause})"
                )
            }
            Self::NotPlaced => f.write_str(
                "no alignment request, the audio left for it could not contain its words",
            ),
            Self::LocalizationUnavailable => f.write_str(
                "no alignment request, no supplied timing or source-bound UTR search evidence",
            ),
            Self::Requested {
                answered,
                unaligned,
                unrequested,
            } => write!(
                f,
                "no positive interval in the final output ({answered} answered, {unaligned} \
                 unaligned, {unrequested} unrequested)"
            ),
            // Each removal renders itself; several are joined so a reader
            // sees every rule that touched the utterance.
            Self::Removed { removals } => {
                for (position, removal) in removals.iter().enumerate() {
                    if position > 0 {
                        f.write_str("; ")?;
                    }
                    removal.fmt(f)?;
                }
                Ok(())
            }
            Self::NeedsRecording { removals } => {
                f.write_str(
                    "retained %wor timing removed as out of order while refreshing bullets with \
                     no recording read; these words need a run with the recording (",
                )?;
                for (position, removal) in removals.iter().enumerate() {
                    if position > 0 {
                        f.write_str("; ")?;
                    }
                    removal.fmt(f)?;
                }
                f.write_str(")")
            }
            Self::Unrecorded => f.write_str("no cause recorded"),
        }
    }
}

impl std::fmt::Display for RetiredUntimedCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoUsableTiming => f.write_str("no usable timing"),
            Self::NotInRecording { postcode } => write!(f, "not in the recording: {postcode}"),
        }
    }
}

impl std::fmt::Display for StoredUntimedCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Written(cause) => cause.fmt(f),
            Self::Retired(cause) => cause.fmt(f),
        }
    }
}

impl std::fmt::Display for OutputShortfallRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UtrUnavailable {
                engine,
                pass,
                category,
                detail,
            } => write!(
                f,
                "UTR {} pass unavailable ({engine}, {category}): {detail}; CHAT and available timing retained",
                pass.name()
            ),
            Self::MorphologyProvenanceIncomplete { detail } => write!(
                f,
                "morphology provenance incomplete: {detail}; morphology and original records retained"
            ),
            // The list-carrying shortfalls give counts only: `detail_lines`
            // names their utterances one per line.
            Self::SecondaryMorphologyUnplaced {
                utterances,
                words,
                first_utterances: _,
            } => write!(
                f,
                "secondary morphology not dispatched: {words} words in {utterances} utterance(s) could not be placed"
            ),
            Self::SecondaryMorphologyIncomplete {
                spans,
                words,
                first_spans: _,
            } => write!(
                f,
                "secondary morphology incomplete: {spans} span(s), {words} words retain L2 placeholders"
            ),
            Self::StageSkipped { stage } => write!(
                f,
                "skipped {}: it requires an admitted document, and the generated \
                 output was written with its diagnostics instead",
                stage.name()
            ),
            Self::StageNotApplied { stage, refusal } => write!(
                f,
                "{} not applied: its output could not be admitted ({refusal}); the \
                 admitted document from before it was written instead",
                stage.name()
            ),
            Self::StageHeldOut {
                stage,
                held_out_utterances,
                first_held_out: _,
            } => write!(
                f,
                "{} applied except to {held_out_utterances} utterance(s) that carry the \
                 findings, which keep their generated form",
                stage.name()
            ),
            Self::TimingIncomplete {
                required_words,
                untimed_words,
                untimed_utterances,
                first_untimed: _,
            } => write!(
                f,
                "timing incomplete: {untimed_words} of {required_words} words in \
                 {untimed_utterances} utterance(s) have no timing and were written without it"
            ),
            Self::WindowSeamAmbiguous {
                seams,
                first_seams: _,
            } => write!(
                f,
                "{seams} recognizer window seam(s) reconciled by rule: a repeated observation \
                 at each was kept once (the earlier window's copy) and marked with a %com on \
                 its utterance for review"
            ),
        }
    }
}

/// One window seam the reconciler settled by rule, as the job reports it.
///
/// Locates the utterance that keeps the copy the way every other first-N
/// record does: its position among the written utterances and its main tier
/// as written, so a reader of the job finds the `%com` the transcript carries
/// at the same place.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct WindowSeamRecord {
    /// Where the utterance that keeps the copy is.
    #[serde(flatten)]
    pub location: StoredUtteranceLocation,
    /// Where the seam is in the recording, in milliseconds.
    pub seam_ms: i64,
    /// The repeated text, as the recognizer emitted it.
    pub text: String,
    /// The decode window whose copy was dropped, counting from 0.
    pub dropped_from_window: u64,
}

impl std::fmt::Display for WindowSeamRecord {
    /// One line a reader can act on: the utterance as written, then the seam
    /// and the text whose later copy was dropped.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}   (seam at {} ms: repeated \"{}\" kept once, window {} copy dropped)",
            self.location, self.seam_ms, self.text, self.dropped_from_window
        )
    }
}

/// Per-file status within a job.
///
/// Tracks processing state, timing, progress, and error details for a
/// single file.  The server updates these entries as workers report
/// results, and they are included in `JobInfo.file_statuses`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FileStatusEntry {
    /// Display path for this file: a bare basename (`"sample.cha"`) or a
    /// relative forward-slash path (`"PWA/TYO_a1.cha"`) for directory input.
    /// Backslashes are normalized to forward slashes on construction.
    pub filename: DisplayPath,
    /// Current lifecycle state of this file.
    pub status: FileStatusKind,
    /// Human-readable error message.  Present only when `status` is `Error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Broad classification of the failure. Helps the dashboard and retry
    /// logic group failures without parsing free-form messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_category: Option<FailureCategory>,
    /// What output admission found in a file written with diagnostics.
    /// Present only when `status` is `Diagnosed`, following the `error`
    /// field's pattern: this entry is a flat record whose status-specific
    /// fields are optional, not a union per status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<FileOutputDiagnostics>,
    /// Parts of the input the command left out of its work on purpose
    /// ([`OutputExclusionRecord`]): information, never a shortfall. Present
    /// only on a file whose output was written (`done` or `diagnosed`), and
    /// only when there is one; a separate channel from `diagnostics`, which
    /// a clean file does not have.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclusions: Vec<OutputExclusionRecord>,
    /// What the command decided about stamping this file with provenance.
    /// Absent when nothing was recorded, which is what `Unrecorded` means: a
    /// command that writes no per-file stamp, or a status restored from the
    /// job database.
    #[serde(default, skip_serializing_if = "FileStampOutcome::is_unrecorded")]
    pub stamp: FileStampOutcome,
    /// When the worker began processing this file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<MachineTime>,
    /// When processing finished (successfully or with error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<MachineTime>,
    /// Seconds from `started_at` to `finished_at`, present when both are.
    /// Computed by the server, never negative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<NonNegativeSeconds>,
    /// When the file is deferred for a retry, the earliest time at which
    /// another attempt should start. `None` when no retry is pending.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_eligible_at: Option<MachineTime>,
    /// Number of sub-steps completed so far (e.g. utterances processed).
    /// Used with `progress_total` to drive progress bars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_current: Option<i64>,
    /// Total number of sub-steps expected for this file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_total: Option<i64>,
    /// Machine-readable code for the current in-flight processing stage.
    ///
    /// This is the stable field clients should use for branching or
    /// conditional UI. [`Self::progress_label`] is derived display text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_stage: Option<FileProgressStage>,
    /// Human-readable label for the current processing stage
    /// (e.g. "Morphosyntax", "Forced Alignment").
    ///
    /// This is derived from `progress_stage` by the server so operators can
    /// render friendly text without hard-coding label mappings in every UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_label: Option<String>,
    /// The file's diagnostics and exclusions as a reader sees them: rendered
    /// by the server from `diagnostics` and `exclusions` when the entry is
    /// built, like `progress_label`, so every client shows the same words
    /// instead of keeping its own copy of the renderer.
    #[serde(default, skip_serializing_if = "FileReview::is_empty")]
    pub review: FileReview,
}

/// A file's record as a reader sees it: what needs review, and what the
/// command left out on purpose, each a headline and the lines under it.
/// Rendered by [`Self::of`] from the record itself; its fields are private,
/// so no client or producer writes its own wording. Absent on the wire when
/// there is nothing in it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FileReview {
    /// What needs review; present only for a file written with diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    needs_review: Option<ReviewBlock>,
    /// What the command left out on purpose; present only when it left
    /// something out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    left_out: Option<ReviewBlock>,
}

/// One part of a review: a headline, and the lines under it (a line that
/// belongs to the one above it is indented). Lines never exist without
/// their headline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct ReviewBlock {
    headline: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    lines: Vec<String>,
}

impl ReviewBlock {
    /// The headline.
    pub fn headline(&self) -> &str {
        &self.headline
    }

    /// The lines under it.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }
}

impl FileReview {
    /// The review of a file with this status, diagnostics and exclusions.
    pub fn of(
        status: FileStatusKind,
        diagnostics: Option<&FileOutputDiagnostics>,
        exclusions: &[OutputExclusionRecord],
    ) -> Self {
        let needs_review = match status {
            FileStatusKind::Diagnosed => Some(match diagnostics {
                Some(diagnostics) => ReviewBlock {
                    headline: diagnostics.summary(),
                    lines: diagnostics.lines(),
                },
                // A diagnosed status restored without its record.
                None => ReviewBlock {
                    headline: "needs review (no details were recorded)".to_owned(),
                    lines: Vec::new(),
                },
            }),
            FileStatusKind::Done
            | FileStatusKind::Queued
            | FileStatusKind::Processing
            | FileStatusKind::Error
            | FileStatusKind::Interrupted => None,
        };
        let left_out = (!exclusions.is_empty()).then(|| ReviewBlock {
            headline: exclusions
                .iter()
                .map(OutputExclusionRecord::summary_phrase)
                .collect::<Vec<_>>()
                .join("; "),
            lines: exclusions
                .iter()
                .flat_map(|exclusion| {
                    std::iter::once(exclusion.to_string()).chain(
                        exclusion
                            .detail_lines()
                            .into_iter()
                            .map(|line| format!("  {line}")),
                    )
                })
                .collect(),
        });
        Self {
            needs_review,
            left_out,
        }
    }

    /// Whether there is nothing to review and nothing left out.
    pub fn is_empty(&self) -> bool {
        self.needs_review.is_none() && self.left_out.is_none()
    }

    /// What needs review, for a file written with diagnostics.
    pub fn needs_review(&self) -> Option<&ReviewBlock> {
        self.needs_review.as_ref()
    }

    /// What was left out on purpose, when anything was.
    pub fn left_out(&self) -> Option<&ReviewBlock> {
        self.left_out.as_ref()
    }
}

/// One line describing a file written with diagnostics: what needs review,
/// in the order a reader acts on it. Validation findings (with the first),
/// each shortfall that is not a stage, stages left out of utterances, words
/// forced alignment left untimed, and how many requested stages were not
/// applied. Each part is present only when it applies, so a file diagnosed
/// only by untimed words never reads as "0 diagnostics".
impl FileOutputDiagnostics {
    /// One line: what needs review, in the order a reader acts on it.
    pub fn summary(&self) -> String {
        let diagnostics = self;
        let mut parts = Vec::new();
        let count = diagnostics.finding_count();
        if count > 0 {
            let noun = if count == 1 { "finding" } else { "findings" };
            let mut part = format!("{count} validation {noun}");
            if let Some(first) = diagnostics.first_findings().first() {
                part.push_str(&format!(" (first: {first})"));
            }
            parts.push(part);
        }
        let mut stages = 0;
        for shortfall in &diagnostics.shortfalls {
            match shortfall {
                OutputShortfallRecord::SecondaryMorphologyIncomplete { .. }
                | OutputShortfallRecord::UtrUnavailable { .. }
                | OutputShortfallRecord::MorphologyProvenanceIncomplete { .. }
                | OutputShortfallRecord::SecondaryMorphologyUnplaced { .. } => {
                    parts.push(shortfall.to_string());
                }
                OutputShortfallRecord::StageSkipped { .. }
                | OutputShortfallRecord::StageNotApplied { .. } => stages += 1,
                OutputShortfallRecord::StageHeldOut {
                    stage,
                    held_out_utterances,
                    ..
                } => parts.push(format!(
                    "{} left out of {held_out_utterances} utterance(s)",
                    stage.name()
                )),
                OutputShortfallRecord::TimingIncomplete {
                    required_words,
                    untimed_words,
                    untimed_utterances,
                    ..
                } => parts.push(format!(
                    "{untimed_words} of {required_words} words in {untimed_utterances} \
                 utterance(s) have no word timing"
                )),
                OutputShortfallRecord::WindowSeamAmbiguous { seams, .. } => parts.push(format!(
                    "{seams} recognizer window seam(s) kept a repeated observation once, \
                     marked %com for review"
                )),
            }
        }
        if stages > 0 {
            parts.push(format!("{stages} requested stage(s) not applied"));
        }
        format!("needs review: {}", parts.join("; "))
    }
}

/// Control-plane backend that currently owns orchestration for a server job.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "kebab-case")]
pub enum JobControlPlaneBackendKind {
    /// In-process local control plane.
    Local,
}

impl std::fmt::Display for JobControlPlaneBackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => f.write_str("local"),
        }
    }
}

/// Backend-owned orchestration metadata for a job projection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JobControlPlaneInfo {
    /// Control-plane backend that produced this projection.
    pub backend: JobControlPlaneBackendKind,
}

impl JobControlPlaneInfo {
    /// Local control-plane marker.
    pub fn local() -> Self {
        Self {
            backend: JobControlPlaneBackendKind::Local,
        }
    }
}

/// `GET /jobs/{id}` response: job progress.
///
/// Contains full detail about a single job, including per-file statuses.
/// Polled by the CLI to drive progress bars and by the dashboard for
/// live-updating tables.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JobInfo {
    /// Server-assigned UUID (v4) for this job.
    pub job_id: JobId,
    /// Current lifecycle state of the job.
    pub status: JobStatus,
    /// Batchalign command that was submitted (e.g. "morphotag", "align").
    pub command: ReleasedCommand,
    /// Typed submitted command options captured at job creation time.
    ///
    /// The dashboard uses this to show the original argument/config payload for
    /// debugging and operator review without reconstructing a lossy CLI string.
    #[cfg_attr(feature = "server", schema(value_type = serde_json::Value))]
    pub options: CommandOptions,
    /// Job language: a resolved ISO 639-3 code, `"auto"`, `"per-file"`, or a
    /// code-switched pair such as `"eng,spa"`. Always written by the server;
    /// a record without one is malformed and refused at deserialization
    /// rather than read as English.
    pub lang: LanguageSpec,
    /// Client's original input directory path, used for display in the
    /// dashboard and CLI output.  May be empty for content-mode submissions.
    #[serde(default)]
    pub source_dir: String,
    /// Total number of files in this job (immutable after submission).
    pub total_files: i64,
    /// Number of files that have finished processing (status `Done`).
    /// Invariant: `0 <= completed_files <= total_files`.
    pub completed_files: i64,
    /// Filename currently being processed.  `None` when the job is queued
    /// or has reached a terminal state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_file: Option<String>,
    /// Job-level error message (e.g. worker pool exhaustion, memory gate
    /// timeout).  Distinct from per-file errors in `file_statuses`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Per-file processing status, one entry per submitted file.
    #[serde(default)]
    pub file_statuses: Vec<FileStatusEntry>,
    /// When the job was submitted (`MachineTime`: UTC, three fractional
    /// digits).
    pub submitted_at: MachineTime,
    /// IP address or Tailscale identifier of the submitting client.
    /// Used for conflict detection (same `submitted_by` + filename =
    /// duplicate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted_by: Option<String>,
    /// Human-readable hostname of the submitting machine, resolved from
    /// Tailscale or reverse DNS.  For dashboard display only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted_by_name: Option<String>,
    /// When the job reached a terminal state (`MachineTime`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<MachineTime>,
    /// Seconds from submission to completion. `None` while the job is
    /// still active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<NonNegativeSeconds>,
    /// When the job is deferred before execution, the earliest time
    /// at which it should be retried. `None` when the job is immediately
    /// eligible to run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_eligible_at: Option<MachineTime>,
    /// Number of concurrent Python workers used for this job.  Determined
    /// by the pool at dispatch time.  `None` for queued jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_workers: Option<i64>,
    /// Active lease information when this job is currently claimed by a node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_lease: Option<LeaseRecord>,
    /// Per-language-group progress for batched text commands (morphotag,
    /// Server control-plane metadata for this job when returned by a server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_plane: Option<JobControlPlaneInfo>,
    /// Execution plan describing where and how this job is processed.
    /// Present for staged-remote jobs; `None` for local execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_plan: Option<crate::types::execution_plan::ExecutionPlan>,
    /// Wall-clock timestamp of the most recent cancel attempt against
    /// this job. `None` until at least one cancel arrives. Denormalized
    /// from the `cancellations` audit table for fast list-view rendering
    /// without a JOIN. See `CancellationRecord` for the full audit
    /// history via `GET /jobs/{id}/cancellations`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_cancelled_at: Option<MachineTime>,
    /// Wire-format source string of the most recent cancel attempt
    /// (`tui`, `api`, `signal`, ...). See `CancelSource`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_cancelled_source: Option<String>,
    /// Caller-reported host of the most recent cancel attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_cancelled_host: Option<String>,
    /// Caller-reported reason of the most recent cancel attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_cancelled_reason: Option<String>,
}

impl JobInfo {
    /// Attach server control-plane metadata to this job projection.
    pub fn with_control_plane(mut self, control_plane: JobControlPlaneInfo) -> Self {
        self.control_plane = Some(control_plane);
        self
    }
}

/// `GET /jobs/{id}/results` response: completed job results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JobResultResponse {
    /// Unique job identifier.
    pub job_id: JobId,
    /// Terminal status of the job.
    pub status: JobStatus,
    /// Per-file results (empty until the job completes).
    #[serde(default)]
    pub files: Vec<FileResult>,
}

/// Summary for job listing (`GET /jobs` response element).
///
/// A lighter-weight projection of [`JobInfo`] without per-file statuses,
/// suitable for listing many jobs at once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JobListItem {
    /// Server-assigned UUID (v4) for this job.
    pub job_id: JobId,
    /// Current lifecycle state of the job.
    pub status: JobStatus,
    /// Batchalign command (e.g. "morphotag", "align").
    pub command: ReleasedCommand,
    /// Job language: a resolved ISO 639-3 code, `"auto"`, `"per-file"`, or a
    /// code-switched pair such as `"eng,spa"`. Always written by the server;
    /// a record without one is malformed and refused at deserialization
    /// rather than read as English.
    pub lang: LanguageSpec,
    /// Client's original input directory path, for display.
    #[serde(default)]
    pub source_dir: String,
    /// Total number of files in this job.
    pub total_files: i64,
    /// Number of files that reached a terminal state: done, diagnosed or
    /// error.
    pub completed_files: i64,
    /// Number of files that ended in `Error` status.
    #[serde(default)]
    pub error_files: i64,
    /// Number of files written with diagnostics (`Diagnosed`): their output
    /// is on disk, but it is not certified complete. A `completed` job with a
    /// nonzero count is not a clean success.
    #[serde(default)]
    pub diagnosed_files: i64,
    /// Job-level error message when the job failed (the aggregated per-file
    /// failure reason). Surfaces the cause in the `/jobs` list and the live
    /// dashboard/TUI feed instead of a bare "failed". `None` for non-failed
    /// jobs. Distinct from per-file errors and from the `error_files` count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// When the job was submitted (`MachineTime`).
    pub submitted_at: MachineTime,
    /// IP address or Tailscale identifier of the submitting client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted_by: Option<String>,
    /// Human-readable hostname of the submitting machine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submitted_by_name: Option<String>,
    /// When the job reached a terminal state (`MachineTime`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<MachineTime>,
    /// Wall-clock duration in seconds.  `None` while the job is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<NonNegativeSeconds>,
    /// When the job is deferred before execution, the earliest time
    /// at which it should be retried. `None` when immediately eligible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_eligible_at: Option<MachineTime>,
    /// Number of concurrent Python workers used for this job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_workers: Option<i64>,
    /// Active lease information when this job is currently claimed by a node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_lease: Option<LeaseRecord>,
    /// Server control-plane metadata for this job when returned by a server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_plane: Option<JobControlPlaneInfo>,
}

impl JobListItem {
    /// Attach server control-plane metadata to this list projection.
    pub fn with_control_plane(mut self, control_plane: JobControlPlaneInfo) -> Self {
        self.control_plane = Some(control_plane);
        self
    }
}

/// The host's memory as the operating system reported it for one response.
///
/// One reading, so the three figures are always from the same moment and are
/// present or absent together: an unreadable host is an absent reading, never
/// a zero total the dashboard would draw.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct SystemMemoryReading {
    /// Total physical memory in MB.
    pub total_mb: MemoryMb,
    /// Available memory in MB (free + reclaimable).  On macOS this is
    /// `free + purgeable` which can undercount; see `sysinfo` docs.
    pub available_mb: MemoryMb,
    /// Used memory in MB (`total - available`).
    pub used_mb: MemoryMb,
}

/// `GET /health` response.
///
/// Provides a snapshot of server liveness, capacity, and operational
/// metrics. The CLI uses this for daemon version checks (`build_hash`) and
/// capability probing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct HealthResponse {
    /// Server health status.  Always `Ok` in the current implementation
    /// (the endpoint itself being reachable implies health), but future
    /// versions may report degraded states.
    ///
    /// REQUIRED, and that is what makes this type an identification rather
    /// than a shape any JSON happens to fit. With `#[serde(default)]` here and
    /// on every other field, `{"error":{"message":"unknown endpoint"}}`
    /// deserialized into a perfectly healthy response, because `HealthStatus`
    /// has a single `#[default] Ok` variant. On 2026-08-27 an unrelated local
    /// service holding port 8000 was therefore accepted as a Batchalign
    /// server and dispatched real jobs. Parse at the boundary into a type
    /// whose existence proves what it claims: a body without these fields is
    /// not a Batchalign health response and must fail to parse, not arrive
    /// pre-filled with agreeable values.
    pub status: HealthStatus,
    /// Server software version (e.g. `"0.6.0"`).  Used by the CLI to
    /// detect stale daemons and auto-restart them.
    ///
    /// REQUIRED for the same reason as `status`. A server that could omit it
    /// would already have broken the stale-daemon check this field exists for.
    pub version: String,
    /// Identifier of the current server node.
    #[serde(default)]
    pub node_id: NodeId,
    /// Whether the Python workers are running on free-threaded Python
    /// (3.14t+).  Affects memory budgets and concurrency strategy.
    #[serde(default)]
    pub free_threaded: bool,
    /// Batchalign commands this server can process (e.g. `["morphotag",
    /// "align", "transcribe"]`).  Determined by probing worker capabilities
    /// at startup.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Pipelines currently loaded in memory (warm workers).  A subset of
    /// `capabilities`, only commands whose workers have been spawned.
    #[serde(default)]
    pub loaded_pipelines: Vec<String>,
    /// Distinct content-addressed Python runtimes observed since server start.
    /// Paths are deliberately absent; hashes identify executable bytes,
    /// Batchalign package bytes, and the installed distribution inventory.
    #[serde(default)]
    pub worker_runtime_identities: Vec<crate::worker::runtime_identity::WorkerRuntimeIdentity>,
    /// The latest capability admission outcome of every worker key that has
    /// reported: admitted with its infer tasks, or refused with the reason, so
    /// an operator sees why a worker is not used.
    #[serde(default)]
    pub worker_capability_admissions: Vec<WorkerCapabilityAdmission>,
    /// Registry daemons the latest discovery sweep found alive and refused to
    /// adopt (another build, or no build named), with why, so an operator sees
    /// why a running daemon is unused.
    #[serde(default)]
    pub refused_registry_workers: Vec<RefusedRegistryWorker>,
    /// Filesystem directories the server searches for media files (audio/video).
    /// Configured via `server.yaml` `media_roots`.
    #[serde(default)]
    pub media_roots: Vec<String>,
    /// Named keys from `server.yaml` `media_mappings`, each mapping a
    /// logical name (e.g. "childes-data") to a filesystem root.
    #[serde(default)]
    pub media_mapping_keys: Vec<String>,
    /// Backward-compat alias for `job_slots_available`.  Older clients read
    /// this field; newer clients prefer `job_slots_available`.
    #[serde(default)]
    pub workers_available: i64,
    /// Number of additional jobs the server can accept right now, based on
    /// memory and concurrency limits.
    #[serde(default)]
    pub job_slots_available: i64,
    /// Number of currently live Python worker processes.
    #[serde(default)]
    pub live_workers: i64,
    /// Active worker keys (`target:lang`) currently loaded in the pool.
    /// Infer-task workers use labels such as `infer:asr:eng`.
    #[serde(default)]
    pub live_worker_keys: Vec<String>,
    /// Number of jobs currently in `Queued` or `Running` state.
    #[serde(default)]
    pub active_jobs: i64,
    /// Utterance cache backend in use: `"sqlite"` (default).
    #[serde(default = "default_cache_backend")]
    pub cache_backend: String,
    /// Cumulative count of worker process crashes since server start.
    /// A high rate suggests resource exhaustion or buggy engine code.
    #[serde(default)]
    pub worker_crashes: i64,
    /// Cumulative count of work-unit attempts started since server start.
    #[serde(default)]
    pub attempts_started: i64,
    /// Cumulative count of attempts that were explicitly marked retryable.
    #[serde(default)]
    pub attempts_retried: i64,
    /// Cumulative count of work units that were deferred for later execution.
    #[serde(default)]
    pub deferred_work_units: i64,
    /// Cumulative count of files that were force-terminated (e.g. OOM-killed
    /// workers, stuck processes) since server start.
    #[serde(default)]
    pub forced_terminal_errors: i64,
    /// Backward-compat counter for job deferrals caused by host-memory
    /// admission pressure.
    #[serde(default)]
    pub memory_gate_aborts: i64,
    /// Build fingerprint: changes on every rebuild.  Used for stale-binary
    /// detection during development.  Empty string from older servers.
    #[serde(default)]
    pub build_hash: String,
    // ── System memory snapshot ──────────────────────────────────────────
    /// The host's memory, absent when the operating system would not report
    /// it (the reason is in `host_memory_error`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_memory: Option<SystemMemoryReading>,
    /// Host-memory reserve threshold in MB from server config.  0 means the
    /// coordinator will not keep explicit free-memory headroom.
    #[serde(default)]
    pub memory_gate_threshold_mb: MemoryMb,
    /// Host-memory pressure classification derived from current OS memory plus
    /// active cross-process reservations.
    #[serde(default)]
    pub host_memory_pressure: HostMemoryPressureLevel,
    /// Total memory currently reserved by the host-memory coordinator.
    #[serde(default)]
    pub host_memory_reserved_mb: MemoryMb,
    /// Number of active worker/model startup leases in the host coordinator.
    #[serde(default)]
    pub host_memory_startup_leases: i64,
    /// Number of active job-execution leases in the host coordinator.
    #[serde(default)]
    pub host_memory_job_leases: i64,
    /// Number of active machine-wide ML test locks in the host coordinator.
    #[serde(default)]
    pub host_memory_ml_test_locks: i64,
    /// Human-readable labels for active coordinator leases.
    #[serde(default)]
    pub host_memory_active_leases: Vec<String>,
    /// Snapshot error surfaced when the host-memory ledger cannot be read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_memory_error: Option<String>,
}

pub(crate) fn default_cache_backend() -> String {
    "sqlite".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first main tier of `chat_text`, as the parser builds it.
    fn first_main_tier(chat_text: &str) -> talkbank_model::model::MainTier {
        use talkbank_parser_tests::test_error::ParseProductTestExt;
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser init");
        let chat = parser.parse_chat_file(chat_text).expect_built();
        chat.lines
            .iter()
            .find_map(|line| match line {
                talkbank_model::model::Line::Utterance(utterance) => Some(utterance.main.clone()),
                _ => None,
            })
            .expect("one utterance")
    }

    /// The excerpt is the tier a reader searches for: as the writer renders
    /// it, without the bullet (timing, not text), and never longer than the
    /// bound, cut on a character so a multibyte transcript cannot split.
    #[test]
    fn a_main_tier_excerpt_drops_the_bullet_and_is_bounded() {
        let header = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
                      @ID:\teng|test|PAR|||||Participant|||\n@Media:\tinput, audio\n";
        let bulleted = first_main_tier(&format!(
            "{header}*PAR:\t&-uh the dog is running . \u{15}1000_2500\u{15}\n@End\n"
        ));
        insta::assert_snapshot!(MainTierExcerpt::of(&bulleted), @"*PAR:	&-uh the dog is running .");

        let long = "très ".repeat(60);
        let excerpt =
            MainTierExcerpt::of(&first_main_tier(&format!("{header}*PAR:\t{long}.\n@End\n")));
        assert_eq!(
            excerpt.as_str().chars().count(),
            MainTierExcerpt::MAX_CHARS + " ...".len()
        );
        assert!(excerpt.as_str().ends_with(" ..."), "{excerpt}");
    }

    /// Records stored by 0.8.6 locate an utterance by position only: an
    /// object without `main_tier`, or (held-out stages) a bare number. Both
    /// still read, as numbered; one this build writes reads as located and
    /// writes back the same shape; a `main_tier` that does not read is an
    /// error, never a numbered location with the text dropped.
    #[test]
    fn a_location_stored_without_its_main_tier_still_reads() {
        let cause = serde_json::json!({ "kind": "localization_unavailable" });
        let stored: UntimedUtteranceRecord = serde_json::from_value(serde_json::json!({
            "utterance": 29, "words": 1, "untimed_words": 1, "cause": cause,
        }))
        .expect("a 0.8.6 record reads");
        assert_eq!(stored.location.main_tier(), None);
        insta::assert_snapshot!(stored, @"utterance 29   (1 of 1 words untimed: no alignment request, no supplied timing or source-bound UTR search evidence)");

        let located_json = serde_json::json!({
            "utterance": 29, "main_tier": "*INV:\tMhm .", "words": 1, "untimed_words": 1,
            "cause": cause,
        });
        let located: UntimedUtteranceRecord =
            serde_json::from_value(located_json.clone()).expect("a located record reads");
        assert_eq!(located.location.utterance(), 29);
        insta::assert_snapshot!(located, @"*INV:	Mhm .   (1 of 1 words untimed: no alignment request, no supplied timing or source-bound UTR search evidence)");
        assert_eq!(serde_json::to_value(&located).unwrap(), located_json);

        let bare: Vec<StoredUtteranceLocation> = serde_json::from_value(serde_json::json!([3, 5]))
            .expect("0.8.6 held-out positions read");
        assert_eq!(
            bare.iter()
                .map(StoredUtteranceLocation::utterance)
                .collect::<Vec<_>>(),
            [3, 5]
        );
        assert!(bare.iter().all(|location| location.main_tier().is_none()));

        let malformed = serde_json::from_value::<UntimedUtteranceRecord>(serde_json::json!({
            "utterance": 29, "main_tier": 7, "words": 1, "untimed_words": 1, "cause": cause,
        }));
        assert!(malformed.is_err(), "{malformed:?}");
    }

    /// The review the server sends every client, for each kind of record:
    /// what was the dashboard's own copy of these renderers is now this one
    /// rendering. A record from 0.8.6 (no `main_tier`) renders by position.
    #[test]
    fn a_file_review_renders_every_record_kind_once() {
        let located = |n: u64, tier: &str| -> StoredUtteranceLocation {
            UtteranceLocator::for_test(n, tier).into()
        };
        let numbered: StoredUtteranceLocation =
            serde_json::from_value(serde_json::json!(4)).expect("a 0.8.6 position");
        let cause = StoredUntimedCause::Written(UntimedCauseRecord::LocalizationUnavailable);
        let span = |location: StoredUtteranceLocation| SecondaryMorphologySpan {
            location,
            first_word: 2,
            words: 2,
            language: "spa".into(),
        };
        let diagnostics = FileOutputDiagnostics::of_findings(
            Vec::new(),
            vec![
                OutputShortfallRecord::TimingIncomplete {
                    required_words: 120,
                    untimed_words: 3,
                    untimed_utterances: 3,
                    first_untimed: vec![
                        UntimedUtteranceRecord {
                            location: located(24, "*INV:\tMhm ."),
                            words: 1,
                            untimed_words: 1,
                            cause: cause.clone(),
                        },
                        UntimedUtteranceRecord {
                            location: numbered.clone(),
                            words: 1,
                            untimed_words: 1,
                            cause,
                        },
                    ],
                },
                OutputShortfallRecord::SecondaryMorphologyIncomplete {
                    spans: 1,
                    words: 2,
                    first_spans: vec![SecondaryMorphologyFailure {
                        span: span(located(3, "*PAR:\tyo quiero agua .")),
                        cause: SecondaryMorphologyFailureKind::Merge.into(),
                        detail: "analysis could not be merged".into(),
                    }],
                },
                OutputShortfallRecord::SecondaryMorphologyUnplaced {
                    utterances: 1,
                    words: 2,
                    first_utterances: vec![SecondaryMorphologyUnplacedUtterance {
                        location: numbered,
                        words: 2,
                        cause: SecondaryExtractionFailure::Alignment,
                        detail: "primary analysis does not align".into(),
                    }],
                },
                OutputShortfallRecord::StageSkipped {
                    stage: OptionalStage::Morphosyntax,
                },
            ],
        );
        let exclusions = [
            OutputExclusionRecord::NotInRecording {
                excluded_utterances: 2,
                excluded_words: 7,
                first_excluded: vec![ExcludedUtteranceRecord {
                    location: located(2, "*MOT:\twent to the park . [+ diary]"),
                    words: 5,
                    postcode: OffRecordPostcode::Diary,
                }],
            },
            OutputExclusionRecord::UnsupportedSecondaryMorphology {
                spans: 1,
                words: 2,
                first_spans: vec![span(located(5, "*PAR:\tbonjour mes amis ."))],
            },
        ];
        let review = FileReview::of(FileStatusKind::Diagnosed, Some(&diagnostics), &exclusions);
        let rendered = [review.needs_review(), review.left_out()]
            .into_iter()
            .flatten()
            .flat_map(|block| {
                std::iter::once(block.headline()).chain(block.lines().iter().map(String::as_str))
            })
            .collect::<Vec<_>>()
            .join("\n");
        insta::assert_snapshot!(rendered, @"
        needs review: 3 of 120 words in 3 utterance(s) have no word timing; secondary morphology incomplete: 1 span(s), 2 words retain L2 placeholders; secondary morphology not dispatched: 2 words in 1 utterance(s) could not be placed; 1 requested stage(s) not applied
        timing incomplete: 3 of 120 words in 3 utterance(s) have no timing and were written without it
          *INV:	Mhm .   (1 of 1 words untimed: no alignment request, no supplied timing or source-bound UTR search evidence)
          utterance 4   (1 of 1 words untimed: no alignment request, no supplied timing or source-bound UTR search evidence)
          ... and 1 more utterance(s) with untimed words
        secondary morphology incomplete: 1 span(s), 2 words retain L2 placeholders
          *PAR:	yo quiero agua .   (spa, 2 word(s) from word 2): analysis could not be merged
        secondary morphology not dispatched: 2 words in 1 utterance(s) could not be placed
          utterance 4   (2 word(s): primary analysis does not align)
        skipped morphosyntax: it requires an admitted document, and the generated output was written with its diagnostics instead
        2 utterance(s) not in the recording, left out of alignment by design; 1 span(s) excluded from secondary morphology: unsupported languages
        left out of alignment by design: 2 utterance(s) (7 words) marked as not in the recording were not looked for in it
          *MOT:	went to the park . [+ diary]   ([+ diary])
          ... and 1 more utterance(s) not in the recording
        secondary morphology excluded: 1 span(s), 2 words in unsupported languages; L2 placeholders retained
          *PAR:	bonjour mes amis .   (spa, 2 word(s) from word 2)
        ");

        // A clean file with nothing left out has nothing to send.
        assert!(FileReview::of(FileStatusKind::Done, None, &[]).is_empty());
        assert_eq!(
            serde_json::to_value(FileReview::of(FileStatusKind::Done, None, &[])).unwrap(),
            serde_json::json!({})
        );
    }

    /// A locator is made from the document that holds the utterance; an index
    /// past its utterances is refused, never a position with no text.
    #[test]
    fn a_locator_names_the_utterance_of_its_own_document() {
        use talkbank_parser_tests::test_error::ParseProductTestExt;
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser init");
        let chat = parser
            .parse_chat_file(
                "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
                 @ID:\teng|test|PAR|||||Participant|||\n*PAR:\tone .\n*PAR:\ttwo words .\n@End\n",
            )
            .expect_built();
        let second = UtteranceLocator::of(&chat, talkbank_model::UtteranceIdx::new(1))
            .expect("the second utterance");
        let location = StoredUtteranceLocation::from(second);
        assert_eq!(location.utterance(), 2);
        insta::assert_snapshot!(location, @"*PAR:	two words .");
        assert_eq!(
            UtteranceLocator::of(&chat, talkbank_model::UtteranceIdx::new(2)),
            Err(UtteranceNotInDocument(talkbank_model::UtteranceIdx::new(2)))
        );
    }

    /// A stored record carries a cause this build writes, or one only an
    /// earlier build wrote; both read back by their own `kind`, with the
    /// wire shape the cause's own. The writer's type has no retired cause.
    /// RED FIRST: removing `registry_unavailable` from the written causes
    /// made every stored record carrying it unreadable, and crash recovery
    /// dropped the file's diagnostics. It reads as a retired cause.
    #[test]
    fn a_stored_registry_unavailable_failure_still_reads() {
        let stored: StoredSecondaryMorphologyFailureKind =
            serde_json::from_value(serde_json::json!("registry_unavailable"))
                .expect("a retired cause still reads");
        assert_eq!(
            stored,
            StoredSecondaryMorphologyFailureKind::Retired(
                RetiredSecondaryMorphologyFailureKind::RegistryUnavailable
            )
        );
        let written: StoredSecondaryMorphologyFailureKind =
            serde_json::from_value(serde_json::json!("merge")).expect("a written cause reads");
        assert_eq!(
            written,
            StoredSecondaryMorphologyFailureKind::Written(SecondaryMorphologyFailureKind::Merge)
        );
        assert!(
            serde_json::from_value::<StoredSecondaryMorphologyFailureKind>(serde_json::json!(
                "invented"
            ))
            .is_err()
        );
    }

    #[test]
    fn a_stored_untimed_cause_reads_written_and_retired_kinds() {
        let written: StoredUntimedCause =
            serde_json::from_value(serde_json::json!({"kind": "unrecorded"})).unwrap();
        assert_eq!(
            written,
            StoredUntimedCause::Written(UntimedCauseRecord::Unrecorded)
        );
        let retired: StoredUntimedCause =
            serde_json::from_value(serde_json::json!({"kind": "no_usable_timing"})).unwrap();
        assert_eq!(
            retired,
            StoredUntimedCause::Retired(RetiredUntimedCause::NoUsableTiming)
        );
        let noted: StoredUntimedCause = serde_json::from_value(
            serde_json::json!({"kind": "not_in_recording", "postcode": "diary"}),
        )
        .unwrap();
        assert_eq!(
            noted.to_string(),
            "not in the recording: [+ diary]",
            "{noted:?}"
        );
        for cause in [written, retired, noted] {
            let json = serde_json::to_value(&cause).unwrap();
            assert!(json.get("kind").is_some(), "{json}");
            assert_eq!(
                serde_json::from_value::<StoredUntimedCause>(json).unwrap(),
                cause
            );
        }
        assert!(
            serde_json::from_value::<StoredUntimedCause>(serde_json::json!({"kind": "unknown"}))
                .is_err()
        );
    }

    /// Regression: the server returns relative paths like "PWA/TYO_a1.cha" as
    /// display names in file_statuses. These must deserialize successfully.
    /// Previously DisplayPath rejected path separators, causing "error decoding
    /// response body" on any job with files in subdirectories.
    #[test]
    fn job_info_deserializes_relative_path_filenames() {
        let json = r#"{
            "job_id": "abc-123",
            "status": "running",
            "command": "morphotag",
            "lang": "per-file",
            "options": {"command": "morphotag", "retokenize": false, "skipmultilang": false, "merge_abbrev": false},
            "total_files": 2,
            "completed_files": 0,
            "submitted_at": "2026-01-15T10:00:00.000Z",
            "file_statuses": [
                {"filename": "PWA/TYO_a1.cha", "status": "queued"},
                {"filename": "Control/TYO_n1.cha", "status": "queued"}
            ]
        }"#;
        let info: JobInfo = serde_json::from_str(json)
            .expect("JobInfo must deserialize file_statuses with relative-path display names");
        assert_eq!(info.file_statuses.len(), 2);
        assert_eq!(&*info.file_statuses[0].filename, "PWA/TYO_a1.cha");
    }
}

/// Host-wide memory pressure level derived from the current memory snapshot and
/// reserved headroom.
///
/// Declared here rather than in `host_memory`, where it lived until
/// 2026-07-30: it is a serde + `ToSchema` wire enum whose consumers are the
/// health response and the health route, and while it lived there it was one of
/// the references keeping `types` dependent on an impure module, and so keeping
/// every module downstream of `types` out of the core crate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum HostMemoryPressureLevel {
    /// Plenty of free headroom remains after the configured reserve.
    #[default]
    Healthy,
    /// Some headroom remains, but operators should expect reduced concurrency.
    Guarded,
    /// Very little headroom remains; only small new reservations should fit.
    Constrained,
    /// The configured reserve is exhausted or nearly exhausted.
    Critical,
}
