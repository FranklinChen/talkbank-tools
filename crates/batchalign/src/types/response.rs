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
use super::request::default_lang;
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
        lines.extend(self.shortfalls.iter().map(ToString::to_string));
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

/// Requested work a written document does not carry, and why.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutputShortfallRecord {
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
        /// The first of them (positions among the file's utterances before
        /// the stage, counting from 1), at most
        /// [`FileOutputDiagnostics::FIRST_FINDINGS`].
        first_held_out: Vec<u64>,
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

/// One utterance forced alignment left with untimed words.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct UntimedUtteranceRecord {
    /// The utterance's position among the file's utterances, counting from 1.
    pub utterance: u64,
    /// Required lexical words in it.
    pub words: u64,
    /// How many of them are untimed, at least one.
    pub untimed_words: u64,
    /// Why.
    pub cause: UntimedCauseRecord,
}

/// Why an utterance's words have no timing.
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
    /// No refusal names it, and no positive interval resulted for these
    /// words: the aligner returned none, or they could not be sent to it.
    NoUsableTiming,
    /// The transcript marks it as not speech in the recording, so alignment
    /// never looks for it there: no request, no bullet, untimed by design.
    NotInRecording {
        /// The postcode that marks it.
        postcode: OffRecordPostcode,
    },
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
            Self::NoUsableTiming => f.write_str("no usable timing"),
            Self::NotInRecording { postcode } => write!(f, "not in the recording: {postcode}"),
        }
    }
}

impl std::fmt::Display for OutputShortfallRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
                first_held_out,
            } => {
                write!(
                    f,
                    "{} applied except to {held_out_utterances} utterance(s) that carry the \
                     findings, which keep their generated form",
                    stage.name()
                )?;
                if let Some(first) = first_held_out.first() {
                    write!(f, " (first: utterance {first})")?;
                }
                Ok(())
            }
            Self::TimingIncomplete {
                required_words,
                untimed_words,
                untimed_utterances,
                first_untimed,
            } => {
                write!(
                    f,
                    "timing incomplete: {untimed_words} of {required_words} words in \
                     {untimed_utterances} utterance(s) have no timing and were written without it"
                )?;
                if let Some(first) = first_untimed.first() {
                    write!(
                        f,
                        " (first: utterance {}, {} of {} words, {})",
                        first.utterance, first.untimed_words, first.words, first.cause
                    )?;
                }
                Ok(())
            }
        }
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
    /// code-switched pair such as `"eng,spa"`.
    #[serde(default = "default_lang")]
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
    /// code-switched pair such as `"eng,spa"`.
    #[serde(default = "default_lang")]
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
    /// Total physical memory in MB.
    #[serde(default)]
    pub system_memory_total_mb: MemoryMb,
    /// Available memory in MB (free + reclaimable).  On macOS this is
    /// `free + purgeable` which can undercount; see `sysinfo` docs.
    #[serde(default)]
    pub system_memory_available_mb: MemoryMb,
    /// Used memory in MB (`total - available`).
    #[serde(default)]
    pub system_memory_used_mb: MemoryMb,
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
