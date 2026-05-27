//! Structural merge of two CHAT transcripts sharing a media timeline.
//!
//! See `book/src/chatter/user-guide/merge.md` for the user contract
//! and `book/src/architecture/merge-test-plan.md` for the cycle plan
//! that drives this module's incremental growth.
//!
//! Phase A cycle 1 (this commit): minimal happy path — parse two
//! files, partition utterances by retain set, sort by `start_ms`,
//! serialize. No tier stripping, no header reconciliation beyond
//! "use File 1's headers verbatim", no preconditions enforced, no
//! domain newtypes. Later cycles tighten each behavior.
//!
//! The signature exposed here is deliberately the bare minimum the
//! cycle-1 smoke test needs; subsequent cycles will introduce
//! `RetainSet`, `MergeError`, `--strip-tiers`, etc.

use talkbank_model::ParseValidateOptions;
use talkbank_model::model::header::{Header, ParticipantEntries, ParticipantEntry};
use talkbank_model::model::{ChatFile, Line};

use crate::PipelineError;
use crate::pipeline::parse_and_validate;
use crate::serialize::to_chat_string;

/// Errors that can arise from the merge operation.
///
/// Each variant maps to a CLI exit code per the user-guide contract:
/// `Parse` → exit 1 (invalid input); everything else → exit 2
/// (precondition violation). The CLI layer is responsible for the
/// mapping; `MergeError` itself just classifies the failure mode.
///
/// Documented design home: `book/src/architecture/merge-domain-types.md`.
/// This enum lives in `talkbank-transform::transcript_merge` for v1;
/// it may move to `talkbank-model::merge::errors` once an
/// out-of-transform consumer needs it.
#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    /// File 1 declares no utterances for any speaker in the retain
    /// set. The merge would produce a file with no retained content
    /// (a degenerate output that researchers would mistake for a
    /// successful merge); we refuse instead.
    #[error("File 1 declares no utterances for any speaker in --retain ({retain:?})")]
    RetainSpeakersMissing { retain: Vec<String> },

    /// File 1 has retained-speaker utterances but none carry a time
    /// bullet. Without a bulleted utterance the merge has no shared
    /// timeline against which to position File 2's content, so any
    /// "merge" would be a meaningless start-time-less concatenation.
    #[error("File 1 has no time-bulleted utterances; cannot merge against a shared timeline")]
    NoTimelineInFile1,

    /// Underlying parse error from either input file.
    #[error("parse error: {0}")]
    Parse(#[from] PipelineError),
}

/// Default set of dependent-tier kinds stripped from inserted-speaker
/// utterances during merge. Each of these has an authoritative
/// producer stage downstream of merge (`align` regenerates `%wor`;
/// `morphotag` regenerates `%mor` / `%gra`; FA owns `%pho`), so
/// carrying them across the merge boundary leaves the merged file in
/// an inconsistent half-state. Stripping at merge time pushes the
/// merged file into a clean "no derived tiers" state that downstream
/// stages can own end-to-end.
///
/// Listed lowercase to match `DependentTier::kind()`. Callers that
/// want a `Vec<String>`-form of this set (e.g. CLI argument
/// defaulting) use [`default_strip_tiers`].
pub const DEFAULT_STRIP_TIERS: &[&str] = &["wor", "mor", "gra", "pho"];

/// `Vec<String>` form of [`DEFAULT_STRIP_TIERS`] for boundary code
/// (CLI argument parsing, library calls that hold owned strings)
/// that needs an allocated owned value rather than the static
/// `&[&str]` constant.
pub fn default_strip_tiers() -> Vec<String> {
    DEFAULT_STRIP_TIERS
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// Merge two CHAT files. Retained-set speakers' utterances come from
/// `file1_content`; all other speakers' utterances come from
/// `file2_content`. The merged file's headers come from `file1_content`.
///
/// `strip_tiers` lists dependent-tier kinds (lowercase, matching
/// `DependentTier::kind()` — `"wor"`, `"mor"`, `"gra"`, `"pho"`, etc.)
/// that should be removed from inserted-speaker utterances before
/// they are emitted. The strip set never affects retained-speaker
/// utterances. Callers that want the standard pipeline behavior
/// pass [`DEFAULT_STRIP_TIERS`]; an empty slice preserves every
/// dependent tier verbatim.
///
/// Utterances are ordered ascending by `start_ms`; utterances missing
/// a main-tier bullet sort to the end.
pub fn merge_chats(
    file1_content: &str,
    file2_content: &str,
    retain: &[String],
    strip_tiers: &[String],
    options: ParseValidateOptions,
) -> Result<String, MergeError> {
    let f1 = parse_and_validate(file1_content, options.clone())?;
    let f2 = parse_and_validate(file2_content, options)?;

    let in_retain = |speaker: &str| retain.iter().any(|s| s == speaker);

    // Precondition: File 1 must declare at least one utterance for
    // some speaker in `retain`. Without this, the merge would emit a
    // file with no retained content — a degenerate output that
    // looks like a successful merge but is actually missing the
    // authoritative data the operator wanted to preserve. Refuse
    // loudly instead.
    let retained_utts_in_f1: Vec<&Line> = f1
        .lines
        .0
        .iter()
        .filter(|line| match line {
            Line::Utterance(u) => in_retain(u.main.speaker.as_str()),
            _ => false,
        })
        .collect();
    if retained_utts_in_f1.is_empty() {
        return Err(MergeError::RetainSpeakersMissing {
            retain: retain.to_vec(),
        });
    }

    // Precondition: at least one retained utterance must carry a
    // time bullet. The merge orders all utterances by `start_ms`
    // and positions File 2's content against File 1's bullets; with
    // zero bullets there is no anchor for the shared timeline.
    let any_bulleted = retained_utts_in_f1.iter().any(|line| match line {
        Line::Utterance(u) => u.main.content.bullet.is_some(),
        _ => false,
    });
    if !any_bulleted {
        return Err(MergeError::NoTimelineInFile1);
    }

    // Collect File 2's participant entries for speakers NOT in
    // `retain` — these will extend File 1's @Participants header.
    let inserted_participants: Vec<ParticipantEntry> = f2
        .lines
        .0
        .iter()
        .filter_map(|line| match line {
            Line::Header { header, .. } => match header.as_ref() {
                Header::Participants { entries } => Some(entries),
                _ => None,
            },
            _ => None,
        })
        .flat_map(|entries| entries.iter().cloned())
        .filter(|entry| !in_retain(entry.speaker_code.as_str()))
        .collect();

    // Collect File 2's @ID rows for speakers NOT in `retain` —
    // these are injected after File 1's last @ID row.
    let inserted_id_lines: Vec<Line> = f2
        .lines
        .0
        .iter()
        .filter(|line| match line {
            Line::Header { header, .. } => match header.as_ref() {
                Header::ID(id) => !in_retain(id.speaker.as_str()),
                _ => false,
            },
            _ => false,
        })
        .cloned()
        .collect();

    // Collect File 2's @Comment rows verbatim. Donor @Comment
    // content carries provenance (ASR engine identification, run
    // timestamps, processing notes) that the merged file's audit
    // trail must preserve.
    let inserted_comment_lines: Vec<Line> = f2
        .lines
        .0
        .iter()
        .filter(|line| match line {
            Line::Header { header, .. } => matches!(header.as_ref(), Header::Comment { .. }),
            _ => false,
        })
        .cloned()
        .collect();

    // Indices of File 1's last @ID and last @Comment lines, if any.
    // We use these as the "insert after" points for the
    // corresponding File 2 rows. The helper centralizes the
    // shared shape (reverse-scan for last matching header).
    let f1_last_id_idx = last_header_index(&f1, |h| matches!(h, Header::ID(_)));
    let f1_last_comment_idx = last_header_index(&f1, |h| matches!(h, Header::Comment { .. }));

    // Split File 1's lines into pre-@End headers and the @End marker.
    // The @Participants header (if any) is rewritten to concatenate
    // File 1's entries with `inserted_participants`. Utterances from
    // File 1 are kept only if their speaker is in `retain`.
    let mut pre_end_headers: Vec<Line> = Vec::new();
    let mut end_marker: Option<Line> = None;
    let mut retained_utts: Vec<Line> = Vec::new();

    for (i, line) in f1.lines.0.iter().enumerate() {
        match line {
            Line::Header { header, span } => {
                if matches!(header.as_ref(), Header::End) {
                    end_marker = Some(line.clone());
                } else if let Header::Participants { entries } = header.as_ref() {
                    let mut combined: Vec<ParticipantEntry> = entries.iter().cloned().collect();
                    combined.extend(inserted_participants.iter().cloned());
                    let merged_header = Header::Participants {
                        entries: ParticipantEntries::new(combined),
                    };
                    pre_end_headers.push(Line::Header {
                        header: Box::new(merged_header),
                        span: *span,
                    });
                } else {
                    pre_end_headers.push(line.clone());
                }
            }
            Line::Utterance(u) => {
                if in_retain(u.main.speaker.as_str()) {
                    retained_utts.push(line.clone());
                }
            }
        }
        // After emitting File 1's last @ID row, inject File 2's
        // non-retained @ID rows so they appear contiguously with
        // File 1's @ID block. After File 1's last @Comment row,
        // inject File 2's @Comment rows so donor provenance is
        // preserved in the audit trail. Both follow the
        // user-guide contract: "File 1's rows first, then File 2's
        // rows in original order."
        if Some(i) == f1_last_id_idx {
            pre_end_headers.extend(inserted_id_lines.iter().cloned());
        }
        if Some(i) == f1_last_comment_idx {
            pre_end_headers.extend(inserted_comment_lines.iter().cloned());
        }
    }

    // From File 2, take only utterances whose speaker is NOT in
    // `retain`. (Header reconciliation beyond "File 1 wins" is a
    // later cycle.) Strip dependent tiers in DEFAULT_STRIP_TIERS so
    // the merged file enters its downstream stages
    // (`batchalign3 align`, `batchalign3 morphotag`) in the
    // expected "no derived tiers" state.
    let mut inserted_utts: Vec<Line> = Vec::new();
    for line in f2.lines.0.iter() {
        if let Line::Utterance(u) = line
            && !in_retain(u.main.speaker.as_str())
        {
            let mut cloned = u.as_ref().clone();
            cloned
                .dependent_tiers
                .retain(|t| !strip_tiers.iter().any(|s| s == t.kind()));
            inserted_utts.push(Line::Utterance(Box::new(cloned)));
        }
    }

    // Combine and sort by start_ms. Utterances without a main-tier
    // bullet sort to the end with `u64::MAX` so they don't disturb
    // the ordering of timed utterances.
    let mut all_utts: Vec<Line> = retained_utts;
    all_utts.extend(inserted_utts);
    all_utts.sort_by_key(line_start_ms);

    // Assemble: pre-@End headers, sorted utterances, @End marker.
    let mut out_lines = pre_end_headers;
    out_lines.extend(all_utts);
    if let Some(end) = end_marker {
        out_lines.push(end);
    }

    let merged = ChatFile::new(out_lines);
    Ok(to_chat_string(&merged))
}

/// Extract an utterance's main-tier `start_ms`. Returns `u64::MAX`
/// for non-utterance lines and for utterances without a main-tier
/// bullet, so those entries sort to the end of the timeline.
fn line_start_ms(line: &Line) -> u64 {
    match line {
        Line::Utterance(u) => u
            .main
            .content
            .bullet
            .as_ref()
            .map(|b| b.timing.start_ms)
            .unwrap_or(u64::MAX),
        Line::Header { .. } => u64::MAX,
    }
}

/// Find the index of the last header line in `chat_file` whose
/// `Header` payload matches `predicate`. Returns `None` if no
/// matching header is present.
///
/// Used by the header-reconciliation logic to identify the slot at
/// which File 2's contributions of a given header kind (e.g. @ID,
/// @Comment) should be inserted to keep the kind contiguous in the
/// merged output.
fn last_header_index<F>(chat_file: &ChatFile, predicate: F) -> Option<usize>
where
    F: Fn(&Header) -> bool,
{
    chat_file
        .lines
        .0
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, line)| match line {
            Line::Header { header, .. } if predicate(header.as_ref()) => Some(i),
            _ => None,
        })
}
