use crate::common::{
    LiveDirectSession, require_live_direct_warmed, require_live_direct_warmed_many,
};
use batchalign::api::{FilePayload, FileResult, JobInfo, ReleasedCommand};
use batchalign::chat_ops::{ChatFile, DependentTier};
use batchalign::options::CommandOptions;
use batchalign::worker::InferTask;
use batchalign_transform::parse::{TreeSitterParser, parse_lenient};
use std::sync::atomic::{AtomicU64, Ordering};

static LIVE_NAME_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct DirectGoldenSession {
    session: LiveDirectSession,
}

impl DirectGoldenSession {
    pub(crate) async fn submit_content_job(
        &self,
        command: ReleasedCommand,
        lang: &str,
        filename: &str,
        content: &str,
        options: CommandOptions,
    ) -> (JobInfo, Vec<FileResult>) {
        crate::common::submit_and_complete_direct(
            &self.session,
            command,
            lang,
            vec![FilePayload {
                filename: filename.into(),
                content: content.into(),
            }],
            options,
        )
        .await
    }

    pub(crate) async fn submit_files_job(
        &self,
        command: ReleasedCommand,
        lang: &str,
        files: Vec<FilePayload>,
        options: CommandOptions,
    ) -> (JobInfo, Vec<FileResult>) {
        crate::common::submit_and_complete_direct(&self.session, command, lang, files, options)
            .await
    }
}

/// Acquire a warmed live session, or FAIL LOUDLY.
///
/// This deliberately panics rather than skipping. Building this crate with
/// `--features ml-golden` is an explicit request to run the ML golden suite,
/// so a session that cannot be acquired means the request could not be
/// honoured, and reporting `ok` for a test that never executed is a false
/// green of exactly the kind this suite exists to prevent.
///
/// Found 2026-07-28: two newly added Italian golden tests reported `ok` in
/// 7.23s having produced no output at all. Only a mutation (replacing an
/// assertion with a deliberate lie and watching it still pass) would have
/// distinguished "passed" from "never ran". The suite had also been
/// unreachable for some time, with no Makefile or CI entry point and only
/// nextest's retired `--profile ml` to invoke it, so nobody noticed.
///
/// If the environment genuinely cannot host these tests (no Python worker, no
/// model weights, no credentials), do not run the suite: it is feature-gated
/// precisely so a plain `cargo test` never reaches it.
pub(crate) async fn require_direct_session_warmed(
    task: InferTask,
    command: ReleasedCommand,
    lang: &str,
    skip_message: &str,
) -> Option<DirectGoldenSession> {
    let session = require_live_direct_warmed(task, command, lang, skip_message)
        .await
        .unwrap_or_else(|| {
            panic!(
                "ml-golden requested but no live session for {command:?}/{lang} \
                 ({task:?}): {skip_message}. The suite is feature-gated, so \
                 reaching here means the environment cannot honour an explicit \
                 request; a silent skip would report a pass for a test that \
                 never ran."
            )
        });
    Some(DirectGoldenSession { session })
}

pub(crate) async fn require_direct_session_warmed_many(
    task: InferTask,
    warmups: Vec<(ReleasedCommand, &str)>,
    skip_message: &str,
) -> Option<DirectGoldenSession> {
    let session = require_live_direct_warmed_many(task, warmups, skip_message).await?;
    Some(DirectGoldenSession { session })
}

pub(crate) fn unique_test_dir(prefix: &str) -> String {
    let counter = LIVE_NAME_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{counter}")
}

pub(crate) fn parse_output(chat: &str, label: &str) -> ChatFile {
    let parser = TreeSitterParser::new().unwrap();
    let (file, errors) = parse_lenient(&parser, chat);
    assert!(errors.is_empty(), "{label}: CHAT parse errors: {errors:?}");
    file
}

pub(crate) fn has_mor_tier(file: &ChatFile) -> bool {
    file.lines.iter().any(|line| {
        if let batchalign::chat_ops::Line::Utterance(utt) = line {
            utt.dependent_tiers
                .iter()
                .any(|t| matches!(t.tier, DependentTier::Mor(_)))
        } else {
            false
        }
    })
}

pub(crate) fn has_gra_tier(file: &ChatFile) -> bool {
    file.lines.iter().any(|line| {
        if let batchalign::chat_ops::Line::Utterance(utt) = line {
            utt.dependent_tiers
                .iter()
                .any(|t| matches!(t.tier, DependentTier::Gra(_)))
        } else {
            false
        }
    })
}

pub(crate) fn has_user_defined_tier(file: &ChatFile, label: &str) -> bool {
    file.lines.iter().any(|line| {
        if let batchalign::chat_ops::Line::Utterance(utt) = line {
            utt.dependent_tiers.iter().any(|t| match &t.tier {
                DependentTier::UserDefined(ud) => ud.label.as_ref() == label,
                _ => false,
            })
        } else {
            false
        }
    })
}

pub(crate) fn find_mor_line_for(chat: &str, at_s_text: &str) -> Option<String> {
    let lines: Vec<&str> = chat.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains(at_s_text)
            && line.starts_with('*')
            && i + 1 < lines.len()
            && lines[i + 1].starts_with("%mor:")
        {
            return Some(lines[i + 1].trim_start_matches("%mor:\t").to_string());
        }
    }
    None
}

/// What a provenance stamp's timestamp reads as in a golden snapshot.
const PINNED_STAMP_TIMESTAMP: &str = "<timestamp>";

/// Pin the wall-clock timestamp of every provenance stamp in `chat`.
///
/// A stamp records what ran (command, engines, options that shape the output)
/// and when it ran. A golden snapshot must hold the first and never the
/// second, or it changes on every run and can never be accepted. Each line is
/// read by `extract_provenance`, the same codec the writer uses, so only a
/// real stamp is touched and a malformed one fails the test instead of being
/// masked. The timestamp is the stamp's closing section, so it is replaced as
/// the exact suffix the codec read back, not found by a pattern.
pub(crate) fn pin_provenance_timestamps(chat: &str) -> String {
    let mut pinned = String::with_capacity(chat.len());
    for line in chat.lines() {
        let entries = batchalign::provenance::extract_provenance(line)
            .unwrap_or_else(|stamp| panic!("golden output carries an unparseable stamp: {stamp}"));
        match entries.as_slice() {
            [] => pinned.push_str(line),
            [entry] => {
                let written = format!("{}]", entry.timestamp);
                let Some(head) = line.strip_suffix(&written) else {
                    panic!(
                        "stamp {line:?} does not close with the timestamp it was read with, \
                         {:?}",
                        entry.timestamp
                    );
                };
                pinned.push_str(head);
                pinned.push_str(PINNED_STAMP_TIMESTAMP);
                pinned.push(']');
            }
            several => panic!("one line read as {} stamps: {line:?}", several.len()),
        }
        pinned.push('\n');
    }
    if !chat.ends_with('\n') {
        pinned.pop();
    }
    pinned
}

/// Snapshot a golden output, with provenance timestamps pinned (see
/// [`pin_provenance_timestamps`]).
macro_rules! assert_golden_snapshot {
    ($name:expr, $value:expr) => {
        insta::with_settings!({snapshot_path => "../snapshots"}, {
            insta::assert_snapshot!(
                $name,
                crate::ml_golden::golden::helpers::pin_provenance_timestamps($value)
            );
        });
    };
}

pub(crate) use assert_golden_snapshot;
