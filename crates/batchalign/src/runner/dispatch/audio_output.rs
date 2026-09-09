//! Shared output-finalization for audio-backed commands.
//!
//! `align` and `transcribe` both produce one primary CHAT artifact per file and
//! both optionally run `merge_abbrev` before persisting it. `speaker-identify`
//! is audio-backed in exactly the same way and produces a JSON evidence
//! document instead. Keeping the persistence policy in one place makes the
//! per-file orchestrators easier to test and stops the three drifting.
//!
//! # Why the document and its kind travel together
//!
//! [`FileOutput`] is a sum, not a `String` beside a flag. The task that
//! produced the document is the only thing that knows what kind it is, and
//! pairing "here is some text" with "and by the way write it as CHAT" at a
//! separate call site is a relationship maintained by convention, where the
//! wrong combination type-checks. Carrying them together also makes the
//! nonsense pair unrepresentable: `merge_abbreviations` is a field of the CHAT
//! variant, so there is no way to ask for abbreviation merging on a JSON
//! evidence file and have it silently ignored.

use crate::api::{DisplayPath, ReleasedCommand};
use crate::pipeline::post_validate::{
    AbbreviationMergeRefused, PostValidated, PostValidationFailure,
};
use crate::recipe_runner::materialize::PlannedMaterializedFile;
use crate::recipe_runner::runtime::{
    ChatOutputTarget, primary_output_artifact, write_chat_output_artifact_with_provenance_gate,
    write_text_output_artifact,
};
use crate::scheduling::FailureCategory;
use crate::store::RunnerFilesystemConfig;

/// Whether a CHAT document has its single-letter abbreviations merged before
/// it is written.
///
/// A sum rather than a `bool` because `write_primary_output_artifact(.., true)`
/// says nothing at a call site about which question `true` answers, and this
/// value used to be the eighth positional argument of a nine-argument shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeAbbreviations {
    /// Collapse runs of single letters that name a known abbreviation.
    Merge,
    /// Write the document as the pipeline produced it.
    Leave,
}

impl MergeAbbreviations {
    /// Read the caller's `merge_abbrev` option into this vocabulary.
    pub(crate) fn from_option(should_merge: bool) -> Self {
        if should_merge {
            Self::Merge
        } else {
            Self::Leave
        }
    }
}

/// What one per-file attempt produced, and therefore how it is persisted.
///
/// `Eq` is deliberately absent: a `PostValidated` now carries the `ChatFile`
/// its bytes were serialized from, and `ChatFile` is `PartialEq` only (it holds
/// spans and floats). Nothing compares two of these for equality outside tests,
/// and a total equality on a document is not a thing this type needs to claim.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FileOutput {
    /// A CHAT document, written through the provenance gate so a re-run that
    /// changes only the `[ba3 ...]` timestamp does not touch the file.
    ///
    /// It carries the [`PostValidated`] PROOF, not a bare `String`. The
    /// producing task is the only party that knows what bar this document was
    /// admitted at, so it is the only party that can judge the output; this
    /// seam used to take text and manufacture a proof of its own at
    /// `StructurallyComplete`, which refused every `@Options: dummy` document
    /// align is required to hand back untouched.
    Chat {
        /// The gate-proven CHAT document.
        document: PostValidated,
        /// Whether to merge abbreviations before writing.
        merge_abbreviations: MergeAbbreviations,
    },
    /// A non-CHAT evidence document, written verbatim.
    ///
    /// Carries no content type of its own: the catalog's `output_policy`
    /// already declares one per command, and a second copy here would be a
    /// value the two could disagree about.
    Evidence {
        /// The serialized document body.
        body: String,
    },
}

/// Why a command's primary output could not be persisted.
///
/// Two different facts, and they call for two different operator actions: a
/// refusal means the bytes are wrong, an I/O error means the disk is. The
/// single call site used to report both as `FailureCategory::System`, because
/// the only error this returned was `std::io::Error`.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OutputWriteFailure {
    /// The finished bytes failed the post-validation gate. Nothing was
    /// written.
    #[error("{0}")]
    Refused(#[from] PostValidationFailure),
    /// The abbreviation merge broke a document that had already passed its
    /// gate. Nothing was written.
    ///
    /// Its own arm rather than a `Refused`, because the two send an operator
    /// to different places: `Refused` means the command's output is wrong,
    /// this means the command's output was RIGHT and a cosmetic transform
    /// applied afterwards is wrong. The refusal carries the admissible
    /// unmerged proof, which this seam declines to write; see the
    /// `MergeAbbreviations::Merge` arm below.
    #[error("{0}")]
    MergeRefused(#[from] AbbreviationMergeRefused),
    /// The filesystem refused the write.
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

impl OutputWriteFailure {
    /// Classify this failure for the control plane.
    pub(crate) fn category(&self) -> FailureCategory {
        match self {
            Self::Refused(_) | Self::MergeRefused(_) => FailureCategory::Validation,
            Self::Io(_) => FailureCategory::System,
        }
    }

    /// Render the operator-facing line for this failure.
    ///
    /// The single call site said "Failed to write {command} output" for both
    /// arms, which is a false statement of fact on the `Refused` arm: the gate
    /// refuses BEFORE the writer runs, so nothing was written and no operator
    /// looking at the disk would find a half-file. The sentence comes off the
    /// typed failure now, so a new arm has to say what it means.
    pub(crate) fn operator_message(&self, command_label: &str) -> String {
        match self {
            Self::Refused(failure) => {
                format!("Refused to write {command_label} output; nothing was written: {failure}")
            }
            Self::MergeRefused(refused) => {
                format!("Refused to write {command_label} output; nothing was written: {refused}")
            }
            Self::Io(error) => format!("Failed to write {command_label} output: {error}"),
        }
    }
}

/// Gate a CHAT document the command BUILT rather than transformed.
///
/// # Why `StructurallyComplete`, and who this is for
///
/// This is the floor, not an admission level, and the reason it can be stated
/// here is that this seam's callers have no input to have admitted.
/// `transcribe` builds its document from audio, so there is no input CHAT and
/// therefore no admitted level to carry forward; L1 plus `validate_output` is
/// the bar its output must meet.
///
/// `align` does NOT come through here, and the paragraph that used to claim it
/// gated at its admitted level "and this second, weaker gate covers the
/// transforms downstream" was false in the way that matters: the second gate
/// ran UNCONDITIONALLY, including over the `@Options: dummy` and `NoAlign`
/// documents `fa::FaAdmission::pass_through` deliberately does not judge, so a
/// dummy file that had always been written started failing at L1. Align now
/// carries its own [`PostValidated`] here, produced at its own admitted level,
/// and each transform downstream of it (the provenance stamp, the abbreviation
/// merge) is a named transition ON that proof rather than a step after it.
pub(crate) fn gate_built_chat_output(
    chat_text: &str,
    command: ReleasedCommand,
) -> Result<PostValidated, PostValidationFailure> {
    let parser = crate::chat_parser();
    let (file, _parse_errors) = batchalign_transform::parse::parse_lenient(&parser, chat_text);
    PostValidated::gate_owned(
        file,
        batchalign_transform::validate::ValidityLevel::StructurallyComplete,
        command,
    )
}

/// Persist the command's primary CHAT output and return its planned artifact.
///
/// The CHAT half of [`write_primary_output_artifact`], kept as its own function
/// because the provenance gate is a CHAT-specific policy with its own reasons,
/// and because the naming and staging behaviour it pins is what `align` and
/// `transcribe` have always done.
pub(crate) async fn write_primary_chat_output_artifact(
    filesystem: &RunnerFilesystemConfig,
    command: ReleasedCommand,
    file_index: usize,
    source_filename: &str,
    document: PostValidated,
    merge_abbreviations: MergeAbbreviations,
) -> Result<PlannedMaterializedFile, OutputWriteFailure> {
    // The merge is the last transition on the proof, and it re-runs the
    // judgement the proof already carries. Nothing between it and the write may
    // touch the bytes.
    //
    // POLICY: a refusal carries `unmerged`, an admissible document this seam
    // declines to write, and `?` is where that choice is made. Writing it
    // instead would ship a file whose requested merge silently did not happen;
    // failing surfaces a defect in the merge, which is what a broken merge is.
    let proof = match merge_abbreviations {
        MergeAbbreviations::Merge => document.with_abbreviations_merged()?,
        MergeAbbreviations::Leave => document,
    };
    let primary_output = primary_output_artifact(command, &DisplayPath::from(source_filename));
    let target = ChatOutputTarget::new(filesystem, file_index, &primary_output.display_path);
    write_chat_output_artifact_with_provenance_gate(&target, &proof).await?;
    Ok(primary_output)
}

/// Persist whatever one per-file attempt produced, and return its planned
/// artifact.
///
/// The single writeback seam for every audio-backed command. Both arms derive
/// the output path from the catalog's own `output_policy`, so a command's
/// artifact name is stated once, in the catalog, whichever kind it writes.
pub(crate) async fn write_primary_output_artifact(
    filesystem: &RunnerFilesystemConfig,
    command: ReleasedCommand,
    file_index: usize,
    source_filename: &str,
    output: FileOutput,
) -> Result<PlannedMaterializedFile, OutputWriteFailure> {
    match output {
        FileOutput::Chat {
            document,
            merge_abbreviations,
        } => {
            write_primary_chat_output_artifact(
                filesystem,
                command,
                file_index,
                source_filename,
                document,
                merge_abbreviations,
            )
            .await
        }
        FileOutput::Evidence { body } => {
            let primary_output =
                primary_output_artifact(command, &DisplayPath::from(source_filename));
            let target =
                ChatOutputTarget::new(filesystem, file_index, &primary_output.display_path);
            // No provenance gate. That gate exists because re-running a CHAT
            // command rewrites a `[ba3 ...]` timestamp line and produces
            // semantically empty corpus diffs. An evidence document IS a
            // record of one run, so a fresh one differing from the last is
            // the information, not noise.
            write_text_output_artifact(&target, &body).await?;
            Ok(primary_output)
        }
    }
}

#[cfg(test)]
mod tests {
    use batchalign_transform::serialize::to_chat_string;

    use super::*;
    use crate::api::ContentType;
    use batchalign_types::paths::{ClientPath, ServerPath};

    fn sample_filesystem(paths_mode: bool) -> RunnerFilesystemConfig {
        RunnerFilesystemConfig {
            paths_mode,
            source_paths: vec![ClientPath::new("/input/test.cha")],
            output_paths: vec![ClientPath::new("/tmp/output/test.cha")],
            before_paths: Vec::new(),
            staging_dir: ServerPath::new("/tmp/staging-audio-output"),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: ClientPath::new("/input"),
        }
    }

    /// The abbreviation merge is a transition on the PROOF, so a document
    /// gated at L1 comes back merged and still proven.
    #[test]
    fn merging_a_built_output_collapses_a_known_abbreviation() {
        let chat = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\tF B I do it .\n@End\n";
        let parser = crate::chat_parser();
        let merged = gate_built_chat_output(chat, ReleasedCommand::Transcribe)
            .expect("a valid transcribe document passes its own gate")
            .with_abbreviations_merged()
            .expect("a valid document with abbreviations merged still passes its gate");
        let (parsed, _) = batchalign_transform::parse::parse_lenient(&parser, merged.as_str());
        let reparsed = to_chat_string(&parsed);
        assert!(
            reparsed.contains("FBI"),
            "merge_abbrev should collapse 'F B I' into 'FBI', got: {reparsed}"
        );
    }

    /// RED FIRST (review item 5): the bytes written must be bytes a gate
    /// judged. This seam returned a bare `String`, so output that fails the
    /// gate reached `write_chat_output_artifact_with_provenance_gate`
    /// unexamined; and when `merge_abbrev` was on, the merge ran AFTER
    /// whatever gate had run upstream and its result was written unjudged.
    #[test]
    fn gating_a_built_output_refuses_a_document_that_lost_its_terminator() {
        let chat = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\tF B I do it\n@End\n";
        let failure = gate_built_chat_output(chat, ReleasedCommand::Transcribe)
            .expect_err("output with no terminator must be refused, not written");
        assert!(
            failure.to_string().contains("lost its terminator"),
            "the refusal must name what broke, got: {failure}"
        );
        let write_failure = OutputWriteFailure::from(failure);
        assert_eq!(
            write_failure.category(),
            crate::scheduling::FailureCategory::Validation,
            "a refusal on validity grounds must not be reported as a system failure"
        );
        // Review item 8: nothing is written when the gate refuses, so the
        // operator line must not claim a failed write.
        let message = write_failure.operator_message("align");
        assert!(
            message.contains("nothing was written"),
            "a gate refusal must not be reported as a failed write, got: {message}"
        );
    }

    /// RED FIRST (review item 1): a `@Options: dummy` document has no
    /// `@Participants` and so cannot satisfy L1. Align hands one back as a
    /// declared pass-through, and the writer must persist it unchanged rather
    /// than refuse it. Before the proof travelled to the writer, this seam
    /// gated every CHAT output at L1 and this document failed.
    #[tokio::test]
    async fn a_declared_pass_through_is_written_unchanged_not_refused() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let filesystem = RunnerFilesystemConfig {
            output_paths: vec![ClientPath::new(
                tmp.path().join("requested/dummy.cha").to_string_lossy(),
            )],
            staging_dir: ServerPath::new(tmp.path().join("staging")),
            ..sample_filesystem(true)
        };
        let dummy = "@UTF8\n@Begin\n@Options:\tdummy\n*PAR:\tF B I hello .\n@End\n";
        assert!(
            gate_built_chat_output(dummy, ReleasedCommand::Align).is_err(),
            "precondition: a dummy document cannot pass an L1 gate"
        );

        let artifact = write_primary_chat_output_artifact(
            &filesystem,
            ReleasedCommand::Align,
            0,
            "nested/dummy.cha",
            PostValidated::pass_through(dummy, ReleasedCommand::Align),
            // Even with the merge requested, a pass-through is left alone.
            MergeAbbreviations::Merge,
        )
        .await
        .expect("a declared pass-through must be written, not refused");

        assert_eq!(artifact.content_type, ContentType::Chat);
        let written = std::fs::read_to_string(tmp.path().join("requested/dummy.cha"))
            .expect("read written output");
        assert_eq!(
            written, dummy,
            "a document the command did not modify must reach disk byte-identical"
        );
    }

    #[tokio::test]
    async fn write_primary_chat_output_artifact_uses_command_primary_path() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let filesystem = RunnerFilesystemConfig {
            output_paths: vec![ClientPath::new(
                tmp.path().join("requested/test.cha").to_string_lossy(),
            )],
            staging_dir: ServerPath::new(tmp.path().join("staging")),
            ..sample_filesystem(true)
        };
        let chat = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";

        let artifact = write_primary_chat_output_artifact(
            &filesystem,
            ReleasedCommand::Transcribe,
            0,
            "nested/test.mp3",
            gate_built_chat_output(chat, ReleasedCommand::Transcribe).expect("valid CHAT"),
            MergeAbbreviations::Leave,
        )
        .await
        .expect("write artifact");

        assert_eq!(artifact.content_type, ContentType::Chat);
        assert_eq!(artifact.display_path, DisplayPath::from("nested/test.cha"));
        let written = std::fs::read_to_string(tmp.path().join("requested/test.cha"))
            .expect("read written output");
        assert!(written.contains("*PAR:\thello ."));
    }
}
