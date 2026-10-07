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
    AbbreviationMergeRefused, OutputReport, PostValidationFailure, ProducedOutput, Shortfall,
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

/// A judged CHAT document with what the writer must report beside it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatOutput {
    /// The judged CHAT document.
    pub(crate) document: ProducedOutput,
    /// Requested work the document does not carry, reported with the
    /// file's outcome whatever the document's standing.
    pub(crate) shortfalls: Vec<Shortfall>,
    /// Whether to merge abbreviations before writing.
    pub(crate) merge_abbreviations: MergeAbbreviations,
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
    /// changes only the `[fc-ba3 ...]` timestamp does not touch the file.
    ///
    /// It carries the judged PROOF, not a bare `String`: an admitted
    /// [`PostValidated`], or a generating producer's [`ProducedOutput`] that
    /// may be diagnosed. The producing task establishes the judgement before
    /// this seam; the writer cannot select a weaker validity level or
    /// manufacture a proof from serialized CHAT.
    Chat(ChatOutput),
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
/// control plane classifies both as system failures, not bad client input.
#[derive(Debug, thiserror::Error)]
pub(crate) enum OutputWriteFailure {
    /// A producer could not bind the output to the submitted command.
    #[error(transparent)]
    Planning(#[from] crate::recipe_runner::planner::PlanningError),
    /// A document proof is not permission to write a binary recording.
    #[error("document writer cannot publish planned {0} content")]
    ContentKind(crate::api::ContentType),
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
    MergeRefused(#[from] Box<AbbreviationMergeRefused>),
    /// The filesystem refused the write.
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

impl OutputWriteFailure {
    /// Classify this failure for the control plane.
    pub(crate) fn category(&self) -> FailureCategory {
        match self {
            Self::Refused(_)
            | Self::MergeRefused(_)
            | Self::Io(_)
            | Self::Planning(_)
            | Self::ContentKind(_) => FailureCategory::System,
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
            Self::Planning(error) => format!("Failed to plan {command_label} output: {error}"),
            Self::ContentKind(kind) => {
                format!("Refused document write for {command_label}: planned {kind} content")
            }
        }
    }
}

/// What a writer persisted, and whether the bytes were admitted.
///
/// A sum, so the runner records a diagnosed file as diagnosed without being
/// able to forget to ask: there is no artifact without its standing.
#[derive(Debug)]
pub(crate) enum WrittenOutput {
    /// Admitted CHAT, or a document kind with no admission (evidence JSON).
    Clean(PlannedMaterializedFile),
    /// A generating producer's CHAT, written with what its admission found.
    Diagnosed {
        /// The written artifact.
        artifact: PlannedMaterializedFile,
        /// Every finding, and every stage skipped because of them.
        diagnostics: crate::api::FileOutputDiagnostics,
    },
}

impl WrittenOutput {
    /// Record the file's terminal phase for what was written: `Done` for a
    /// clean output, `Diagnosed` (never an error, never retried) otherwise.
    pub(crate) async fn record(self, lifecycle: &crate::runner::util::FileRunTracker<'_>) {
        match self {
            Self::Clean(artifact) => {
                lifecycle
                    .complete_with_result(artifact.display_path, artifact.content_type)
                    .await;
            }
            Self::Diagnosed {
                artifact,
                diagnostics,
            } => {
                lifecycle
                    .complete_diagnosed(artifact.display_path, artifact.content_type, diagnostics)
                    .await;
            }
        }
    }
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
    options: &crate::options::CommandOptions,
    file_index: usize,
    source_filename: &str,
    output: ChatOutput,
) -> Result<WrittenOutput, OutputWriteFailure> {
    let ChatOutput {
        document,
        shortfalls,
        merge_abbreviations,
    } = output;
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
    let primary_output =
        primary_output_artifact(command, options, &DisplayPath::from(source_filename))?;
    if primary_output.content_type != crate::api::ContentType::Chat {
        return Err(OutputWriteFailure::ContentKind(primary_output.content_type));
    }
    let target = ChatOutputTarget::new(filesystem, file_index, &primary_output.display_path);
    write_chat_output_artifact_with_provenance_gate(&target, &proof).await?;
    // Read from the proof that was WRITTEN (after the merge), so the reported
    // findings describe the bytes on disk, with every shortfall of the run.
    Ok(match OutputReport::of(&proof, &shortfalls) {
        OutputReport::Clean => WrittenOutput::Clean(primary_output),
        OutputReport::Diagnosed(draft) => {
            // A finding list too long for the record is written once, in
            // full, beside the job's staged outputs, never into the user's
            // output tree.
            let sidecar = filesystem
                .staging_dir
                .as_path()
                .join("diagnostics")
                .join(format!("{}.findings.json", primary_output.display_path));
            WrittenOutput::Diagnosed {
                diagnostics: draft.record(&sidecar).await,
                artifact: primary_output,
            }
        }
    })
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
    options: &crate::options::CommandOptions,
    file_index: usize,
    source_filename: &str,
    output: FileOutput,
) -> Result<WrittenOutput, OutputWriteFailure> {
    match output {
        FileOutput::Chat(chat) => {
            write_primary_chat_output_artifact(
                filesystem,
                command,
                options,
                file_index,
                source_filename,
                chat,
            )
            .await
        }
        FileOutput::Evidence { body } => {
            let primary_output =
                primary_output_artifact(command, options, &DisplayPath::from(source_filename))?;
            if primary_output.content_type.is_binary()
                || primary_output.content_type == crate::api::ContentType::Chat
            {
                return Err(OutputWriteFailure::ContentKind(primary_output.content_type));
            }
            let target =
                ChatOutputTarget::new(filesystem, file_index, &primary_output.display_path);
            // No provenance gate. That gate exists because re-running a CHAT
            // command rewrites a `[fc-ba3 ...]` timestamp line and produces
            // semantically empty corpus diffs. An evidence document IS a
            // record of one run, so a fresh one differing from the last is
            // the information, not noise.
            write_text_output_artifact(&target, &body).await?;
            Ok(WrittenOutput::Clean(primary_output))
        }
    }
}

#[cfg(test)]
mod tests {
    use batchalign_transform::serialize::to_chat_string;

    use super::*;
    use crate::api::ContentType;
    use crate::pipeline::post_validate::PostValidated;
    use batchalign_types::paths::{ClientPath, ServerPath};

    #[tokio::test]
    async fn document_output_cannot_publish_a_native_binary_destination() {
        let root = tempfile::tempdir().unwrap();
        let filesystem = RunnerFilesystemConfig {
            staging_dir: ServerPath::new(root.path()),
            ..sample_filesystem(false)
        };
        let options = crate::options::CommandOptions::Convert(crate::options::ConvertOptions {
            common: crate::options::CommonOptions::default(),
            format: crate::media::export::AudioExportFormat::Wav,
        });
        let failure = write_primary_output_artifact(
            &filesystem,
            ReleasedCommand::Convert,
            &options,
            0,
            "source.wav",
            FileOutput::Evidence {
                body: "not audio".into(),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(
            failure,
            OutputWriteFailure::ContentKind(ContentType::Wav)
        ));
        assert_eq!(failure.category(), FailureCategory::System);
        assert!(!root.path().join("output").exists());
    }

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
        let merged = PostValidated::gate_owned(
            batchalign_transform::parse::parse_lenient(&parser, chat).0,
            ReleasedCommand::Transcribe,
        )
        .expect("a valid transcribe document passes its own gate")
        .with_abbreviations_merged()
        .expect("a valid document with abbreviations merged still passes its gate");
        let reparsed = to_chat_string(merged.document());
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
        let failure = PostValidated::gate_owned(
            batchalign_transform::parse::parse_lenient(&crate::chat_parser(), chat).0,
            ReleasedCommand::Transcribe,
        )
        .expect_err("output with no terminator must be refused, not written");
        assert!(
            failure.to_string().contains("lost its terminator"),
            "the refusal must name what broke, got: {failure}"
        );
        let write_failure = OutputWriteFailure::from(failure);
        assert_eq!(
            write_failure.category(),
            crate::scheduling::FailureCategory::System,
            "invalid produced output is a tool failure, not invalid client input"
        );
        // Review item 8: nothing is written when the gate refuses, so the
        // operator line must not claim a failed write.
        let message = write_failure.operator_message("align");
        assert!(
            message.contains("nothing was written"),
            "a gate refusal must not be reported as a failed write, got: {message}"
        );
    }

    /// A valid NoAlign source retains exact bytes through the real writer,
    /// including when cosmetic merging was requested.
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
        let dummy = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Child\n@Options:\tNoAlign\n\
@ID:\teng|test|CHI|||||Child|||\n*CHI:\tF B I hello .\n@End\n";
        let unchanged = crate::fa::read_fa_source(dummy)
            .expect("the NoAlign source is valid")
            .unchanged()
            .expect("NoAlign carries source admission")
            .clone();

        let artifact = write_primary_chat_output_artifact(
            &filesystem,
            ReleasedCommand::Align,
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Align),
            0,
            "nested/dummy.cha",
            ChatOutput {
                document: PostValidated::pass_through(unchanged, ReleasedCommand::Align).into(),
                shortfalls: Vec::new(),
                // Even with the merge requested, a pass-through is left alone.
                merge_abbreviations: MergeAbbreviations::Merge,
            },
        )
        .await
        .expect("a declared pass-through must be written, not refused");
        let WrittenOutput::Clean(artifact) = artifact else {
            panic!("an unchanged admitted source is written clean");
        };

        assert_eq!(artifact.content_type, ContentType::Chat);
        let written = std::fs::read_to_string(tmp.path().join("requested/dummy.cha"))
            .expect("read written output");
        assert_eq!(
            written, dummy,
            "a document the command did not modify must reach disk byte-identical"
        );
    }

    #[test]
    fn no_align_cannot_exempt_corrupt_retained_morphology() {
        let valid = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Child\n@Options:\tNoAlign\n\
@ID:\teng|test|CHI|||||Child|||\n*CHI:\thello world .\n@End\n";
        let admitted = crate::fa::read_fa_source(valid).expect("valid control");
        assert!(admitted.unchanged().is_some());
        let corrupt = valid.replace("@End", "%mor:\tnoun|hello\n@End");
        assert_ne!(corrupt, valid);
        let error = crate::fa::read_fa_source(&corrupt)
            .err()
            .expect("NoAlign retains every tier, so corrupt morphology must refuse");
        assert!(matches!(
            error,
            crate::error::ServerError::ChatReplacementAdmission(_)
        ));
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            FailureCategory::Validation
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
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Transcribe),
            0,
            "nested/test.mp3",
            ChatOutput {
                document: PostValidated::gate_owned(
                    batchalign_transform::parse::parse_lenient(&crate::chat_parser(), chat).0,
                    ReleasedCommand::Transcribe,
                )
                .expect("valid CHAT")
                .into(),
                shortfalls: Vec::new(),
                merge_abbreviations: MergeAbbreviations::Leave,
            },
        )
        .await
        .expect("write artifact");
        let WrittenOutput::Clean(artifact) = artifact else {
            panic!("an admitted document is written clean");
        };

        assert_eq!(artifact.content_type, ContentType::Chat);
        assert_eq!(artifact.display_path, DisplayPath::from("nested/test.cha"));
        let written = std::fs::read_to_string(tmp.path().join("requested/test.cha"))
            .expect("read written output");
        assert!(written.contains("*PAR:\thello ."));
    }

    /// RED FIRST (2026-10-06): a generating producer's diagnosed output
    /// reaches disk through the real writer, with the merge it asked for, and
    /// the writer reports the diagnosis instead of a clean write or a refusal.
    #[tokio::test]
    async fn a_diagnosed_transcript_is_written_and_reported_diagnosed() {
        use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};

        let tmp = tempfile::tempdir().expect("tempdir");
        let filesystem = RunnerFilesystemConfig {
            output_paths: vec![ClientPath::new(
                tmp.path().join("requested/test.cha").to_string_lossy(),
            )],
            staging_dir: ServerPath::new(tmp.path().join("staging")),
            ..sample_filesystem(true)
        };
        let chat = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";
        let mut generated =
            batchalign_transform::parse::parse_lenient(&crate::chat_parser(), chat).0;
        for line in &mut generated.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.content = TierContentItems::new(
                    ["F", "B", "I", "said", "b2"]
                        .into_iter()
                        .map(|word| UtteranceContent::Word(Box::new(Word::simple(word))))
                        .collect(),
                );
            }
        }

        let written = write_primary_chat_output_artifact(
            &filesystem,
            ReleasedCommand::Transcribe,
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Transcribe),
            0,
            "nested/test.mp3",
            ChatOutput {
                document: PostValidated::produced(generated, ReleasedCommand::Transcribe),
                shortfalls: Vec::new(),
                merge_abbreviations: MergeAbbreviations::Merge,
            },
        )
        .await
        .expect("a diagnosed transcript is written, not refused");

        let WrittenOutput::Diagnosed {
            artifact,
            diagnostics,
        } = written
        else {
            panic!("the writer must report the diagnosis");
        };
        assert_eq!(artifact.display_path, DisplayPath::from("nested/test.cha"));
        assert!(
            diagnostics
                .first_findings()
                .iter()
                .any(|finding| finding.code.as_deref() == Some("E220")),
            "{diagnostics:?}"
        );
        let on_disk = std::fs::read_to_string(tmp.path().join("requested/test.cha"))
            .expect("the diagnosed output is on disk");
        assert!(on_disk.contains("*PAR:\tFBI said b2 ."), "{on_disk}");
    }
}
