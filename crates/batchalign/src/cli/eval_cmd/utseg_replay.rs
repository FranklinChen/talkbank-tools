//! Offline replay of retained utterance-segmentation evidence.
//!
//! A run that segments a file retains exactly what it applied: the words it
//! dispatched and the boundary decision it applied to each of them
//! (`crate::utseg_evidence`). This replay reapplies that evidence with the
//! current build and asks one question: does it still produce the document the
//! run wrote? Nothing here infers: no model loads, no worker starts, and no
//! artifact is modified.
//!
//! # Two passes, two typed inputs
//!
//! Transcribe segments twice, over different populations, and each pass
//! retains its own artifact. The two are separate subcommands consuming
//! different inputs rather than one argument list read two ways:
//!
//! - `post-chat` reapplies post-CHAT evidence to the CHAT document the run
//!   segmented, which is what the standalone `utseg` command does.
//! - `pre-asr` reapplies pre-CHAT evidence to the retained ASR response, then
//!   builds CHAT from the result, which is what transcribe does before the
//!   document exists.
//!
//! Each mode names the evidence phase it reproduces when it admits the
//! artifact, so a pass can never be replayed against the other pass's evidence.
//!
//! # What the comparison ignores, and why
//!
//! A run also writes comments that say a run happened: its per-command stamp
//! and, for transcribe, the unchecked-ASR warning. A stamp carries a timestamp
//! and the warning carries a build identity, so neither can ever match by
//! equality. Both are recognized through the provenance codec that writes them
//! (`crate::provenance::recognize_generated_comment`), set aside on both sides,
//! and reported. Every other line is compared for CHAT semantics.

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::chat_ops::{ChatFile, DependentTier, Header, LanguageCode, Line};
use crate::cli::args::{
    UtsegPostChatReplayArgs, UtsegPreAsrReplayArgs, UtsegReplayAction, UtsegReplayArgs,
};
use crate::cli::error::CliError;
use crate::cli::eval_cmd::InputIdentity;
use crate::provenance::{GeneratedComment, UnparseableStamp, recognize_generated_comment};
use crate::utseg::{
    AdmittedUtsegPrediction, collect_utseg_batch_items, integrate_admitted_assignments,
};
use crate::utseg_evidence::{
    AdmittedUtsegEvidence, AdmittedUtsegEvidenceItem, UtsegEvidenceAdmissionError,
    UtsegEvidencePhase,
};
use batchalign_transform::asr_postprocess;
use batchalign_transform::build_chat;
use batchalign_transform::parse::parse_lenient;
use batchalign_transform::serialize::to_chat_string;
use batchalign_transform::utseg::{UtsegBatchItem, apply_utseg_results};
use batchalign_transform::validate::{ValidityLevel, validate_to_level};
use talkbank_model::SemanticEq;

/// Run one utterance-segmentation replay.
///
/// Publishes the typed report on stdout whatever the comparison found, then
/// returns the verdict: a document that did not reproduce is an error, so a
/// script sees it without parsing the report.
pub fn run(args: &UtsegReplayArgs) -> Result<(), CliError> {
    let report = match &args.action {
        UtsegReplayAction::PostChat(args) => replay_post_chat(args),
        UtsegReplayAction::PreAsr(args) => replay_pre_asr(args),
    }?;

    let mut rendered = serde_json::to_vec_pretty(&report)?;
    rendered.push(b'\n');
    std::io::stdout().write_all(&rendered)?;

    match report.outcome {
        UtsegReproduction::Reproduced => Ok(()),
        UtsegReproduction::Differing(difference) => Err(CliError::ReplayDiffers(difference)),
    }
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// What the comparison found.
///
/// Returned from the replay and reported by its caller, so the verdict travels
/// as a value rather than as text printed from inside the comparison.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum UtsegReproduction {
    /// Every comparable line of the recomputed document matches the retained
    /// one.
    Reproduced,
    /// They differ, in this exact way.
    Differing(UtsegReproductionDifference),
}

/// Where a recomputed document stops matching the retained one.
///
/// A closed set: either a shared line disagrees, or every shared line agrees
/// and one document has more of them. There is no "somewhere" state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FirstDifference {
    /// The documents disagree at this comparable line, counting from zero.
    Line {
        /// Position among comparable lines.
        index: usize,
    },
    /// Every comparable line they share matches; one document simply has more.
    Length,
}

impl fmt::Display for FirstDifference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Line { index } => write!(f, "they first differ at comparable line {index}"),
            Self::Length => f.write_str("every shared line matches and one document has more"),
        }
    }
}

/// Exactly how a recomputed document differs from the retained one.
///
/// Public because a difference is the one outcome the CLI reports as an error
/// ([`CliError::ReplayDiffers`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UtsegReproductionDifference {
    /// Comparable lines the replay produced.
    recomputed_lines: usize,
    /// Comparable lines the retained document has.
    retained_lines: usize,
    /// Where they stop agreeing.
    first_difference: FirstDifference,
}

impl UtsegReproductionDifference {
    /// Record one difference between two compared documents.
    pub(crate) fn new(
        recomputed_lines: usize,
        retained_lines: usize,
        first_difference: FirstDifference,
    ) -> Self {
        Self {
            recomputed_lines,
            retained_lines,
            first_difference,
        }
    }
}

impl fmt::Display for UtsegReproductionDifference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "reapplying the retained evidence did not reproduce the retained output: \
             {} comparable lines against {}, and {}",
            self.recomputed_lines, self.retained_lines, self.first_difference
        )
    }
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// How one retained evidence item fails to describe the request this build
/// collected at the same position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BindingMismatch {
    /// The transcript positions disagree.
    Ordinal {
        /// The position this build collected.
        collected: usize,
        /// The position the artifact records.
        retained: usize,
    },
    /// The dispatched words disagree.
    Words {
        /// The words this build collected.
        collected: Vec<String>,
        /// The words the artifact records.
        retained: Vec<String>,
    },
    /// The dispatched text disagrees.
    Text {
        /// The text this build collected.
        collected: String,
        /// The text the artifact records.
        retained: String,
    },
}

impl fmt::Display for BindingMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ordinal {
                collected,
                retained,
            } => write!(
                f,
                "this build collected transcript position {collected} where the evidence \
                 records {retained}"
            ),
            Self::Words {
                collected,
                retained,
            } => write!(
                f,
                "this build collected {} words where the evidence records {}",
                collected.len(),
                retained.len()
            ),
            Self::Text { .. } => f.write_str(
                "this build collected different text from the text the evidence records",
            ),
        }
    }
}

/// Why a replay cannot run at all.
///
/// Every variant names the artifact and what about it was refused, so an
/// operator learns which input to correct or regenerate. A refusal is not a
/// comparison result: nothing was compared.
#[derive(Debug, thiserror::Error)]
enum UtsegReplayRefusal {
    /// An artifact could not be read.
    #[error("cannot read {}: {source}", path.display())]
    Read {
        /// The artifact named on the command line.
        path: PathBuf,
        /// Why the read failed.
        #[source]
        source: std::io::Error,
    },
    /// A CHAT artifact is not UTF-8.
    #[error("{} is not UTF-8: {source}", path.display())]
    NotUtf8 {
        /// The artifact named on the command line.
        path: PathBuf,
        /// Where the encoding failed.
        #[source]
        source: std::str::Utf8Error,
    },
    /// A CHAT artifact does not parse.
    #[error("{} did not parse as CHAT: {details}", path.display())]
    Unparseable {
        /// The artifact named on the command line.
        path: PathBuf,
        /// The parse errors, joined.
        details: String,
    },
    /// The input document is not valid enough for segmentation, so the run
    /// being replayed could not have accepted it either.
    #[error("{} does not pass the utseg input gate: {details}", path.display())]
    InputRefused {
        /// The artifact named on the command line.
        path: PathBuf,
        /// The gate failures, joined.
        details: String,
    },
    /// ASR post-processing refused the retained response, so there is nothing
    /// to compare the retained transcript against.
    #[error("the retained ASR response could not be prepared: {details}")]
    Preparation {
        /// Why preparation refused.
        details: String,
    },
    /// The retained evidence could not be admitted.
    #[error(transparent)]
    Evidence(#[from] UtsegEvidenceAdmissionError),
    /// The retained ASR response could not be read as one.
    #[error("{} is not a retained ASR response: {source}", path.display())]
    AsrResponse {
        /// The artifact named on the command line.
        path: PathBuf,
        /// Why deserialization failed.
        #[source]
        source: serde_json::Error,
    },
    /// The ASR response and the evidence come from different runs.
    #[error(
        "the retained ASR response is {response} and the retained evidence is {evidence}; \
         these are artifacts of different runs"
    )]
    LanguageDisagreement {
        /// The language the response records.
        response: String,
        /// The language the evidence records.
        evidence: String,
    },
    /// This build collects a different number of requests than the run did.
    #[error(
        "this build collected {collected} segmentation requests and the retained evidence \
         records {retained}; the evidence does not belong to this input"
    )]
    PopulationMismatch {
        /// Requests this build collected.
        collected: usize,
        /// Items the artifact records.
        retained: usize,
    },
    /// One request does not match the evidence item at its position.
    #[error("retained evidence item {index} does not describe this input: {mismatch}")]
    Binding {
        /// Position in the artifact's item list.
        index: usize,
        /// What disagreed.
        mismatch: BindingMismatch,
    },
    /// The transcript could not be rebuilt from the replayed utterances.
    #[error("could not rebuild the transcript from the replayed segmentation: {0}")]
    Rebuild(String),
    /// A document this replay rebuilt does not parse back.
    #[error("the {stage} this replay rebuilt did not parse back as CHAT: {details}")]
    RebuiltUnparseable {
        /// What was rebuilt.
        stage: &'static str,
        /// The parse errors, joined.
        details: String,
    },
    /// The retained output declares languages this pass cannot reproduce.
    #[error(
        "the retained output declares @Languages {declared} and this replay builds with \
         {language} alone: a run that detected languages (--lang auto) cannot be reproduced \
         by this pass, which neither detects per-utterance languages nor writes code-switch \
         markup"
    )]
    DetectedLanguages {
        /// The codes the retained output declares.
        declared: String,
        /// The language the retained evidence names.
        language: String,
    },
    /// An utterance carries code-switch markup the replay never writes.
    #[error(
        "the retained output has an utterance carrying a [- {code}] code-switch precode: a run \
         that detected languages (--lang auto) cannot be reproduced by this pass, which tags no \
         utterance with a language"
    )]
    CodeSwitchPrecode {
        /// The code on the first such utterance.
        code: String,
    },
    /// The rebuild names a different media file than the retained output.
    #[error(
        "the retained output's media header names {retained} and this replay rebuilt {recomputed}: \
         pass the run's own --media-name, or omit it for a run that recorded none. The retained \
         evidence records only the phase, the language and the items, so the media name is \
         operator input and nothing but the retained output can check it"
    )]
    MediaNameDisagreement {
        /// What the rebuild wrote, from `--media-name`.
        recomputed: String,
        /// What the retained output carries.
        retained: String,
    },
    /// One side carries `%wor` tiers and the other does not.
    #[error(
        "the retained output {retained} %wor tiers and this replay rebuilt a transcript that \
         {recomputed}: pass --wor exactly as the run did. The retained evidence records no wor \
         flag, so it is operator input and nothing but the retained output can check it"
    )]
    WorDisagreement {
        /// How the rebuild came out, from `--wor`.
        recomputed: String,
        /// How the retained output came out.
        retained: String,
    },
    /// A comment that opens as one of our stamps does not parse, so what
    /// produced the document cannot be established.
    #[error(transparent)]
    Stamp(#[from] UnparseableStamp),
}

impl From<UtsegReplayRefusal> for CliError {
    /// Every refusal is an answer about the artifacts the operator named: they
    /// cannot be replayed together, and the remedy is different inputs.
    fn from(refusal: UtsegReplayRefusal) -> Self {
        Self::InvalidArgument(refusal.to_string())
    }
}

// ---------------------------------------------------------------------------
// Artifacts
// ---------------------------------------------------------------------------

/// What a replay requires of one CHAT artifact it reads.
///
/// Closed, and matched exhaustively where the artifact is parsed, so a new kind
/// of input cannot silently inherit another's gate.
#[derive(Debug, Clone, Copy)]
enum ChatAdmission {
    /// A document a run segmented. Gated exactly as the `utseg` command gates
    /// its own input, so the replay refuses what that command would refuse.
    SegmentationInput,
    /// A document a run wrote. It must parse, which is all a comparison of its
    /// lines needs; gating an output would refuse files the run itself
    /// produced.
    RetainedOutput,
}

/// One artifact a replay read, held with the path it came from.
struct ReplayArtifact {
    path: PathBuf,
    bytes: Vec<u8>,
}

impl ReplayArtifact {
    /// Read one artifact whole.
    fn read(path: &Path) -> Result<Self, UtsegReplayRefusal> {
        let bytes = std::fs::read(path).map_err(|source| UtsegReplayRefusal::Read {
            path: path.to_owned(),
            source,
        })?;
        Ok(Self {
            path: path.to_owned(),
            bytes,
        })
    }

    /// How this artifact is named in the report.
    fn identity(&self) -> InputIdentity {
        InputIdentity::of(&self.path, &self.bytes)
    }

    /// This artifact's CHAT text.
    fn text(&self) -> Result<&str, UtsegReplayRefusal> {
        std::str::from_utf8(&self.bytes).map_err(|source| UtsegReplayRefusal::NotUtf8 {
            path: self.path.clone(),
            source,
        })
    }

    /// Parse this artifact with the parser every other BA3 command uses.
    fn parse_chat(&self, admission: ChatAdmission) -> Result<ChatFile, UtsegReplayRefusal> {
        let text = self.text()?;
        let parser = crate::chat_parser();
        let (file, parse_errors) = parse_lenient(&parser, text);
        match admission {
            // Exactly what the utseg pipeline does with an input: parse
            // leniently, then hand the gate the document AND its parse errors
            // and let the gate decide. L0 is part of L1, so a parse error still
            // refuses, but it refuses through the same gate, with the same
            // message, as the run being replayed.
            ChatAdmission::SegmentationInput => {
                validate_to_level(&file, &parse_errors, ValidityLevel::StructurallyComplete)
                    .map_err(|errors| UtsegReplayRefusal::InputRefused {
                        path: self.path.clone(),
                        details: join_display(errors.iter()),
                    })?;
            }
            // An output is not gated: production never re-gates what it wrote,
            // and gating one here would refuse documents a run really produced.
            // It must parse, because comparing lines needs lines.
            ChatAdmission::RetainedOutput => {
                if !parse_errors.is_empty() {
                    return Err(UtsegReplayRefusal::Unparseable {
                        path: self.path.clone(),
                        details: join_display(parse_errors.iter()),
                    });
                }
            }
        }
        Ok(file)
    }
}

/// The one basis both passes compare on: the AST of the CHAT TEXT a run writes.
///
/// Production writes text, and every later stage and every reader sees that
/// text rather than the AST behind it, so serializing and parsing back is what
/// makes this a comparison of what the run actually wrote. Both passes do it,
/// so a serialization-only defect is visible in both rather than in whichever
/// one happened to reparse.
fn comparison_basis(file: &ChatFile, stage: &'static str) -> Result<ChatFile, UtsegReplayRefusal> {
    let serialized = to_chat_string(file);
    let parser = crate::chat_parser();
    let (reparsed, parse_errors) = parse_lenient(&parser, &serialized);
    if !parse_errors.is_empty() {
        return Err(UtsegReplayRefusal::RebuiltUnparseable {
            stage,
            details: join_display(parse_errors.iter()),
        });
    }
    Ok(reparsed)
}

/// Refuse a retained output whose languages this pass cannot reproduce.
///
/// A `--lang auto` run performs two detections the replay does not: a
/// file-level one that can put several codes in `@Languages`, and a
/// per-utterance one that writes a `[- code]` precode wherever an utterance
/// differs from the primary language. The replay builds with the one resolved
/// language the evidence names and tags no utterance, so such a transcript
/// differs for reasons that have nothing to do with segmentation. Saying so is
/// the honest answer; comparing would report a difference and blame the
/// boundaries for it.
fn refuse_detected_languages(
    retained: &ChatFile,
    language: &str,
) -> Result<(), UtsegReplayRefusal> {
    for line in retained.lines.as_slice() {
        if let Line::Header { header, .. } = line
            && let Header::Languages { codes } = header.as_ref()
        {
            let declared: Vec<&str> = codes.as_slice().iter().map(LanguageCode::as_str).collect();
            if declared.as_slice() != [language] {
                return Err(UtsegReplayRefusal::DetectedLanguages {
                    declared: declared.join(", "),
                    language: language.to_owned(),
                });
            }
        }
        if let Line::Utterance(utterance) = line
            && let Some(code) = &utterance.main.content.language_code
        {
            return Err(UtsegReplayRefusal::CodeSwitchPrecode {
                code: code.as_str().to_owned(),
            });
        }
    }
    Ok(())
}

/// Refuse a rebuild whose media header or `%wor` tiers disagree with the
/// retained output.
///
/// Both properties are OPERATOR INPUT: `--media-name` and `--wor` are what the
/// caller remembers about the run, and the pre-CHAT evidence records neither,
/// so until this check existed nothing compared them with anything. A wrong
/// `--wor` adds or removes a dependent tier under every timed utterance and a
/// wrong `--media-name` rewrites the media header; both are then compared as
/// ordinary lines, so the command reported that the retained evidence did not
/// reproduce and pointed at a line, implicitly blaming segmentation for an
/// argument. Refusing is the answer this pass already gives for a `--lang auto`
/// transcript it cannot rebuild: these inputs cannot be replayed together, and
/// the remedy is different arguments rather than a verdict about boundaries.
///
/// Compared on the REBUILT document rather than on the raw argument, because
/// that is where the argument's effect is visible: `--media-name recording.mp3`
/// and a retained `@Media:\trecording, audio` agree, and only the built header
/// can say so.
fn refuse_rebuild_property_disagreement(
    recomputed: &ChatFile,
    retained: &ChatFile,
) -> Result<(), UtsegReplayRefusal> {
    let (rebuilt_media, retained_media) = (media_name(recomputed), media_name(retained));
    if rebuilt_media != retained_media {
        return Err(UtsegReplayRefusal::MediaNameDisagreement {
            recomputed: describe_media(rebuilt_media),
            retained: describe_media(retained_media),
        });
    }

    let (rebuilt_wor, retained_wor) = (has_wor_tier(recomputed), has_wor_tier(retained));
    if rebuilt_wor != retained_wor {
        return Err(UtsegReplayRefusal::WorDisagreement {
            recomputed: describe_wor(rebuilt_wor),
            retained: describe_wor(retained_wor),
        });
    }
    Ok(())
}

/// The media file a document's `@Media` header names, if it has one.
fn media_name(file: &ChatFile) -> Option<&str> {
    file.lines.as_slice().iter().find_map(|line| match line {
        Line::Header { header, .. } => match header.as_ref() {
            Header::Media(media) => Some(media.filename.as_str()),
            _ => None,
        },
        _ => None,
    })
}

/// Whether any utterance in a document carries a `%wor` tier.
fn has_wor_tier(file: &ChatFile) -> bool {
    file.lines.as_slice().iter().any(|line| match line {
        Line::Utterance(utterance) => utterance
            .dependent_tiers
            .iter()
            .any(|tier| matches!(tier.tier, DependentTier::Wor(_))),
        _ => false,
    })
}

/// How a media name reads in a refusal: quoted, or named as absent.
fn describe_media(name: Option<&str>) -> String {
    name.map_or_else(
        || "no media header".to_owned(),
        |name| format!("\"{name}\""),
    )
}

/// How a `%wor` presence reads in a refusal.
fn describe_wor(present: bool) -> String {
    if present {
        "carries".to_owned()
    } else {
        "does not carry".to_owned()
    }
}

/// Render a list of failures into one line.
fn join_display(items: impl Iterator<Item = impl fmt::Display>) -> String {
    items
        .map(|item| item.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// One comment left out of a comparison because this build generates it.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SetAsideComment {
    /// A per-command stamp, whose timestamp changes on every run.
    Stamp {
        /// The command the stamp records.
        command: String,
    },
    /// The unchecked-ASR warning, which names the build that wrote it.
    UncheckedAsrWarning {
        /// The ASR engine the warning records.
        engine: String,
    },
}

impl From<GeneratedComment> for SetAsideComment {
    fn from(comment: GeneratedComment) -> Self {
        match comment {
            GeneratedComment::Stamp(entry) => Self::Stamp {
                command: entry.command,
            },
            GeneratedComment::UncheckedAsrWarning { engine } => {
                Self::UncheckedAsrWarning { engine }
            }
        }
    }
}

/// The generated comments each side of the comparison carried.
///
/// Both sides are recorded, not only the retained one: the replay writes no
/// stamp of its own, so a non-empty recomputed list would mean this command had
/// started generating provenance, and the report would say so.
#[derive(Debug, Default, Serialize)]
struct SetAsideComments {
    recomputed: Vec<SetAsideComment>,
    retained: Vec<SetAsideComment>,
}

/// The pass a report covers, with the facts only that pass has.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ReplayPass {
    /// Segmentation reapplied to an existing CHAT document.
    PostChat {
        /// The document the run segmented.
        input_chat: InputIdentity,
        /// Utterances before the retained boundaries were reapplied.
        utterances_before: usize,
        /// Utterances after.
        utterances_after: usize,
    },
    /// Segmentation reapplied to a retained ASR response, then rebuilt to CHAT.
    PreAsr {
        /// The response the run transcribed.
        asr_response: InputIdentity,
        /// Prepared chunks the current build derived from it.
        prepared_chunks: usize,
        /// Utterances the replayed boundaries produced.
        utterances: usize,
        /// Words the CHAT-legality gate refused and emitted anyway, as the
        /// production path does; a nonzero count explains an odd transcript
        /// without changing what was compared.
        language_invalid_words: usize,
    },
}

/// One replay, reported in full.
#[derive(Debug, Serialize)]
struct UtsegReplayReport {
    schema_version: u8,
    build: &'static str,
    pass: ReplayPass,
    evidence: InputIdentity,
    retained_output: InputIdentity,
    /// The language the retained evidence records.
    language: String,
    /// Requests bound one-to-one to retained evidence items.
    bound_items: usize,
    set_aside: SetAsideComments,
    outcome: UtsegReproduction,
}

impl UtsegReplayReport {
    /// First schema. A reader can tell one shape from the next without
    /// guessing from the fields present.
    const SCHEMA_VERSION: u8 = 1;
}

// ---------------------------------------------------------------------------
// Binding and comparison
// ---------------------------------------------------------------------------

/// The predictions of a bound replay: evidence proven to describe the requests
/// this build collected, in transcript order.
///
/// The field is private and [`bind`] is the only constructor, which CONSUMES
/// the admitted items. A caller therefore cannot reach a prediction except
/// through the proof that it belongs to this input. `bind` used to return unit
/// and both callers then took `item.prediction` off the items themselves, so
/// deleting the call, or reordering it after the reach, still compiled and
/// silently compared a population nothing had checked, in the one command
/// whose entire output is a reproduce-or-not verdict.
struct BoundPredictions(Vec<AdmittedUtsegPrediction>);

impl BoundPredictions {
    /// The bound predictions, in transcript order.
    fn as_slice(&self) -> &[AdmittedUtsegPrediction] {
        &self.0
    }

    /// How many requests were bound.
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Bind the requests this build collected to the retained evidence items.
///
/// One-to-one and in order: the same count, the same transcript positions, and
/// the same words and text. Evidence that passes this describes exactly the
/// work this build would dispatch, so reapplying it is a replay rather than a
/// new segmentation of a different population.
fn bind(
    collected: &[(usize, UtsegBatchItem)],
    retained: Vec<AdmittedUtsegEvidenceItem>,
) -> Result<BoundPredictions, UtsegReplayRefusal> {
    if collected.len() != retained.len() {
        return Err(UtsegReplayRefusal::PopulationMismatch {
            collected: collected.len(),
            retained: retained.len(),
        });
    }
    for (index, ((ordinal, request), item)) in collected.iter().zip(&retained).enumerate() {
        let mismatch = if *ordinal != item.item_ordinal {
            Some(BindingMismatch::Ordinal {
                collected: *ordinal,
                retained: item.item_ordinal,
            })
        } else if request.words != item.request.words {
            Some(BindingMismatch::Words {
                collected: request.words.clone(),
                retained: item.request.words.clone(),
            })
        } else if request.text != item.request.text {
            Some(BindingMismatch::Text {
                collected: request.text.clone(),
                retained: item.request.text.clone(),
            })
        } else {
            None
        };
        if let Some(mismatch) = mismatch {
            return Err(UtsegReplayRefusal::Binding { index, mismatch });
        }
    }
    Ok(BoundPredictions(
        retained.into_iter().map(|item| item.prediction).collect(),
    ))
}

/// The lines of `file` that take part in a comparison, with the comments this
/// build generates recorded into `set_aside` instead.
fn comparable_lines<'a>(
    file: &'a ChatFile,
    set_aside: &mut Vec<SetAsideComment>,
) -> Result<Vec<&'a Line>, UnparseableStamp> {
    let mut lines = Vec::new();
    for line in file.lines.as_slice() {
        if let Line::Header { header, .. } = line
            && let Header::Comment { content } = header.as_ref()
            && let Some(generated) = recognize_generated_comment(&content.to_chat_string())?
        {
            set_aside.push(SetAsideComment::from(generated));
            continue;
        }
        lines.push(line);
    }
    Ok(lines)
}

/// Compare a recomputed document with the retained one.
fn compare(
    recomputed: &ChatFile,
    retained: &ChatFile,
) -> Result<(UtsegReproduction, SetAsideComments), UnparseableStamp> {
    let mut set_aside = SetAsideComments::default();
    let recomputed_lines = comparable_lines(recomputed, &mut set_aside.recomputed)?;
    let retained_lines = comparable_lines(retained, &mut set_aside.retained)?;

    let differing = recomputed_lines
        .iter()
        .zip(&retained_lines)
        .position(|(actual, expected)| !actual.semantic_eq(expected));

    let outcome = match (differing, recomputed_lines.len() == retained_lines.len()) {
        (None, true) => UtsegReproduction::Reproduced,
        (first, _) => UtsegReproduction::Differing(UtsegReproductionDifference::new(
            recomputed_lines.len(),
            retained_lines.len(),
            match first {
                Some(index) => FirstDifference::Line { index },
                None => FirstDifference::Length,
            },
        )),
    };
    Ok((outcome, set_aside))
}

// ---------------------------------------------------------------------------
// The two passes
// ---------------------------------------------------------------------------

/// Reapply post-CHAT evidence to the document the run segmented.
fn replay_post_chat(
    args: &UtsegPostChatReplayArgs,
) -> Result<UtsegReplayReport, UtsegReplayRefusal> {
    let input = ReplayArtifact::read(&args.input_chat)?;
    let evidence = ReplayArtifact::read(&args.evidence)?;
    let retained = ReplayArtifact::read(&args.output_chat)?;

    let (language, items) =
        AdmittedUtsegEvidence::admit(&evidence.bytes, UtsegEvidencePhase::PostChat)?.into_parts();

    let mut chat = input.parse_chat(ChatAdmission::SegmentationInput)?;
    let retained_chat = retained.parse_chat(ChatAdmission::RetainedOutput)?;

    // The collector the utseg pipeline itself uses: the population is defined
    // by the current build, and binding then proves the evidence describes it.
    let collected = collect_utseg_batch_items(&chat);
    let bound = bind(&collected, items)?;

    let utterances_before = chat.utterances().count();
    let mut assignments: HashMap<usize, Vec<usize>> = HashMap::new();
    integrate_admitted_assignments(&mut assignments, &collected, bound.as_slice());
    apply_utseg_results(&mut chat, &assignments);
    let utterances_after = chat.utterances().count();

    let recomputed = comparison_basis(&chat, "reapplied segmentation")?;
    let (outcome, set_aside) = compare(&recomputed, &retained_chat)?;
    Ok(UtsegReplayReport {
        schema_version: UtsegReplayReport::SCHEMA_VERSION,
        build: crate::build_hash(),
        pass: ReplayPass::PostChat {
            input_chat: input.identity(),
            utterances_before,
            utterances_after,
        },
        evidence: evidence.identity(),
        retained_output: retained.identity(),
        language,
        bound_items: bound.len(),
        set_aside,
        outcome,
    })
}

/// Reapply pre-CHAT evidence to the retained ASR response, then rebuild CHAT.
///
/// This mirrors transcribe's own pre-CHAT path and calls its functions, so the
/// replay cannot drift from the code it is checking. Speakers come from the
/// retained response, as they do when no separate diarization artifact was
/// projected onto the chunks; a run whose speakers came from such an artifact
/// is `eval transcribe-replay`'s subject, not this one's.
fn replay_pre_asr(args: &UtsegPreAsrReplayArgs) -> Result<UtsegReplayReport, UtsegReplayRefusal> {
    let response_artifact = ReplayArtifact::read(&args.asr_response)?;
    let evidence = ReplayArtifact::read(&args.evidence)?;
    let retained = ReplayArtifact::read(&args.output_chat)?;

    let response: crate::transcribe::AsrResponse =
        serde_json::from_slice(&response_artifact.bytes).map_err(|source| {
            UtsegReplayRefusal::AsrResponse {
                path: args.asr_response.clone(),
                source,
            }
        })?;
    let (language, items) =
        AdmittedUtsegEvidence::admit(&evidence.bytes, UtsegEvidencePhase::PreChat)?.into_parts();

    // Two artifacts of one run, or the transcript would be built under a
    // language the boundaries were never produced for.
    if response.lang.as_ref() != language.as_str() {
        return Err(UtsegReplayRefusal::LanguageDisagreement {
            response: response.lang.to_string(),
            evidence: language,
        });
    }
    let lang = response.lang.clone();
    let retained_chat = retained.parse_chat(ChatAdmission::RetainedOutput)?;
    refuse_detected_languages(&retained_chat, lang.as_ref())?;

    let asr_output = crate::transcribe::convert_asr_response(&response);
    // Transcribe's own preparation, not a second implementation of it. Its one
    // refusal (Cantonese normalization changing a character count) is carried
    // through rather than unwrapped: a replay that cannot prepare its input has
    // not reproduced anything.
    let chunks = crate::pipeline::transcribe::prepare_asr_chunks(&asr_output, lang.as_ref())
        .map_err(|error| UtsegReplayRefusal::Preparation {
            details: error.to_string(),
        })?;
    let collected: Vec<(usize, UtsegBatchItem)> =
        crate::pipeline::transcribe::build_prechat_utseg_items(&chunks)
            .into_iter()
            .enumerate()
            .collect();
    let bound = bind(&collected, items)?;

    let split = crate::pipeline::transcribe::apply_prechat_assignments(&chunks, bound.as_slice());
    let mut utterances = asr_postprocess::utterances_from_prepared_chunks(split);
    asr_postprocess::finalize_utterances(&mut utterances, lang.as_ref());

    let transcript = build_chat::NamedAsrUtterances::numbered(&utterances)
        .into_transcript(&[lang.to_string()], args.media_name.as_deref(), args.wor)
        .map_err(|error| UtsegReplayRefusal::Rebuild(error.to_string()))?;
    let language_invalid_words = transcript.language_invalid.len();
    let built = build_chat::build_chat(&transcript.description)
        .map_err(|error| UtsegReplayRefusal::Rebuild(error.to_string()))?;

    let recomputed = comparison_basis(&built, "rebuilt transcript")?;
    // Before any line is compared: the two properties the evidence cannot
    // check are the operator's, and a disagreement in either is not a
    // segmentation difference.
    refuse_rebuild_property_disagreement(&recomputed, &retained_chat)?;
    let (outcome, set_aside) = compare(&recomputed, &retained_chat)?;
    Ok(UtsegReplayReport {
        schema_version: UtsegReplayReport::SCHEMA_VERSION,
        build: crate::build_hash(),
        pass: ReplayPass::PreAsr {
            asr_response: response_artifact.identity(),
            prepared_chunks: chunks.len(),
            utterances: utterances.len(),
            language_invalid_words,
        },
        evidence: evidence.identity(),
        retained_output: retained.identity(),
        language,
        bound_items: bound.len(),
        set_aside,
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One multi-word utterance, so the collector dispatches exactly one
    /// request and a split is observable.
    const INPUT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello there how are you .\n@End\n";

    /// The words the collector extracts from `INPUT`.
    fn words() -> Vec<&'static str> {
        vec!["hello", "there", "how", "are", "you"]
    }

    /// Evidence for `INPUT` recording `assignments`, in the shape a run writes.
    fn evidence_json(assignments: &[usize]) -> String {
        json!({
            "schema_version": 4,
            "phase": "post_chat",
            "language": "eng",
            "items": [{
                "item_ordinal": 0,
                "words": words(),
                "text": words().join(" "),
                "prediction": {
                    "source": "unobserved_assignments",
                    "assignments": assignments,
                }
            }]
        })
        .to_string()
    }

    /// The document a run applying `assignments` to `INPUT` would have written,
    /// produced by the transform the production path uses.
    fn retained_output(assignments: Vec<usize>) -> String {
        let parser = crate::chat_parser();
        let (mut chat, errors) = parse_lenient(&parser, INPUT);
        assert!(errors.is_empty(), "fixture must parse cleanly");
        let mut map = HashMap::new();
        map.insert(0, assignments);
        apply_utseg_results(&mut chat, &map);
        to_chat_string(&chat)
    }

    /// Write the three post-CHAT artifacts and replay them.
    fn replay(
        evidence: &str,
        retained: &str,
    ) -> (tempfile::TempDir, Result<UtsegReplayReport, UtsegReplayRefusal>) {
        let dir = tempfile::tempdir().expect("temporary directory");
        let input_chat = dir.path().join("input.cha");
        let evidence_path = dir.path().join("evidence.json");
        let output_chat = dir.path().join("output.cha");
        std::fs::write(&input_chat, INPUT).expect("write input");
        std::fs::write(&evidence_path, evidence).expect("write evidence");
        std::fs::write(&output_chat, retained).expect("write retained output");
        let report = replay_post_chat(&UtsegPostChatReplayArgs {
            input_chat,
            evidence: evidence_path,
            output_chat,
        });
        (dir, report)
    }

    #[test]
    fn reapplied_evidence_reproduces_the_document_the_run_wrote() {
        let split = vec![0, 0, 1, 1, 1];
        let retained = retained_output(split.clone());
        let (_dir, report) = replay(&evidence_json(&split), &retained);
        let report = report.expect("the retained artifacts must be replayable");

        assert!(
            matches!(report.outcome, UtsegReproduction::Reproduced),
            "reapplying the same evidence must reproduce the same document, got {:?}",
            report.outcome
        );
        assert_eq!(report.bound_items, 1);
        assert_eq!(report.language, "eng");
        let ReplayPass::PostChat {
            utterances_before,
            utterances_after,
            ..
        } = report.pass
        else {
            panic!("a post-CHAT replay reports the post-CHAT pass");
        };
        assert_eq!(utterances_before, 1);
        assert_eq!(
            utterances_after, 2,
            "the fixture must actually split, or the comparison proves nothing"
        );
    }

    #[test]
    fn different_boundaries_are_reported_as_a_difference_not_a_failure() {
        let retained = retained_output(vec![0, 0, 1, 1, 1]);
        // Evidence saying the utterance was never split, against an output
        // where it was.
        let (_dir, report) = replay(&evidence_json(&[0, 0, 0, 0, 0]), &retained);
        let report = report.expect("a difference is an outcome, never a refusal");

        let UtsegReproduction::Differing(difference) = report.outcome else {
            panic!("boundaries that disagree must be reported as a difference");
        };
        assert_ne!(difference.recomputed_lines, difference.retained_lines);
        assert!(
            difference.to_string().contains("did not reproduce"),
            "the difference must say what it means, got: {difference}"
        );
    }

    #[test]
    fn evidence_from_an_older_schema_is_refused_by_name() {
        let retained = retained_output(vec![0, 0, 1, 1, 1]);
        let legacy = evidence_json(&[0, 0, 1, 1, 1])
            .replace("\"schema_version\":4", "\"schema_version\":2");
        // The substitution is the whole fixture. If the writer's shape changes
        // and this stops matching, the "legacy" artifact would still be current
        // and readable, and the test would quietly prove nothing instead of
        // failing.
        assert!(
            legacy.contains("\"schema_version\":2"),
            "the legacy fixture must actually be downgraded, got: {legacy}"
        );
        let (_dir, report) = replay(&legacy, &retained);

        let refusal = report.expect_err("an unreadable schema must refuse, never compare");
        let rendered = refusal.to_string();
        assert!(
            rendered.contains("schema 2") && rendered.contains("schema 4"),
            "the refusal must name both schemas, got: {rendered}"
        );
    }

    #[test]
    fn evidence_whose_prediction_does_not_fit_its_request_is_refused_by_position() {
        let retained = retained_output(vec![0, 0, 1, 1, 1]);
        // Four assignments for five dispatched words: admission refuses this
        // exactly as it refuses a live worker that returns the wrong count.
        let (_dir, report) = replay(&evidence_json(&[0, 0, 1, 1]), &retained);

        let refusal = report.expect_err("a prediction that does not fit must refuse");
        let rendered = refusal.to_string();
        assert!(
            rendered.contains("item 0") && rendered.contains("4 assignments"),
            "the refusal must name the item and what was wrong, got: {rendered}"
        );
    }

    #[test]
    fn a_stamp_on_the_retained_output_is_set_aside_not_compared() {
        let split = vec![0, 0, 1, 1, 1];
        let stamped = retained_output(split.clone()).replace(
            "@ID:\teng|test|CHI|3;|male|||Target_Child|||\n",
            "@ID:\teng|test|CHI|3;|male|||Target_Child|||\n\
@Comment:\t[fc-ba3 utseg | engine=talkbank/boundary@r1 ; lang=eng | 2026-09-15T20:00:00-04:00]\n",
        );
        let (_dir, report) = replay(&evidence_json(&split), &stamped);
        let report = report.expect("a stamped output must still be replayable");

        assert!(
            matches!(report.outcome, UtsegReproduction::Reproduced),
            "a stamp the replay does not write must not count as a difference"
        );
        assert!(
            report.set_aside.recomputed.is_empty(),
            "the replay writes no provenance of its own"
        );
        let [SetAsideComment::Stamp { command }] = report.set_aside.retained.as_slice() else {
            panic!(
                "the retained stamp must be reported as set aside, got {:?}",
                report.set_aside.retained
            );
        };
        assert_eq!(command, "utseg");
    }

    /// The same transcript as `INPUT`, carrying the `%wor` tier a `--wor` run
    /// writes for timed words.
    const WITH_WOR: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello there .\n\
%wor:\thello \u{15}0_500\u{15} there \u{15}500_1000\u{15} .\n@End\n";

    /// The same transcript without it, as a run that passed no `--wor` writes.
    const WITHOUT_WOR: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello there .\n@End\n";

    /// Parse a fixture the way this module parses every artifact.
    fn parse_fixture(text: &str) -> ChatFile {
        let parser = crate::chat_parser();
        let (chat, errors) = parse_lenient(&parser, text);
        assert!(errors.is_empty(), "fixture must parse cleanly: {errors:?}");
        chat
    }

    /// The two properties the evidence cannot check are refused BY NAME rather
    /// than compared as ordinary lines.
    ///
    /// Both come from the command line, and a wrong one changes the rebuilt
    /// document in a way that has nothing to do with boundaries: the media
    /// header, or every timed utterance's dependent tiers. Compared, they made
    /// the command answer "the retained evidence did not reproduce" and point
    /// at a line, which blames segmentation for an argument.
    #[test]
    fn a_rebuild_property_the_evidence_cannot_check_is_refused_not_compared() {
        let with_media = parse_fixture(&INPUT.replace("*CHI:", "@Media:\tclip, audio\n*CHI:"));
        let without_media = parse_fixture(INPUT);

        let refusal = refuse_rebuild_property_disagreement(&with_media, &without_media)
            .expect_err("a media header on one side only must refuse");
        assert!(
            matches!(refusal, UtsegReplayRefusal::MediaNameDisagreement { .. }),
            "{refusal:?}"
        );
        let rendered = refusal.to_string();
        assert!(rendered.contains("clip"), "{rendered}");
        assert!(rendered.contains("--media-name"), "{rendered}");

        let refusal =
            refuse_rebuild_property_disagreement(&parse_fixture(WITH_WOR), &parse_fixture(WITHOUT_WOR))
                .expect_err("a %wor tier on one side only must refuse");
        assert!(
            matches!(refusal, UtsegReplayRefusal::WorDisagreement { .. }),
            "{refusal:?}"
        );
        assert!(refusal.to_string().contains("--wor"), "{refusal}");

        refuse_rebuild_property_disagreement(&parse_fixture(WITH_WOR), &parse_fixture(WITH_WOR))
            .expect("documents agreeing on both properties are compared, not refused");
    }
}
