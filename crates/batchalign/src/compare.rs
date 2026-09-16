//! Server-side compare orchestrator.
//!
//! Owns the full CHAT lifecycle for compare jobs:
//! 1. Parse main + gold files
//! 2. Run morphosyntax on main (via existing pipeline)
//! 3. DP-align main vs gold words
//! 4. Project compare annotations onto the gold/reference transcript
//! 5. Serialize projected CHAT + CSV metrics
//!
//! Gold file convention: for each `FILE.cha`, expects `FILE.gold.cha` in the
//! same directory. Files ending in `.gold.cha` are skipped.

use std::path::Path;

use crate::api::LanguageCode3;
use crate::chat_ops::morphosyntax_ops::MwtDict;
use crate::pipeline::PipelineServices;
use tracing::warn;

use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, TokenizationMode};
use crate::chat_ops::{DependentTier, Header, Line};
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::post_validate::{PostValidated, UtteranceCensus};
use crate::text_batch::TextBatchFileInput;
use batchalign_transform::compare::{
    clear_comparison, format_metrics_csv, inject_comparison, project_gold_structurally,
};
use batchalign_transform::parse::parse_lenient;

/// Released compare outputs.
pub(crate) struct CompareMaterializedOutputs {
    /// The CHAT document written by the released compare command, as a PROOF.
    ///
    /// A `String` until 2026-09-07, which is what let `execution::kernel` run
    /// `merge_abbreviations_in_chat_text` over it and write the RESULT: the
    /// bytes on disk were then a transform past anything judged, and this
    /// materializer had the `ChatFile` in hand the whole time. See
    /// [`gate_comparison_output`] for what it is judged against and why.
    pub chat_output: PostValidated,
    /// CSV sidecar containing aggregate and per-POS compare metrics.
    pub metrics_csv: String,
}

/// Internal main-annotated compare output used by benchmark-style flows.
pub(crate) struct MainAnnotatedCompareOutputs {
    /// The main transcript annotated with `%xsrep` and `%xsmor`, as a PROOF.
    ///
    /// A `String` until 2026-09-07, for the same reason and with the same
    /// consequence as [`CompareMaterializedOutputs::chat_output`]; the writer
    /// there was `runner::dispatch::benchmark_pipeline`.
    pub annotated_main_chat: PostValidated,
    /// CSV sidecar containing aggregate and per-POS compare metrics.
    pub metrics_csv: String,
}

/// Judge a comparison artifact by PRESERVATION, against the document it
/// descends from.
///
/// # Why not a validity level
///
/// Because neither of these two commands admits its input at one:
///
/// - compare's released output is the GOLD companion, structurally projected.
///   That companion is read from disk and [`parse_lenient`]'d, its parse errors
///   are `warn!`ed and not refused, and nothing calls `validate_to_level` on it
///   anywhere on this path.
/// - benchmark's main-annotated output descends from ASR text that
///   `transcribe::process_transcribe` returns as a bare `String`, and it too is
///   re-parsed leniently here before the comparison runs.
///
/// A level gate therefore judges these outputs against a bar their inputs were
/// never held to, and picking a weak level does not fix it. This gated at
/// [`batchalign_transform::validate::ValidityLevel::Parseable`] until
/// 2026-09-07 on exactly that reasoning, and the reasoning was wrong twice
/// over: L0 inspects only the parse errors it is passed, and the caller always
/// passed none, so the level was vacuous; while `validate_output`'s terminator
/// loop runs for every command regardless of the level, and asks whether the
/// document HAS terminators rather than whether this command kept them. A gold
/// companion with one terminator-less non-CA utterance had always been written
/// and became a hard `ServerError::Validation` with no output at all, which is
/// the very failure that docstring claimed to have avoided.
///
/// # What it judges instead
///
/// The honest property for a command whose input nothing vouched for is that
/// the output did not LOSE what the input had, so the input's own faults are
/// carried through and this command's damage is not. `input` is the census of
/// the document the artifact DESCENDS from, taken before this module edited
/// it: the gold companion for compare, the morphotagged main transcript for
/// benchmark. Both materializers below only add headers and rewrite dependent
/// tiers, so nothing on this path can legitimately drop a main-tier
/// terminator, and a build that starts doing so is refused.
///
/// The judgement travels with the proof, so the abbreviation merge the writers
/// run afterwards is judged the same way rather than against a fresh bar.
fn gate_comparison_output(
    input: UtteranceCensus,
    output: crate::chat_ops::ChatFile,
    command: crate::api::ReleasedCommand,
) -> Result<PostValidated, ServerError> {
    // BY VALUE: both call sites drop the model immediately afterwards, and a
    // borrowing form would clone the whole document straight back.
    PostValidated::preserving(input, output, command)
        .map_err(|failure| ServerError::Validation(failure.to_string()))
}

/// The comparison's states, and the only transitions between them.
///
/// The types live in a module of their own so their fields are private to it.
/// That is the whole mechanism: [`MorphotaggedMain`] can only be made from a
/// morphotag PROOF and [`ComparisonArtifacts`] can only be made from a
/// [`MorphotaggedMain`], so there is no route into a comparison that begins
/// with a `String`. The lenient re-parse this module replaced was reachable
/// precisely because such a route existed, and deleting the parse without
/// closing the route would leave it free to grow back.
mod artifacts {
    use batchalign_transform::compare::{ComparisonBundle, GoldCoverage, compare};
    use tracing::info;

    use crate::chat_ops::ChatFile;
    use crate::error::ServerError;
    use crate::pipeline::post_validate::PostValidated;

    /// The morphotagged main transcript, as a document only morphotag's own
    /// proof can produce.
    pub(super) struct MorphotaggedMain {
        /// Private to this module, which is the point: [`Self::from_proof`] is
        /// the only thing that can fill it, so no caller anywhere, this file's
        /// own tests included, can mint a main side out of text.
        file: ChatFile,
    }

    impl MorphotaggedMain {
        /// THE transition into the comparison: consume morphotag's proof and
        /// go on in the document it judged.
        ///
        /// Fails only for a proof that carries no document at all, which is a
        /// pass-through; morphotag's two routes both carry one, so the compare
        /// pipeline does not meet that error. It is returned rather than
        /// papered over with a parse because parsing here would be the very
        /// re-parse this graph exists to remove.
        pub(super) fn from_proof(proof: PostValidated) -> Result<Self, ServerError> {
            proof
                .into_judged_document()
                .map(|file| Self { file })
                .map_err(|refused| ServerError::Validation(refused.to_string()))
        }
    }

    /// One comparison: the two documents, and the bundle comparing THEM.
    pub(super) struct ComparisonArtifacts {
        main_file: ChatFile,
        gold_file: ChatFile,
        bundle: ComparisonBundle,
    }

    /// A finished comparison, opened up for the materializers that consume it.
    ///
    /// Destructuring only. Nothing turns these parts back into a
    /// [`ComparisonArtifacts`], so the single-constructor rule above still
    /// holds: this is how a materializer takes ownership of the documents it
    /// edits, not a second way to build a comparison.
    pub(super) struct ComparisonParts {
        pub(super) main_file: ChatFile,
        pub(super) gold_file: ChatFile,
        pub(super) bundle: ComparisonBundle,
    }

    impl ComparisonArtifacts {
        /// Compare a proof-carried main side against a gold companion.
        ///
        /// The only constructor, and it RUNS the comparison rather than
        /// accepting one, so the bundle a materializer reads is always the
        /// comparison of the two documents beside it. Passing a bundle built
        /// from different documents is not a mistake a caller can make here,
        /// because a caller does not supply the bundle at all.
        pub(super) fn build(main: MorphotaggedMain, gold_file: ChatFile) -> Self {
            // A `FILE.gold.cha` companion is a re-transcription of the same
            // recording, so main material it does not account for is genuinely
            // unmatched output.
            let bundle = compare(&main.file, &gold_file, GoldCoverage::Complete);

            info!(
                matches = bundle.metrics.matches,
                insertions = bundle.metrics.insertions,
                deletions = bundle.metrics.deletions,
                wer = %format!("{:.4}", bundle.metrics.wer),
                cwer = %format!("{:.4}", bundle.metrics.cwer),
                "Compare alignment complete"
            );

            Self {
                main_file: main.file,
                gold_file,
                bundle,
            }
        }

        /// Hand the two documents and their comparison to a materializer.
        pub(super) fn into_parts(self) -> ComparisonParts {
            ComparisonParts {
                main_file: self.main_file,
                gold_file: self.gold_file,
                bundle: self.bundle,
            }
        }
    }
}

use artifacts::{ComparisonArtifacts, ComparisonParts, MorphotaggedMain};

/// Build the comparison from morphotag's proof and the gold companion's text.
///
/// The main side arrives as a PROOF and is never parsed here. The gold side is
/// the one input nothing vouched for: it is read off disk as it is, parsed
/// leniently, and its parse errors are reported rather than refused. That
/// asymmetry is deliberate and is the same one [`gate_comparison_output`]
/// documents: compare's output is judged against what the gold companion HAD,
/// so refusing the companion for its own faults would refuse a document this
/// command never damaged.
fn build_comparison_artifacts_from_proof(
    morphotagged_main: PostValidated,
    gold_text: &str,
) -> Result<ComparisonArtifacts, ServerError> {
    let main = MorphotaggedMain::from_proof(morphotagged_main)?;

    let (gold_file, gold_errors) = parse_lenient(&crate::chat_parser(), gold_text);
    if !gold_errors.is_empty() {
        warn!(
            num_errors = gold_errors.len(),
            "Parse errors in gold file (continuing)"
        );
    }

    Ok(ComparisonArtifacts::build(main, gold_file))
}

async fn build_comparison_artifacts(
    main_text: &str,
    gold_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    mwt: &MwtDict,
) -> Result<ComparisonArtifacts, ServerError> {
    let mor_params = MorphosyntaxParams {
        lang,
        tokenization_mode: TokenizationMode::Preserve,
        multilingual_policy: MultilingualPolicy::ProcessAll,
        mwt,
        policy: crate::params::MorphotagExecutionPolicy {
            l2: crate::params::L2MorphotagPolicy::Placeholder,
            pos_hints: crate::params::PosHintPolicy::Ignore,
            ca_policy: crate::options::CaMorphotagPolicy::Honor,
        },
        // Compare's internal morphotag never surfaces review tiers.
        review_level: crate::chat_ops::fa::ReviewLevel::None,
        // No job-level reporter on this path: see the field doc.
        progress: None,
        // compare-runs is an offline analysis command with no job and no
        // job-level cancellation token; genuinely NotWired, not a stand-in.
        cancellation: crate::infer_retry::Cancellation::NotWired {
            reason: "compare-runs has no job-level cancellation",
        },
    };
    // The INTERNAL morphotag's proof is CARRIED, not discharged: its bytes are
    // never written and its DOCUMENT is the comparison's main side. It used to
    // become a `String` here and be parsed again below, which is how a document
    // the gate had judged came to be re-read by a parser that tolerates what
    // the gate refuses. What the command writes is the comparison artifact, and
    // that gets its own proof in the materializers below.
    let morphotagged_main =
        crate::morphosyntax::process_morphosyntax(main_text, services, &mor_params).await?;
    build_comparison_artifacts_from_proof(morphotagged_main, gold_text)
}

fn materialize_main_annotated(
    artifacts: ComparisonArtifacts,
) -> Result<MainAnnotatedCompareOutputs, ServerError> {
    let ComparisonParts {
        mut main_file,
        bundle,
        ..
    } = artifacts.into_parts();
    // Taken BEFORE the edits below, because this output descends from the main
    // transcript and preservation is a claim about that descent.
    let input = UtteranceCensus::of(&main_file);
    clear_comparison(&mut main_file);
    inject_comparison(&mut main_file, &bundle.main_utterances).map_err(|err| {
        ServerError::Persistence(format!("compare tier serialization failed: {err}"))
    })?;
    Ok(MainAnnotatedCompareOutputs {
        annotated_main_chat: gate_comparison_output(
            input,
            main_file,
            crate::api::ReleasedCommand::Benchmark,
        )?,
        metrics_csv: format_metrics_csv(&bundle.metrics).map_err(|err| {
            ServerError::Persistence(format!("compare CSV serialization failed: {err}"))
        })?,
    })
}

fn materialize_released(
    artifacts: ComparisonArtifacts,
) -> Result<CompareMaterializedOutputs, ServerError> {
    let ComparisonParts {
        main_file,
        gold_file,
        bundle,
    } = artifacts.into_parts();
    // Taken BEFORE the projection, because the released output descends from
    // the gold companion, faults included.
    let input = UtteranceCensus::of(&gold_file);
    let mut gold_file = project_gold_structurally(&main_file, &gold_file, &bundle);
    apply_media_header_from_main(&main_file, &mut gold_file);
    clear_comparison(&mut gold_file);
    inject_comparison(&mut gold_file, &bundle.gold_utterances).map_err(|err| {
        ServerError::Persistence(format!("compare tier serialization failed: {err}"))
    })?;
    strip_mor_gra_tiers(&mut gold_file);
    Ok(CompareMaterializedOutputs {
        chat_output: gate_comparison_output(
            input,
            gold_file,
            crate::api::ReleasedCommand::Compare,
        )?,
        metrics_csv: format_metrics_csv(&bundle.metrics).map_err(|err| {
            ServerError::Persistence(format!("compare CSV serialization failed: {err}"))
        })?,
    })
}

fn strip_mor_gra_tiers(chat_file: &mut crate::chat_ops::ChatFile) {
    for line in &mut chat_file.lines {
        if let Line::Utterance(utterance) = line {
            utterance
                .dependent_tiers
                .retain(|tier| !matches!(tier.tier, DependentTier::Mor(_) | DependentTier::Gra(_)));
        }
    }
}

fn apply_media_header_from_main(
    main_file: &crate::chat_ops::ChatFile,
    gold_file: &mut crate::chat_ops::ChatFile,
) {
    let Some(media) = main_file.media.clone() else {
        return;
    };

    gold_file.media = Some(media.clone());
    for line in &mut gold_file.lines {
        if let Line::Header { header, .. } = line
            && matches!(header.as_ref(), Header::Media(_))
        {
            **header = Header::Media((*media).clone());
            return;
        }
    }

    let insert_at = gold_file
        .lines
        .iter()
        .position(|line| matches!(line, Line::Utterance(_)))
        .unwrap_or(gold_file.lines.len());
    gold_file
        .lines
        .insert(insert_at, Line::header(Header::Media((*media).clone())));
}

/// Process a single CHAT file through the compare pipeline.
///
/// Returns the released compare outputs for the current projected-reference
/// workflow materialization.
///
/// Steps:
/// 1. Run morphosyntax on `main_text` (so it has %mor/%gra).
/// 2. Parse gold file.
/// 3. Build the comparison bundle from main vs gold.
/// 4. Materialize the projected reference-side output.
pub(crate) async fn process_compare(
    main_text: &str,
    gold_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    mwt: &MwtDict,
) -> Result<CompareMaterializedOutputs, ServerError> {
    materialize_released(
        build_comparison_artifacts(main_text, gold_text, lang, services, mwt).await?,
    )
}

/// Materialize compare outputs from morphotag's own PROOF of the main
/// transcript.
///
/// It takes the proof rather than the bytes, and that is what makes the lenient
/// re-parse unreachable rather than merely deleted: the kernel has no `String`
/// to offer here, and no constructor anywhere below would accept one.
pub(crate) fn process_compare_morphotagged_main(
    morphotagged_main: PostValidated,
    gold_text: &str,
) -> Result<CompareMaterializedOutputs, ServerError> {
    materialize_released(build_comparison_artifacts_from_proof(
        morphotagged_main,
        gold_text,
    )?)
}

/// Process one compare flow and keep the main transcript as the structural anchor.
pub(crate) async fn process_compare_main_annotated(
    main_text: &str,
    gold_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    mwt: &MwtDict,
) -> Result<MainAnnotatedCompareOutputs, ServerError> {
    materialize_main_annotated(
        build_comparison_artifacts(main_text, gold_text, lang, services, mwt).await?,
    )
}

/// Derive the gold file path from a main file path.
///
/// Convention: `FILE.cha` -> `FILE.gold.cha` (in the same directory).
pub fn gold_path_for(main_path: &str) -> String {
    let p = Path::new(main_path);
    let stem = p.file_stem().unwrap_or_default().to_string_lossy();
    let parent = p.parent().unwrap_or_else(|| Path::new(""));
    parent
        .join(format!("{stem}.gold.cha"))
        .to_string_lossy()
        .to_string()
}

/// Derive the directory-level template gold path for a main file path.
///
/// Convention: `DIR/FILE.cha` -> `DIR/template.gold.cha`.
pub fn template_gold_path_for(main_path: &str) -> String {
    let p = Path::new(main_path);
    let parent = p.parent().unwrap_or_else(|| Path::new(""));
    parent
        .join("template.gold.cha")
        .to_string_lossy()
        .to_string()
}

/// Returns `true` if the filename is a gold reference file (ends with `.gold.cha`).
pub fn is_gold_file(filename: &str) -> bool {
    filename.ends_with(".gold.cha")
}

/// Process multiple CHAT files through the compare pipeline.
///
/// For each `(filename, chat_text)`:
/// 1. Skip `.gold.cha` files
/// 2. Look up the companion gold file
/// 3. Run morphosyntax + compare
/// 4. Return `(filename, Ok(outputs) | Err(error_msg))`
#[allow(dead_code)]
pub(crate) async fn process_compare_batch(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    mwt: &MwtDict,
    read_gold_fn: &dyn Fn(&str) -> Option<String>,
) -> Vec<(String, Result<CompareMaterializedOutputs, String>)> {
    let mut results = Vec::with_capacity(files.len());

    for file in files {
        let filename = file.filename.as_ref();
        let chat_text = file.chat_text.as_ref();
        // Skip gold files: they're companions, not inputs
        if is_gold_file(filename) {
            continue;
        }

        let gold_filename = gold_path_for(filename);
        let template_gold_filename = template_gold_path_for(filename);
        let gold_text =
            match read_gold_fn(&gold_filename).or_else(|| read_gold_fn(&template_gold_filename)) {
                Some(text) => text,
                None => {
                    results.push((
                        file.filename.to_string(),
                        Err(format!(
                            "No gold .cha file found for comparison. \
                         main: {filename}, expected: {gold_filename} or {template_gold_filename}"
                        )),
                    ));
                    continue;
                }
            };

        match process_compare(chat_text, &gold_text, lang, services, mwt).await {
            Ok(result) => {
                results.push((file.filename.to_string(), Ok(result)));
            }
            Err(e) => {
                results.push((file.filename.to_string(), Err(e.to_string())));
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_transform::parse::TreeSitterParser;
    use batchalign_transform::parse::parse_lenient;

    /// Build a comparison the way production does: out of a morphotag PROOF.
    ///
    /// There is no other route, here or anywhere else. `ComparisonArtifacts`
    /// has private fields and a single constructor, which consumes a
    /// `MorphotaggedMain`, which only a proof can produce. These tests cannot
    /// assemble a comparison out of two strings any more than the kernel can,
    /// which is what keeps the deleted lenient parse from growing back: a
    /// future caller reaching for text has nothing to call.
    fn comparison_of(main: &str, gold: &str) -> ComparisonArtifacts {
        let (gold_file, _) = parse_lenient(&TreeSitterParser::new().expect("parser"), gold);
        let main = MorphotaggedMain::from_proof(PostValidated::for_test(
            main,
            crate::api::ReleasedCommand::Morphotag,
        ))
        .expect("a gated proof carries the document it judged");
        ComparisonArtifacts::build(main, gold_file)
    }

    fn make_chat(utterances: &[(&str, &str)]) -> String {
        let mut lines = vec![
            "@UTF8".to_string(),
            "@Begin".to_string(),
            "@Languages:\teng".to_string(),
            "@Participants:\tPAR Participant".to_string(),
            "@ID:\teng|test|PAR|||||Participant|||".to_string(),
        ];
        for (speaker, text) in utterances {
            lines.push(format!("*{speaker}:\t{text}"));
        }
        lines.push("@End".to_string());
        lines.join("\n")
    }

    #[test]
    fn gold_path_derivation() {
        assert_eq!(gold_path_for("test.cha"), "test.gold.cha");
        assert_eq!(
            gold_path_for("/data/corpus/01DM.cha"),
            "/data/corpus/01DM.gold.cha"
        );
        assert_eq!(gold_path_for("dir/sub/file.cha"), "dir/sub/file.gold.cha");
    }

    #[test]
    fn template_gold_path_derivation() {
        assert_eq!(template_gold_path_for("test.cha"), "template.gold.cha");
        assert_eq!(
            template_gold_path_for("/data/corpus/01DM.cha"),
            "/data/corpus/template.gold.cha"
        );
        assert_eq!(
            template_gold_path_for("dir/sub/file.cha"),
            "dir/sub/template.gold.cha"
        );
    }

    #[test]
    fn gold_file_detection() {
        assert!(is_gold_file("test.gold.cha"));
        assert!(is_gold_file("/data/01DM.gold.cha"));
        assert!(!is_gold_file("test.cha"));
        assert!(!is_gold_file("test.gold.txt"));
    }

    #[test]
    fn released_compare_surface_should_match_ba2_projected_gold_chat() {
        let main = make_chat(&[("PAR", "hello big world .")]);
        let gold = make_chat(&[("PAR", "hello world today .")]);

        let output = materialize_released(comparison_of(&main, &gold)).expect("materialized");

        assert_eq!(
            output.chat_output.command(),
            crate::api::ReleasedCommand::Compare,
            "the writer reads the command off the proof, so the materializer names it"
        );
        assert!(
            output
                .chat_output
                .as_str()
                .contains("*PAR:\thello world today .")
        );
        assert!(
            output
                .chat_output
                .as_str()
                .contains("%xsrep:\thello +big world -today")
        );
    }

    #[test]
    fn main_materializer_keeps_main_anchor() {
        let main = make_chat(&[("PAR", "hello big world .")]);
        let gold = make_chat(&[("PAR", "hello world today .")]);

        let output =
            materialize_main_annotated(comparison_of(&main, &gold)).expect("materialized");

        assert_eq!(
            output.annotated_main_chat.command(),
            crate::api::ReleasedCommand::Benchmark,
            "the main-annotated materializer is benchmark's output, not compare's"
        );
        assert!(
            output
                .annotated_main_chat
                .as_str()
                .contains("*PAR:\thello big world .")
        );
        assert!(
            output
                .annotated_main_chat
                .as_str()
                .contains("%xsrep:\thello +big world -today")
        );
    }

    #[test]
    fn gold_materializer_projects_structural_tiers_for_exact_match() {
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello world .\n%mor:\tintj|hello noun|world .\n%gra:\t1|2|COM 2|0|ROOT 3|2|PUNCT\n%wor:\thello \u{15}0_100\u{15} world \u{15}100_200\u{15} .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello world .\n@End\n";

        let output = materialize_released(comparison_of(main, gold)).expect("materialized");

        assert!(!output.chat_output.as_str().contains("%mor:"));
        assert!(!output.chat_output.as_str().contains("%gra:"));
        assert!(output.chat_output.as_str().contains("%wor:\thello"));
        assert!(output.chat_output.as_str().contains("\u{15}0_100\u{15}"));
    }

    /// RED FIRST (2026-09-07): the comparison gate refuses a LOSS, and the
    /// same output is admitted when the input was already like that.
    ///
    /// Both assertions are one test on purpose, because either alone is
    /// satisfiable by a gate that does the wrong thing. The predecessor of
    /// this test stripped the terminators from the INPUTS and asserted a
    /// refusal, which a gate reading only the output passes without ever
    /// comparing anything: its verdict was identical under "the command
    /// degraded the file" and "the file arrived that way", so it was not
    /// evidence of a preservation check. Here the two calls differ ONLY in
    /// what the input had, so a gate that ignores the input fails one of them
    /// whichever way it leans.
    #[test]
    fn the_comparison_gate_refuses_a_lost_terminator_and_admits_an_absent_one() {
        let parser = TreeSitterParser::new().expect("parser");
        let intact = make_chat(&[("PAR", "hello world today .")]);
        let (intact_file, _) = parse_lenient(&parser, &intact);
        let mut stripped_file = intact_file.clone();
        for line in &mut stripped_file.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.terminator = None;
            }
        }

        let refusal = gate_comparison_output(
            UtteranceCensus::of(&intact_file),
            stripped_file.clone(),
            crate::api::ReleasedCommand::Compare,
        )
        .expect_err("an output that lost a terminator its input had must be refused");
        assert!(
            refusal.to_string().contains("lost its terminator"),
            "the refusal must name what broke, got: {refusal}"
        );

        let admitted = gate_comparison_output(
            UtteranceCensus::of(&stripped_file),
            stripped_file,
            crate::api::ReleasedCommand::Compare,
        )
        .expect("the very same bytes must be admitted when the input had no terminator either");
        assert!(admitted.as_str().contains("*PAR:\thello world today"));
    }

    /// RED FIRST (2026-09-07): a gold companion that ARRIVES without a
    /// terminator is still written. Compare cannot have caused an absence its
    /// input already had, and refusing it is refusing a document the command
    /// never damaged.
    #[test]
    fn a_gold_companion_that_arrives_without_a_terminator_is_still_written() {
        let main = make_chat(&[("PAR", "hello big world .")]);
        // Read from disk and parsed leniently: nothing admits a gold companion
        // at any level, so a terminator-less utterance in one is an INPUT
        // property, not a loss.
        let gold = make_chat(&[("PAR", "hello world today")]);

        let output = materialize_released(comparison_of(&main, &gold))
            .expect("a gold companion compare never damaged must still be written");
        assert!(
            output
                .chat_output
                .as_str()
                .contains("*PAR:\thello world today"),
            "got: {}",
            output.chat_output.as_str()
        );
    }

    #[test]
    fn released_compare_output_copies_media_header_from_main() {
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\tsample, audio\n*PAR:\thello world .\n%mor:\tintj|hello noun|world .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello world .\n@End\n";

        let output = materialize_released(comparison_of(main, gold)).expect("materialized");

        assert!(
            output
                .chat_output
                .as_str()
                .contains("@Media:\tsample, audio")
        );
        assert!(!output.chat_output.as_str().contains("%mor:"));
        assert!(!output.chat_output.as_str().contains("%gra:"));
    }
}
