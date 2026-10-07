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

use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, TokenizationMode};
use crate::chat_ops::{DependentTier, Header, Line};
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::post_validate::{PostValidated, UtteranceCensus};
use crate::text_batch::TextBatchFileInput;
use batchalign_transform::compare::{
    CompareMetricsCsvTable, clear_comparison, inject_comparison, project_gold_structurally,
};

/// A completely admitted reference transcript. Only source admission produces
/// this capability; reference content has no regeneration exemption.
#[derive(Debug, Clone)]
pub(crate) struct AdmittedComparisonReference {
    file: Box<talkbank_model::validation::ValidChatFile>,
}

impl AdmittedComparisonReference {
    /// Admit the reference once, before transcription or morphology inference.
    pub(crate) fn admit(text: &str) -> Result<Self, ServerError> {
        let source = crate::pipeline::text_infer::admit_retained_text(
            &crate::chat_parser(),
            text,
            talkbank_model::model::TranscriptName::Anonymous,
        )?;
        Ok(Self {
            file: Box::new(source.into_valid_file()),
        })
    }

    fn into_document(self) -> crate::chat_ops::ChatFile {
        self.file.into_unchecked()
    }
}

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
    /// Producer-built metrics, retained structurally until each CSV boundary.
    pub metrics: CompareMetricsCsvTable,
}

/// Internal main-annotated compare output used by benchmark-style flows.
pub(crate) struct MainAnnotatedCompareOutputs {
    /// The main transcript annotated with `%xsrep` and `%xsmor`, as a PROOF.
    ///
    /// A `String` until 2026-09-07, for the same reason and with the same
    /// consequence as [`CompareMaterializedOutputs::chat_output`]; the writer
    /// there was `runner::dispatch::benchmark_pipeline`.
    pub annotated_main_chat: PostValidated,
    /// Producer-built metrics, retained structurally until CSV encoding.
    pub metrics: CompareMetricsCsvTable,
}

/// Establish complete checked construction AND preservation against the
/// admitted document this artifact descends from. Neither property substitutes
/// for the other: a valid projection can still lose source structure.
///
/// `input` is taken before edits: the completely admitted reference for released
/// compare, or the morphotagged main document for benchmark. Both documents reach
/// the materializer through producer-owned admission proofs, without reparsing.
/// The output proof carries the judgment through any later abbreviation merge.
fn gate_comparison_output(
    input: UtteranceCensus,
    output: crate::chat_ops::ChatFile,
    command: crate::api::ReleasedCommand,
) -> Result<PostValidated, ServerError> {
    // BY VALUE: both call sites drop the model immediately afterwards, and a
    // borrowing form would clone the whole document straight back.
    PostValidated::preserving(input, output, command).map_err(|failure| failure.into_server_error())
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
        /// Every admitted output owns its model, including unchanged source;
        /// no serialized-output reparse is needed or permitted.
        pub(super) fn from_proof(proof: PostValidated) -> Self {
            Self {
                file: proof.into_judged_document(),
            }
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
        pub(super) fn build(
            main: MorphotaggedMain,
            reference: super::AdmittedComparisonReference,
        ) -> Self {
            let gold_file = reference.into_document();
            // A `FILE.gold.cha` companion is a re-transcription of the same
            // recording, so main material it does not account for is genuinely
            // unmatched output.
            let bundle = compare(&main.file, &gold_file, GoldCoverage::Complete);

            info!(
                matches = bundle.metrics.matches(),
                insertions = bundle.metrics.insertions(),
                deletions = bundle.metrics.deletions(),
                wer = %format!("{:.4}", bundle.metrics.wer()),
                cwer = %format!("{:.4}", bundle.metrics.cwer()),
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

/// Consume the two admitted documents without parsing either serialization.
fn build_comparison_artifacts_from_proof(
    morphotagged_main: PostValidated,
    reference: AdmittedComparisonReference,
) -> ComparisonArtifacts {
    let main = MorphotaggedMain::from_proof(morphotagged_main);
    ComparisonArtifacts::build(main, reference)
}

enum ComparisonMain<'a> {
    Source(&'a str),
    /// A constructed main transcript, admitted by its type.
    Constructed(crate::pipeline::post_validate::PostValidated),
}

async fn build_comparison_artifacts(
    main: ComparisonMain<'_>,
    reference: AdmittedComparisonReference,
    _lang: &LanguageCode3,
    services: PipelineServices<'_>,
    mwt: &MwtDict,
) -> Result<ComparisonArtifacts, ServerError> {
    let mor_params = MorphosyntaxParams {
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
    let parsed = match main {
        ComparisonMain::Source(text) => {
            crate::pipeline::morphosyntax::ParsedFile::parse(text, mor_params.policy.ca_policy)?
        }
        ComparisonMain::Constructed(output) => {
            crate::pipeline::morphosyntax::ParsedFile::from_output(
                output,
                mor_params.policy.ca_policy,
            )?
        }
    };
    let morphotagged_main =
        crate::pipeline::morphosyntax::run_admitted_morphosyntax(parsed, services, &mor_params)
            .await?;
    Ok(build_comparison_artifacts_from_proof(
        morphotagged_main,
        reference,
    ))
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
        metrics: CompareMetricsCsvTable::from_metrics(&bundle.metrics).map_err(|err| {
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
    // the completely admitted gold companion.
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
        metrics: CompareMetricsCsvTable::from_metrics(&bundle.metrics).map_err(|err| {
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
/// 1. Completely admit the retained reference.
/// 2. Admit and run morphosyntax on `main_text` (regenerating %mor/%gra).
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
        build_comparison_artifacts(
            ComparisonMain::Source(main_text),
            AdmittedComparisonReference::admit(gold_text)?,
            lang,
            services,
            mwt,
        )
        .await?,
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
    reference: AdmittedComparisonReference,
) -> Result<CompareMaterializedOutputs, ServerError> {
    materialize_released(build_comparison_artifacts_from_proof(
        morphotagged_main,
        reference,
    ))
}

/// Continue benchmark comparison from its admitted constructed transcript.
pub(crate) async fn process_compare_constructed_main(
    main: crate::pipeline::post_validate::PostValidated,
    reference: AdmittedComparisonReference,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    mwt: &MwtDict,
) -> Result<MainAnnotatedCompareOutputs, ServerError> {
    materialize_main_annotated(
        build_comparison_artifacts(
            ComparisonMain::Constructed(main),
            reference,
            lang,
            services,
            mwt,
        )
        .await?,
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
        let reference = AdmittedComparisonReference::admit(gold)
            .expect("the positive reference fixture must be completely valid");
        let main = MorphotaggedMain::from_proof(PostValidated::for_test(
            main,
            crate::api::ReleasedCommand::Morphotag,
        ));
        ComparisonArtifacts::build(main, reference)
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

        let output = materialize_main_annotated(comparison_of(&main, &gold)).expect("materialized");

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
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\tsample, audio\n*PAR:\thello world .\n%mor:\tintj|hello noun|world .\n%gra:\t1|2|COM 2|0|ROOT 3|2|PUNCT\n%wor:\thello \u{15}0_100\u{15} world \u{15}100_200\u{15} .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello world .\n@End\n";

        let output = materialize_released(comparison_of(main, gold)).expect("materialized");

        assert!(!output.chat_output.as_str().contains("%mor:"));
        assert!(!output.chat_output.as_str().contains("%gra:"));
        assert!(output.chat_output.as_str().contains("%wor:\thello"));
        assert!(output.chat_output.as_str().contains("\u{15}0_100\u{15}"));
        assert!(
            output
                .chat_output
                .as_str()
                .contains("@Media:\tsample, audio")
        );
    }

    /// Preservation detects new losses, while complete admission also refuses
    /// inherited invalidity. Neither case can authorize malformed output.
    #[test]
    fn comparison_requires_validity_even_when_input_already_lacked_a_terminator() {
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

        let inherited = gate_comparison_output(
            UtteranceCensus::of(&stripped_file),
            stripped_file,
            crate::api::ReleasedCommand::Compare,
        )
        .expect_err("inherited invalidity cannot authorize output");
        assert!(matches!(inherited, ServerError::OutputAdmission { .. }));
    }

    /// Invalid reference source cannot acquire comparison admission.
    #[test]
    fn invalid_gold_cannot_authorize_a_successful_comparison_output() {
        let gold = make_chat(&[("PAR", "hello world today")]);
        let refusal = AdmittedComparisonReference::admit(&gold)
            .expect_err("invalid reference input must refuse before comparison");
        assert!(matches!(refusal, ServerError::ChatAdmission(_)));
    }

    #[test]
    fn released_compare_output_copies_media_header_from_main() {
        let main = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\tsample, audio, unlinked\n*PAR:\thello world .\n%mor:\tintj|hello noun|world .\n@End\n";
        let gold = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello world .\n@End\n";

        let output = materialize_released(comparison_of(main, gold)).expect("materialized");

        assert!(
            output
                .chat_output
                .as_str()
                .contains("@Media:\tsample, audio, unlinked")
        );
        assert!(!output.chat_output.as_str().contains("%mor:"));
        assert!(!output.chat_output.as_str().contains("%gra:"));
    }
}
