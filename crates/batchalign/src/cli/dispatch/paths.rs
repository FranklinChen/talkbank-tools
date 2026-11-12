//! Paths-mode submission preparation for local filesystem execution.

use std::path::Path;

use crate::ReleasedCommand;
use crate::api::{JobSubmission, LanguageSpec};
use crate::options::CommandOptions;

use crate::cli::discover::{
    PassthroughMode, PassthroughReport, copy_nonmatching, infer_base_dir, plan_server_inputs,
};
use crate::cli::error::CliError;
use crate::cli::output::{Overwrite, ResultDestinations};

use super::helpers::{
    classify_files, filter_files_for_command, inject_lexicon, order_files_for_command,
};
use crate::cli::args::InputKind;

pub(super) struct PreparedSubmission {
    pub submission: JobSubmission,
    pub destinations: ResultDestinations,
    pub total_files: usize,
    pub passthrough: PassthroughReport,
}

/// Where the inputs' non-input neighbours would be copied, and the report of
/// that walk, for every input directory of one submission. Nothing is copied
/// under [`PassthroughMode::Plan`].
fn pass_through_neighbours(
    command: ReleasedCommand,
    input_kind: InputKind,
    inputs: &[std::path::PathBuf],
    out_dir: Option<&Path>,
    mode: PassthroughMode,
) -> Result<PassthroughReport, CliError> {
    let mut passthrough = PassthroughReport::default();
    if command != ReleasedCommand::Convert
        && let Some(od) = out_dir
    {
        for inp in inputs {
            if Path::new(inp).is_dir() {
                let dir_report =
                    copy_nonmatching(Path::new(inp), Path::new(od), input_kind, command, mode)?;
                passthrough.extend_from(dir_report);
            }
        }
    }
    Ok(passthrough)
}

/// Discover, plan and package one content submission: transcripts shipped by
/// content to a server on another host (which resolves recordings itself).
///
/// The same preparation `prepare_paths_submission` is for a shared
/// filesystem, so a plan of either transport is the submission the run would
/// send. `None` when nothing was found to submit.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_content_submission(
    options: &CommandOptions,
    lang: &LanguageSpec,
    num_speakers: u32,
    input_kind: InputKind,
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    lexicon: Option<&str>,
    overwrite: Option<Overwrite>,
    mode: PassthroughMode,
) -> Result<Option<PreparedSubmission>, CliError> {
    let command = options.command();
    let (files, outputs) =
        crate::cli::discover::discover_server_inputs(inputs, out_dir, input_kind)?;
    let (files, outputs) = filter_files_for_command(command, files, outputs);
    let (files, outputs) = order_files_for_command(command, files, outputs)?;
    let plan = plan_server_inputs(options, &files, &outputs, inputs, out_dir, overwrite)?;
    let passthrough = pass_through_neighbours(command, input_kind, inputs, out_dir, mode)?;

    let base_dir = infer_base_dir(inputs)?;
    let source_paths = plan
        .inputs()
        .iter()
        .map(|input| input.source().to_path_buf())
        .collect::<Vec<_>>();
    let server_names = plan
        .inputs()
        .iter()
        .map(|input| input.server_name().to_owned())
        .collect::<Vec<_>>();
    let (file_payloads, media_file_names) = classify_files(&source_paths, &server_names)?;
    if file_payloads.is_empty() && media_file_names.is_empty() {
        return Ok(None);
    }
    let total_files = file_payloads.len() + media_file_names.len();

    let mut opts = options.clone();
    inject_lexicon(&mut opts, lexicon)?;
    let debug_traces = opts.common().debug_dir.is_some();

    Ok(Some(PreparedSubmission {
        submission: JobSubmission {
            command,
            lang: lang.clone(),
            num_speakers: num_speakers.into(),
            files: file_payloads,
            media_files: media_file_names,
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: base_dir.to_string_lossy().to_string().into(),
            options: opts,
            paths_mode: false,
            source_paths: vec![],
            output_paths: vec![],
            display_names: vec![],
            debug_traces,
            before_paths: vec![],
        },
        destinations: plan.into_destinations(),
        total_files,
        passthrough,
    }))
}

/// Discover, plan and package one shared-filesystem submission.
///
/// The command is read from `options`, never passed beside it: typed options
/// name exactly one command, so a submission whose command and options
/// disagree has no way to be built.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_paths_submission(
    options: &CommandOptions,
    lang: &LanguageSpec,
    num_speakers: u32,
    input_kind: InputKind,
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    lexicon: Option<&str>,
    before: Option<&std::path::Path>,
    media_mapping_keys: &[String],
    overwrite: Option<Overwrite>,
    mode: PassthroughMode,
) -> Result<Option<PreparedSubmission>, CliError> {
    let command = options.command();
    let (files, outputs) =
        crate::cli::discover::discover_server_inputs(inputs, out_dir, input_kind)?;
    let (files, outputs) = filter_files_for_command(command, files, outputs);
    let (files, outputs) = order_files_for_command(command, files, outputs)?;
    let plan = plan_server_inputs(options, &files, &outputs, inputs, out_dir, overwrite)?;
    let passthrough = pass_through_neighbours(command, input_kind, inputs, out_dir, mode)?;

    if files.is_empty() {
        return Ok(None);
    }

    let server_names = plan
        .inputs()
        .iter()
        .map(|input| input.server_name().to_owned())
        .collect();
    let source_paths: Vec<String> = plan
        .inputs()
        .iter()
        .map(|input| input.source().to_string_lossy().to_string())
        .collect();
    let output_paths: Vec<String> = plan
        .inputs()
        .iter()
        .map(|input| input.output_anchor().to_string_lossy().to_string())
        .collect();

    let base_dir = infer_base_dir(inputs)?;
    let (mapping_key, mapping_subdir) = detect_media_mapping(&base_dir, media_mapping_keys)?;

    let mut opts = options.clone();
    inject_lexicon(&mut opts, lexicon)?;
    let debug_traces = opts.common().debug_dir.is_some();

    let before_paths = if let Some(before_arg) = before {
        let before_path = Path::new(before_arg);
        if before_path.is_dir() {
            let mut matches = Vec::new();
            for src in &files {
                let src_path = Path::new(src);
                let Some(filename) = src_path.file_name() else {
                    return Err(CliError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("input path has no filename: {}", src_path.display()),
                    )));
                };
                let candidate = before_path.join(filename);
                if candidate.exists() {
                    matches.push(Some(
                        std::fs::canonicalize(&candidate)
                            .map_err(CliError::Io)?
                            .to_string_lossy()
                            .to_string(),
                    ));
                } else {
                    matches.push(None);
                }
            }
            matches
        } else if before_path.is_file() && files.len() == 1 {
            std::fs::canonicalize(before_path)
                .map_err(CliError::Io)
                .map(|p| vec![Some(p.to_string_lossy().to_string())])?
        } else {
            eprintln!("warning: --before path is not a valid file or directory, ignoring");
            Vec::new()
        }
    } else {
        Vec::new()
    };

    Ok(Some(PreparedSubmission {
        submission: JobSubmission {
            command,
            lang: lang.clone(),
            num_speakers: num_speakers.into(),
            files: vec![],
            media_files: vec![],
            media_mapping: mapping_key.into(),
            media_subdir: mapping_subdir,
            source_dir: base_dir.to_string_lossy().to_string().into(),
            options: opts,
            paths_mode: true,
            source_paths: source_paths.into_iter().map(Into::into).collect(),
            output_paths: output_paths.into_iter().map(Into::into).collect(),
            display_names: server_names,
            debug_traces,
            before_paths: before_paths
                .into_iter()
                .map(|path| path.map(Into::into))
                .collect(),
        },
        destinations: plan.into_destinations(),
        total_files: files.len(),
        passthrough,
    }))
}

fn detect_media_mapping(
    in_dir: &Path,
    mapping_keys: &[String],
) -> Result<(String, batchalign_types::paths::RepoRelativePath), CliError> {
    use batchalign_types::paths::RepoRelativePath;
    if mapping_keys.is_empty() {
        return Ok((String::new(), RepoRelativePath::default()));
    }

    let abs = std::fs::canonicalize(in_dir).map_err(CliError::Io)?;
    let parts: Vec<String> = abs
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();

    for key in mapping_keys {
        if let Some(idx) = parts.iter().position(|p| p == key) {
            // Folder names of a canonical path: always plain names.
            let subdir = RepoRelativePath::new(parts[idx + 1..].join("/"))
                .map_err(|refusal| CliError::InvalidArgument(refusal.to_string()))?;
            return Ok((key.clone(), subdir));
        }
    }

    Ok((String::new(), RepoRelativePath::default()))
}

#[cfg(test)]
mod tests {
    use super::{detect_media_mapping, prepare_paths_submission};
    use crate::api::{
        ContentType, FileProvenance, FileResult, LanguageCode3, LanguageSpec, ReleasedCommand,
    };
    use crate::cli::args::InputKind;
    use crate::cli::discover::PassthroughMode;
    use crate::cli::output::write_result;
    use crate::options::{AlignOptions, CommandOptions, MorphotagOptions};

    fn morphotag() -> CommandOptions {
        CommandOptions::Morphotag(MorphotagOptions::default())
    }

    fn align() -> CommandOptions {
        CommandOptions::Align(AlignOptions::default())
    }

    const CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";

    #[test]
    fn partial_before_directory_preserves_every_source_slot() {
        for options in [morphotag(), align()] {
            let root = tempfile::tempdir().unwrap();
            let inputs = root.path().join("current");
            let priors = root.path().join("before");
            let output = root.path().join("output");
            std::fs::create_dir_all(&inputs).unwrap();
            std::fs::create_dir_all(&priors).unwrap();
            for name in ["a.cha", "b.cha", "c.cha"] {
                std::fs::write(inputs.join(name), CHAT).unwrap();
            }
            // Exercise no, partial and complete companion populations through
            // real discovery and request validation, not a copied index loop.
            for added in [None, Some("b.cha"), Some("a.cha"), Some("c.cha")] {
                if let Some(name) = added {
                    std::fs::write(priors.join(name), CHAT).unwrap();
                }
                // Both commands here are per-file: neither has a `--lang`.
                let prepared = prepare_paths_submission(
                    &options,
                    &LanguageSpec::PerFile,
                    1,
                    InputKind::Chat,
                    std::slice::from_ref(&inputs),
                    Some(&output),
                    None,
                    Some(&priors),
                    &[],
                    None,
                    PassthroughMode::Copy,
                )
                .unwrap()
                .unwrap();
                let request = prepared.submission;
                request.validate().unwrap();
                assert_eq!(request.before_paths.len(), request.source_paths.len());
                for (source, prior) in request.source_paths.iter().zip(&request.before_paths) {
                    // The CLI and the files are on one machine here, so a
                    // wire path is read as a local one.
                    let candidate =
                        priors.join(std::path::Path::new(source.as_str()).file_name().unwrap());
                    let expected = candidate
                        .exists()
                        .then(|| std::fs::canonicalize(candidate).unwrap());
                    assert_eq!(
                        prior
                            .as_ref()
                            .map(|path| std::path::PathBuf::from(path.as_str())),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn shared_submission_and_copy_back_share_exact_mixed_root_destinations() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("directory");
        let individual = root.path().join("individual");
        let output = root.path().join("out");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::create_dir_all(&individual).unwrap();
        std::fs::write(directory.join("one.cha"), CHAT).unwrap();
        let two = individual.join("two.cha");
        let three = individual.join("nested/three.cha");
        std::fs::create_dir_all(three.parent().unwrap()).unwrap();
        std::fs::write(&two, CHAT).unwrap();
        std::fs::write(&three, CHAT).unwrap();
        let prepared = prepare_paths_submission(
            &morphotag(),
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            1,
            InputKind::Chat,
            &[directory, two, three],
            Some(&output),
            None,
            None,
            &[],
            None,
            PassthroughMode::Copy,
        )
        .unwrap()
        .unwrap();
        assert_eq!(prepared.total_files, 3);
        for (name, destination) in prepared
            .submission
            .display_names
            .iter()
            .zip(&prepared.submission.output_paths)
        {
            let result = FileResult {
                filename: name.as_str().into(),
                content: CHAT.into(),
                content_type: ContentType::Chat,
                error: None,
                provenance: FileProvenance::NotRead,
            };
            write_result(&result, &prepared.destinations).unwrap();
            assert_eq!(std::fs::read_to_string(destination.as_str()).unwrap(), CHAT);
        }
        assert!(output.join("one.cha").exists());
        assert!(output.join("two.cha").exists());
        assert!(output.join("three.cha").exists());
        assert!(!output.join("nested").exists());
        assert!(!output.join("individual").exists());
    }

    #[test]
    fn collision_refusal_precedes_dummy_and_passthrough_copies() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("directory");
        let individual = root.path().join("individual");
        let output = root.path().join("not-created");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::create_dir_all(&individual).unwrap();
        std::fs::write(directory.join("same.cha"), CHAT).unwrap();
        std::fs::write(directory.join("dummy.cha"), "@Options:\tdummy\n").unwrap();
        std::fs::write(directory.join("notes.txt"), "notes").unwrap();
        let other = individual.join("same.cha");
        std::fs::write(&other, CHAT).unwrap();
        let err = prepare_paths_submission(
            &morphotag(),
            &LanguageSpec::Resolved(LanguageCode3::eng()),
            1,
            InputKind::Chat,
            &[directory, other],
            Some(&output),
            None,
            None,
            &[],
            None,
            PassthroughMode::Copy,
        )
        .err()
        .expect("colliding outputs must be refused");
        assert!(matches!(
            err,
            crate::cli::error::CliError::InputNameCollision { .. }
                | crate::cli::error::CliError::OutputCollision { .. }
        ));
        assert!(!output.exists());
    }

    #[test]
    fn detect_media_mapping_keeps_local_subdir() {
        let root = tempfile::tempdir().unwrap();
        let dir = root
            .path()
            .join("slabank-data/French/Newcastle/Discussion/12");
        std::fs::create_dir_all(&dir).unwrap();

        let (key, subdir) = detect_media_mapping(&dir, &["slabank-data".to_string()]).unwrap();
        assert_eq!(key, "slabank-data");
        assert_eq!(subdir.as_str(), "French/Newcastle/Discussion/12");
    }

    #[test]
    fn paths_submission_carries_detected_media_mapping() {
        let root = tempfile::tempdir().unwrap();
        let input_dir = root
            .path()
            .join("slabank-data/French/Newcastle/Discussion/12");
        let output_dir = root.path().join("out");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::write(input_dir.join("d01oma12a.cha"), "@Begin\n@End\n").unwrap();

        let prepared = prepare_paths_submission(
            &align(),
            &LanguageSpec::PerFile,
            1,
            InputKind::Chat,
            std::slice::from_ref(&input_dir),
            Some(output_dir.as_path()),
            None,
            None,
            &["slabank-data".to_string()],
            None,
            PassthroughMode::Copy,
        )
        .unwrap()
        .expect("expected one prepared submission");

        assert_eq!(prepared.submission.media_mapping.as_str(), "slabank-data");
        assert_eq!(
            prepared.submission.media_subdir.as_str(),
            "French/Newcastle/Discussion/12"
        );
    }

    /// An existing `<media>.cha` beside the recording refuses the whole
    /// preparation: no `JobSubmission` is built, so nothing can be submitted,
    /// started or paid for. `--overwrite` lets the same preparation through.
    #[test]
    fn transcribe_preparation_refuses_an_existing_cha_before_any_submission_exists() {
        let root = tempfile::tempdir().unwrap();
        let input_dir = root.path().join("recordings");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::write(input_dir.join("session.wav"), b"recording bytes").unwrap();
        std::fs::write(input_dir.join("session.cha"), CHAT).unwrap();
        let transcribe = crate::recipe_runner::runtime::test_options(ReleasedCommand::Transcribe);
        let prepare = |overwrite| {
            prepare_paths_submission(
                &transcribe,
                &LanguageSpec::Resolved(LanguageCode3::eng()),
                2,
                InputKind::Media,
                std::slice::from_ref(&input_dir),
                None,
                None,
                None,
                &[],
                overwrite,
                PassthroughMode::Copy,
            )
        };
        assert!(matches!(
            prepare(None),
            Err(crate::cli::error::CliError::ExistingOutput { .. })
        ));
        let prepared = prepare(Some(crate::cli::output::Overwrite::by_flag()))
            .unwrap()
            .expect("one recording to submit");
        assert_eq!(prepared.total_files, 1);
        // The anchor is canonical (on macOS, under `/private`), beside the
        // recording: the daemon writes `session.cha` next to it.
        assert_eq!(
            prepared.submission.output_paths[0].as_str(),
            input_dir
                .join("session.wav")
                .canonicalize()
                .unwrap()
                .to_string_lossy()
        );
    }

    #[test]
    fn paths_submission_withholds_provenance_json_from_output() {
        let root = tempfile::tempdir().unwrap();
        let input_dir = root.path().join("in");
        let output_dir = root.path().join("out");
        std::fs::create_dir_all(&input_dir).unwrap();
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::write(input_dir.join("a.cha"), "@Begin\n@End\n").unwrap();
        std::fs::write(
            input_dir.join("PROVENANCE.json"),
            r#"{"inputs":["some other run"]}"#,
        )
        .unwrap();

        let prepared = prepare_paths_submission(
            &align(),
            &LanguageSpec::PerFile,
            1,
            InputKind::Chat,
            std::slice::from_ref(&input_dir),
            Some(output_dir.as_path()),
            None,
            None,
            &[],
            None,
            PassthroughMode::Copy,
        )
        .unwrap()
        .expect("expected one prepared submission");

        assert!(!output_dir.join("PROVENANCE.json").exists());
        let skipped: Vec<_> = prepared.passthrough.skipped().collect();
        assert_eq!(skipped.len(), 1);
        assert_eq!(
            skipped[0].relative_path,
            std::path::PathBuf::from("PROVENANCE.json")
        );
    }
}
