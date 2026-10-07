//! Paths-mode submission preparation for local filesystem execution.

use std::path::Path;

use crate::ReleasedCommand;
use crate::api::{JobSubmission, LanguageSpec};
use crate::options::CommandOptions;

use crate::cli::discover::{
    PassthroughReport, copy_nonmatching, infer_base_dir, plan_server_inputs,
};
use crate::cli::error::CliError;
use crate::cli::output::ResultDestinations;

use super::helpers::{filter_files_for_command, inject_lexicon, order_files_for_command};
use crate::cli::args::InputKind;

pub(super) struct PreparedPathsSubmission {
    pub submission: JobSubmission,
    pub destinations: ResultDestinations,
    pub total_files: usize,
    pub passthrough: PassthroughReport,
}

/// Discover, plan and package one shared-filesystem submission.
///
/// The command is read from `options`, never passed beside it: typed options
/// name exactly one command, so a submission whose command and options
/// disagree has no way to be built.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_paths_submission(
    options: &CommandOptions,
    lang: &str,
    num_speakers: u32,
    input_kind: InputKind,
    inputs: &[std::path::PathBuf],
    out_dir: Option<&std::path::Path>,
    lexicon: Option<&str>,
    before: Option<&std::path::Path>,
    media_mapping_keys: &[String],
) -> Result<Option<PreparedPathsSubmission>, CliError> {
    let command = options.command();
    let (files, outputs) =
        crate::cli::discover::discover_server_inputs(inputs, out_dir, input_kind)?;
    let (files, outputs) = filter_files_for_command(command, files, outputs);
    let (files, outputs) = order_files_for_command(command, files, outputs)?;
    let plan = plan_server_inputs(options, &files, &outputs, inputs, out_dir)?;

    let mut passthrough = PassthroughReport::default();
    if command != ReleasedCommand::Convert
        && let Some(od) = out_dir
    {
        for inp in inputs {
            if Path::new(inp).is_dir() {
                let dir_report =
                    copy_nonmatching(Path::new(inp), Path::new(od), input_kind, command)?;
                passthrough.extend_from(dir_report);
            }
        }
    }

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
                    matches.push(
                        std::fs::canonicalize(&candidate)
                            .map_err(CliError::Io)?
                            .to_string_lossy()
                            .to_string(),
                    );
                }
            }
            matches
        } else if before_path.is_file() && files.len() == 1 {
            std::fs::canonicalize(before_path)
                .map_err(CliError::Io)
                .map(|p| vec![p.to_string_lossy().to_string()])?
        } else {
            eprintln!("warning: --before path is not a valid file or directory, ignoring");
            Vec::new()
        }
    } else {
        Vec::new()
    };

    Ok(Some(PreparedPathsSubmission {
        submission: JobSubmission {
            command,
            lang: LanguageSpec::try_from(lang)
                .map_err(|e| CliError::InvalidArgument(format!("invalid language: {e}")))?,
            num_speakers: num_speakers.into(),
            files: vec![],
            media_files: vec![],
            media_mapping: mapping_key.into(),
            media_subdir: mapping_subdir.into(),
            source_dir: base_dir.to_string_lossy().to_string().into(),
            options: opts,
            paths_mode: true,
            source_paths: source_paths.into_iter().map(Into::into).collect(),
            output_paths: output_paths.into_iter().map(Into::into).collect(),
            display_names: server_names,
            debug_traces,
            before_paths: before_paths.into_iter().map(Into::into).collect(),
        },
        destinations: plan.into_destinations(),
        total_files: files.len(),
        passthrough,
    }))
}

fn detect_media_mapping(
    in_dir: &Path,
    mapping_keys: &[String],
) -> Result<(String, String), CliError> {
    if mapping_keys.is_empty() {
        return Ok((String::new(), String::new()));
    }

    let abs = std::fs::canonicalize(in_dir).map_err(CliError::Io)?;
    let parts: Vec<String> = abs
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();

    for key in mapping_keys {
        if let Some(idx) = parts.iter().position(|p| p == key) {
            let subdir = if idx + 1 < parts.len() {
                parts[idx + 1..].join("/")
            } else {
                String::new()
            };
            return Ok((key.clone(), subdir));
        }
    }

    Ok((String::new(), String::new()))
}

#[cfg(test)]
mod tests {
    use super::{detect_media_mapping, prepare_paths_submission};
    use crate::api::{ContentType, FileProvenance, FileResult};
    use crate::cli::args::InputKind;
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
            "eng",
            1,
            InputKind::Chat,
            &[directory, two, three],
            Some(&output),
            None,
            None,
            &[],
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
            "eng",
            1,
            InputKind::Chat,
            &[directory, other],
            Some(&output),
            None,
            None,
            &[],
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
        assert_eq!(subdir, "French/Newcastle/Discussion/12");
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
            "eng",
            1,
            InputKind::Chat,
            std::slice::from_ref(&input_dir),
            Some(output_dir.as_path()),
            None,
            None,
            &["slabank-data".to_string()],
        )
        .unwrap()
        .expect("expected one prepared submission");

        assert_eq!(prepared.submission.media_mapping.as_str(), "slabank-data");
        assert_eq!(
            prepared.submission.media_subdir.as_str(),
            "French/Newcastle/Discussion/12"
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
            "eng",
            1,
            InputKind::Chat,
            std::slice::from_ref(&input_dir),
            Some(output_dir.as_path()),
            None,
            None,
            &[],
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
