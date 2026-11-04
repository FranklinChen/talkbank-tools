use super::*;
use crate::api::FileProvenance;
use crate::cli::discover::{ServerInputPlan, plan_server_inputs};

#[test]
fn native_export_collisions_and_existing_outputs_refuse_before_directory_creation() {
    use crate::options::{CommandOptions, CommonOptions, ConvertOptions};
    let root = tempfile::tempdir().unwrap();
    let options = CommandOptions::Convert(ConvertOptions {
        common: CommonOptions::default(),
        format: crate::media::export::AudioExportFormat::Wav,
    });
    let sources = [root.path().join("same.wav"), root.path().join("same.mp3")];
    for source in &sources {
        std::fs::write(source, b"admission placeholder").unwrap();
    }
    let out = root.path().join("out");
    let anchors = [out.join("same.wav"), out.join("same.mp3")];
    assert!(matches!(
        plan_server_inputs(&options, &sources, &anchors, &sources, Some(&out), None),
        Err(CliError::OutputCollision { .. })
    ));
    assert!(!out.exists());
    std::fs::create_dir(&out).unwrap();
    let retained = out.join("same.converted.wav");
    std::fs::write(&retained, b"existing output").unwrap();
    assert!(
        plan_server_inputs(
            &options,
            &sources[..1],
            &anchors[..1],
            &sources[..1],
            Some(&out),
            None
        )
        .is_err()
    );
    assert_eq!(std::fs::read(retained).unwrap(), b"existing output");
}

/// A `transcribe` plan over one recording: `audio.wav` beside an optional
/// existing `audio.cha`, with or without `-o` and `--overwrite`.
fn transcribe_plan(
    root: &Path,
    existing_cha: bool,
    out_dir: Option<&Path>,
    overwrite: Option<Overwrite>,
) -> Result<ResultDestinations, CliError> {
    let media = root.join("in/audio.wav");
    std::fs::create_dir_all(media.parent().unwrap()).unwrap();
    std::fs::write(&media, b"recording bytes").unwrap();
    let anchor = match out_dir {
        Some(dir) => dir.join("audio.wav"),
        None => media.clone(),
    };
    if existing_cha {
        let cha = anchor.with_extension("cha");
        std::fs::create_dir_all(cha.parent().unwrap()).unwrap();
        std::fs::write(cha, "@UTF8\n@Begin\n@End\n").unwrap();
    }
    plan_server_inputs(
        &crate::recipe_runner::runtime::test_options(ReleasedCommand::Transcribe),
        std::slice::from_ref(&media),
        std::slice::from_ref(&anchor),
        std::slice::from_ref(&media),
        out_dir,
        overwrite,
    )
    .map(ServerInputPlan::into_destinations)
}

/// An existing `<media>.cha` beside the recording, with neither `--overwrite`
/// nor `-o`, is refused by name at admission: no destinations exist, so no
/// submission can be built, and the file is untouched.
#[test]
fn transcribe_refuses_to_replace_an_existing_cha_beside_the_media() {
    let dir = tempfile::tempdir().unwrap();
    let error = transcribe_plan(dir.path(), true, None, None).expect_err("refused");
    let cha = dir.path().join("in/audio.cha");
    match error {
        CliError::ExistingOutput { destination, input } => {
            // Planned paths are canonical (on macOS, under `/private`).
            assert_eq!(destination, cha.canonicalize().unwrap());
            assert_eq!(
                input,
                dir.path().join("in/audio.wav").canonicalize().unwrap()
            );
        }
        other => panic!("expected ExistingOutput, got {other:?}"),
    }
    assert!(error_names_the_flag(&transcribe_plan(
        dir.path(),
        true,
        None,
        None
    )));
    assert_eq!(
        std::fs::read_to_string(cha).unwrap(),
        "@UTF8\n@Begin\n@End\n",
        "the refusal must not touch the existing transcript"
    );
}

fn error_names_the_flag(result: &Result<ResultDestinations, CliError>) -> bool {
    match result {
        Err(error) => {
            let text = error.to_string();
            text.contains("--overwrite") && text.contains("-o DIR")
        }
        Ok(_) => false,
    }
}

/// `--overwrite` is the authority to replace it, recorded on the destination.
#[test]
fn transcribe_overwrite_flag_authorizes_replacing_the_existing_cha() {
    let dir = tempfile::tempdir().unwrap();
    let destinations = transcribe_plan(dir.path(), true, None, Some(Overwrite::by_flag())).unwrap();
    assert_eq!(
        destinations.write_decision("audio.cha"),
        Some(WriteDecision::Replace(Overwrite::by_flag()))
    );
    assert!(write_result(&result("audio.cha", ContentType::Chat), &destinations).unwrap());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("in/audio.cha")).unwrap(),
        "returned bytes"
    );
}

/// An explicit output directory is its own authority: results were asked
/// for under it, so an earlier result there is replaced.
#[test]
fn transcribe_output_directory_authorizes_replacing_its_existing_cha() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let destinations = transcribe_plan(dir.path(), true, Some(&out), None).unwrap();
    assert_eq!(
        destinations
            .write_decision("audio.cha")
            .map(|d| d.to_string()),
        Some("replaces an existing file (the output directory)".to_owned())
    );
    assert!(write_result(&result("audio.cha", ContentType::Chat), &destinations).unwrap());
    assert_eq!(
        std::fs::read_to_string(out.join("audio.cha")).unwrap(),
        "returned bytes"
    );
}

/// With nothing at the destination the plan is fresh, flag or no flag.
#[test]
fn transcribe_with_no_existing_cha_is_a_fresh_destination() {
    let dir = tempfile::tempdir().unwrap();
    let destinations = transcribe_plan(dir.path(), false, None, None).unwrap();
    assert_eq!(
        destinations.write_decision("audio.cha"),
        Some(WriteDecision::Fresh)
    );
    assert!(write_result(&result("audio.cha", ContentType::Chat), &destinations).unwrap());
    assert!(dir.path().join("in/audio.cha").exists());
}

/// An in-place text command's destination is the file it was given, and
/// rewriting it needs no authority: the existing in-place test below proves
/// the write; this pins the decision it was admitted under.
#[test]
fn in_place_text_commands_rewrite_their_source_without_authority() {
    let dir = tempfile::tempdir().unwrap();
    let in_place = dir.path().join("corpus/file.cha");
    std::fs::create_dir_all(in_place.parent().unwrap()).unwrap();
    std::fs::write(&in_place, "previous bytes").unwrap();
    let destinations = plan_in_place(ReleasedCommand::Morphotag, &in_place);
    assert_eq!(
        destinations.write_decision("file.cha"),
        Some(WriteDecision::RewritesSource)
    );
}

fn result(name: &str, content_type: ContentType) -> FileResult {
    FileResult {
        filename: name.into(),
        content: "returned bytes".into(),
        content_type,
        error: None,
        provenance: FileProvenance::NotRead,
    }
}

fn plan(command: ReleasedCommand, source: &Path, output: &Path) -> ResultDestinations {
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(source, "source bytes").unwrap();
    plan_with_root(command, source, output, Some(output.parent().unwrap()))
}

/// An in-place job: the source is its own destination, no `-o`.
fn plan_in_place(command: ReleasedCommand, source: &Path) -> ResultDestinations {
    plan_with_root(command, source, source, None)
}

fn plan_with_root(
    command: ReleasedCommand,
    source: &Path,
    output: &Path,
    out_dir: Option<&Path>,
) -> ResultDestinations {
    plan_server_inputs(
        &crate::recipe_runner::runtime::test_options(command),
        std::slice::from_ref(&source.to_owned()),
        std::slice::from_ref(&output.to_owned()),
        std::slice::from_ref(&source.to_owned()),
        out_dir,
        None,
    )
    .unwrap()
    .into_destinations()
}

/// An in-place job's destination is its input, the user's only copy: the
/// copy-back replaces it (a text result's `Existing` rule is `Replace`,
/// unlike a media export's `Keep`) through the atomic writer, so the path
/// names a new inode afterwards; a truncating write keeps the inode.
/// `atomic_file`'s own tests prove what a new inode implies.
#[cfg(unix)]
#[test]
fn write_result_replaces_an_existing_text_output_rather_than_rewriting_it() {
    use std::os::unix::fs::MetadataExt;
    let dir = tempfile::tempdir().unwrap();
    let in_place = dir.path().join("corpus/file.cha");
    std::fs::create_dir_all(in_place.parent().unwrap()).unwrap();
    std::fs::write(&in_place, "previous bytes").unwrap();
    let destinations = plan_in_place(ReleasedCommand::Morphotag, &in_place);
    let old_inode = std::fs::metadata(&in_place).unwrap().ino();

    assert!(write_result(&result("file.cha", ContentType::Chat), &destinations).unwrap());

    assert_eq!(
        std::fs::read_to_string(&in_place).unwrap(),
        "returned bytes"
    );
    assert_ne!(std::fs::metadata(&in_place).unwrap().ino(), old_inode);
}

#[test]
fn write_result_creates_only_the_admitted_nested_destination() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out/sub/deep/file.cha");
    let destinations = plan(
        ReleasedCommand::Morphotag,
        &dir.path().join("in/file.cha"),
        &output,
    );
    assert!(!output.parent().unwrap().exists());
    assert!(write_result(&result("file.cha", ContentType::Chat), &destinations).unwrap());
    assert_eq!(std::fs::read_to_string(output).unwrap(), "returned bytes");
}

#[test]
fn unknown_or_hostile_results_create_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out/file.cha");
    let destinations = plan(
        ReleasedCommand::Morphotag,
        &dir.path().join("in/file.cha"),
        &output,
    );
    for name in [
        "other.cha",
        "../../../escaped.cha",
        "sub/../../escaped.cha",
        "/etc/stuff",
    ] {
        assert!(matches!(
            write_result(&result(name, ContentType::Chat), &destinations),
            Err(CliError::UnplannedResult(_))
        ));
    }
    assert!(!dir.path().join("out").exists());
}

#[test]
fn wrong_content_type_creates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out/file.cha");
    let destinations = plan(
        ReleasedCommand::Morphotag,
        &dir.path().join("in/file.cha"),
        &output,
    );
    assert!(matches!(
        write_result(&result("file.cha", ContentType::Csv), &destinations),
        Err(CliError::ResultTypeMismatch(_))
    ));
    assert!(!dir.path().join("out").exists());
}

#[test]
fn failed_result_is_not_written_even_if_its_identity_is_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out/file.cha");
    let destinations = plan(
        ReleasedCommand::Morphotag,
        &dir.path().join("in/file.cha"),
        &output,
    );
    let mut failed = result("unknown.cha", ContentType::Chat);
    failed.error = Some("processing failed".into());
    assert!(!write_result(&failed, &destinations).unwrap());
    assert!(!dir.path().join("out").exists());
}

#[test]
fn write_result_preserves_empty_content() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out/empty.cha");
    let destinations = plan(
        ReleasedCommand::Morphotag,
        &dir.path().join("in/empty.cha"),
        &output,
    );
    let mut empty = result("empty.cha", ContentType::Chat);
    empty.content = ResultContent::default();
    assert!(write_result(&empty, &destinations).unwrap());
    assert_eq!(std::fs::read_to_string(output).unwrap(), "");
}

#[test]
fn compare_primary_and_sidecar_use_the_same_destination_parent() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out/sub/sample.cha");
    let destinations = plan(
        ReleasedCommand::Compare,
        &dir.path().join("in/sample.cha"),
        &output,
    );
    assert!(write_result(&result("sample.cha", ContentType::Chat), &destinations).unwrap());
    assert!(
        write_result(
            &result("sample.compare.csv", ContentType::Csv),
            &destinations
        )
        .unwrap()
    );
    assert!(output.exists());
    assert!(output.with_extension("compare.csv").exists());
    assert!(!dir.path().join("out/sample.compare.csv").exists());
}

#[test]
fn media_primary_uses_command_naming_not_a_writer_extension_guess() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out/audio.mp3");
    let destinations = plan(
        ReleasedCommand::Transcribe,
        &dir.path().join("in/audio.mp3"),
        &output,
    );
    assert!(write_result(&result("audio.cha", ContentType::Chat), &destinations).unwrap());
    assert!(output.with_extension("cha").exists());
    assert!(!output.exists());
    assert!(matches!(
        write_result(&result("audio.mp3", ContentType::Chat), &destinations),
        Err(CliError::UnplannedResult(_))
    ));
}

#[test]
fn planned_traversal_is_refused_before_creating_directories() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("file.cha");
    std::fs::write(&source, "source").unwrap();
    let root = dir.path().join("out");
    let err = plan_server_inputs(
        &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
        std::slice::from_ref(&source),
        &[root.join("../escape/file.cha")],
        std::slice::from_ref(&source),
        Some(&root),
        None,
    )
    .unwrap_err();
    assert!(matches!(err, CliError::PathTraversal(_)));
    assert!(!root.exists());
    assert!(!dir.path().join("escape").exists());
}

#[cfg(unix)]
#[test]
fn symlink_escape_is_refused_at_admission_and_after_admission() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in/file.cha");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(&source, "source").unwrap();
    let root = dir.path().join("out");
    let outside = dir.path().join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    symlink(&outside, root.join("escape")).unwrap();
    assert!(matches!(
        plan_server_inputs(
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
            std::slice::from_ref(&source),
            &[root.join("escape/file.cha")],
            std::slice::from_ref(&source),
            Some(&root),
            None
        ),
        Err(CliError::PathTraversal(_))
    ));
    let destinations = plan_server_inputs(
        &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
        std::slice::from_ref(&source),
        &[root.join("future/file.cha")],
        std::slice::from_ref(&source),
        Some(&root),
        None,
    )
    .unwrap()
    .into_destinations();
    symlink(&outside, root.join("future")).unwrap();
    assert!(matches!(
        write_result(&result("file.cha", ContentType::Chat), &destinations),
        Err(CliError::PathTraversal(_))
    ));
    assert!(!outside.join("file.cha").exists());
}

#[cfg(unix)]
#[test]
fn an_existing_leaf_symlink_keeps_submission_and_copy_back_on_the_same_target() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in/file.cha");
    let root = dir.path().join("out");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::create_dir_all(root.join("nested")).unwrap();
    std::fs::write(&source, "source").unwrap();
    let target = root.join("nested/renamed.cha");
    std::fs::write(&target, "prior").unwrap();
    std::os::unix::fs::symlink(&target, root.join("file.cha")).unwrap();
    let plan = plan_server_inputs(
        &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
        std::slice::from_ref(&source),
        &[root.join("file.cha")],
        std::slice::from_ref(&source),
        Some(&root),
        None,
    )
    .unwrap();
    assert_eq!(
        plan.inputs()[0].output_anchor(),
        std::fs::canonicalize(&root).unwrap().join("file.cha")
    );
    write_result(
        &result("file.cha", ContentType::Chat),
        &plan.into_destinations(),
    )
    .unwrap();
    assert_eq!(std::fs::read_to_string(target).unwrap(), "returned bytes");
    assert!(!root.join("nested/file.cha").exists());
}
