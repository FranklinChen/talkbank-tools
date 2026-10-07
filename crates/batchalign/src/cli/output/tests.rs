use super::*;
use crate::api::FileProvenance;
use crate::cli::discover::plan_server_inputs;

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
        plan_server_inputs(&options, &sources, &anchors, &sources, Some(&out)),
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
            Some(&out)
        )
        .is_err()
    );
    assert_eq!(std::fs::read(retained).unwrap(), b"existing output");
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
    plan_server_inputs(
        &crate::recipe_runner::runtime::test_options(command),
        &[source.to_owned()],
        &[output.to_owned()],
        &[source.to_owned()],
        Some(output.parent().unwrap()),
    )
    .unwrap()
    .into_destinations()
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
            Some(&root)
        ),
        Err(CliError::PathTraversal(_))
    ));
    let destinations = plan_server_inputs(
        &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
        std::slice::from_ref(&source),
        &[root.join("future/file.cha")],
        std::slice::from_ref(&source),
        Some(&root),
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
