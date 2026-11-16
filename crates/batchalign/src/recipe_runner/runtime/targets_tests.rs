//! External-declaration admission controls through the public target API.

use super::{PlanningError, SourceOutputs, planned_output_artifacts, test_options};
use crate::api::ReleasedCommand;
use crate::store::test_support::{PathsFixture, only_file, paths_inputs};
use crate::store::{ContentInput, InputKind, JobInputs, RunnerFilesystemConfig};

enum ChangedDeclaration {
    Name,
    Kind,
    Source,
    Output,
    Prior,
    Ordinal,
    Mode,
}

/// Guards: Output admission refuses a file from a different declaration, a differently sourced layout and an incompatible artifact shape before publication.
/// Kind: boundary
/// Not a type: Pending files and layouts have private constructors, while equality of independently observed names, ordinals, kinds and admitted source/output/prior paths must be established when they meet.
/// State: One isolated temporary root supplies admitted path identities. The contract compares declarations and checks that no output directory is created. No input bytes, worker or store is used.
/// Concurrency: Each candidate configuration is local and immutable during admission. The test owns its root and observations.
#[test]
fn output_admission_binds_complete_input_observations_and_layout_source() {
    let root = PathsFixture::create();
    let base = root.input("sample.cha", InputKind::Chat);
    let other = root.input("other.cha", InputKind::Chat);
    let filesystem = RunnerFilesystemConfig {
        inputs: paths_inputs(vec![base.clone()]),
        staging_dir: batchalign_types::paths::ServerPath::new(root.path().join("job")),
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: batchalign_types::paths::ClientPath::new(root.path().to_string_lossy()),
    };
    let file = only_file(&filesystem.inputs);
    let options = test_options(ReleasedCommand::Morphotag);
    let layout = planned_output_artifacts(ReleasedCommand::Morphotag, &options, file.filename())
        .expect("catalog layout");
    SourceOutputs::admit_planned(&filesystem, &file, &layout).expect("matching source");

    let mut accepted_mismatches = Vec::new();
    for change in [
        ChangedDeclaration::Name,
        ChangedDeclaration::Kind,
        ChangedDeclaration::Source,
        ChangedDeclaration::Output,
        ChangedDeclaration::Prior,
        ChangedDeclaration::Ordinal,
        ChangedDeclaration::Mode,
    ] {
        let mut input = base.clone();
        let inputs = match change {
            ChangedDeclaration::Name => {
                input.display = other.display.clone();
                paths_inputs(vec![input])
            }
            ChangedDeclaration::Kind => {
                input.kind = InputKind::Media;
                paths_inputs(vec![input])
            }
            ChangedDeclaration::Source => {
                input.source = other.source.clone();
                paths_inputs(vec![input])
            }
            ChangedDeclaration::Output => {
                input.output = other.output.clone();
                paths_inputs(vec![input])
            }
            ChangedDeclaration::Prior => {
                input.before = Some(other.source.clone());
                paths_inputs(vec![input])
            }
            ChangedDeclaration::Ordinal => paths_inputs(vec![other.clone(), input]),
            ChangedDeclaration::Mode => JobInputs::Content(vec![ContentInput {
                display: base.display.clone(),
                kind: base.kind,
            }]),
        };
        let mut changed = filesystem.clone();
        changed.inputs = inputs;
        match SourceOutputs::admit_planned(&changed, &file, &layout) {
            Err(PlanningError::OutputSourceMismatch(name)) => assert_eq!(name, *file.filename()),
            Err(error) => panic!("unexpected refusal: {error}"),
            Ok(_) => accepted_mismatches.push(changed.inputs),
        }
    }
    let different =
        planned_output_artifacts(ReleasedCommand::Morphotag, &options, &"other.cha".into())
            .expect("independently planned source");
    assert!(matches!(
        SourceOutputs::admit_planned(&filesystem, &file, &different),
        Err(PlanningError::OutputSourceMismatch(_))
    ));
    assert!(
        accepted_mismatches.is_empty(),
        "mismatched declarations were admitted: {accepted_mismatches:?}"
    );
    assert!(matches!(
        SourceOutputs::plan(&filesystem, &file, ReleasedCommand::Morphotag, &options)
            .expect("primary plan")
            .into_chat_metrics(),
        Err(PlanningError::OutputLayoutMismatch(_))
    ));
    assert!(!root.path().join("job/output").exists());
}
