use super::*;

struct Fixture {
    _root: tempfile::TempDir,
    input: PathBuf,
    output: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        let output = root.path().join("output");
        fs::create_dir(&input).unwrap();
        fs::create_dir(&output).unwrap();
        Self {
            _root: root,
            input,
            output,
        }
    }

    fn sources(&self, names: &[&str]) -> Vec<PathBuf> {
        names
            .iter()
            .map(|name| {
                let path = self.input.join(name);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(&path, format!("retained {name}")).unwrap();
                path
            })
            .collect()
    }

    fn proposal(&self, command: PathCommand, names: &[&str]) -> PathProposal {
        PathProposal {
            command,
            source_root: self.input.clone(),
            inputs: self.sources(names),
            destination: PathDestination::Separate {
                folder: self.output.clone(),
            },
        }
    }
}

#[test]
fn nested_outputs_are_derived_from_canonical_inputs_without_creating_them() {
    let fixture = Fixture::new();
    let receipt = fixture
        .proposal(PathCommand::Morphotag, &["nested/a.cha"])
        .admit()
        .unwrap();
    assert_eq!(
        receipt.inputs,
        [wire(&fixture.input.join("nested/a.cha").canonicalize().unwrap()).unwrap()]
    );
    assert_eq!(
        receipt.outputs,
        [wire(&fixture.output.canonicalize().unwrap().join("nested/a.cha")).unwrap()]
    );
    assert!(!fixture.output.join("nested").exists());
    assert_eq!(
        fs::read_to_string(fixture.input.join("nested/a.cha")).unwrap(),
        "retained nested/a.cha"
    );
}

#[test]
fn separate_output_trees_may_neither_contain_nor_be_contained_by_input() {
    let fixture = Fixture::new();
    let child = fixture.input.join("results");
    fs::create_dir(&child).unwrap();
    for output in [&child, fixture.input.parent().unwrap(), &fixture.input] {
        let mut proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha", "results/a.cha"]);
        proposal.destination = PathDestination::Separate {
            folder: output.to_owned(),
        };
        assert!(matches!(
            proposal.admit(),
            Err(PathPlanError::OverlappingTrees)
        ));
        assert_eq!(
            fs::read_to_string(fixture.input.join("a.cha")).unwrap(),
            "retained a.cha"
        );
    }
}

#[cfg(unix)]
#[test]
fn a_different_folder_name_cannot_hide_the_input_directory() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let alias = fixture._root.path().join("alias");
    symlink(&fixture.input, &alias).unwrap();
    let mut proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha"]);
    proposal.destination = PathDestination::Separate { folder: alias };
    assert!(matches!(
        proposal.admit(),
        Err(PathPlanError::OverlappingTrees)
    ));
}

#[test]
fn an_existing_hard_link_to_an_input_cannot_be_a_separate_output() {
    let fixture = Fixture::new();
    let proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha"]);
    fs::hard_link(&proposal.inputs[0], fixture.output.join("a.cha")).unwrap();
    assert!(matches!(
        proposal.admit(),
        Err(PathPlanError::InputOverwrite(_))
    ));
    assert_eq!(
        fs::read_to_string(fixture.input.join("a.cha")).unwrap(),
        "retained a.cha"
    );
}

#[cfg(unix)]
#[test]
fn an_output_subdirectory_link_cannot_escape_the_selected_output_root() {
    let fixture = Fixture::new();
    let proposal = fixture.proposal(PathCommand::Morphotag, &["nested/a.cha"]);
    std::os::unix::fs::symlink(fixture.input.join("nested"), fixture.output.join("nested"))
        .unwrap();
    assert!(matches!(
        proposal.admit(),
        Err(PathPlanError::OutputEscape(_))
    ));
}

#[test]
fn two_names_for_the_same_input_are_refused() {
    let fixture = Fixture::new();
    let mut proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha"]);
    let alias = fixture.input.join("alias.cha");
    fs::hard_link(&proposal.inputs[0], &alias).unwrap();
    proposal.inputs.push(alias);
    assert!(matches!(
        proposal.admit(),
        Err(PathPlanError::DuplicateInput(_))
    ));
}

#[cfg(unix)]
#[test]
fn source_links_outside_the_selected_tree_are_refused() {
    let fixture = Fixture::new();
    let other = fixture.output.join("outside.cha");
    fs::write(&other, "keep outside").unwrap();
    let alias = fixture.input.join("alias.cha");
    std::os::unix::fs::symlink(&other, &alias).unwrap();
    let proposal = PathProposal {
        command: PathCommand::Morphotag,
        source_root: fixture.input,
        inputs: vec![alias],
        destination: PathDestination::Separate {
            folder: fixture.output,
        },
    };
    assert!(matches!(proposal.admit(), Err(PathPlanError::Input(_))));
    assert_eq!(fs::read_to_string(other).unwrap(), "keep outside");
}

#[test]
fn audio_outputs_refuse_equal_case_and_unicode_normalization_aliases() {
    for names in [
        ["clip.wav", "clip.mp3"],
        ["Clip.wav", "clip.mp3"],
        ["é.wav", "e\u{301}.mp3"],
    ] {
        let fixture = Fixture::new();
        assert!(matches!(
            fixture.proposal(PathCommand::Transcribe, &names).admit(),
            Err(PathPlanError::OutputCollision(_))
        ));
        assert_eq!(fs::read_dir(&fixture.output).unwrap().count(), 0);
    }
}

#[test]
fn preexisting_output_aliases_cannot_merge_two_results() {
    let fixture = Fixture::new();
    let proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha", "b.cha"]);
    fs::write(fixture.output.join("a.cha"), "old output").unwrap();
    fs::hard_link(fixture.output.join("a.cha"), fixture.output.join("b.cha")).unwrap();
    assert!(matches!(
        proposal.admit(),
        Err(PathPlanError::OutputCollision(_))
    ));
    assert_eq!(
        fs::read_to_string(fixture.output.join("a.cha")).unwrap(),
        "old output"
    );
}

#[cfg(unix)]
#[test]
fn a_dangling_output_link_is_an_inspection_error_not_absence() {
    let fixture = Fixture::new();
    let proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha"]);
    std::os::unix::fs::symlink(fixture.output.join("missing"), fixture.output.join("a.cha"))
        .unwrap();
    assert!(matches!(
        proposal.admit(),
        Err(PathPlanError::Inspect { .. })
    ));
    assert!(!fixture.output.join("missing").exists());
}

#[test]
fn explicit_in_place_chat_is_allowed_and_benchmark_is_not() {
    let fixture = Fixture::new();
    let mut proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha"]);
    proposal.destination = PathDestination::InPlace;
    let result = proposal.admit().unwrap();
    assert_eq!(result.inputs, result.outputs);
    let mut benchmark = fixture.proposal(PathCommand::Benchmark, &["audio.wav"]);
    benchmark.destination = PathDestination::InPlace;
    assert!(matches!(
        benchmark.admit(),
        Err(PathPlanError::BenchmarkInPlace)
    ));
}

#[test]
fn audio_in_place_cannot_replace_an_input_under_a_different_extension() {
    let fixture = Fixture::new();
    let mut proposal = fixture.proposal(PathCommand::Transcribe, &["a.wav"]);
    fs::hard_link(&proposal.inputs[0], fixture.input.join("a.cha")).unwrap();
    proposal.destination = PathDestination::InPlace;
    assert!(matches!(
        proposal.admit(),
        Err(PathPlanError::InputOverwrite(_))
    ));
}

#[test]
fn unrelated_existing_outputs_remain_eligible_and_untouched_by_admission() {
    let fixture = Fixture::new();
    let proposal = fixture.proposal(PathCommand::Morphotag, &["a.cha"]);
    fs::write(fixture.output.join("a.cha"), "old independent result").unwrap();
    assert!(proposal.admit().is_ok());
    assert_eq!(
        fs::read_to_string(fixture.output.join("a.cha")).unwrap(),
        "old independent result"
    );
}

#[test]
fn wire_requests_refuse_unknown_commands_modes_and_fields() {
    let valid = serde_json::json!({"command":"morphotag", "source_root":"/input", "inputs":["/input/a.cha"], "destination":{"mode":"separate", "folder":"/output"}});
    assert!(serde_json::from_value::<PathProposal>(valid.clone()).is_ok());
    for (key, value) in [
        ("command", serde_json::json!("unknown")),
        ("destination", serde_json::json!({"mode":"guess"})),
        ("extra", serde_json::json!(true)),
    ] {
        let mut bad = valid.clone();
        bad[key] = value;
        assert!(serde_json::from_value::<PathProposal>(bad).is_err());
    }
}
