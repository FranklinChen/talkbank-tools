//! Submission-frame tests without changing the process working directory.

use super::*;
use crate::cli::args::Cli;
use clap::Parser;
use std::path::Path;

fn align(root: Option<&str>) -> Cli {
    let mut args = vec!["batchalign3", "align", "input.cha"];
    if let Some(root) = root {
        args.extend(["--media-dir", root]);
    }
    Cli::parse_from(args)
}

#[test]
fn media_root_cli_relative_anchors_to_submission_cwd() {
    let temp = tempfile::tempdir().unwrap();
    let cli = align(Some("media/../声 recordings"));
    let Some(CommandOptions::Align(options)) =
        build_typed_options_with_cwd(&cli.command, &cli.global, || Ok(temp.path().to_owned()))
            .expect("admitted CLI options")
    else {
        panic!("align options")
    };
    let root = options.media_dir.unwrap().admit_absolute().unwrap();
    assert_eq!(root.as_path(), temp.path().join("media/../声 recordings"));
    assert!(!root.as_path().exists());
}

#[test]
fn media_root_cli_absolute_and_absent_do_not_read_cwd() {
    let temp = tempfile::tempdir().unwrap();
    for root in [None, temp.path().to_str()] {
        let cli = align(root);
        let Some(CommandOptions::Align(options)) =
            build_typed_options_with_cwd(&cli.command, &cli.global, || panic!("cwd unnecessary"))
                .unwrap()
        else {
            panic!("align options")
        };
        assert_eq!(options.media_dir.is_some(), root.is_some());
        if let Some(declaration) = options.media_dir {
            assert_eq!(declaration.admit_absolute().unwrap().as_path(), temp.path());
        }
    }
}

#[test]
fn media_root_cli_refuses_empty_relative_base_and_unreadable_cwd() {
    let empty = align(Some(""));
    assert!(matches!(
        build_typed_options_with_cwd(&empty.command, &empty.global, || panic!(
            "empty root must fail before reading cwd"
        )),
        Err(InvalidCommandOptions::MediaRoot(MediaRootRefusal::Empty))
    ));
    let relative = align(Some("media"));
    assert!(matches!(
        build_typed_options_with_cwd(&relative.command, &relative.global, || Ok(Path::new(
            "relative-base"
        )
        .to_owned())),
        Err(InvalidCommandOptions::MediaRoot(
            MediaRootRefusal::Relative(_)
        ))
    ));
    assert!(matches!(
        build_typed_options_with_cwd(&relative.command, &relative.global, || Err(
            std::io::Error::new(std::io::ErrorKind::NotFound, "cwd removed")
        )),
        Err(InvalidCommandOptions::WorkingDirectory(_))
    ));
}
