//! Producer-admitted CLI inputs and their exact result destinations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(test)]
use crate::ReleasedCommand;
use crate::cli::error::CliError;
use crate::cli::output::ResultDestinations;

/// One immutable source/name/output-anchor relationship.
#[derive(Debug)]
pub struct PlannedServerInput {
    source: PathBuf,
    server_name: String,
    output_anchor: PathBuf,
}

impl PlannedServerInput {
    /// Canonical source read by either transport.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Source-relative identity submitted to the server.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Absolute output anchor supplied to the shared-filesystem runner.
    pub fn output_anchor(&self) -> &Path {
        &self.output_anchor
    }
}

/// An admitted submission source set with its associated write capability.
///
/// There is no empty-map/default constructor. Both transports derive their
/// payload from these inputs and transfer the same destinations to copy-back.
#[derive(Debug)]
pub struct ServerInputPlan {
    inputs: Vec<PlannedServerInput>,
    destinations: ResultDestinations,
}

impl ServerInputPlan {
    /// Ordered, immutable source relationships.
    pub fn inputs(&self) -> &[PlannedServerInput] {
        &self.inputs
    }

    /// Hand the admitted write capability to the result consumer.
    pub fn into_destinations(self) -> ResultDestinations {
        self.destinations
    }
}

/// Admit discovery output before any output directory, pass-through copy or
/// job is created. Command-owned naming covers primary artifacts and sidecars.
pub fn plan_server_inputs(
    options: &crate::options::CommandOptions,
    files: &[PathBuf],
    outputs: &[PathBuf],
    input_roots: &[PathBuf],
    out_dir: Option<&Path>,
) -> Result<ServerInputPlan, CliError> {
    if files.len() != outputs.len() {
        return Err(CliError::InvalidArgument(
            "discovered sources and output anchors have different lengths".into(),
        ));
    }
    let names = super::build_server_names(files, input_roots)?;
    let mut identities = HashMap::new();
    let mut planned = Vec::with_capacity(files.len());
    for ((source, output), name) in files.iter().zip(outputs).zip(names) {
        let source = super::canonicalize_path(source, "resolve planned input")?;
        if let Some(first) = identities.insert(name.clone(), source.clone()) {
            return Err(CliError::InputNameCollision {
                name,
                first,
                second: source,
            });
        }
        planned.push(PlannedServerInput {
            source,
            server_name: name,
            output_anchor: crate::cli::output::absolute_output_anchor(output)?,
        });
    }
    let destinations = ResultDestinations::admit(options, &planned, out_dir)?;
    Ok(ServerInputPlan {
        inputs: planned,
        destinations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ContentType, FileProvenance, FileResult};
    use crate::cli::output::write_result;

    fn sources(root: &Path, names: &[&str]) -> Vec<PathBuf> {
        names
            .iter()
            .map(|name| {
                let path = root.join(name);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, "input bytes").unwrap();
                path
            })
            .collect()
    }

    #[test]
    fn distinct_inputs_cannot_share_a_destination() {
        let root = tempfile::tempdir().unwrap();
        let files = sources(root.path(), &["a/same.cha", "b/same.cha"]);
        let out = root.path().join("not-created");
        let err = plan_server_inputs(
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
            &files,
            &[out.join("same.cha"), out.join("same.cha")],
            &files,
            Some(&out),
        )
        .unwrap_err();
        assert!(matches!(err, CliError::OutputCollision { .. }));
        assert_eq!(err.exit_code(), CliError::EXIT_USAGE);
        assert!(!out.exists());
    }

    #[test]
    fn extension_rewriting_cannot_hide_a_media_collision() {
        let root = tempfile::tempdir().unwrap();
        let files = sources(root.path(), &["same.mp3", "same.wav"]);
        let out = root.path().join("not-created");
        let err = plan_server_inputs(
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Transcribe),
            &files,
            &[out.join("same.mp3"), out.join("same.wav")],
            &files,
            Some(&out),
        )
        .unwrap_err();
        assert!(matches!(err, CliError::OutputCollision { .. }));
        assert!(!out.exists());
    }

    #[test]
    fn duplicate_server_identities_are_not_silently_overwritten() {
        let root = tempfile::tempdir().unwrap();
        let files = sources(root.path(), &["a/same.cha", "b/same.cha"]);
        let roots = vec![root.path().join("a"), root.path().join("b")];
        let err = plan_server_inputs(
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
            &files,
            &files,
            &roots,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::InputNameCollision { .. }));
    }

    #[test]
    fn mismatched_discovery_vectors_are_not_truncated() {
        let root = tempfile::tempdir().unwrap();
        let files = sources(root.path(), &["a.cha", "b.cha"]);
        assert!(matches!(
            plan_server_inputs(
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
                &files,
                &files[..1],
                &files,
                None
            ),
            Err(CliError::InvalidArgument(_))
        ));
    }

    #[test]
    fn mixed_root_in_place_destinations_retain_each_sources_parent() {
        let root = tempfile::tempdir().unwrap();
        let files = sources(root.path(), &["a/one.cha", "b/two.cha"]);
        let plan = plan_server_inputs(
            &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
            &files,
            &files,
            &files,
            None,
        )
        .unwrap();
        let names = plan
            .inputs()
            .iter()
            .map(|input| input.server_name().to_owned())
            .collect::<Vec<_>>();
        let destinations = plan.into_destinations();
        for (source, name) in files.iter().zip(names) {
            write_result(
                &FileResult {
                    filename: name.into(),
                    content: "replacement".into(),
                    content_type: ContentType::Chat,
                    error: None,
                    provenance: FileProvenance::NotRead,
                },
                &destinations,
            )
            .unwrap();
            assert_eq!(std::fs::read_to_string(source).unwrap(), "replacement");
        }
        assert!(!root.path().join("a/b").exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_aliases_cannot_disguise_a_destination_collision() {
        let root = tempfile::tempdir().unwrap();
        let files = sources(root.path(), &["a/same.cha", "b/same.cha"]);
        let out = root.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        std::os::unix::fs::symlink(&out, root.path().join("alias")).unwrap();
        assert!(matches!(
            plan_server_inputs(
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Morphotag),
                &files,
                &[out.join("same.cha"), root.path().join("alias/same.cha")],
                &files,
                Some(&out)
            ),
            Err(CliError::OutputCollision { .. })
        ));
        assert!(!out.join("same.cha").exists());
    }
}
