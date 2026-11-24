//! Per-file fixture input admission and source-bound pending projections.

/// Admit complete per-file records through the same set boundary as production.
/// Each record already owns its admitted paths and explicit input kind.
pub fn paths_inputs(files: Vec<crate::store::PathsInput>) -> crate::store::JobInputs {
    crate::store::JobInputs::Paths {
        files: crate::store::PathsFiles::admit(files).expect("noncolliding fixture inputs"),
        source_dir: None,
    }
}

/// A fixture path admitted as the user's own process admits it.
pub fn admitted(path: &str) -> batchalign_types::paths::AdmittedPath {
    batchalign_types::paths::AllowedRoots::same_user()
        .admit(&batchalign_types::paths::ClientPath::new(path))
        .expect("an absolute fixture path")
}

/// Every file of `inputs` as pending runner work: what a fresh job's
/// snapshot carries.
pub fn pending_files(inputs: &crate::store::JobInputs) -> Vec<crate::store::PendingJobFile> {
    inputs.entries(|_| true).collect()
}

/// A snapshot whose pending files are every file of its own inputs, as a
/// fresh job's projection gives.
pub fn with_pending(
    mut snapshot: crate::store::RunnerJobSnapshot,
) -> crate::store::RunnerJobSnapshot {
    snapshot.pending_files = pending_files(&snapshot.filesystem.inputs);
    snapshot
}

/// The complete source proof for a one-file fixture.
pub fn only_file(inputs: &crate::store::JobInputs) -> crate::store::PendingJobFile {
    let [file] = pending_files(inputs).try_into().expect("a one-file job");
    file
}

/// Own the isolated root and the authority admitted for that same root.
/// The root observation is retained once for every input in the fixture batch.
pub struct PathsFixture {
    root: tempfile::TempDir,
    authority: batchalign_types::paths::AllowedRoots,
}

impl PathsFixture {
    pub(crate) fn create() -> Self {
        let root = tempfile::tempdir().expect("create fixture root");
        let authority =
            batchalign_types::paths::AllowedRoots::admit_roots(&[root.path().to_path_buf()])
                .expect("admit fixture root");
        Self { root, authority }
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        self.root.path()
    }

    pub(crate) fn source(&self, name: &str) -> batchalign_types::paths::AdmittedPath {
        self.authority
            .admit(&batchalign_types::paths::ClientPath::new(
                self.path().join(name).to_string_lossy(),
            ))
            .expect("admit fixture source")
    }

    /// Per-file admission uses this root's retained authority. A caller
    /// cannot supply a different root alongside the admitted authority.
    pub(crate) fn input(
        &self,
        name: &str,
        kind: crate::store::InputKind,
    ) -> crate::store::PathsInput {
        use batchalign_types::paths::ClientPath;
        let output = ClientPath::new(
            self.path()
                .join(format!("{name}.out.cha"))
                .to_string_lossy(),
        );
        crate::store::PathsInput {
            display: name.into(),
            kind,
            source: self.source(name),
            output: self.authority.admit(&output).expect("admit fixture output"),
            before: None,
        }
    }
}
