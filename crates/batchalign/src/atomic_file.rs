//! Writing a whole file so a reader sees the old contents or the new ones,
//! never a mixture.
//!
//! The bytes go to a uniquely named temporary file beside the target (so a
//! rename stays on one filesystem), are flushed to disk, and are then renamed
//! over the target; on Unix the directory is synced too, so the rename itself
//! survives a crash. A uniquely named temporary file matters: the copies this
//! replaced used a fixed `<target>.tmp`, so two writers racing could truncate
//! each other's temporary file mid-write.
//!
//! The daemon state file, the server handshake, the worker registry, the
//! debug artifacts, the comparison reports and the evaluation evidence all
//! write through here, as does every CHAT, CSV and evidence text artifact a
//! command produces for the user ([`write_text_artifact`]). The text
//! artifacts matter most: an in-place `morphotag` run's output path IS the
//! user's only copy of the file, and a truncating write (what
//! `std::fs::write` and `tokio::fs::write` do) empties that file first and
//! fills it afterwards, so a crash, a kill or a full disk between the two left
//! a truncated transcript behind. Here the old bytes are never opened for
//! writing: they stay whole until the rename lands.
//!
//! # File mode
//!
//! Every caller names who the file is for ([`Audience`]), because the
//! temporary file's own mode becomes the target's on rename, and the two
//! audiences want different modes. Runtime state and debug dumps are
//! [`Audience::Owner`] (`0600`): they live in the per-user state directory,
//! the debug dumps hold transcript content, and every reader is the same
//! user's CLI, server or worker. Reports a command was asked to write are
//! [`Audience::UmaskDefault`]: the mode `std::fs::write` gives (`0666` less
//! the umask, `0644` under the usual `022`), so a report written into a
//! shared corpus or output directory is as readable as any other file the
//! user creates there.

use std::io::Write;
use std::path::Path;

/// Whether a write may replace a file already at the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Existing {
    /// Replace whatever is there.
    Replace,
    /// Refuse (`AlreadyExists`) rather than replace an existing file.
    Keep,
}

/// Who a written file is for, which decides its mode and whether an alias at
/// the target is written through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Audience {
    /// Per-user runtime state or debug content: owner read/write only
    /// (`0600`), regardless of the umask. The path is this program's own and
    /// is never an alias, so it is written exactly where named.
    Owner,
    /// A file at a path the user named (a command's output, a report): `0666`
    /// less the umask, as `std::fs::write` would create it. An existing leaf
    /// symlink there is followed, so the bytes land in the file the link
    /// names, as a truncating write would have put them; a rename over the
    /// link itself would quietly turn the alias into a private copy.
    UmaskDefault,
}

impl Audience {
    /// The mode the temporary file is created with. The umask still applies
    /// on creation, so `Owner` stays `0600` and `UmaskDefault` becomes what
    /// the umask allows.
    #[cfg(unix)]
    fn permissions(self) -> std::fs::Permissions {
        use std::os::unix::fs::PermissionsExt;
        std::fs::Permissions::from_mode(match self {
            Self::Owner => 0o600,
            Self::UmaskDefault => 0o666,
        })
    }
}

/// Write `bytes` to `path` atomically, creating missing parent directories,
/// with the mode and alias handling `audience` calls for.
pub(crate) fn write_atomically(
    path: &Path,
    bytes: &[u8],
    existing: Existing,
    audience: Audience,
) -> std::io::Result<()> {
    let resolved;
    let path = match audience {
        Audience::Owner => path,
        Audience::UmaskDefault => match std::fs::canonicalize(path) {
            Ok(target) => {
                resolved = target;
                &resolved
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => path,
            Err(error) => return Err(error),
        },
    };
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "cannot write a file at {}: it has no parent",
                path.display()
            ),
        )
    })?;
    // A bare file name has the empty path as its parent: the current directory.
    let directory = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    std::fs::create_dir_all(directory)?;
    let mut builder = tempfile::Builder::new();
    #[cfg(unix)]
    builder.permissions(audience.permissions());
    #[cfg(not(unix))]
    let _ = audience;
    let mut staged = builder.tempfile_in(directory)?;
    staged.write_all(bytes)?;
    staged.as_file().sync_all()?;
    // The staged file's bytes are already on disk; the rename is what the
    // directory sync below makes durable. Syncing the renamed file again
    // would flush the same inode a second time (a full drive flush on macOS).
    match existing {
        Existing::Replace => staged.persist(path),
        Existing::Keep => staged.persist_noclobber(path),
    }
    .map_err(|error| error.error)?;
    #[cfg(unix)]
    std::fs::File::open(directory)?.sync_all()?;
    Ok(())
}

/// Write a text artifact a command produced for the user (a CHAT output, a
/// CSV sidecar, an evidence document, a merge-verify deliverable) over
/// `path`: replace whatever is there, as readable as any file the user
/// creates, an alias written through ([`Audience::UmaskDefault`]).
///
/// The one writer for such artifacts on the daemon side (the runner's output
/// targets and the consolidated `compare.csv`), on the CLI side (copy-back of
/// a job's results) and in merge-verify, each of which deleted its raw
/// `fs::write` when this arrived. All of them replace on purpose: an in-place
/// run's output path is its input, and an explicit `-o` destination is where
/// the user asked for the result. Media exports, which must never replace,
/// publish through their own `persist_noclobber` path.
pub(crate) fn write_text_artifact(path: &Path, text: &str) -> std::io::Result<()> {
    write_atomically(
        path,
        text.as_bytes(),
        Existing::Replace,
        Audience::UmaskDefault,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_and_keeps_as_asked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("state.json");
        write_atomically(&path, b"one", Existing::Replace, Audience::Owner).expect("first write");
        write_atomically(&path, b"two", Existing::Replace, Audience::Owner).expect("replace");
        assert_eq!(std::fs::read(&path).expect("read"), b"two");

        let refused =
            write_atomically(&path, b"three", Existing::Keep, Audience::Owner).expect_err("kept");
        assert_eq!(refused.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&path).expect("read"), b"two");

        let leftovers: Vec<_> = std::fs::read_dir(path.parent().expect("parent"))
            .expect("list")
            .collect();
        assert_eq!(leftovers.len(), 1, "no temporary file is left behind");
    }

    /// Runtime state is owner-only; a requested report gets the mode
    /// `std::fs::write` would give a file in the same directory.
    #[cfg(unix)]
    #[test]
    fn the_audience_decides_the_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let mode = |path: &Path| {
            std::fs::metadata(path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777
        };

        let state = dir.path().join("daemon.json");
        write_atomically(&state, b"{}", Existing::Replace, Audience::Owner).expect("state");
        assert_eq!(mode(&state), 0o600);

        let report = dir.path().join("report.csv");
        write_atomically(&report, b"a,b", Existing::Replace, Audience::UmaskDefault)
            .expect("report");
        let plain = dir.path().join("plain.csv");
        std::fs::write(&plain, b"a,b").expect("plain write");
        assert_eq!(mode(&report), mode(&plain));
    }

    /// The old bytes are never opened for writing: a reader holding the old
    /// file open still reads every one of them after the replacement, and
    /// the path now names a different inode. A truncating write fails this
    /// (the held descriptor would read the new bytes, or nothing), which is
    /// also why an interruption at any point before the rename leaves the
    /// previous file whole.
    #[cfg(unix)]
    #[test]
    fn a_text_artifact_replaces_without_touching_the_old_inode() {
        use std::io::Read;
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sample.cha");
        std::fs::write(&path, "@UTF8\n@Begin\n*CHI:\told .\n@End\n").expect("previous file");
        let mut held = std::fs::File::open(&path).expect("open the previous file");
        let old_inode = held.metadata().expect("metadata").ino();

        write_text_artifact(&path, "@UTF8\n@Begin\n*CHI:\tnew .\n@End\n").expect("replace");

        let mut still_old = String::new();
        held.read_to_string(&mut still_old).expect("read held");
        assert_eq!(still_old, "@UTF8\n@Begin\n*CHI:\told .\n@End\n");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read path"),
            "@UTF8\n@Begin\n*CHI:\tnew .\n@End\n"
        );
        assert_ne!(
            std::fs::metadata(&path).expect("metadata").ino(),
            old_inode,
            "the path names the newly staged file, not the old one rewritten"
        );
    }

    /// Writing through an alias changes the file the alias names and keeps
    /// the alias an alias, as a truncating write did; a state file is written
    /// where named, so an alias there becomes the file.
    #[cfg(unix)]
    #[test]
    fn a_user_named_path_is_written_through_an_existing_leaf_symlink() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("store").join("sample.cha");
        std::fs::create_dir_all(real.parent().expect("parent")).expect("store dir");
        std::fs::write(&real, "prior").expect("real file");
        let alias = dir.path().join("sample.cha");
        std::os::unix::fs::symlink(&real, &alias).expect("alias");

        write_text_artifact(&alias, "returned bytes").expect("write through alias");

        assert_eq!(
            std::fs::read_to_string(&real).expect("real"),
            "returned bytes"
        );
        assert!(
            std::fs::symlink_metadata(&alias)
                .expect("alias metadata")
                .file_type()
                .is_symlink(),
            "the alias is still a symlink"
        );

        write_atomically(&alias, b"{}", Existing::Replace, Audience::Owner).expect("state write");
        assert_eq!(
            std::fs::read_to_string(&real).expect("real"),
            "returned bytes"
        );
        assert!(
            !std::fs::symlink_metadata(&alias)
                .expect("alias metadata")
                .file_type()
                .is_symlink(),
            "a state file replaces the alias"
        );
    }
}
