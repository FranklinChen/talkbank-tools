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
//! write through here.
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

/// Who a written file is for, which decides its mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Audience {
    /// Per-user runtime state or debug content: owner read/write only
    /// (`0600`), regardless of the umask.
    Owner,
    /// A report the user asked a command to write: `0666` less the umask,
    /// as `std::fs::write` would create it.
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
/// with the mode `audience` calls for.
pub(crate) fn write_atomically(
    path: &Path,
    bytes: &[u8],
    existing: Existing,
    audience: Audience,
) -> std::io::Result<()> {
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
    let written = match existing {
        Existing::Replace => staged.persist(path),
        Existing::Keep => staged.persist_noclobber(path),
    }
    .map_err(|error| error.error)?;
    written.sync_all()?;
    #[cfg(unix)]
    std::fs::File::open(directory)?.sync_all()?;
    Ok(())
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
}
