//! Which file extensions name media batchalign can consume.
//!
//! One owner, and the verb that goes with it. Split out of `media.rs`
//! because that file had grown past the workspace's 400-line guidance and
//! the module already keeps one concept per file (`probe`, `tools`,
//! `transcode`, `window`).

use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

/// The media formats batchalign can consume, and the two questions asked of
/// them.
///
/// Adding an extension here makes it discoverable via `resolve()`,
/// `list_files()` and the `/media/list` endpoint, AND resolvable by the forced
/// alignment pipeline, because both now ask this type rather than each keeping
/// a list.
///
/// **This used to be two lists, and they disagreed.** A dotted one lived here
/// and an undotted `KNOWN_MEDIA_EXTENSIONS` lived in `runner::util`, and the
/// runner's carried `wma` and `webm` while this one did not. The consequence
/// was reachable: a `.webm` recording could be resolved by the pipeline and
/// transcoded by `ensure_wav` (whose `FORCED_CONVERSION` lists it), yet was
/// invisible to the media walk, so it never appeared in `/media/list`. The doc
/// comment here asserted the opposite the whole time, claiming the list
/// "intentionally mirrors the formats that the engines can consume".
///
/// The dot was the tell. An extension is a bare token; the leading dot belongs
/// to how a FILENAME is spelled, not to what the format is, so storing it
/// dotted made a second representation whose only job was to suit one
/// `ends_with` call. Two of the three callers here had to re-add the dot
/// themselves to compare.
pub struct MediaExtensions;

impl MediaExtensions {
    /// Deliberately PRIVATE. The two questions below are the whole interface;
    /// a public list is an invitation to write a fourth `for ext in ...` loop
    /// building `{stem}.{ext}`, which is how the duplication arose.
    const ALL: &'static [&'static str] = &[
        "wav", "mp3", "mp4", "m4a", "flac", "ogg", "aac", "wma", "webm",
    ];

    /// Whether this filename names a media file batchalign can consume.
    pub fn matches(filename: &str) -> bool {
        Path::new(filename)
            .extension()
            .is_some_and(|extension| Self::is_known(&extension.to_string_lossy()))
    }

    /// Whether this bare extension, with or without a leading dot, is known.
    pub fn is_known(extension: &str) -> bool {
        let bare = extension.trim_start_matches('.').to_lowercase();
        Self::ALL.contains(&bare.as_str())
    }

    /// Every filename this stem could have, in a stable order.
    ///
    /// The order is the declaration order, which puts the formats the corpus
    /// actually uses first: measured over the kept corpus, every one of the
    /// 73,092 resolvable recordings is `mp3`, `mp4` or `wav`.
    pub fn candidates(stem: &str) -> impl Iterator<Item = String> {
        let stem = stem.to_owned();
        Self::ALL
            .iter()
            .map(move |extension| format!("{stem}.{extension}"))
    }

    /// The media file for `stem` in `root/subdir`, every component of `subdir`
    /// and the file name compared exactly.
    ///
    /// **This is the verb, and it was written out four times**: in
    /// `resolve_audio_for_chat_with_media_dir` (twice), in
    /// `staging::prepare::resolve_adjacent_media`, and in `fa_pipeline`'s
    /// `find_media_in_root`. Each was the same `for ext in LIST { join, then
    /// try_exists }` loop, and each therefore had to be found and edited
    /// separately whenever the list changed, which is how the list came to have
    /// two versions. A public extension list invites the fifth copy; this
    /// function is what callers actually wanted.
    ///
    /// **It lists directories rather than asking whether a path exists.**
    /// Until 2026-09-29 it asked, and macOS's case-insensitive filesystem
    /// answered yes to `foo.wav` for a file named `Foo.WAV` while a
    /// case-sensitive one says no. See [`MediaLookup`]. It is blocking, like
    /// the listing it does; an async caller runs it on a blocking thread.
    pub fn find_under(root: &Path, subdir: &Path, stem: &str) -> MediaLookup {
        let walk = match DirWalk::of(root, subdir) {
            Ok(walk) => walk,
            Err(lookup) => return lookup,
        };
        let listing = match Listing::read(&walk.on_disk) {
            Ok(listing) => listing,
            Err(lookup) => return lookup,
        };
        // An exact name under ANY extension beats a near miss under an earlier
        // one, which is the answer a case-sensitive host gives.
        if let Some(name) = Self::candidates(stem).find(|name| listing.contains(OsStr::new(name))) {
            return walk.lookup(OsStr::new(&name), OsStr::new(&name));
        }
        let folded = listing.folded();
        Self::candidates(stem)
            .find_map(|name| {
                let on_disk = folded.get(&fold(OsStr::new(&name)))?;
                Some(walk.lookup(OsStr::new(&name), on_disk))
            })
            .unwrap_or(MediaLookup::Absent)
    }
}

/// A directory reached from a root by exact names: where the caller asked to
/// go and where that is on disk. The two differ exactly when some component
/// is spelled differently (letter case or Unicode form), so no separate
/// flag is kept.
pub(crate) struct DirWalk {
    /// The directory as the caller spelled it.
    pub(crate) wanted: PathBuf,
    /// The same directory as spelled on disk.
    pub(crate) on_disk: PathBuf,
}

impl DirWalk {
    /// Walks `subdir` under `root` one component at a time. `root` is taken as
    /// given (it is configuration, not a name being checked). A `..`, `.` or
    /// absolute component is not a name that can be misspelled, so it is
    /// [`MediaLookup::Absent`].
    pub(crate) fn of(root: &Path, subdir: &Path) -> Result<Self, MediaLookup> {
        let mut on_disk = root.to_path_buf();
        for component in subdir.components() {
            let std::path::Component::Normal(wanted) = component else {
                return Err(MediaLookup::Absent);
            };
            let listing = Listing::read(&on_disk)?;
            let actual = if listing.contains(wanted) {
                wanted.to_os_string()
            } else {
                listing
                    .folded()
                    .get(&fold(wanted))
                    .map(|actual| (*actual).clone())
                    .ok_or(MediaLookup::Absent)?
            };
            on_disk.push(actual);
        }
        Ok(Self {
            wanted: root.join(subdir),
            on_disk,
        })
    }

    /// Whether every component was spelled exactly.
    pub(crate) fn is_exact(&self) -> bool {
        self.wanted == self.on_disk
    }

    /// The answer for a file wanted as `name` and present as `on_disk`.
    fn lookup(&self, name: &OsStr, on_disk: &OsStr) -> MediaLookup {
        if self.is_exact() && name == on_disk {
            MediaLookup::Found(self.on_disk.join(on_disk))
        } else {
            MediaLookup::Missed(Missed::SpelledDifferently {
                wanted: self.wanted.join(name),
                on_disk: self.on_disk.join(on_disk),
            })
        }
    }
}

/// A directory's entry names, listed once and compared exactly.
struct Listing {
    names: HashSet<OsString>,
}

impl Listing {
    fn read(dir: &Path) -> Result<Self, MediaLookup> {
        let entries = std::fs::read_dir(dir).map_err(|error| match error.kind() {
            // A missing directory is an ordinary miss; any other failure to
            // list it leaves the answer unknown.
            std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory => MediaLookup::Absent,
            _ => Missed::unreadable(dir, &error),
        })?;
        let names = entries
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<Result<_, _>>()
            .map_err(|error| Missed::unreadable(dir, &error))?;
        Ok(Self { names })
    }

    fn contains(&self, name: &OsStr) -> bool {
        self.names.contains(name)
    }

    /// Lowercased name to on-disk spelling, built only after an exact miss.
    /// Two entries that differ only in case (possible on Linux) keep one;
    /// either is a correct witness that the wanted name is spelled otherwise.
    fn folded(&self) -> HashMap<String, &OsString> {
        self.names.iter().map(|name| (fold(name), name)).collect()
    }
}

/// The key two spellings share when they differ only in letter case or in
/// Unicode form (composed versus decomposed accents). macOS's filesystem
/// ignores both differences when it looks a name up; a lookup here must
/// notice them instead, so it compares exactly and uses this only to name
/// the near miss.
fn fold(name: &OsStr) -> String {
    use unicode_normalization::UnicodeNormalization;
    name.to_string_lossy()
        .nfc()
        .collect::<String>()
        .to_lowercase()
}

/// What a directory holds for one stem, with names compared exactly.
///
/// **Why a case-only near miss is its own variant.** A lookup that asks the
/// filesystem gets a different answer on macOS (case-insensitive by default)
/// than on a case-sensitive filesystem such as Linux's, so the same
/// transcript would align on a Mac and fail on a server that publishes its
/// media from Linux. The project wants names consistent rather than
/// tolerated on one platform. So every caller gets the case-sensitive
/// answer, and a file spelled differently (in letter case, or in Unicode form:
/// macOS also treats composed and decomposed accents as one name) is carried
/// as evidence for the error message, never used as the media.
///
/// **Explicit inputs are a different question.** [`MediaExtensions::matches`]
/// and [`MediaExtensions::is_known`] still recognise `recording.WAV` named on
/// the command line, whose extension never reaches CHAT. What must be exact is
/// a recording found BY NAME from a transcript, because the public server
/// builds that same name.
#[derive(Debug, PartialEq, Eq)]
pub enum MediaLookup {
    /// `{stem}.{extension}` present exactly, for the first extension in
    /// declaration order that is present.
    Found(PathBuf),
    /// Not there under any spelling, or the directory does not exist.
    Absent,
    /// Not usable, with evidence a caller reports rather than acts on.
    Missed(Missed),
}

/// Why a lookup found no usable media although something was there, or
/// could not tell. Kept apart from [`MediaLookup`] so a search that collects
/// these can hold nothing else.
#[derive(Debug, PartialEq, Eq)]
pub enum Missed {
    /// A file whose path differs only in letter case or Unicode form (in its
    /// stem, extension or a directory above it) is there.
    SpelledDifferently {
        /// The path as it would have to be spelled.
        wanted: PathBuf,
        /// The path as it is spelled on disk.
        on_disk: PathBuf,
    },
    /// The directory exists but could not be listed, so absence is unknown.
    Unreadable {
        /// The directory that could not be listed.
        dir: PathBuf,
        /// The operating system's reason.
        error: String,
    },
}

impl Missed {
    fn unreadable(dir: &Path, error: &impl std::fmt::Display) -> MediaLookup {
        MediaLookup::Missed(Self::Unreadable {
            dir: dir.to_path_buf(),
            error: error.to_string(),
        })
    }
}

impl std::fmt::Display for Missed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpelledDifferently { wanted, on_disk } => write!(
                f,
                "found '{}', whose name differs from '{}' only in letter case or \
                 Unicode form; media names must match exactly (macOS's filesystem \
                 forgives both differences, a Linux one forgives neither), so \
                 rename the file",
                on_disk.display(),
                wanted.display()
            ),
            Self::Unreadable { dir, error } => {
                write!(f, "could not list '{}': {error}", dir.display())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().expect("test path has a parent"))
            .expect("create test directory");
        std::fs::write(path, b"data").expect("create test file");
    }

    #[test]
    fn an_exact_name_is_found() {
        let dir = tempfile::tempdir().expect("tempdir");
        touch(&dir.path().join("session.mp3"));
        assert_eq!(
            MediaExtensions::find_under(dir.path(), Path::new(""), "session"),
            MediaLookup::Found(dir.path().join("session.mp3"))
        );
    }

    /// The case that motivated the type: on macOS the filesystem itself would
    /// say `session.wav` exists, and the answer must not depend on that.
    #[test]
    fn a_name_spelled_with_other_case_is_a_near_miss_not_the_media() {
        let dir = tempfile::tempdir().expect("tempdir");
        touch(&dir.path().join("Session.WAV"));
        assert_eq!(
            MediaExtensions::find_under(dir.path(), Path::new(""), "session"),
            MediaLookup::Missed(Missed::SpelledDifferently {
                wanted: dir.path().join("session.wav"),
                on_disk: dir.path().join("Session.WAV"),
            })
        );
    }

    /// An exact name under a later extension beats a near miss under an
    /// earlier one: the case-sensitive answer, which every host must give.
    #[test]
    fn an_exact_later_extension_wins_over_an_earlier_near_miss() {
        let dir = tempfile::tempdir().expect("tempdir");
        touch(&dir.path().join("session.WAV"));
        touch(&dir.path().join("session.mp3"));
        assert_eq!(
            MediaExtensions::find_under(dir.path(), Path::new(""), "session"),
            MediaLookup::Found(dir.path().join("session.mp3"))
        );
    }

    /// macOS stores a decomposed accent as written yet finds it under the
    /// composed spelling; the answer here must not depend on that either.
    #[test]
    fn a_name_in_another_unicode_form_is_a_near_miss_not_the_media() {
        let dir = tempfile::tempdir().expect("tempdir");
        let decomposed = "Schlu\u{308}ssel.mp3";
        touch(&dir.path().join(decomposed));
        assert_eq!(
            MediaExtensions::find_under(dir.path(), Path::new(""), "Schl\u{fc}ssel"),
            MediaLookup::Missed(Missed::SpelledDifferently {
                wanted: dir.path().join("Schl\u{fc}ssel.mp3"),
                on_disk: dir.path().join(decomposed),
            })
        );
    }

    #[test]
    fn a_directory_spelled_with_other_case_is_a_near_miss() {
        let root = tempfile::tempdir().expect("tempdir");
        touch(&root.path().join("Eng-NA/Brown/session.mp3"));
        assert_eq!(
            MediaExtensions::find_under(root.path(), Path::new("Eng-NA/brown"), "session"),
            MediaLookup::Missed(Missed::SpelledDifferently {
                wanted: root.path().join("Eng-NA/brown/session.mp3"),
                on_disk: root.path().join("Eng-NA/Brown/session.mp3"),
            })
        );
        assert_eq!(
            MediaExtensions::find_under(root.path(), Path::new("Eng-NA/Brown"), "session"),
            MediaLookup::Found(root.path().join("Eng-NA/Brown/session.mp3"))
        );
    }

    #[test]
    fn a_missing_directory_or_a_climbing_subdir_is_absent() {
        let root = tempfile::tempdir().expect("tempdir");
        touch(&root.path().join("a/session.mp3"));
        assert_eq!(
            MediaExtensions::find_under(root.path(), Path::new("b"), "session"),
            MediaLookup::Absent
        );
        assert_eq!(
            MediaExtensions::find_under(root.path(), Path::new("../a"), "session"),
            MediaLookup::Absent
        );
    }
}
