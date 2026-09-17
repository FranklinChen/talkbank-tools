//! Where forced alignment looks for a transcript's recording.
//!
//! The corpus root and the media root can differ on one host, and differ again
//! between deployment modes, so resolution tries several ROOTS. The name is the
//! same in every one of them: the transcript's stem, which valid CHAT requires
//! to equal the `@Media` basename (chatter reports a mismatch as an error).
//!
//! **Looking IS recording, and that is the point of this module.** The rungs
//! used to be a chain of `if original_audio_path.is_none()` blocks with a
//! `searched.push(format!(...))` beside each, which is two things a human keeps
//! in step. They did not stay in step: one rung searched and recorded nothing,
//! so the failure message omitted the last place it had looked, while the
//! comment above the chain asserted that could not happen. Before that, the
//! message reconstructed the list from the same inputs that drove the search,
//! in a second if/else that could name only ONE place however many had run.
//!
//! [`MediaSearch::try_place`] is the only way to search, and it takes the place
//! by value, so a rung that looks somewhere has recorded it by construction.
//! There is no `push`, and nothing to forget.

use std::fmt;
use std::path::{Path, PathBuf};

use tracing::info;

use crate::media::MediaExtensions;
use crate::runner::DispatchHostContext;
use crate::store::RunnerJobSnapshot;

/// One place resolution looked.
///
/// A variant per rung rather than a rendered sentence. A `Vec<String>` of
/// prose can be printed and nothing else: no test can ask whether every rung
/// recorded itself, which is exactly the property that broke. These can be
/// compared, counted and asserted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SearchedPlace {
    /// An explicit `--media-dir` root, with the corpus subdir mapped under it.
    MediaDirRoot { root: PathBuf, subdir: String },
    /// Beside the transcript on the execution host, optionally preceded by
    /// `--media-dir` itself.
    TranscriptAdjacent {
        transcript: PathBuf,
        media_dir: Option<PathBuf>,
    },
    /// The client's shared source directory.
    ClientSourceDir {
        source_dir: String,
        transcript: PathBuf,
        media_dir: Option<PathBuf>,
    },
    /// A media mapping configured on this host.
    LocalMediaMapping {
        key: String,
        root: PathBuf,
        subdir: String,
    },
    /// A media mapping inferred from the client's source path.
    InferredMediaMapping { dir: PathBuf },
    /// One of the server's configured media roots.
    ServerMediaRoot { root: PathBuf },
}

struct SearchDirectory {
    root: PathBuf,
    subdir: PathBuf,
}

impl SearchDirectory {
    fn at(root: PathBuf) -> Self {
        Self {
            root,
            subdir: PathBuf::new(),
        }
    }
}

impl SearchedPlace {
    /// Every directory this place means, in the order to try them.
    ///
    /// The whole difference between the rungs lives here. Two of them are
    /// "`--media-dir` first, then beside the file", which is what
    /// `resolve_audio_for_chat_with_media_dir` did; the rest are a single
    /// root, optionally with a subdir joined under it.
    fn directories(&self) -> Vec<SearchDirectory> {
        // A subdir joined only when non-empty, matching what the previous
        // `find_media_in_root` did: an empty subdir means the root itself.
        fn under(root: &Path, subdir: &str) -> SearchDirectory {
            SearchDirectory {
                root: root.to_owned(),
                subdir: subdir.into(),
            }
        }
        // `--media-dir` is consulted before the file's own directory, which is
        // the order the previous helper used.
        fn beside(media_dir: Option<&PathBuf>, transcript: &Path) -> Vec<SearchDirectory> {
            media_dir
                .cloned()
                .into_iter()
                .chain(transcript.parent().map(Path::to_path_buf))
                .map(SearchDirectory::at)
                .collect()
        }
        match self {
            Self::MediaDirRoot { root, subdir } => vec![under(root, subdir)],
            Self::TranscriptAdjacent {
                transcript,
                media_dir,
            } => beside(media_dir.as_ref(), transcript),
            Self::ClientSourceDir {
                transcript,
                media_dir,
                ..
            } => beside(media_dir.as_ref(), transcript),
            Self::LocalMediaMapping { root, subdir, .. } => vec![under(root, subdir)],
            Self::InferredMediaMapping { dir } => vec![SearchDirectory::at(dir.clone())],
            Self::ServerMediaRoot { root } => vec![SearchDirectory::at(root.clone())],
        }
    }
}

impl fmt::Display for SearchedPlace {
    /// The operator-facing sentence, derived from the value rather than
    /// written at the call site, so it cannot describe a place that was not
    /// actually tried.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MediaDirRoot { root, subdir } => {
                write!(f, "--media-dir root '{}' subdir '{subdir}'", root.display())
            }
            Self::TranscriptAdjacent {
                transcript,
                media_dir: Some(dir),
            } => write!(
                f,
                "--media-dir '{}' and alongside '{}'",
                dir.display(),
                transcript.display()
            ),
            Self::TranscriptAdjacent {
                transcript,
                media_dir: None,
            } => write!(f, "alongside '{}'", transcript.display()),
            Self::ClientSourceDir { source_dir, .. } => {
                write!(f, "shared source directory '{source_dir}'")
            }
            Self::LocalMediaMapping { key, root, subdir } => write!(
                f,
                "local media mapping '{key}' root '{}' subdir '{subdir}'",
                root.display()
            ),
            Self::InferredMediaMapping { dir } => {
                write!(f, "inferred media mapping '{}'", dir.display())
            }
            Self::ServerMediaRoot { root } => {
                write!(f, "server media root '{}'", root.display())
            }
        }
    }
}

/// A search in progress: the stem being looked for, and everywhere looked.
pub(crate) struct MediaSearch {
    stem: String,
    searched: Vec<SearchedPlace>,
}

impl MediaSearch {
    pub(crate) fn for_stem(stem: impl Into<String>) -> Self {
        Self {
            stem: stem.into(),
            searched: Vec::new(),
        }
    }

    /// Search one place, recording it.
    ///
    /// The ONLY search primitive. It takes the place by value and records it
    /// before looking, so "every place searched appears in the report" is true
    /// by construction rather than by everyone remembering a `push`.
    ///
    /// A place already tried is skipped rather than searched again. That is
    /// only expressible because places are values: in paths_mode with no
    /// `--media-dir`, the adjacency rung and the final fallback are the same
    /// directory, and the old chain probed it twice under nine extensions.
    pub(crate) async fn try_place(
        &mut self,
        place: SearchedPlace,
    ) -> Result<Option<PathBuf>, UnresolvedMedia> {
        if self.searched.contains(&place) {
            return Ok(None);
        }
        let directories = place.directories();
        self.searched.push(place);
        for directory in directories {
            let stem = self.stem.clone();
            match crate::media::access::access(directory.root, move |root| {
                MediaExtensions::find_in_blocking(&root.path().join(directory.subdir), &stem)
            })
            .await
            {
                Ok(Some(found)) => return Ok(Some(found)),
                Ok(None) | Err(crate::media::access::MediaAccessError::Missing { .. }) => {}
                Err(error) => {
                    return Err(UnresolvedMedia {
                        message: error.to_string(),
                    });
                }
            }
        }
        Ok(None)
    }

    /// Everywhere this search looked.
    ///
    /// Test-only, and that is the point rather than an omission: production
    /// wants the rendered sentence, while a test wants to assert that a rung
    /// RECORDED itself. The old `Vec<String>` could not answer the second
    /// question at all, which is why one rung searched silently for as long as
    /// it did.
    #[cfg(test)]
    pub(crate) fn places(&self) -> &[SearchedPlace] {
        &self.searched
    }

    /// The rendered list, joined for an error message.
    pub(crate) fn describe(&self) -> String {
        self.searched
            .iter()
            .map(SearchedPlace::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path) {
        std::fs::create_dir_all(path.parent().expect("has a parent")).expect("mkdir");
        std::fs::write(path, b"media").expect("write");
    }

    #[tokio::test]
    async fn a_place_that_is_searched_is_recorded() {
        // The property the previous shape could not hold: one rung searched
        // and recorded nothing, and no test could notice, because the record
        // was a `Vec<String>` a caller had to remember to push to.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut search = MediaSearch::for_stem("absent");
        let place = SearchedPlace::ServerMediaRoot {
            root: dir.path().to_path_buf(),
        };
        assert_eq!(search.try_place(place.clone()).await.unwrap(), None);
        assert_eq!(search.places(), [place]);
    }

    #[tokio::test]
    async fn a_repeated_place_is_not_searched_twice() {
        let dir = tempfile::tempdir().expect("tempdir");
        let transcript = dir.path().join("talk.cha");
        write(&transcript);
        let place = SearchedPlace::TranscriptAdjacent {
            transcript: transcript.clone(),
            media_dir: None,
        };
        let mut search = MediaSearch::for_stem("talk");
        assert!(search.try_place(place.clone()).await.unwrap().is_none());
        assert!(search.try_place(place).await.unwrap().is_none());
        assert_eq!(search.places().len(), 1, "the second attempt was skipped");
    }

    #[tokio::test]
    async fn adjacency_finds_media_beside_the_transcript() {
        // Covers what `resolve_audio_for_chat` used to test before it became
        // dead: the commonest case, a recording sitting next to its transcript.
        let dir = tempfile::tempdir().expect("tempdir");
        write(&dir.path().join("talk.wav"));
        let mut search = MediaSearch::for_stem("talk");
        let found = search
            .try_place(SearchedPlace::TranscriptAdjacent {
                transcript: dir.path().join("talk.cha"),
                media_dir: None,
            })
            .await
            .unwrap();
        assert_eq!(found, Some(dir.path().join("talk.wav")));
    }

    #[tokio::test]
    async fn a_server_media_root_is_searched_flat() {
        // The media_roots fallback, likewise.
        let root = tempfile::tempdir().expect("tempdir");
        write(&root.path().join("ACWT01a.wav"));
        let mut search = MediaSearch::for_stem("ACWT01a");
        let found = search
            .try_place(SearchedPlace::ServerMediaRoot {
                root: root.path().to_path_buf(),
            })
            .await
            .unwrap();
        assert_eq!(found, Some(root.path().join("ACWT01a.wav")));
    }

    #[tokio::test]
    async fn media_dir_is_tried_before_the_transcript_directory() {
        let media = tempfile::tempdir().expect("tempdir");
        let corpus = tempfile::tempdir().expect("tempdir");
        write(&media.path().join("talk.mp3"));
        write(&corpus.path().join("talk.wav"));

        let mut search = MediaSearch::for_stem("talk");
        let found = search
            .try_place(SearchedPlace::TranscriptAdjacent {
                transcript: corpus.path().join("talk.cha"),
                media_dir: Some(media.path().to_path_buf()),
            })
            .await
            .unwrap();

        assert_eq!(found, Some(media.path().join("talk.mp3")));
    }

    #[tokio::test]
    async fn a_subdir_is_joined_under_the_root() {
        let root = tempfile::tempdir().expect("tempdir");
        write(&root.path().join("French/Newcastle/d01.mp3"));
        let mut search = MediaSearch::for_stem("d01");
        let found = search
            .try_place(SearchedPlace::MediaDirRoot {
                root: root.path().to_path_buf(),
                subdir: "French/Newcastle".to_owned(),
            })
            .await
            .unwrap();
        assert_eq!(found, Some(root.path().join("French/Newcastle/d01.mp3")));
    }

    #[test]
    fn every_place_renders_a_distinct_sentence() {
        // The message is derived from the value, so a place cannot describe
        // itself as somewhere it did not look.
        let places = [
            SearchedPlace::MediaDirRoot {
                root: PathBuf::from("/m"),
                subdir: "s".to_owned(),
            },
            SearchedPlace::TranscriptAdjacent {
                transcript: PathBuf::from("/c/talk.cha"),
                media_dir: None,
            },
            SearchedPlace::ServerMediaRoot {
                root: PathBuf::from("/r"),
            },
        ];
        let rendered: Vec<String> = places.iter().map(SearchedPlace::to_string).collect();
        assert_eq!(rendered.len(), 3);
        assert!(rendered.iter().all(|text| !text.is_empty()));
        assert_eq!(
            rendered[1], "alongside '/c/talk.cha'",
            "adjacency without --media-dir names only the transcript"
        );
    }
}

/// Return the parent directory of a filename as a `Path`, or `None` if the
/// filename has no directory component (i.e. it is a bare name like `foo.cha`).
///
/// Used when constructing the `infer_client` path for `infer_media_mapping`:
/// joining this onto `source_dir` makes the repo key visible when `source_dir`
/// is a top-level data root and the filename carries a repo-key prefix.
pub(super) fn filename_parent_dir(filename: &str) -> Option<&Path> {
    Path::new(filename)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
}

pub(super) fn media_search_subdir(filename: &str, media_subdir: &str) -> String {
    let file_parent = Path::new(filename)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    if file_parent.is_empty() {
        media_subdir.to_string()
    } else if media_subdir.is_empty() {
        file_parent
    } else {
        format!("{media_subdir}/{file_parent}")
    }
}

/// Media the six-rung search could not find, with the places it looked.
///
/// A typed refusal rather than a lifecycle side effect, so the same search can
/// serve a per-file task shell that owns its own failure reporting. The message
/// is built here because only the search knows what it looked at, and only the
/// transcript's own `@Media` header distinguishes "the recording is missing"
/// from "this transcript was published without one".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnresolvedMedia {
    /// The operator-facing explanation, naming every place searched.
    pub message: String,
}

/// Find the recording for one transcript, by the one search every audio-backed
/// command uses.
///
/// Extracted from `fa_pipeline::process_one_fa_file` on 2026-09-02, unchanged,
/// when `speaker-identify` needed the same six rungs. Copying them would have
/// been the third place this workspace states where a corpus keeps its media,
/// and the copies would have drifted the first time a rung was added.
pub(crate) async fn resolve_transcript_media(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    filename: &str,
    read_path: &Path,
    chat_text: &str,
    media_dir: Option<&str>,
) -> Result<PathBuf, UnresolvedMedia> {
    let media_mapping = job.filesystem.media_mapping.clone();
    let media_subdir = job.filesystem.media_subdir.clone();
    let source_dir = job.filesystem.source_dir.clone();
    // Resolve audio path.
    //
    // Everything is local to the execution host now, but the corpus root and
    // media root can still differ on that host, so what varies between these
    // is WHICH ROOT to look under. The name is the same in every one of them:
    // the transcript's stem, which valid CHAT requires to equal the @Media
    // basename (chatter reports a mismatch as an error). Search order:
    //   1. explicit --media-dir root replacement using the known corpus subdir
    //   2. paths_mode adjacency (or content-mode source_dir when shared)
    //   3. local media_mappings root replacement on the execution host
    //   4. media mapping INFERRED from the client's source path
    //   5. server media_roots fallback
    //   6. flat --media-dir / staged adjacency fallback
    //
    // This list said five for as long as it had six; derive it by reading the
    // `searched` pushes below, which are the same order and cannot drift.
    let stem = Path::new(filename)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let mapped_subdir = media_search_subdir(filename, media_subdir.as_str());
    let media_dir_path = media_dir.map(Path::new);

    // Every rung goes through `search.try_place`, which records the place and
    // then looks. There is no separate push to forget, which is what went
    // wrong when the two were kept in step by hand.
    let mut search = MediaSearch::for_stem(&stem);
    let mut original_audio_path = None;

    if let Some(root) = media_dir_path {
        original_audio_path = search
            .try_place(SearchedPlace::MediaDirRoot {
                root: root.to_path_buf(),
                subdir: mapped_subdir.clone(),
            })
            .await?;
        if original_audio_path.is_some() {
            info!(
                filename,
                media_dir = %root.display(),
                mapped_subdir = %mapped_subdir,
                "Resolved audio via --media-dir root mapping"
            );
        }
    }

    if original_audio_path.is_none() && job.filesystem.paths_mode {
        original_audio_path = search
            .try_place(SearchedPlace::TranscriptAdjacent {
                transcript: read_path.to_path_buf(),
                media_dir: None,
            })
            .await?;
    }

    if original_audio_path.is_none() && !source_dir.is_empty() {
        // paths_mode is active here, convert to a ServerPath for I/O.
        let server_source_dir = source_dir.assume_shared_filesystem();
        let source_path =
            server_source_dir.join(Path::new(filename).file_name().unwrap_or_default());
        original_audio_path = search
            .try_place(SearchedPlace::ClientSourceDir {
                source_dir: source_dir.to_string(),
                transcript: source_path.as_path().to_path_buf(),
                media_dir: media_dir_path.map(Path::to_path_buf),
            })
            .await?;
        if original_audio_path.is_some() {
            info!(
                filename,
                source_dir = %source_dir,
                "Resolved audio via client source directory"
            );
        }
    }

    if original_audio_path.is_none()
        && !media_mapping.is_empty()
        && let Some(root) = host.media_mapping_root(media_mapping.as_str())
    {
        original_audio_path = search
            .try_place(SearchedPlace::LocalMediaMapping {
                key: media_mapping.to_string(),
                root: root.as_path().to_path_buf(),
                subdir: mapped_subdir.clone(),
            })
            .await?;
        if original_audio_path.is_some() {
            info!(
                filename,
                media_mapping = %media_mapping,
                mapped_subdir = %mapped_subdir,
                "Resolved audio via local media mapping"
            );
        }
    }

    // Auto-detect a media mapping from the client's source path.
    // `infer_media_mapping()` is a pure string operation on the ClientPath: it
    // extracts the repo-name component and the repo-relative subdir with no
    // filesystem I/O, so it works for a local daemon (paths_mode) AND for
    // remote `--server` jobs whose client path is not on this filesystem.
    //
    // Do NOT join `repo_subdir` with `mapped_subdir` afterwards. `infer_client`
    // already embeds the file's parent, so `repo_subdir` is the complete
    // media-volume-relative path and joining again double-counts it.
    if original_audio_path.is_none() && media_mapping.is_empty() {
        // The inferred path must contain the repo key as a component. When
        // `source_dir` is a top-level data root (content mode / `--server`),
        // filenames carry the key as a prefix, so joining the filename's parent
        // exposes it; when `source_dir` already embeds a subdir (paths_mode),
        // the join is a no-op for bare names and still correct for nested ones.
        let infer_client: Option<batchalign_types::paths::ClientPath> = if !source_dir.is_empty() {
            Some(match filename_parent_dir(filename) {
                Some(parent) => source_dir.join(parent),
                None => source_dir.clone(),
            })
        } else if job.filesystem.paths_mode {
            // read_path is server-local; its parent contains the repo key.
            read_path
                .parent()
                .map(|p| batchalign_types::paths::ClientPath::new(p.to_string_lossy()))
        } else {
            None
        };

        if let Some(client_path) = infer_client
            && let Some((inferred_key, inferred_root, repo_subdir)) =
                batchalign_types::paths::infer_media_mapping(
                    &client_path,
                    &host.config().media_mappings,
                )
        {
            let search_dir = repo_subdir.resolve_on_server(&inferred_root);
            original_audio_path = search
                .try_place(SearchedPlace::InferredMediaMapping {
                    dir: search_dir.as_path().to_path_buf(),
                })
                .await?;
            if original_audio_path.is_some() {
                info!(
                    filename,
                    inferred_key = %inferred_key,
                    repo_subdir = %repo_subdir,
                    "Resolved audio via auto-detected media mapping"
                );
            }
        }
    }

    if original_audio_path.is_none() {
        for root in host.media_roots() {
            original_audio_path = search
                .try_place(SearchedPlace::ServerMediaRoot {
                    root: root.as_path().to_path_buf(),
                })
                .await?;
            if original_audio_path.is_some() {
                break;
            }
        }
    }

    if original_audio_path.is_none() {
        // Identical to the paths_mode rung above when no `--media-dir` was
        // given, and `try_place` skips a place it has already searched, so the
        // nine probes that pair used to repeat no longer happen.
        original_audio_path = search
            .try_place(SearchedPlace::TranscriptAdjacent {
                transcript: read_path.to_path_buf(),
                media_dir: media_dir_path.map(Path::to_path_buf),
            })
            .await?;
    }

    let original_audio_path = match original_audio_path {
        Some(p) => p,
        None => {
            // Asked only now, never before the search: a transcript declaring
            // its media absent may still have it on disk (2,618 do, corpus
            // wide), and those align today.
            // Always non-empty: the final fallback rung runs unconditionally
            // when nothing else matched, and records itself. It did not, which
            // made a branch here print "nowhere" for a file that had just been
            // searched under nine extensions alongside its own transcript.
            let places = search.describe();
            let err_msg = match crate::media::DeclaredMedia::read(chat_text) {
                crate::media::DeclaredMedia::Absent { as_written } => format!(
                    "{filename} declares its own media absent (@Media status \
                     '{as_written}') and none was found. This is a fact about the \
                     transcript, not a misconfigured run. Searched: {places}."
                ),
                crate::media::DeclaredMedia::Undeclared => format!(
                    "Cannot find audio file for {filename}, which declares no \
                     @Media header at all, so it may be a transcript published \
                     without a recording rather than one whose media went \
                     astray. Searched: {places}."
                ),
                crate::media::DeclaredMedia::Expected => {
                    format!("Cannot find audio file for {filename}. Searched: {places}.")
                }
            };
            return Err(UnresolvedMedia { message: err_msg });
        }
    };
    Ok(original_audio_path)
}
