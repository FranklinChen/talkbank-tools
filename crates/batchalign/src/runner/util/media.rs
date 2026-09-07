//! Media resolution, preflight validation, and output path handling.

use std::collections::HashMap;
#[cfg(test)]
use std::path::{Path, PathBuf};

use crate::store::PendingJobFile;
use batchalign_types::paths::ClientPath;

use crate::media::MediaExtensions;
use crate::media::probe::MediaProbe;
use tracing::warn;

/// Pre-validate media files before dispatch.
///
/// For non-CHAT files in paths_mode, checks:
/// 1. File exists on disk
/// 2. File is non-empty
/// 3. File extension is a known audio/video format
///
/// Returns the set of file indices that failed validation.
pub(in crate::runner) async fn preflight_validate_media(
    file_list: &[PendingJobFile],
    source_paths: &[ClientPath],
    paths_mode: bool,
) -> HashMap<usize, String> {
    if !paths_mode {
        return HashMap::new();
    }

    let mut failures = HashMap::new();

    for file in file_list {
        // Only validate non-CHAT (media) files
        if file.has_chat {
            continue;
        }

        let Some(client_path) = source_paths.get(file.file_index) else {
            failures.insert(file.file_index, "No source path for file index".to_string());
            continue;
        };

        // In paths_mode the client and server share a filesystem.
        let path = client_path.assume_shared_filesystem();

        // Check file exists and non-empty via metadata (one syscall)
        match tokio::fs::metadata(&path).await {
            Err(_) => {
                failures.insert(file.file_index, format!("File not found: {}", path));
                continue;
            }
            Ok(meta) if meta.len() == 0 => {
                failures.insert(file.file_index, format!("File is empty: {}", path));
                continue;
            }
            Ok(_) => {}
        }

        // Check known extension
        let ext = path
            .as_path()
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        if let Some(ref ext) = ext {
            if !MediaExtensions::is_known(ext) {
                failures.insert(
                    file.file_index,
                    format!("Unknown media format '.{ext}': {}", path),
                );
            }
        } else {
            failures.insert(file.file_index, format!("File has no extension: {}", path));
        }
    }

    failures
}

/// Compute audio identity for cache keying.
///
/// Returns an [`AudioIdentity`] built from the file's resolved path,
/// modification time, and size.
pub(in crate::runner) async fn compute_audio_identity(
    audio_path: &str,
) -> Option<crate::chat_ops::fa::AudioIdentity> {
    let canonical_path = tokio::fs::canonicalize(audio_path).await.ok()?;
    let audio_path = canonical_path.to_str()?;
    let meta = tokio::fs::metadata(&canonical_path).await.ok()?;
    let size = meta.len();
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(crate::chat_ops::fa::AudioIdentity::from_metadata(
        audio_path, mtime, size,
    ))
}

/// Get audio duration in milliseconds via ffprobe.
///
/// `None` is a DOWNGRADE the caller asks for, not an erasure: the duration is
/// optional for untimed-utterance estimation, so a failure is survivable, but
/// the reason is logged before it is dropped. This function used to return
/// `Option` with no logging, folding "ffprobe is not installed", "ffprobe was
/// killed", "ffprobe refused the file" and "ffprobe printed nonsense" into one
/// silent `None`, so an operator could not tell a missing dependency from a
/// corrupt recording.
pub(in crate::runner) async fn get_audio_duration_ms(audio_path: &str) -> Option<u64> {
    match MediaProbe::new(audio_path).duration().await {
        Ok(duration) => Some(duration.0),
        Err(error) => {
            warn!(
                audio = %audio_path,
                error = %error,
                "Could not determine audio duration; untimed utterances will be estimated without it"
            );
            None
        }
    }
}

/// Replace the filename in `output_path` with `result_filename`.
#[cfg(test)]
pub(in crate::runner) fn apply_result_filename(
    output_path: &Path,
    result_filename: &str,
) -> PathBuf {
    let result_name = Path::new(result_filename).file_name().unwrap_or_default();
    output_path
        .parent()
        .map(|p| p.join(result_name))
        .unwrap_or_else(|| result_name.into())
}
