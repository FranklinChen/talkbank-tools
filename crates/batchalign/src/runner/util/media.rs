//! Media resolution, preflight validation, and output path handling.

use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

use crate::media::probe::{AudioDuration, MediaProbe};
use tracing::{debug, warn};

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

/// How long the audio at `audio_path` runs, as the media probe establishes it.
///
/// `None` is a DOWNGRADE the caller asks for, not an erasure: the reason is
/// logged before it is dropped, and FA, which cannot run without a recording
/// bound, probes again and reports the failure as the file's error. This
/// function used to return `Option` with no logging, folding "ffprobe is not
/// installed", "ffprobe was killed", "ffprobe refused the file" and "ffprobe
/// printed nonsense" into one silent `None`, so an operator could not tell a
/// missing dependency from a corrupt recording.
///
/// Returns the probe's [`AudioDuration`] rather than its milliseconds, so every
/// bound built from it is the probe's measurement (see `media::probe`).
///
/// The one place that measures an audio file's length AND logs the failure.
/// Crate-visible so the transcribe pipeline's ASR stage, which needs the length
/// to size a provider request's decode budget, reuses it instead of keeping a
/// second probe-and-log of its own.
pub(crate) async fn probe_audio_duration(audio_path: &Path) -> Option<AudioDuration> {
    match MediaProbe::new(audio_path).duration().await {
        Ok(duration) => {
            debug!(
                audio = %audio_path.display(),
                length_ms = duration.length().0,
                basis = ?duration.basis(),
                "Probed audio duration"
            );
            Some(duration)
        }
        Err(error) => {
            warn!(
                audio = %audio_path.display(),
                error = %error,
                "Could not determine audio duration; steps that need it fall back to their defaults"
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
