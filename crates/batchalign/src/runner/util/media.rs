//! Media resolution, preflight validation, and output path handling.

use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

use crate::media::probe::{AudioDuration, MediaProbe};
use tracing::{debug, warn};

/// Best-effort duration used by standalone transcription to size its decode
/// budget. A refusal is logged and the caller uses its default budget.
pub async fn probe_audio_duration(audio_path: &Path) -> Option<AudioDuration> {
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
