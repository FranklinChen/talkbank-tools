use std::path::{Path, PathBuf};

use crate::error::ServerError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedAsrMediaInput {
    pub original_audio_path: PathBuf,
    pub inference_audio_path: PathBuf,
    /// The `@Media` name, parsed here, before any recognition is paid for.
    pub media_name: Option<talkbank_model::model::MediaFilename>,
}

pub(crate) fn preserved_media_name_for_chat(
    original_audio_path: &Path,
    inference_audio_path: &Path,
) -> Option<String> {
    original_audio_path
        .file_stem()
        .or_else(|| original_audio_path.file_name())
        .or_else(|| inference_audio_path.file_stem())
        .or_else(|| inference_audio_path.file_name())
        .map(|s| s.to_string_lossy().into_owned())
}

pub(crate) async fn prepare_asr_media_input(
    original_audio_path: PathBuf,
    media_name: Option<String>,
    context_label: &str,
) -> Result<PreparedAsrMediaInput, ServerError> {
    // Reject an unusable media name HERE, before the ASR call is made.
    //
    // `@Media` delimits the filename with a comma, so a stem like
    // `interview,part2` cannot be written to a header. The failure used to
    // surface at CHAT serialization, AFTER transcription had been paid for,
    // which is the same late-failure pattern that made this defect expensive
    // in the field. Nothing here is recoverable by the pipeline: the operator
    // renames the file. So the cheapest possible moment to say so is now,
    // while the only thing spent is a path lookup.
    let media_name = media_name
        .map(|name| {
            talkbank_model::model::MediaFilename::parse(&name).map_err(|error| {
                ServerError::Validation(format!(
                    "Media name {name:?} for {context_label} cannot be written to an @Media \
                     header ({error}); rename the media file and resubmit"
                ))
            })
        })
        .transpose()?;

    let inference_audio_path = crate::ensure_wav::ensure_wav(&original_audio_path, None)
        .await
        .map_err(|error| {
            ServerError::Validation(format!(
                "Media conversion failed for {context_label}: {error}"
            ))
        })?;

    Ok(PreparedAsrMediaInput {
        original_audio_path,
        inference_audio_path,
        media_name,
    })
}

#[cfg(test)]
mod tests {
    use super::preserved_media_name_for_chat;

    /// An unusable media name is rejected BEFORE ASR runs, not at CHAT
    /// serialization afterwards.
    ///
    /// This is the whole point of the check's placement: the operator's fix is
    /// to rename the file, and learning that after paying for transcription is
    /// the expensive version of the same message.
    #[tokio::test]
    async fn an_unusable_media_name_is_rejected_before_asr() {
        // `tempfile`'s RAII guard, not a hand-rolled directory: the manual
        // version cleaned up AFTER the assertions, so a failing run leaked a
        // directory every time, which is the run you hit while iterating.
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("interview,part2.wav");
        std::fs::write(&audio, b"not really audio").expect("write");

        let err = super::prepare_asr_media_input(
            audio.clone(),
            Some("interview,part2".to_string()),
            "test job",
        )
        .await
        .expect_err("a comma-bearing media name must be rejected up front");

        let message = err.to_string();
        assert!(
            message.contains("@Media") && message.contains("rename"),
            "the error must say what is wrong and what to do: {message}"
        );
    }

    use std::path::PathBuf;

    #[test]
    fn preserved_media_name_prefers_original_basename() {
        let original = PathBuf::from("/corpus/interview.mp4");
        let inference = PathBuf::from("/cache/worker/interview.wav");
        assert_eq!(
            preserved_media_name_for_chat(&original, &inference).as_deref(),
            Some("interview")
        );
    }
}
