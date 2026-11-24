//! Rust-owned worker-protocol V2 prepared-artifact lookup and reads.
//!
//! **See also:** [`INTERFACE_MAP.md`](../../../INTERFACE_MAP.md) section "8. Worker Artifact Loading" for:
//! - Python callers: All V2 executors use these functions
//! - Design: File-backed artifacts with future shared-memory migration path.

use std::fs;
use std::path::Path;

use batchalign_types::worker_v2::{
    ArtifactRefV2, PreparedAudioEncodingV2, PreparedAudioRefV2, PreparedTextEncodingV2,
    PreparedTextRefV2, ProtocolErrorCodeV2,
};

use crate::worker_execute::ExecuteFailure;

/// Why a prepared artifact could not be produced, carrying the distinction the
/// protocol actually reports.
///
/// It lives here, beside the only code that can fail this way, because the
/// distinction is RECORDED where the failure is born rather than re-derived
/// later. It used to be flattened into a sentence and recovered downstream with
/// `message.contains("missing worker protocol V2 attachment")`: a value
/// standing in for a richer fact the producer already had. Rewording a
/// diagnostic in this file would have silently re-labelled every missing
/// attachment as unreadable, and nothing would have noticed, because the
/// coupling ran between a `format!` here and a substring in another file.
///
/// Deliberately has no `Display` and no message accessor. Either would make
/// `format!("{failure}")` type-check while discarding the category, which is
/// the loss this type exists to prevent.
pub enum ArtifactFailure {
    /// The request never carried the attachment the task asked for.
    Missing(String),
    /// The attachment was declared but could not be read or decoded.
    Unreadable(String),
    /// The attachment's DESCRIPTOR is self-inconsistent (zero channels, zero
    /// sample rate). RULED 2026-08-21: this is `invalid_payload` on the wire,
    /// through every door, because the descriptor is part of the request
    /// payload; a request lying about its artifact is a malformed request,
    /// not an unreadable artifact.
    Descriptor(DescriptorFailure),
}

impl ArtifactFailure {
    /// Convert into the wire outcome, code and message together, so neither can
    /// be taken without the other.
    pub(crate) fn into_code_and_message(self) -> (ProtocolErrorCodeV2, String) {
        match self {
            Self::Missing(message) => (ProtocolErrorCodeV2::MissingAttachment, message),
            Self::Unreadable(message) => (ProtocolErrorCodeV2::AttachmentUnreadable, message),
            Self::Descriptor(DescriptorFailure(message)) => {
                (ProtocolErrorCodeV2::InvalidPayload, message)
            }
        }
    }
}

/// A self-inconsistent prepared-artifact descriptor, born in the validators
/// below and nowhere else.
///
/// Four call sites used to classify this same failure three ways
/// (`attachment_unreadable`, `invalid_payload`, or a bare internal error)
/// depending only on which door it came through, because the validators
/// handed back prose (`Result<(), String>`) and each caller had to guess the
/// category. The category now travels with the value: every consumer
/// taxonomy's `From` impl maps it to `invalid_payload`, the ruled code.
pub struct DescriptorFailure(String);

fn validate_prepared_audio_attachment(
    attachment: &PreparedAudioRefV2,
) -> Result<(), DescriptorFailure> {
    if attachment.channels.0 == 0 {
        return Err(DescriptorFailure(format!(
            "prepared audio attachment {:?} must declare at least one channel",
            attachment.id.as_ref()
        )));
    }
    if attachment.sample_rate_hz.0 == 0 {
        return Err(DescriptorFailure(format!(
            "prepared audio attachment {:?} must declare positive sample_rate_hz",
            attachment.id.as_ref()
        )));
    }
    Ok(())
}

pub fn validate_attachment_descriptors(
    attachments: &[ArtifactRefV2],
) -> Result<(), DescriptorFailure> {
    for attachment in attachments {
        if let ArtifactRefV2::PreparedAudio(value) = attachment {
            validate_prepared_audio_attachment(value)?;
        }
    }
    Ok(())
}

impl From<DescriptorFailure> for ArtifactFailure {
    /// Lets the artifact loaders propagate a descriptor failure with `?`
    /// without re-deciding its category.
    fn from(failure: DescriptorFailure) -> Self {
        Self::Descriptor(failure)
    }
}

pub fn find_attachment<'a>(
    attachments: &'a [ArtifactRefV2],
    artifact_id: &str,
) -> Result<&'a ArtifactRefV2, ArtifactFailure> {
    attachments
        .iter()
        .find(|attachment| match attachment {
            ArtifactRefV2::PreparedAudio(value) => value.id.as_ref() == artifact_id,
            ArtifactRefV2::PreparedText(value) => value.id.as_ref() == artifact_id,
            ArtifactRefV2::InlineJson(value) => value.id.as_ref() == artifact_id,
        })
        .ok_or_else(|| {
            ArtifactFailure::Missing(format!(
                "missing worker protocol V2 attachment {artifact_id:?}"
            ))
        })
}

fn read_attachment_slice(
    path: &Path,
    byte_offset: usize,
    byte_len: usize,
) -> Result<Vec<u8>, ArtifactFailure> {
    let raw = fs::read(path).map_err(|error| ArtifactFailure::Unreadable(error.to_string()))?;
    slice_prepared_bytes(raw, byte_offset, byte_len, path)
}

/// Narrow an artifact buffer to the declared window, consuming it.
///
/// Split out from the read so the bounds logic is testable without a
/// filesystem; `read_attachment_slice` is then just "read, then narrow".
///
/// Takes the buffer BY VALUE and narrows it in place. The previous form ended
/// in `raw[byte_offset..end].to_vec()`, which duplicated the window into a
/// second allocation. Every producer in the Rust server writes
/// `byte_offset: 0` with `byte_len` equal to the whole file, so that window is
/// in practice always the entire artifact: the copy doubled peak memory for a
/// hundreds-of-megabyte audio buffer and bought nothing. `truncate` and `drain`
/// are in-place, so the common whole-file case now costs nothing at all.
fn slice_prepared_bytes(
    mut raw: Vec<u8>,
    byte_offset: usize,
    byte_len: usize,
    path: &Path,
) -> Result<Vec<u8>, ArtifactFailure> {
    let end = byte_offset.checked_add(byte_len).ok_or_else(|| {
        ArtifactFailure::Unreadable(format!(
            "prepared artifact slice overflow for {}",
            path.display()
        ))
    })?;
    if end > raw.len() {
        return Err(ArtifactFailure::Unreadable(format!(
            "prepared artifact slice {byte_offset}:{end} is outside {}",
            path.display()
        )));
    }
    raw.truncate(end);
    raw.drain(..byte_offset);
    Ok(raw)
}

pub fn load_prepared_text_json_impl(
    attachment: &PreparedTextRefV2,
) -> Result<String, ArtifactFailure> {
    if attachment.encoding != PreparedTextEncodingV2::Utf8Json {
        return Err(ArtifactFailure::Unreadable(format!(
            "unsupported prepared text encoding utf8_json for {:?}",
            attachment.id.as_ref()
        )));
    }
    let raw = read_attachment_slice(
        Path::new(attachment.path.as_ref()),
        attachment.byte_offset.0 as usize,
        attachment.byte_len.0 as usize,
    )?;
    String::from_utf8(raw).map_err(|error| ArtifactFailure::Unreadable(error.to_string()))
}

fn load_prepared_audio_bytes_impl(
    attachment: &PreparedAudioRefV2,
) -> Result<Vec<u8>, ArtifactFailure> {
    validate_prepared_audio_attachment(attachment)?;
    if attachment.encoding != PreparedAudioEncodingV2::PcmF32le {
        return Err(ArtifactFailure::Unreadable(format!(
            "unsupported prepared audio encoding pcm_f32le for {:?}",
            attachment.id.as_ref()
        )));
    }

    let raw = read_attachment_slice(
        Path::new(attachment.path.as_ref()),
        attachment.byte_offset.0 as usize,
        attachment.byte_len.0 as usize,
    )?;
    let expected_values = attachment.frame_count.0 as usize * attachment.channels.0 as usize;
    let expected_bytes = expected_values * std::mem::size_of::<f32>();
    if raw.len() != expected_bytes {
        return Err(ArtifactFailure::Unreadable(format!(
            "prepared audio artifact {:?} has {} bytes, expected {expected_bytes}",
            attachment.id.as_ref(),
            raw.len()
        )));
    }
    Ok(raw)
}

/// The concrete type name of an attachment, for the wrong-type diagnostics
/// below. One owner for what were two identical inline `match` arms.
const fn attachment_type_name(attachment: &ArtifactRefV2) -> &'static str {
    match attachment {
        ArtifactRefV2::PreparedAudio(_) => "PreparedAudioRefV2",
        ArtifactRefV2::PreparedText(_) => "PreparedTextRefV2",
        ArtifactRefV2::InlineJson(_) => "InlineJsonRefV2",
    }
}

fn require_prepared_audio_attachment<'a>(
    attachments: &'a [ArtifactRefV2],
    artifact_id: &str,
) -> Result<&'a PreparedAudioRefV2, ArtifactFailure> {
    match find_attachment(attachments, artifact_id)? {
        ArtifactRefV2::PreparedAudio(value) => {
            validate_prepared_audio_attachment(value)?;
            Ok(value)
        }
        // Present but of the wrong kind: the attachment IS there, so this is
        // unreadable rather than missing.
        other => Err(ArtifactFailure::Unreadable(format!(
            "worker protocol V2 attachment {artifact_id:?} had type {}, expected PreparedAudioRefV2",
            attachment_type_name(other)
        ))),
    }
}

/// Resolve a JSON-payload attachment (inline JSON, or a prepared UTF-8 JSON
/// text artifact) into its JSON text.
///
/// The Rust text executors' route to a prepared batch. For a prepared-text
/// artifact this is the FROZEN BYTES, unparsed, so a multi-megabyte batch
/// pays no parse here and the downstream consumer sees exactly what Rust
/// froze; only an inline value (already parsed by the envelope) is
/// serialized. Whether the text is valid JSON is the CONSUMER's question,
/// asked where the answer can carry the right category.
pub fn load_json_attachment_text(
    attachments: &[ArtifactRefV2],
    artifact_id: &str,
) -> Result<String, ArtifactFailure> {
    match find_attachment(attachments, artifact_id)? {
        ArtifactRefV2::InlineJson(value) => serde_json::to_string(&value.value).map_err(|error| {
            ArtifactFailure::Unreadable(format!(
                "worker protocol V2 attachment {artifact_id:?} could not be serialized: {error}"
            ))
        }),
        ArtifactRefV2::PreparedText(value) => load_prepared_text_json_impl(value),
        other => Err(ArtifactFailure::Unreadable(format!(
            "worker protocol V2 attachment {artifact_id:?} had type {}, expected JSON payload data",
            attachment_type_name(other)
        ))),
    }
}

pub fn require_prepared_text_attachment<'a>(
    attachments: &'a [ArtifactRefV2],
    artifact_id: &str,
) -> Result<&'a PreparedTextRefV2, ArtifactFailure> {
    match find_attachment(attachments, artifact_id)? {
        ArtifactRefV2::PreparedText(value) => Ok(value),
        other => Err(ArtifactFailure::Unreadable(format!(
            "worker protocol V2 attachment {artifact_id:?} had type {}, expected PreparedTextRefV2",
            attachment_type_name(other)
        ))),
    }
}

/// Decode a prepared-audio payload of little-endian `f32` samples.
///
/// One owner for what were three byte-identical copies in the ASR, FA and
/// media executors. Clippy found them (`chunks_exact_to_as_chunks`) only
/// because a lint moved; the duplication had been there the whole time, which
/// is the actual defect: three places to fix when the wire encoding changes.
///
/// The `as_chunks` remainder is discarded, and cannot be non-empty on any
/// current path: the only producer of these bytes is
/// `load_prepared_audio_bytes_impl` above, which refuses the payload unless its
/// length is exactly `frame_count * channels * size_of::<f32>()`. The
/// divisibility question is therefore already answered, once, at the boundary
/// where the artifact is read. Re-asking it here would be a second copy of a
/// check that already has an owner, which is the shape this file just finished
/// removing.
///
/// What is NOT structural is that the guarantee travels as a bare `Vec<u8>`, so
/// a future caller could hand this function bytes from somewhere else and
/// silently lose a partial sample. The cure is a newtype only that loader can
/// construct; the trigger to write it is a SECOND producer of prepared-audio
/// bytes, which does not exist today.
fn decode_f32le_audio(raw: Vec<u8>) -> Vec<f32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Proof that a prepared-audio attachment was found and checked to be mono.
///
/// Every audio executor used to open with the same three-step prologue
/// (require the attachment, check `channels != 1`, decode), written six times
/// with five wordings of one constraint. The only constructor is
/// [`require_mono_prepared_audio`], and the raw primitives it composes
/// (`require_prepared_audio_attachment`, `load_prepared_audio_bytes_impl`,
/// `decode_f32le_audio`) are PRIVATE to this module, so this type really is
/// the only route from an executor to decoded samples: the check travels with
/// the value, and the un-proved path has no signature to reach.
pub struct MonoPreparedAudio<'a>(&'a PreparedAudioRefV2);

impl<'a> MonoPreparedAudio<'a> {
    /// The underlying descriptor, for callers that need the sample rate or
    /// source path alongside the samples.
    pub(crate) const fn descriptor(&self) -> &'a PreparedAudioRefV2 {
        self.0
    }

    /// Decode the attachment's PCM f32le bytes into samples.
    pub(crate) fn samples(&self) -> Result<Vec<f32>, ExecuteFailure> {
        Ok(decode_f32le_audio(load_prepared_audio_bytes_impl(self.0)?))
    }
}

/// Resolve a prepared-audio attachment and require it to be mono.
///
/// Returns [`ExecuteFailure`] deliberately: the channel check is a
/// payload-shape complaint (`InvalidPayload` on the wire). Classifying it as
/// an [`ArtifactFailure`] would silently re-label it `AttachmentUnreadable`.
///
/// `task_label` keeps each executor's established message wording
/// ("<label> currently expects mono prepared audio").
pub fn require_mono_prepared_audio<'a>(
    attachments: &'a [ArtifactRefV2],
    artifact_id: &str,
    task_label: &str,
) -> Result<MonoPreparedAudio<'a>, ExecuteFailure> {
    let attachment = require_prepared_audio_attachment(attachments, artifact_id)?;
    if attachment.channels.0 != 1 {
        return Err(ExecuteFailure::InvalidPayload(format!(
            "{task_label} currently expects mono prepared audio"
        )));
    }
    Ok(MonoPreparedAudio(attachment))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ROUNDTRIP between the declared window and the bytes it selects. No
    /// type can express "these three integers agree with that buffer's length",
    /// so this stays a test. It exists because the in-place narrowing replaced
    /// a `to_vec()` copy, and an off-by-one there would hand a model audio that
    /// starts or ends in the wrong place rather than failing loudly.
    #[test]
    fn slicing_selects_the_declared_window() {
        let path = Path::new("artifact.pcm");
        let raw = || vec![0_u8, 1, 2, 3, 4, 5, 6, 7];

        // The whole-file case every current producer actually emits.
        assert_eq!(slice_prepared_bytes(raw(), 0, 8, path).ok(), Some(raw()));
        // A window inside the buffer, which nothing else exercises.
        assert_eq!(
            slice_prepared_bytes(raw(), 2, 3, path).ok(),
            Some(vec![2_u8, 3, 4])
        );
        // Degenerate but legal: an empty window.
        assert_eq!(slice_prepared_bytes(raw(), 4, 0, path).ok(), Some(vec![]));
    }

    /// A window reaching past the buffer is rejected rather than truncated
    /// silently, so a short or corrupt artifact cannot be handed to a model as
    /// though it were complete.
    #[test]
    fn slicing_refuses_a_window_outside_the_buffer() {
        let path = Path::new("artifact.pcm");
        let raw = vec![0_u8, 1, 2, 3];

        assert!(slice_prepared_bytes(raw.clone(), 2, 8, path).is_err());
        assert!(slice_prepared_bytes(raw.clone(), 5, 0, path).is_err());
        // Offset + length overflowing usize must not wrap into a valid window.
        assert!(slice_prepared_bytes(raw, usize::MAX, 2, path).is_err());
    }

    /// The category is a POLICY with a ruling behind it (2026-08-21): a
    /// self-inconsistent descriptor is `invalid_payload` on the wire, through
    /// every door. Four call sites used to classify this same failure three
    /// ways; the `DescriptorFailure` type carries the category from where the
    /// invalidity is detected, and this pins the ruled wire code.
    #[test]
    fn descriptor_failure_maps_to_invalid_payload() {
        let descriptor: PreparedAudioRefV2 = serde_json::from_str(
            r#"{
                "id": "audio-1",
                "path": "/nonexistent/audio.pcm",
                "encoding": "pcm_f32le",
                "channels": 0,
                "sample_rate_hz": 16000,
                "frame_count": 4,
                "byte_offset": 0,
                "byte_len": 16
            }"#,
        )
        .expect("descriptor JSON must parse; zero channels is a semantic, not schema, failure");

        let failure = validate_prepared_audio_attachment(&descriptor)
            .expect_err("zero channels must be refused");
        assert_eq!(
            ArtifactFailure::from(failure).into_code_and_message(),
            (
                ProtocolErrorCodeV2::InvalidPayload,
                r#"prepared audio attachment "audio-1" must declare at least one channel"#
                    .to_owned(),
            )
        );
    }
}
