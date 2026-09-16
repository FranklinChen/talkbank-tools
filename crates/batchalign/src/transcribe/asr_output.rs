//! ASR response conversion and speaker-track admission.

use batchalign_transform::asr_postprocess::{
    AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs, SpeakerIndex,
};
use tracing::warn;

use super::types::AsrResponse;

/// Convert flat ASR tokens (with speaker labels) into speaker-grouped monologues.
///
/// Groups consecutive tokens by speaker. Adjacent tokens with the same speaker
/// are combined into a single monologue. Speaker changes create new monologues.
pub(crate) fn convert_asr_response(response: &AsrResponse) -> AsrOutput {
    if let Some(monologues) = &response.source_monologues {
        return AsrOutput {
            monologues: monologues.clone(),
        };
    }

    if response.tokens.is_empty() {
        return AsrOutput {
            monologues: Vec::new(),
        };
    }

    let mut monologues: Vec<AsrMonologue> = Vec::new();
    let mut current_speaker: Option<SpeakerIndex> = None;
    let mut current_elements: Vec<AsrElement> = Vec::new();

    for token in &response.tokens {
        let speaker_idx = admit_token_speaker(token.speaker.as_deref(), &token.text);

        if current_speaker != Some(speaker_idx) {
            // Flush previous monologue
            if let Some(spk) = current_speaker
                && !current_elements.is_empty()
            {
                monologues.push(AsrMonologue {
                    speaker: spk,
                    elements: std::mem::take(&mut current_elements),
                });
            }
            current_speaker = Some(speaker_idx);
        }

        current_elements.push(AsrElement {
            value: AsrRawText::new(token.text.clone()),
            ts: AsrTimestampSecs::from(token.start_s.map(|s| s.0)),
            end_ts: AsrTimestampSecs::from(token.end_s.map(|s| s.0)),
            kind: AsrElementKind::Text,
        });
    }

    // Flush last monologue
    if let Some(spk) = current_speaker
        && !current_elements.is_empty()
    {
        monologues.push(AsrMonologue {
            speaker: spk,
            elements: current_elements,
        });
    }

    AsrOutput { monologues }
}

/// The track one flat ASR token belongs to.
///
/// This is the LEGACY admission, and it is the only one left: the live worker
/// path carries a typed `SpeakerAttributionV2` and admits it in
/// `worker::asr_result_v2`. What still arrives here as a bare
/// `Option<String>` is Rev's own projection and replayed
/// `_asr_response.json` evidence, both of which number their speakers, and
/// replayed evidence carrying speaker `"0"` must keep working exactly as it
/// did.
///
/// Absence means the engine attributed nobody, which is the single track of an
/// undiarized recording. A label that is not a speaker NUMBER is reported and
/// takes the first track; it is no longer put through `rsplit('_')`, which
/// silently merged distinct labels (`A_0` and `B_0` became one speaker) and
/// read a suffix out of labels that never had that shape.
pub(super) fn admit_token_speaker(label: Option<&str>, token_text: &str) -> SpeakerIndex {
    let Some(label) = label else {
        return SpeakerIndex(0);
    };
    let trimmed = label.trim();
    match trimmed.parse::<usize>() {
        Ok(speaker) => SpeakerIndex(speaker),
        Err(_) => {
            warn!(
                speaker = %label,
                token = %token_text,
                "ASR token speaker label is not a speaker number; it cannot number a \
                 transcript tier, so this token takes the first track"
            );
            SpeakerIndex(0)
        }
    }
}
