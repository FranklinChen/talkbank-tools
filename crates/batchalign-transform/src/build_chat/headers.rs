use talkbank_model::Location;

use talkbank_model::model::{
    Header, IDHeader, LanguageCodes, Line, MediaHeader, MediaType, ParticipantDetails,
    ParticipantEntries, ParticipantEntry, ParticipantName, ParticipantRole, SpeakerCode,
};

use super::BuildChatError;
use super::TranscriptDescription;

pub(super) fn build_header_lines(
    desc: &TranscriptDescription,
) -> Result<Vec<Line>, BuildChatError> {
    let participant_entries = build_participant_entries(desc);
    let id_headers = build_id_headers(desc);
    let mut lines: Vec<Line> = vec![
        Line::header(Location::Unlocated, Header::Utf8),
        Line::header(Location::Unlocated, Header::Begin),
        Line::header(
            Location::Unlocated,
            Header::Languages {
                codes: LanguageCodes::new(desc.langs.all()),
            },
        ),
        Line::header(
            Location::Unlocated,
            Header::Participants {
                entries: ParticipantEntries::new(participant_entries),
            },
        ),
    ];

    for id in id_headers {
        lines.push(Line::header(Location::Unlocated, Header::ID(id)));
    }

    if let Some(media_header) = build_media_header(desc) {
        lines.push(Line::header(
            Location::Unlocated,
            Header::Media(media_header),
        ));
    }

    Ok(lines)
}

fn build_participant_entries(desc: &TranscriptDescription) -> Vec<ParticipantEntry> {
    desc.participants
        .iter()
        .map(|participant| ParticipantEntry {
            speaker_code: SpeakerCode::new(participant.id.as_str()),
            details: ParticipantDetails::Complete {
                name: participant.name.as_ref().map(ParticipantName::new),
                role: ParticipantRole::new(participant.role.as_str()),
            },
        })
        .collect()
}

fn build_id_headers(desc: &TranscriptDescription) -> Vec<IDHeader> {
    // @ID headers carry the primary language.
    let lang_code = desc.langs.primary();

    desc.participants
        .iter()
        .map(|participant| {
            let corpus = if participant.corpus.is_empty() {
                "corpus_name"
            } else {
                participant.corpus.as_str()
            };
            IDHeader::new(
                lang_code.clone(),
                participant.id.as_str(),
                participant.role.as_str(),
            )
            .with_corpus(corpus)
        })
        .collect()
}

/// Builds the `@Media` header, if the description names any media. The name
/// arrives parsed ([`MediaFilename`], made where the name is born), so an
/// unwritable name was refused before recognition, not here after it.
fn build_media_header(desc: &TranscriptDescription) -> Option<MediaHeader> {
    let filename = desc.media_name.clone()?;
    let media_type = match desc.media_type.as_deref() {
        Some("video") => MediaType::Video,
        Some("audio") | None => MediaType::Audio,
        other => {
            tracing::warn!(media_type = ?other, "unrecognized media_type, defaulting to audio");
            MediaType::Audio
        }
    };
    let mut header = MediaHeader::new(filename, media_type);
    if let Some(status) = &desc.media_status {
        header = header.with_status(status.clone());
    }
    Some(header)
}
