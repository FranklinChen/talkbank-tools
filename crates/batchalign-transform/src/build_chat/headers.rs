use std::path::Path;
use talkbank_model::Location;

use talkbank_model::model::{
    Header, IDHeader, LanguageCodes, Line, MediaFilename, MediaHeader, MediaType,
    ParticipantDetails, ParticipantEntries, ParticipantEntry, ParticipantName, ParticipantRole,
    SpeakerCode,
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

    if let Some(media_header) = build_media_header(desc)? {
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

/// Builds the `@Media` header, if the description names any media.
///
/// FALLIBLE, and the `Result` is the point. The media name is external input
/// (a file stem the operator chose), and `@Media` cannot represent every
/// string: the comma separates the filename from the media type. Swallowing
/// that with `.ok()` produced a transcript with NO `@Media` line and no
/// diagnostic, so a job on `interview,part2.mp3` reported success and silently
/// lost its audio link. `Ok(None)` means "the description named no media";
/// an unusable name is an error, and the two must not share a signature.
fn build_media_header(desc: &TranscriptDescription) -> Result<Option<MediaHeader>, BuildChatError> {
    let Some(media_name) = desc.media_name.as_ref() else {
        return Ok(None);
    };
    let normalized_media_name = normalize_media_name(media_name);
    let media_type = match desc.media_type.as_deref() {
        Some("video") => MediaType::Video,
        Some("audio") | None => MediaType::Audio,
        other => {
            tracing::warn!(media_type = ?other, "unrecognized media_type, defaulting to audio");
            MediaType::Audio
        }
    };

    let filename = MediaFilename::parse(&normalized_media_name)?;
    let mut header = MediaHeader::new(filename, media_type);
    if let Some(status) = &desc.media_status {
        header = header.with_status(status.clone());
    }
    Ok(Some(header))
}

fn normalize_media_name(raw: &str) -> String {
    let candidate = Path::new(raw);
    candidate
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .or_else(|| candidate.file_name())
        .filter(|name| !name.is_empty())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| raw.to_string())
}
