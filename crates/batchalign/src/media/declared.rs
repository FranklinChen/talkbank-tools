//! What a transcript says about its own recording.
//!
//! Split out of `media.rs` for the same reason as `extensions`: one
//! concept per file, matching the rest of the module.

/// What a transcript SAYS about its own recording.
///
/// A sum type rather than a bool, because the three cases are three different
/// things for an operator to DO. When resolution fails, the message previously
/// said "Cannot find audio file" for all three, which is only true of one:
///
/// * [`Expected`](Self::Expected): the transcript names a recording and means
///   it. Nothing found means a staging or media-mapping fault, and the roots
///   are what to look at.
/// * [`Absent`](Self::Absent): the transcript declares the recording gone. The
///   pipeline found nothing because there is nothing, which is a fact about the
///   corpus and not a fault of the run.
/// * [`Undeclared`](Self::Undeclared): no `@Media` header at all, so there was
///   never a declared name to look for and stem matching was the only route.
///
/// **Read only AFTER the search has failed, never before it**, and the reason
/// is not the one first written here. That said 2,618 transcripts "declare
/// their media `missing` or `unlinked` while a file of that name is on disk",
/// which conflated the two: `unlinked` never declared anything absent, and
/// measured over the same corpus, ZERO of the 3,018 transcripts declaring
/// `missing` have a recording on disk. Short-circuiting would cost nothing
/// today.
///
/// It stays after the search anyway, because a declaration is a hand-written
/// CLAIM about a file somewhere else, and the run has the actual filesystem in
/// front of it. Trusting the claim over the evidence would make a stale header
/// silently skip a file that can be aligned, and the ordering costs nothing:
/// this is reached only once the search has already failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredMedia {
    /// `@Media: name, audio|video` with no status qualifier.
    Expected,
    /// `@Media: name, missing`, or a `missing` STATUS. Only those two.
    Absent {
        /// The status token as the transcript wrote it, for the message.
        as_written: String,
    },
    /// No `@Media` header.
    Undeclared,
}

impl DeclaredMedia {
    /// Inspect the already-produced header structure, without a second parse.
    /// This classifies a declaration, not the presence of an actual recording.
    pub fn from_document(file: &crate::chat_ops::ChatFile) -> Self {
        use talkbank_model::model::Header;
        file.lines
            .iter()
            .take_while(|line| line.is_header())
            .filter_map(|line| line.as_header())
            .find_map(|header| match header {
                Header::Media(media) => Some(Self::from_media(media)),
                _ => None,
            })
            .unwrap_or(Self::Undeclared)
    }

    fn from_media(media: &talkbank_model::model::MediaHeader) -> Self {
        use talkbank_model::model::header::{MediaStatus, MediaType};
        match (&media.media_type, media.status.as_ref()) {
            (MediaType::Missing, _) => Self::Absent {
                as_written: MediaType::Missing.as_str().to_owned(),
            },
            (_, Some(MediaStatus::Missing)) => Self::Absent {
                as_written: MediaStatus::Missing.as_str().to_owned(),
            },
            (_, Some(MediaStatus::Unlinked | MediaStatus::Notrans))
            | (_, Some(MediaStatus::Unsupported(_)))
            | (_, None) => Self::Expected,
        }
    }

    /// Reads the declaration from a transcript's text.
    ///
    /// Only the HEADER BLOCK is parsed: headers precede the first `*` speaker
    /// tier, and building a whole `ChatFile` to read one header is what made a
    /// corpus-wide pass of this question take minutes instead of seconds
    /// elsewhere in the workspace. Each candidate line goes through chatter's
    /// `parse_header_fragment`, so what an `@Media` payload MEANS stays
    /// chatter's decision rather than becoming a split on a comma here.
    pub fn read(chat_text: &str) -> Self {
        use talkbank_model::errors::NullErrorSink;
        use talkbank_model::model::header::Header;

        // `crate::chat_parser()` rather than a private construction: it is what
        // the other 22 parse sites in this crate use, and it goes through the
        // `batchalign_transform` facade. The private version also had to handle
        // its own failure, and did so by returning `Undeclared`, which reports
        // "this transcript has no @Media header" as a fact about the transcript
        // when the truth is that the tool could not tell. A fabricated value in
        // exactly the sense the workspace bans.
        let parser = crate::chat_parser();
        for line in chat_text.lines() {
            if line.starts_with('*') {
                break;
            }
            if !line.starts_with('@') {
                continue;
            }
            let Some(Header::Media(media)) = parser
                .parse_header_fragment(line, 0, &NullErrorSink)
                .into_option()
            else {
                continue;
            };
            // ONLY `missing` means absent. This was `status.is_some()`, which
            // is wrong in the direction that matters most on this exact path:
            // chatter documents `unlinked` as "the media file EXISTS but
            // utterances have not been aligned to timestamps yet ... the NORMAL
            // state for a transcript before forced alignment ... Processing
            // commands SHOULD resolve and use the media, the whole point of
            // `align` is to create the links that are currently absent."
            // Telling an operator that such a transcript declares its media
            // absent is the reverse of true, and on the align path it steers
            // them away from the staging fault that actually caused the miss.
            // `notrans` says the media exists as well.
            //
            // Matched exhaustively rather than with a catch-all, so a status
            // added to chatter has to be classified here instead of defaulting
            // to "expected".
            return Self::from_media(&media);
        }
        Self::Undeclared
    }
}

#[cfg(test)]
mod declared_media_tests {
    use super::DeclaredMedia;

    /// Wraps a `@Media` line in the minimum around it that parses.
    fn transcript(media_line: &str) -> String {
        format!("@UTF8\n@Begin\n@Languages:\teng\n{media_line}\n*CHI:\thi .\n@End\n")
    }

    #[test]
    fn a_plain_declaration_expects_media() {
        assert_eq!(
            DeclaredMedia::read(&transcript("@Media:\tfoo, audio")),
            DeclaredMedia::Expected
        );
    }

    #[test]
    fn produced_header_structure_uses_the_same_declaration_policy() {
        for (payload, expected) in [
            ("audio, unlinked", DeclaredMedia::Expected),
            ("video, notrans", DeclaredMedia::Expected),
            (
                "audio, missing",
                DeclaredMedia::Absent {
                    as_written: "missing".into(),
                },
            ),
            (
                "missing",
                DeclaredMedia::Absent {
                    as_written: "missing".into(),
                },
            ),
        ] {
            let source = format!(
                "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\tsample, {payload}\n*CHI:\thello .\n@End\n"
            );
            let admitted = crate::pipeline::text_infer::admit_retained_text(
                &crate::chat_parser(),
                &source,
                talkbank_model::model::TranscriptName::Anonymous,
            )
            .expect("the declaration fixture is completely valid");
            assert_eq!(DeclaredMedia::from_document(admitted.document()), expected);
            assert_eq!(DeclaredMedia::read(&source), expected);
        }
    }

    #[test]
    fn unlinked_expects_media_because_the_recording_exists() {
        // The case this type got wrong, and the reason it now has tests at all.
        // chatter documents `unlinked` as "the media file EXISTS but utterances
        // have not been aligned yet ... the NORMAL state for a transcript before
        // forced alignment", which is precisely the state `align` is run to
        // change. Classifying it as absent told operators the opposite, on the
        // one command where it matters most.
        assert_eq!(
            DeclaredMedia::read(&transcript("@Media:\tfoo, audio, unlinked")),
            DeclaredMedia::Expected
        );
    }

    #[test]
    fn notrans_expects_media_too() {
        // "the media exists but no transcription has been done."
        assert_eq!(
            DeclaredMedia::read(&transcript("@Media:\tfoo, audio, notrans")),
            DeclaredMedia::Expected
        );
    }

    #[test]
    fn a_missing_status_is_the_only_absent_status() {
        assert_eq!(
            DeclaredMedia::read(&transcript("@Media:\tfoo, audio, missing")),
            DeclaredMedia::Absent {
                as_written: "missing".to_owned()
            }
        );
    }

    #[test]
    fn a_missing_media_type_is_absent() {
        assert_eq!(
            DeclaredMedia::read(&transcript("@Media:\tfoo, missing")),
            DeclaredMedia::Absent {
                as_written: "missing".to_owned()
            }
        );
    }

    #[test]
    fn no_header_is_undeclared() {
        assert_eq!(
            DeclaredMedia::read("@UTF8\n@Begin\n*CHI:\thi .\n@End\n"),
            DeclaredMedia::Undeclared
        );
    }

    #[test]
    fn the_scan_stops_at_the_first_speaker_tier() {
        // The header block is the bound. A `@Media`-looking line in the body is
        // not a declaration, and reading the whole file to find out is what this
        // function exists to avoid.
        let text = "@UTF8\n@Begin\n*CHI:\thi .\n@Media:\tfoo, audio\n@End\n";
        assert_eq!(DeclaredMedia::read(text), DeclaredMedia::Undeclared);
    }
}
