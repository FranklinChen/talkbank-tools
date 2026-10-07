//! Timed-CHAT diarization: admitted source, explicit identities, checked output.

use std::collections::BTreeMap;

use talkbank_model::model::{ChatFileLines, Header, Line};
use talkbank_model::validation::ValidChatFile;
use talkbank_model::{SpeakerCode, Utterance};
use talkbank_transform::rediarize::{DiarizationTimeline, DiarizationTurn, TimeSpanMs};
use talkbank_transform::utterance_split::{
    WordSpeakerPartition, WordSpeakerSource, WordSpeakerSplitRefusal,
};

use crate::api::ReleasedCommand;
use crate::chat_ops::speaker::DiarizationLabelCoordinates;
use crate::options::SpeakerTrackMapping;
use crate::pipeline::post_validate::{PostValidated, PostValidationFailure};
use crate::scheduling::FailureCategory;
use crate::types::worker_v2::SpeakerSegmentV2;

/// Refusals retain input eligibility versus producer failure, not diagnostic prose.
#[derive(Debug, thiserror::Error)]
pub(super) enum DiarizeChatRefusal {
    #[error("--speaker-map PAR{track} names undeclared participant {participant}")]
    UndeclaredParticipant {
        track: usize,
        participant: SpeakerCode,
    },
    #[error("input utterance {ordinal} cannot be diarized: {source}")]
    WordTiming {
        ordinal: usize,
        source: WordSpeakerSplitRefusal,
    },
    #[error("observed anonymous track PAR{0} has no explicit --speaker-map entry")]
    UnmappedTrack(usize),
    #[error("speaker producer omitted its own label coordinate")]
    ProducerCoordinate,
    #[error("speaker producer supplied an unusable timeline interval")]
    ProducerInterval,
    #[error(transparent)]
    Output(#[from] PostValidationFailure),
}

impl DiarizeChatRefusal {
    pub(super) fn category(&self) -> FailureCategory {
        match self {
            Self::UndeclaredParticipant { .. } | Self::UnmappedTrack(_) => {
                FailureCategory::Validation
            }
            Self::WordTiming {
                source: WordSpeakerSplitRefusal::ProducerChildCount { .. },
                ..
            }
            | Self::ProducerCoordinate
            | Self::ProducerInterval
            | Self::Output(_) => FailureCategory::System,
            Self::WordTiming { .. } => FailureCategory::Validation,
        }
    }
}

enum UtteranceSource<'source> {
    /// Admitted word timing, beside the utterance it was admitted from: a
    /// partition that relabels the whole turn relabels this utterance.
    Timed {
        ordinal: usize,
        utterance: &'source Utterance,
        source: WordSpeakerSource<'source>,
    },
    /// No lexical timing slots: preserve the original identity, never guess one.
    NonLexical(&'source Utterance),
}

/// A source-bound preflight retained across inference. No independently supplied
/// document, timing vector or participant table can be substituted at application.
pub(super) struct MappedDiarizeSource<'source> {
    transcript: &'source ValidChatFile,
    mapping: BTreeMap<usize, SpeakerCode>,
    utterances: Vec<UtteranceSource<'source>>,
}

impl<'source> MappedDiarizeSource<'source> {
    pub(super) fn admit(
        transcript: &'source ValidChatFile,
        mapping: &SpeakerTrackMapping,
    ) -> Result<Self, DiarizeChatRefusal> {
        let document = transcript.document();
        let mut declared_mapping = BTreeMap::new();
        for (track, participant) in mapping.entries() {
            if !document
                .participants
                .values()
                .any(|declared| &declared.code == participant)
            {
                return Err(DiarizeChatRefusal::UndeclaredParticipant {
                    track,
                    participant: participant.clone(),
                });
            }
            declared_mapping.insert(track, participant.clone());
        }
        let mut utterances = Vec::new();
        for (ordinal, source) in document.utterances().enumerate() {
            utterances.push(match WordSpeakerSource::admit(source) {
                Ok(timed) => UtteranceSource::Timed {
                    ordinal: ordinal + 1,
                    utterance: source,
                    source: timed,
                },
                Err(WordSpeakerSplitRefusal::EmptyWordTiming) => {
                    UtteranceSource::NonLexical(source)
                }
                Err(source) => {
                    return Err(DiarizeChatRefusal::WordTiming {
                        ordinal: ordinal + 1,
                        source,
                    });
                }
            });
        }
        Ok(Self {
            transcript,
            mapping: declared_mapping,
            utterances,
        })
    }

    pub(super) fn needs_inference(&self) -> bool {
        self.utterances
            .iter()
            .any(|source| matches!(source, UtteranceSource::Timed { .. }))
    }

    pub(super) fn document(&self) -> &talkbank_model::ChatFile {
        self.transcript.document()
    }

    /// Model evidence is translated directly into the shared typed timeline,
    /// using the same anonymous coordinates as the turns artifact, not via JSON.
    pub(super) fn apply(
        self,
        segments: &[SpeakerSegmentV2],
    ) -> Result<PostValidated, DiarizeChatRefusal> {
        let coordinates = DiarizationLabelCoordinates::from_labels(
            segments.iter().map(|segment| segment.speaker.as_str()),
        );
        let mut turns = Vec::with_capacity(segments.len());
        for segment in segments {
            let track = coordinates
                .index_for(segment.speaker.as_str())
                .ok_or(DiarizeChatRefusal::ProducerCoordinate)?
                .as_usize();
            let participant = self
                .mapping
                .get(&track)
                .ok_or(DiarizeChatRefusal::UnmappedTrack(track))?;
            let span = TimeSpanMs::new(
                segment.interval.start_millis(),
                segment.interval.end_millis(),
            )
            .map_err(|_| DiarizeChatRefusal::ProducerInterval)?;
            turns.push(DiarizationTurn {
                track: participant.clone(),
                span,
            });
        }
        let timeline = DiarizationTimeline::new(turns);
        let mut replacements = Vec::with_capacity(self.utterances.len());
        let mut losses = Vec::new();
        let mut retained_nonlexical = 0usize;
        for utterance in self.utterances {
            replacements.push(match utterance {
                UtteranceSource::NonLexical(source) => {
                    retained_nonlexical += 1;
                    vec![source.clone()]
                }
                UtteranceSource::Timed {
                    ordinal,
                    utterance,
                    source,
                } => {
                    let parts = source
                        .bind_timeline(&timeline)
                        .and_then(|plan| plan.execute())
                        .map_err(|source| DiarizeChatRefusal::WordTiming { ordinal, source })?
                        .into_parts();
                    match parts.partition {
                        // One track owns every word: the turn keeps its content
                        // and every dependent tier, under that speaker.
                        WordSpeakerPartition::Relabeled { speaker } => {
                            let mut relabeled = utterance.clone();
                            relabeled.main.speaker = speaker;
                            vec![relabeled]
                        }
                        WordSpeakerPartition::Split(split) => {
                            let (children, invalidated) = split.into_parts();
                            losses.extend(invalidated.into_iter().map(|loss| {
                                format!(
                                    "input utterance {ordinal} %{} ({:?})",
                                    loss.tier().kind(),
                                    loss.reason(),
                                )
                            }));
                            children
                        }
                    }
                }
            });
        }
        // Rebuild only after every source-bound split has succeeded. Headers and
        // participant facts are cloned unchanged; anonymous tracks add no facts.
        let mut output = self.transcript.document().clone();
        let mut replacements = replacements.into_iter();
        let mut lines = Vec::new();
        for line in output.lines.take() {
            match line {
                Line::Utterance(_) => {
                    let children = replacements
                        .next()
                        .ok_or(DiarizeChatRefusal::ProducerCoordinate)?;
                    lines.extend(
                        children
                            .into_iter()
                            .map(|child| Line::Utterance(Box::new(child))),
                    );
                }
                header => lines.push(header),
            }
        }
        if replacements.next().is_some() {
            return Err(DiarizeChatRefusal::ProducerCoordinate);
        }
        output.lines = ChatFileLines::new(lines);
        if !losses.is_empty() || retained_nonlexical > 0 {
            let text = format!(
                "Speaker diarization: retained {retained_nonlexical} nonlexical utterances with original speakers; invalidated dependent tiers: {}. Regenerate analysis if required.",
                if losses.is_empty() {
                    "none".to_owned()
                } else {
                    losses.join("; ")
                },
            );
            let position = crate::provenance::insert_pos_after_constant_headers(&output);
            output.lines.insert(
                position,
                Line::header(Header::Comment {
                    content: crate::chat_ops::BulletContent::from_text(text),
                }),
            );
        }
        Ok(PostValidated::gate_owned(output, ReleasedCommand::Diarize)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::model::TranscriptName;

    const SOURCE: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n\
        @ID:\teng|sample|CHI|2;00.||||Child|||\n@ID:\teng|sample|MOT|||||Mother|||\n\
        @Media:\ttimed, audio\n*CHI:\tone two three . \u{15}100_400\u{15}\n\
        %wor:\tone \u{15}100_200\u{15} two \u{15}200_300\u{15} three \u{15}300_400\u{15} .\n\
        %com:\tcontributor annotation\n@End\n";

    fn admitted(text: &str) -> ValidChatFile {
        crate::pipeline::text_infer::admit_retained_text(
            &crate::chat_parser(),
            text,
            TranscriptName::for_path(std::path::Path::new("timed.cha")),
        )
        .expect("complete valid source CHAT")
        .into_valid_file()
    }

    fn mapping() -> SpeakerTrackMapping {
        "PAR0=CHI,PAR1=MOT".parse().unwrap()
    }

    fn segment(speaker: &str, start: i64, end: i64) -> SpeakerSegmentV2 {
        SpeakerSegmentV2 {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(start, end)
                .unwrap(),
            speaker: speaker.to_owned(),
        }
    }

    #[test]
    fn declared_mapping_preserves_facts_and_all_measured_returning_speaker_runs() {
        let input = admitted(SOURCE);
        let source = MappedDiarizeSource::admit(&input, &mapping()).unwrap();
        assert!(source.needs_inference());
        let output = source
            .apply(&[
                segment("b", 200, 300),
                segment("a", 300, 400),
                segment("a", 100, 200),
            ])
            .unwrap();
        let before: Vec<_> = input
            .document()
            .lines
            .iter()
            .filter_map(Line::as_header)
            .collect();
        let after: Vec<_> = output
            .document()
            .lines
            .iter()
            .filter_map(Line::as_header)
            .collect();
        assert_eq!(
            before, after,
            "participant and recording facts are unchanged"
        );
        assert_eq!(
            input.document().utterances().count(),
            1,
            "source remains immutable"
        );
        let records: Vec<_> = output
            .document()
            .utterances()
            .map(|child| {
                let timing = &child.main.content.bullet.as_ref().unwrap().timing;
                (child.main.speaker.as_str(), timing.start_ms, timing.end_ms)
            })
            .collect();
        assert_eq!(
            records,
            [("CHI", 100, 200), ("MOT", 200, 300), ("CHI", 300, 400)]
        );
        insta::assert_snapshot!("mapped_diarize_returning_speaker", output.as_str());
    }

    #[test]
    fn undeclared_participants_and_missing_or_partial_word_timing_refuse_before_inference() {
        let input = admitted(SOURCE);
        assert!(matches!(
            MappedDiarizeSource::admit(&input, &"PAR0=INV".parse().unwrap()),
            Err(DiarizeChatRefusal::UndeclaredParticipant { .. })
        ));
        let partial = admitted(&SOURCE.replace("two \u{15}200_300\u{15}", "two"));
        assert!(matches!(
            MappedDiarizeSource::admit(&partial, &mapping()),
            Err(DiarizeChatRefusal::WordTiming {
                source: WordSpeakerSplitRefusal::IncompleteWordTiming { .. },
                ..
            })
        ));
        let no_wor = SOURCE
            .lines()
            .filter(|line| !line.starts_with("%wor:"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let missing = admitted(&no_wor);
        assert!(matches!(
            MappedDiarizeSource::admit(&missing, &mapping()),
            Err(DiarizeChatRefusal::WordTiming {
                source: WordSpeakerSplitRefusal::MissingWordTiming,
                ..
            })
        ));
    }

    #[test]
    fn acoustic_gaps_ties_and_unmapped_tracks_never_acquire_write_permission() {
        let input = admitted(SOURCE);
        let tie = MappedDiarizeSource::admit(&input, &mapping())
            .unwrap()
            .apply(&[segment("a", 100, 400), segment("b", 100, 400)]);
        assert!(matches!(
            tie,
            Err(DiarizeChatRefusal::WordTiming {
                source: WordSpeakerSplitRefusal::TiedWord { .. },
                ..
            })
        ));
        let gap = MappedDiarizeSource::admit(&input, &mapping())
            .unwrap()
            .apply(&[segment("a", 100, 200)]);
        assert!(matches!(
            gap,
            Err(DiarizeChatRefusal::WordTiming {
                source: WordSpeakerSplitRefusal::UncoveredWord { .. },
                ..
            })
        ));
        let unmapped = MappedDiarizeSource::admit(&input, &"PAR0=CHI".parse().unwrap())
            .unwrap()
            .apply(&[segment("a", 100, 200), segment("b", 200, 400)]);
        assert!(matches!(
            unmapped,
            Err(DiarizeChatRefusal::UnmappedTrack(1))
        ));
    }

    #[test]
    fn analysis_loss_is_reported_without_discarding_contributor_comments() {
        let input = admitted(&SOURCE.replace("%com:", "%mor:\tn|one n|two n|three .\n%com:"));
        let output = MappedDiarizeSource::admit(&input, &mapping())
            .unwrap()
            .apply(&[segment("a", 100, 200), segment("b", 200, 400)])
            .unwrap();
        assert!(output.as_str().contains("input utterance 1 %mor"));
        assert!(!output.as_str().contains("%mor:"));
        assert_eq!(
            output
                .as_str()
                .matches("%com:\tcontributor annotation")
                .count(),
            1
        );
    }

    #[test]
    fn a_speaker_boundary_inside_an_annotated_group_refuses_instead_of_guessing() {
        let input =
            admitted(&SOURCE.replace("*CHI:\tone two three", "*CHI:\t<one two> [= group] three"));
        let result = MappedDiarizeSource::admit(&input, &mapping())
            .unwrap()
            .apply(&[segment("a", 100, 200), segment("b", 200, 400)]);
        assert!(matches!(
            result,
            Err(DiarizeChatRefusal::WordTiming {
                source: WordSpeakerSplitRefusal::Partition(_),
                ..
            })
        ));
    }
}
