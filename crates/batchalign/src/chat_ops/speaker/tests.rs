use super::*;
use batchalign_transform::asr_postprocess::AsrWord;

fn word(text: &str, start_ms: Option<i64>, end_ms: Option<i64>) -> AsrWord {
    AsrWord::new(text, start_ms, end_ms)
}

#[test]
fn splits_at_diarization_boundary_and_attaches_punctuation_backward() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(0),
        words: vec![
            word("hello", Some(0), Some(500)),
            word("world", Some(500), Some(1_000)),
            word(".", None, None),
        ],
    }];
    let segments = vec![
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
                .expect("an ordered fixture span"),
            speaker: "A".into(),
        },
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(500, 1_000)
                .expect("an ordered fixture span"),
            speaker: "B".into(),
        },
    ];

    let observation = project_speakers_onto_chunks(chunks, &segments);
    let stats = observation.stats();
    let projected = observation.admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);

    assert_eq!(projected.chunks().len(), 2);
    assert_eq!(projected.chunks()[0].speaker, SpeakerIndex(0));
    assert_eq!(projected.chunks()[1].speaker, SpeakerIndex(1));
    assert_eq!(projected.chunks()[1].words.len(), 2);
    assert_eq!(stats.speaker_boundaries, 1);
}

#[test]
fn speaker_indices_use_the_same_lexical_label_coordinates_as_turn_artifacts() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(0),
        words: vec![
            word("first", Some(0), Some(500)),
            word("second", Some(500), Some(1_000)),
        ],
    }];
    let segments = vec![
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
                .expect("an ordered fixture span"),
            speaker: "SPEAKER_01".into(),
        },
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(500, 1_000)
                .expect("an ordered fixture span"),
            speaker: "SPEAKER_00".into(),
        },
    ];

    let observation = project_speakers_onto_chunks(chunks, &segments);
    let projected = observation.admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);

    assert_eq!(projected.chunks()[0].speaker, SpeakerIndex(1));
    assert_eq!(projected.chunks()[1].speaker, SpeakerIndex(0));
}

#[test]
fn uncovered_words_remain_in_the_dedicated_label_space() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(0),
        words: vec![
            word("hello", Some(0), Some(500)),
            word("later", Some(2_000), Some(2_500)),
        ],
    }];
    let segments = vec![SpeakerSegment {
        interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 500)
            .expect("an ordered fixture span"),
        speaker: "A".into(),
    }];

    let observation = project_speakers_onto_chunks(chunks, &segments);
    let stats = observation.stats();
    let projected = observation.admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);

    assert_eq!(projected.chunks().len(), 1);
    assert_eq!(projected.chunks()[0].speaker, SpeakerIndex(0));
    assert_eq!(stats.unattested_timed_words, 1);
}

#[test]
fn gaps_choose_the_nearest_dedicated_segment_without_phantom_speakers() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(7),
        words: vec![
            word("left", Some(0), Some(200)),
            word("near_right", Some(850), Some(950)),
            word("right", Some(1_000), Some(1_200)),
        ],
    }];
    let segments = vec![
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 200)
                .expect("an ordered fixture span"),
            speaker: "A".into(),
        },
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(1_000, 1_200)
                .expect("an ordered fixture span"),
            speaker: "B".into(),
        },
    ];

    let observation = project_speakers_onto_chunks(chunks, &segments);
    let stats = observation.stats();
    let projected = observation.admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);

    assert_eq!(projected.chunks().len(), 2);
    assert_eq!(projected.chunks()[0].speaker, SpeakerIndex(0));
    assert_eq!(projected.chunks()[1].speaker, SpeakerIndex(1));
    assert_eq!(projected.chunks()[1].words[0].text, "near_right");
    assert_eq!(stats.unattested_timed_words, 1);
    assert!(
        projected
            .chunks()
            .iter()
            .all(|chunk| chunk.speaker.as_usize() < 2)
    );
}

#[test]
fn reports_contested_words_and_uses_greatest_total_overlap() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(0),
        words: vec![word("hello", Some(0), Some(1_000))],
    }];
    let segments = vec![
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(0, 700)
                .expect("an ordered fixture span"),
            speaker: "A".into(),
        },
        SpeakerSegment {
            interval: batchalign_types::interval::AdmittedInterval::admit_millis(600, 1_000)
                .expect("an ordered fixture span"),
            speaker: "B".into(),
        },
    ];

    let observation = project_speakers_onto_chunks(chunks, &segments);
    let stats = observation.stats();
    let projected = observation.admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);

    assert_eq!(projected.chunks()[0].speaker, SpeakerIndex(0));
    assert_eq!(stats.contested_timed_words, 1);
}

#[test]
fn empty_segments_preserve_chunks_exactly() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(4),
        words: vec![word("hello", Some(0), Some(500))],
    }];

    let observation = project_speakers_onto_chunks(chunks.clone(), &[]);
    let stats = observation.stats();
    let projected = observation.admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);

    assert_eq!(projected.chunks(), chunks);
    assert_eq!(stats, SpeakerProjectionStats::default());
    assert!(projected.evidence().needs_review());
    assert_eq!(
        projected.evidence().observations().summary().retained_asr,
        1
    );
    assert_eq!(
        projected.evidence().observations().assignments()[0].coordinate(),
        SpeakerCoordinate::Asr(4)
    );
}

fn segment(start: i64, end: i64, label: &str) -> SpeakerSegment {
    SpeakerSegment {
        interval: batchalign_types::interval::AdmittedInterval::admit_millis(start, end)
            .expect("ordered fixture"),
        speaker: label.into(),
    }
}

#[test]
fn speaker_evidence_retains_untimed_default_and_neighbor_witnesses() {
    let segments = [segment(0, 500, "B"), segment(500, 1000, "A")];
    let chunks = vec![
        PreparedMonologueChunk {
            speaker: SpeakerIndex(7),
            words: vec![
                word("leading", None, None),
                word("timed", Some(0), Some(500)),
                word(".", None, None),
                word("broken", Some(-1), Some(100)),
            ],
        },
        PreparedMonologueChunk {
            speaker: SpeakerIndex(8),
            words: vec![word("untimed", None, None)],
        },
    ];
    let admitted = project_speakers_onto_chunks(chunks, &segments)
        .admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);
    let evidence = admitted.evidence().observations();
    let entries = evidence.assignments();
    assert_eq!(
        entries[0].basis(),
        &AssignmentBasis::FollowingWord {
            witness: SourceWordAddress { chunk: 0, word: 1 },
        }
    );
    assert_eq!(
        entries[2].basis(),
        &AssignmentBasis::PreviousWord {
            witness: SourceWordAddress { chunk: 0, word: 1 },
        }
    );
    assert_eq!(entries[3].timing(), &TimingSupport::InvalidInterval);
    assert_eq!(
        entries[3].basis(),
        &AssignmentBasis::PreviousWord {
            witness: SourceWordAddress { chunk: 0, word: 1 },
        }
    );
    assert_eq!(entries[4].basis(), &AssignmentBasis::FirstLabelDefault);
    assert_eq!(entries[4].coordinate(), SpeakerCoordinate::Diarization(0));
    assert_eq!(admitted.chunks()[0].speaker, SpeakerIndex(1));
    assert_eq!(admitted.chunks()[1].speaker, SpeakerIndex(0));
    assert_eq!(
        evidence.summary(),
        SpeakerAssignmentSummary {
            words: 5,
            directly_supported: 1,
            contested: 0,
            inferred: 3,
            defaulted: 1,
            retained_asr: 0,
        }
    );
    assert!(
        admitted
            .evidence()
            .review_warning()
            .expect("review warning")
            .contains("defaulted=1")
    );
}

#[test]
fn speaker_evidence_distinguishes_contested_ties_and_nearest_gap_fallback() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(9),
        words: vec![
            word("tie", Some(0), Some(100)),
            word("gap", Some(200), Some(300)),
        ],
    }];
    // Reversing labels keeps lexical overlap ties but the gap's original-order tie.
    let segments = [segment(0, 100, "B"), segment(0, 100, "A")];
    let admitted = project_speakers_onto_chunks(chunks, &segments)
        .admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);
    let entries = admitted.evidence().observations().assignments();
    assert_eq!(entries[0].coordinate(), SpeakerCoordinate::Diarization(0));
    assert_eq!(
        entries[0].timing(),
        &TimingSupport::Overlap {
            speakers: 2,
            tied_winners: 2,
            winning_overlap_ms: 100,
        }
    );
    assert_eq!(entries[1].coordinate(), SpeakerCoordinate::Diarization(1));
    assert_eq!(
        entries[1].timing(),
        &TimingSupport::Gap {
            segment: 0,
            distance_ms: 100
        }
    );
    assert_eq!(entries[1].basis(), &AssignmentBasis::NearestSegment);
    assert_eq!(
        admitted.evidence().observations().stats(),
        SpeakerProjectionStats {
            contested_timed_words: 1,
            unattested_timed_words: 1,
            speaker_boundaries: 1,
        }
    );
    assert!(admitted.evidence().needs_review());
}

#[test]
fn speaker_evidence_wire_summary_is_derived_and_has_an_explicit_policy() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(3),
        words: vec![word("direct", Some(0), Some(100))],
    }];
    let admitted = project_speakers_onto_chunks(chunks, &[segment(0, 100, "A")])
        .admit(SpeakerProjectionPolicy::BestEffortWithReviewEvidenceV1);
    let wire = serde_json::to_value(admitted.evidence()).expect("speaker evidence wire");
    assert_eq!(wire["schema_version"], 1);
    assert_eq!(wire["policy"], "best_effort_with_review_evidence_v1");
    assert_eq!(wire["needs_review"], false);
    assert_eq!(wire["summary"]["words"], 1);
    assert_eq!(wire["summary"]["directly_supported"], 1);
    assert_eq!(
        wire["evidence"]["assignments"][0]["source"],
        serde_json::json!({"chunk":0,"word":0})
    );
    assert_eq!(
        wire["evidence"]["assignments"][0]["coordinate"],
        serde_json::json!({"space":"diarization","index":0})
    );
    assert!(admitted.evidence().review_warning().is_none());
}

#[test]
fn speaker_evidence_fingerprints_bind_exact_prepared_source_and_ordered_segments() {
    let chunks = vec![PreparedMonologueChunk {
        speaker: SpeakerIndex(3),
        words: vec![word("direct", Some(0), Some(100))],
    }];
    let segments = vec![segment(0, 100, "A"), segment(100, 200, "B")];
    let hash = |chunks: Vec<PreparedMonologueChunk>, segments: &[SpeakerSegment]| {
        serde_json::to_value(project_speakers_onto_chunks(chunks, segments).observations())
            .expect("source fingerprints")
    };
    let original = hash(chunks.clone(), &segments);
    assert_eq!(original, hash(chunks.clone(), &segments));
    for change in 0..5 {
        let mut changed = chunks.clone();
        match change {
            0 => changed[0].speaker = SpeakerIndex(4),
            1 => {
                changed[0].words[0].text =
                    batchalign_transform::asr_postprocess::AsrNormalizedText::new("different")
            }
            2 => changed[0].words[0].start_ms = None,
            3 => changed[0].words[0].end_ms = Some(99),
            _ => {
                changed[0].words[0].kind = batchalign_transform::asr_postprocess::WordKind::Retrace
            }
        }
        assert_ne!(
            original["input_chunks_blake3"],
            hash(changed, &segments)["input_chunks_blake3"]
        );
    }
    let mut reordered = segments.clone();
    reordered.reverse();
    assert_ne!(
        original["segments_blake3"],
        hash(chunks.clone(), &reordered)["segments_blake3"]
    );
    let mut relabeled = segments.clone();
    relabeled[0].speaker = "C".into();
    assert_ne!(
        original["segments_blake3"],
        hash(chunks.clone(), &relabeled)["segments_blake3"]
    );
    let mut retimed = segments.clone();
    retimed[0] = segment(0, 99, "A");
    assert_ne!(
        original["segments_blake3"],
        hash(chunks, &retimed)["segments_blake3"]
    );
}
