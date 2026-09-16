//! FunASR output admission: pairing FunASR's own units with its timestamps.
//!
//! FunASR reports timing as a flat `timestamp` array holding one
//! `[start_ms, end_ms]` entry per recognized unit. Which list those units live
//! in depends on the checkpoint:
//!
//! - **Paraformer**, run with the `ct-punc-c` punctuation model: `text` is the
//!   PUNCTUATED surface with no spaces at all, so it cannot be split back into
//!   units. The pre-punctuation surface `raw_text` (present when the recognizer
//!   asks for `return_raw_text=True`) holds exactly one whitespace token per
//!   timestamp, a Han character or a Latin word.
//! - **SenseVoice**: `words` is built by FunASR in lockstep with `timestamp`
//!   (`SenseVoiceSmall.post`), punctuation units included.
//!
//! The defect this module makes unrepresentable: batchalign3 used to
//! re-tokenize `text` itself and pair the i-th token with the i-th timestamp.
//! For Paraformer that paired punctuation-delimited CLAUSES with per-character
//! timestamps, so a 437 s recording came out with every bullet inside its first
//! 50 s. For SenseVoice, stripping punctuation from the text but not from the
//! timestamps shifted every later character by one entry per punctuation mark.
//!
//! The graph, each edge consuming the node before it:
//!
//! ```text
//! FunasrSegmentWire --select--> UnitSource::{Words, RawText, DisplayText}
//!   --admit--> AdmittedSegment::Timed(Units<AdmittedInterval>) | Untimed(Units<NoTiming>)
//! Units<T>          --into_lexical--> LexicalUnits<T>   (punctuation and markup leave whole)
//! LexicalUnits<T>   --> monologue elements, plus timed words when T is UnitSpan
//! ```
//!
//! Only [`AdmittedSegment::admit`] builds [`Units`], and a timed one only after
//! the unit and timestamp counts agree, so no later stage can pair by position.
//! Display text is an untimed source only: timing must arrive with FunASR's own
//! unit list.
//!
//! Surfaces leave this module exactly as FunASR wrote them. Cantonese
//! normalization used to run here as well, over the joined segment; it now has
//! one owner, `batchalign_transform::asr_postprocess::AlignedNormalization`,
//! which the server applies once per monologue.

use batchalign_transform::asr_postprocess::cantonese as cantonese_ops;
use batchalign_transform::asr_postprocess::is_cjk_ideograph;
use batchalign_transform::asr_postprocess::{
    AdmittedInterval, IntervalRefusal, UntimedCause, WordTiming,
};
use serde::Deserialize;

use super::{
    HkAsrElement, HkAsrMonologue, HkAsrProjection, HkTimedWord, ProviderSpeakerAttribution,
    push_admitted_word,
};

/// One FunASR result exactly as Python forwards it, before admission.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct FunasrSegmentWire {
    /// Display surface: punctuated and space-free for Paraformer, carrying
    /// `<|...|>` markup for SenseVoice. A unit source only without timing.
    #[serde(default)]
    text: String,
    /// One `[start_ms, end_ms]` entry per unit, when FunASR produced timing.
    #[serde(default)]
    timestamp: Vec<serde_json::Value>,
    /// SenseVoice's unit list, parallel to `timestamp` by construction.
    #[serde(default)]
    words: Option<Vec<String>>,
    /// Paraformer's pre-punctuation surface: one whitespace token per timestamp.
    #[serde(default)]
    raw_text: Option<String>,
}

/// Why a FunASR segment could not be admitted.
///
/// Not `Eq`: an inadmissible timing carries the owner's refusal, which names
/// the offending bound as the float the provider actually sent.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(super) enum FunasrAdmissionError {
    /// Two unit lists arrived, so which one the timing indexes is unknown.
    #[error(
        "FunASR segment {segment} carries both `words` and `raw_text`; the list its \
         timestamps belong to is ambiguous"
    )]
    AmbiguousUnits { segment: usize },
    /// Timing arrived with only display text, which FunASR never pairs it with.
    #[error(
        "FunASR segment {segment} reported {timestamps} timestamps but no unit list \
         (`words` or `raw_text`) to pair them with"
    )]
    MissingUnits { segment: usize, timestamps: usize },
    /// The unit list and the timing disagree in length.
    #[error(
        "FunASR segment {segment} has {units} units but {timestamps} timestamps; pairing \
         them by position would misattribute timing"
    )]
    CountMismatch {
        segment: usize,
        units: usize,
        timestamps: usize,
    },
    /// One timing entry is not a pair of numbers at all, so there is nothing
    /// to admit: the shape is wrong before any bound can be judged.
    #[error(
        "FunASR segment {segment} timestamp {index} is not a [start_ms, end_ms] pair of numbers"
    )]
    MalformedTimestamp { segment: usize, index: usize },
    /// The entry is a pair of numbers, and the interval owner refused it.
    #[error("FunASR segment {segment} timestamp {index} is not an admissible interval: {refusal}")]
    InadmissibleTimestamp {
        segment: usize,
        index: usize,
        /// The owner's refusal, naming the bound and the value.
        refusal: IntervalRefusal,
    },
}

/// Admit one raw FunASR timing entry through the one owner of interval
/// rounding and range admission.
///
/// This module used to admit the pair itself: it rounded with `as i64`, which
/// is the cast [`AdmittedInterval`] exists to eliminate, and it enforced no
/// upper bound, so a FunASR build reporting epoch milliseconds produced a
/// plausible-looking time here while the same value was refused by name at
/// every other provider.
fn admit_timestamp(
    value: &serde_json::Value,
    segment: usize,
    index: usize,
) -> Result<AdmittedInterval, FunasrAdmissionError> {
    let malformed = FunasrAdmissionError::MalformedTimestamp { segment, index };
    let Some([start, end]) = value.as_array().map(Vec::as_slice) else {
        return Err(malformed);
    };
    let (Some(start_ms), Some(end_ms)) = (start.as_f64(), end.as_f64()) else {
        return Err(malformed);
    };
    AdmittedInterval::admit_millis_f64(start_ms, end_ms).map_err(|refusal| {
        FunasrAdmissionError::InadmissibleTimestamp {
            segment,
            index,
            refusal,
        }
    })
}

/// The timing phase of a unit FunASR reported no timestamp for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NoTiming;

/// One FunASR unit and its timing phase ([`UnitSpan`] or [`NoTiming`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Unit<T> {
    surface: String,
    timing: T,
}

/// Admitted FunASR units.
///
/// The field is private and the only constructor is [`AdmittedSegment::admit`],
/// which for timed units proves the unit and timestamp counts agree first.
#[derive(Debug)]
struct Units<T>(Vec<Unit<T>>);

/// A FunASR segment after admission.
#[derive(Debug)]
enum AdmittedSegment {
    /// FunASR reported timing, and every unit has exactly one admitted span.
    Timed(Units<AdmittedInterval>),
    /// FunASR reported no timing for this segment at all.
    Untimed(Units<NoTiming>),
}

/// Where a segment's units come from, decided before timing is considered.
enum UnitSource {
    /// SenseVoice's own unit list, parallel to its timestamps.
    Words(Vec<String>),
    /// Paraformer's pre-punctuation surface: one whitespace token per timestamp.
    RawText(String),
    /// Display text only, usable solely when there is no timing to pair.
    DisplayText(String),
}

impl UnitSource {
    /// Choose the unit list FunASR sent, refusing two at once.
    fn select(
        segment: usize,
        text: String,
        words: Option<Vec<String>>,
        raw_text: Option<String>,
    ) -> Result<Self, FunasrAdmissionError> {
        match (words, raw_text) {
            (Some(words), None) => Ok(Self::Words(words)),
            (None, Some(raw_text)) => Ok(Self::RawText(raw_text)),
            (None, None) => Ok(Self::DisplayText(text)),
            (Some(_), Some(_)) => Err(FunasrAdmissionError::AmbiguousUnits { segment }),
        }
    }

    /// Unit surfaces in FunASR's own granularity, for an untimed segment.
    fn into_surfaces(self) -> Vec<String> {
        match self {
            Self::Words(words) => words,
            Self::RawText(raw_text) => whitespace_surfaces(&raw_text),
            Self::DisplayText(text) => display_text_surfaces(&text),
        }
    }
}

impl AdmittedSegment {
    /// Admit one wire segment, refusing every way its timing could be paired
    /// with the wrong unit.
    fn admit(segment: usize, wire: FunasrSegmentWire) -> Result<Self, FunasrAdmissionError> {
        let FunasrSegmentWire {
            text,
            timestamp,
            words,
            raw_text,
        } = wire;
        let source = UnitSource::select(segment, text, words, raw_text)?;
        if timestamp.is_empty() {
            let units = source
                .into_surfaces()
                .into_iter()
                .map(|surface| Unit {
                    surface,
                    timing: NoTiming,
                })
                .collect();
            return Ok(Self::Untimed(Units(units)));
        }

        let surfaces = match source {
            UnitSource::Words(words) => words,
            UnitSource::RawText(raw_text) => whitespace_surfaces(&raw_text),
            UnitSource::DisplayText(_) => {
                return Err(FunasrAdmissionError::MissingUnits {
                    segment,
                    timestamps: timestamp.len(),
                });
            }
        };
        if surfaces.len() != timestamp.len() {
            return Err(FunasrAdmissionError::CountMismatch {
                segment,
                units: surfaces.len(),
                timestamps: timestamp.len(),
            });
        }

        // The zip is total: the lengths were proven equal just above.
        surfaces
            .into_iter()
            .zip(&timestamp)
            .enumerate()
            .map(|(index, (surface, raw))| {
                admit_timestamp(raw, segment, index).map(|timing| Unit { surface, timing })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|units| Self::Timed(Units(units)))
    }
}

/// Whitespace tokens, the unit granularity of Paraformer's `raw_text`.
fn whitespace_surfaces(text: &str) -> Vec<String> {
    text.split_whitespace().map(ToOwned::to_owned).collect()
}

/// Remove every closed `<|...|>` tag; an unclosed `<|` stays ordinary text.
fn strip_markup(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("<|") {
        let (before, tagged) = rest.split_at(open);
        match tagged[2..].find("|>") {
            Some(close) => {
                stripped.push_str(before);
                rest = &tagged[2 + close + 2..];
            }
            None => break,
        }
    }
    stripped.push_str(rest);
    stripped
}

/// Characters that end a display-text run: whitespace, CJK punctuation, and
/// ASCII punctuation other than the apostrophe and hyphen that occur inside
/// words such as `don't` and `well-known`.
fn is_display_separator(ch: char) -> bool {
    ch.is_whitespace()
        || cantonese_ops::is_cjk_punctuation(ch)
        || (ch.is_ascii_punctuation() && !matches!(ch, '\'' | '-'))
}

/// Split display text the way FunASR tokenizes: each Han character is a unit,
/// other runs stay whole, and markup, punctuation and whitespace separate.
fn display_text_surfaces(text: &str) -> Vec<String> {
    fn flush(run: &mut String, surfaces: &mut Vec<String>) {
        if !run.is_empty() {
            surfaces.push(std::mem::take(run));
        }
    }

    let mut surfaces = Vec::new();
    let mut run = String::new();
    for ch in strip_markup(text).chars() {
        match ch {
            separator if is_display_separator(separator) => flush(&mut run, &mut surfaces),
            ideograph if is_cjk_ideograph(ideograph) => {
                flush(&mut run, &mut surfaces);
                surfaces.push(ideograph.to_string());
            }
            other => run.push(other),
        }
    }
    flush(&mut run, &mut surfaces);
    surfaces
}

/// Whether a unit carries no lexical content: blank, `<|...|>` markup, or
/// punctuation only. A word such as `don't` is lexical because not every
/// character is punctuation.
fn is_non_lexical(surface: &str) -> bool {
    surface.is_empty()
        || (surface.starts_with("<|") && surface.ends_with("|>"))
        || surface
            .chars()
            .all(|ch| cantonese_ops::is_cjk_punctuation(ch) || ch.is_ascii_punctuation())
}

/// Units with every non-lexical unit removed together with its timing.
#[derive(Debug)]
struct LexicalUnits<T>(Vec<Unit<T>>);

impl<T> Units<T> {
    /// Drop punctuation and markup units TOGETHER WITH their timing, which is
    /// what keeps every remaining span attached to its own unit.
    fn into_lexical(self) -> LexicalUnits<T> {
        LexicalUnits(
            self.0
                .into_iter()
                .filter_map(|Unit { surface, timing }| {
                    let trimmed = surface.trim();
                    if is_non_lexical(trimmed) {
                        return None;
                    }
                    let already_trimmed = trimmed.len() == surface.len();
                    let surface = if already_trimmed {
                        surface
                    } else {
                        trimmed.to_owned()
                    };
                    Some(Unit { surface, timing })
                })
                .collect(),
        )
    }
}

impl LexicalUnits<AdmittedInterval> {
    /// Emit timed monologue elements plus the timed words usable for timing
    /// injection.
    ///
    /// The zero-width rule is NOT decided here: [`WordTiming::from_admitted`]
    /// owns it, and it reports such a unit as untimed, so the element carries
    /// no timestamp. This module used to decide it a second time and keep the
    /// zero on the element, which made a zero-width FunASR unit claim a time
    /// where the same unit from Tencent or Aliyun carried none.
    fn into_elements_and_timed_words(self) -> (Vec<HkAsrElement>, Vec<HkTimedWord>) {
        let mut elements = Vec::with_capacity(self.0.len());
        let mut timed_words = Vec::new();
        for Unit { surface, timing } in self.0 {
            push_admitted_word(
                surface,
                WordTiming::from_admitted(timing),
                &mut elements,
                &mut timed_words,
            );
        }
        (elements, timed_words)
    }
}

impl LexicalUnits<NoTiming> {
    /// Emit untimed monologue elements, through the same constructor every
    /// other provider's elements go through.
    fn into_elements(self) -> Vec<HkAsrElement> {
        self.0
            .into_iter()
            .map(|Unit { surface, .. }| {
                HkAsrElement::text(
                    surface,
                    WordTiming::Untimed(UntimedCause::ProviderReportedNoTiming),
                )
            })
            .collect()
    }
}

/// Admit and project FunASR segments into the shared HK ASR projection.
///
/// Each segment becomes one UNATTRIBUTED monologue: this path performs no
/// diarization, so there is no speaker label to carry and none is invented.
/// (The FunASR recognizer passes `spk_model="cam++"` to `generate`, but funasr
/// builds its speaker model in `AutoModel.__init__` and only ever consults
/// `self.spk_model`; a `spk_model` given to `generate` is inert, and no
/// `sentence_info` with `spk` labels is produced. Verified against the
/// installed funasr 1.4.12.) A segment whose units are all punctuation or markup
/// contributes no monologue. The first segment that cannot be admitted
/// refuses the whole result: a partially mistimed transcript is worse than
/// none.
pub(super) fn project_funasr_segments(
    segments: Vec<FunasrSegmentWire>,
) -> Result<HkAsrProjection, FunasrAdmissionError> {
    let mut monologues = Vec::with_capacity(segments.len());
    let mut timed_words = Vec::new();

    for (segment, wire) in segments.into_iter().enumerate() {
        let elements = match AdmittedSegment::admit(segment, wire)? {
            AdmittedSegment::Timed(units) => {
                let (elements, words) = units.into_lexical().into_elements_and_timed_words();
                timed_words.extend(words);
                elements
            }
            AdmittedSegment::Untimed(units) => units.into_lexical().into_elements(),
        };
        if !elements.is_empty() {
            monologues.push(HkAsrMonologue {
                speaker: ProviderSpeakerAttribution::Undiarized,
                elements,
            });
        }
    }

    timed_words.sort_by_key(|word| word.start_ms);
    Ok(HkAsrProjection {
        monologues,
        timed_words,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// Build a wire segment directly (this module owns its private fields).
    fn wire(
        text: &str,
        timestamp: Vec<Value>,
        words: Option<&[&str]>,
        raw_text: Option<&str>,
    ) -> FunasrSegmentWire {
        FunasrSegmentWire {
            text: text.to_owned(),
            timestamp,
            words: words.map(|units| units.iter().map(|unit| (*unit).to_owned()).collect()),
            raw_text: raw_text.map(ToOwned::to_owned),
        }
    }

    /// FunASR timing entries from `(start_ms, end_ms)` pairs.
    fn spans(pairs: &[(i64, i64)]) -> Vec<Value> {
        pairs
            .iter()
            .map(|(start, end)| json!([start, end]))
            .collect()
    }

    fn values(projection: &HkAsrProjection) -> Vec<&str> {
        projection
            .monologues
            .iter()
            .flat_map(|monologue| {
                monologue
                    .elements
                    .iter()
                    .map(|element| element.value.as_str())
            })
            .collect()
    }

    /// The module's reason to exist: units come from `raw_text`, and the last
    /// unit keeps the last span rather than one from the recording's start.
    #[test]
    fn paraformer_pairs_characters_from_raw_text_not_clauses_from_text()
    -> Result<(), FunasrAdmissionError> {
        let projection = project_funasr_segments(vec![wire(
            "你好，世界。再见？",
            spans(&[
                (0, 100),
                (100, 200),
                (5000, 5100),
                (5100, 5200),
                (9000, 9100),
                (9100, 9200),
            ]),
            None,
            Some("你 好 世 界 再 见"),
        )])?;

        assert_eq!(
            values(&projection),
            vec!["你", "好", "世", "界", "再", "见"]
        );
        let last = projection.monologues[0].elements.last();
        assert_eq!(
            last.map(|element| (element.ts, element.end_ts)),
            Some((Some(9.1), Some(9.2)))
        );
        assert_eq!(projection.timed_words.len(), 6);
        Ok(())
    }

    /// SenseVoice reports punctuation as units with spans of their own. They
    /// must leave together with those spans, or every later unit shifts.
    #[test]
    fn sensevoice_punctuation_units_leave_with_their_own_spans() -> Result<(), FunasrAdmissionError>
    {
        let projection = project_funasr_segments(vec![wire(
            "<|en|><|NEUTRAL|><|Speech|><|withitn|>I, ok.",
            spans(&[(0, 100), (100, 110), (200, 300), (300, 310)]),
            Some(&["I", ",", "ok", "."]),
            None,
        )])?;

        assert_eq!(values(&projection), vec!["I", "ok"]);
        assert_eq!(projection.monologues[0].elements[1].ts, Some(0.2));
        Ok(())
    }

    /// Surfaces cross this module unchanged, Cantonese included, and a Latin
    /// word stays one unit instead of being split into letters.
    ///
    /// `系` used to leave here as `係`. Normalizing at the provider boundary
    /// made a transcript's text depend on which engine had produced it (Qwen
    /// and Whisper never reach this module), so it moved to one owner in the
    /// server, which normalizes each monologue once.
    #[test]
    fn cantonese_surfaces_cross_unchanged_and_keep_each_span() -> Result<(), FunasrAdmissionError> {
        let projection = project_funasr_segments(vec![wire(
            "真系，ok",
            spans(&[(0, 100), (100, 200), (200, 210), (300, 400)]),
            Some(&["真", "系", "，", "ok"]),
            None,
        )])?;

        assert_eq!(values(&projection), vec!["真", "系", "ok"]);
        assert_eq!(projection.monologues[0].elements[1].ts, Some(0.1));
        Ok(())
    }

    #[test]
    fn a_count_mismatch_is_refused_instead_of_paired_by_position() {
        let result = project_funasr_segments(vec![
            wire("hello", spans(&[(0, 100)]), Some(&["hello"]), None),
            wire(
                "hello world bye",
                spans(&[(0, 200)]),
                Some(&["hello", "world", "bye"]),
                None,
            ),
        ]);
        assert_eq!(
            result,
            Err(FunasrAdmissionError::CountMismatch {
                segment: 1,
                units: 3,
                timestamps: 1
            })
        );
    }

    #[test]
    fn timing_with_only_display_text_is_refused() {
        let result = project_funasr_segments(vec![wire(
            "你好。",
            spans(&[(0, 100), (100, 200)]),
            None,
            None,
        )]);
        assert_eq!(
            result,
            Err(FunasrAdmissionError::MissingUnits {
                segment: 0,
                timestamps: 2
            })
        );
    }

    #[test]
    fn two_unit_lists_are_refused_as_ambiguous() {
        let result = project_funasr_segments(vec![wire(
            "好",
            spans(&[(0, 100)]),
            Some(&["好"]),
            Some("好"),
        )]);
        assert_eq!(
            result,
            Err(FunasrAdmissionError::AmbiguousUnits { segment: 0 })
        );
    }

    /// An entry that is not a pair of numbers is refused on SHAPE, before any
    /// bound can be judged, and names its position.
    #[test]
    fn a_timestamp_that_is_not_a_pair_of_numbers_is_refused_with_its_position() {
        for bad in [
            json!("x"),
            json!([0]),
            json!([0, "x"]),
            json!([0, 100, 200]),
        ] {
            let result = project_funasr_segments(vec![wire(
                "a b",
                vec![json!([0, 100]), bad.clone()],
                Some(&["a", "b"]),
                None,
            )]);
            assert_eq!(
                result,
                Err(FunasrAdmissionError::MalformedTimestamp {
                    segment: 0,
                    index: 1
                }),
                "{bad}"
            );
        }
    }

    /// A pair of numbers the interval owner refuses comes back as ITS refusal,
    /// naming the bound and the offending value.
    ///
    /// The out-of-range case is the one this module could not express before:
    /// it enforced no upper bound, so a build reporting epoch milliseconds
    /// produced a plausible-looking time here while the same value was refused
    /// by name at both cloud providers.
    #[test]
    fn an_inadmissible_pair_carries_the_owners_refusal() {
        use batchalign_transform::asr_postprocess::IntervalBound;

        #[allow(clippy::cast_precision_loss)]
        let beyond_range = AdmittedInterval::MAX_MS as f64 + 1.0;
        for (bad, expected) in [
            (
                json!([300, 200]),
                IntervalRefusal::Inverted {
                    start_ms: 300,
                    end_ms: 200,
                },
            ),
            (
                json!([-5, 10]),
                IntervalRefusal::Negative {
                    bound: IntervalBound::Start,
                    value_ms: -5.0,
                },
            ),
            (
                json!([beyond_range, beyond_range]),
                IntervalRefusal::OutOfRange {
                    bound: IntervalBound::Start,
                    value_ms: beyond_range,
                },
            ),
        ] {
            let result = project_funasr_segments(vec![wire(
                "a b",
                vec![json!([0, 100]), bad.clone()],
                Some(&["a", "b"]),
                None,
            )]);
            assert_eq!(
                result,
                Err(FunasrAdmissionError::InadmissibleTimestamp {
                    segment: 0,
                    index: 1,
                    refusal: expected,
                }),
                "{bad}"
            );
        }
    }

    /// Untimed display text is split the way FunASR tokenizes, and goes
    /// through the same lexical stage as timed units.
    #[test]
    fn untimed_display_text_splits_like_funasr() -> Result<(), FunasrAdmissionError> {
        let cantonese = project_funasr_segments(vec![wire("<|zh|> 真系", Vec::new(), None, None)])?;
        assert_eq!(values(&cantonese), vec!["真", "系"]);
        assert!(
            cantonese.monologues[0]
                .elements
                .iter()
                .all(|element| element.ts.is_none())
        );
        assert!(cantonese.timed_words.is_empty());

        let mixed = project_funasr_segments(vec![wire(
            "<|en|><|NEUTRAL|>I don't know，你好。ok",
            Vec::new(),
            None,
            None,
        )])?;
        assert_eq!(values(&mixed), vec!["I", "don't", "know", "你", "好", "ok"]);
        Ok(())
    }

    #[test]
    fn untimed_segments_still_prefer_funasr_unit_lists() -> Result<(), FunasrAdmissionError> {
        let projection =
            project_funasr_segments(vec![wire("你好。", Vec::new(), None, Some("你 好"))])?;
        assert_eq!(values(&projection), vec!["你", "好"]);
        Ok(())
    }

    /// A zero-width span locates nothing, so its element carries NO time and
    /// contributes no timed word.
    ///
    /// The element used to keep the zero-width timestamp, because this module
    /// re-decided the rule its owner already owns; the same unit from Tencent
    /// or Aliyun came out null.
    #[test]
    fn a_zero_length_span_leaves_its_element_untimed() -> Result<(), FunasrAdmissionError> {
        let projection = project_funasr_segments(vec![wire(
            "hello",
            spans(&[(100, 100)]),
            Some(&["hello"]),
            None,
        )])?;

        assert_eq!(values(&projection), vec!["hello"]);
        let element = &projection.monologues[0].elements[0];
        assert_eq!((element.ts, element.end_ts), (None, None));
        assert!(projection.timed_words.is_empty());
        Ok(())
    }

    #[test]
    fn markup_is_stripped_only_when_closed() {
        assert_eq!(
            strip_markup("<|zh|><|HAPPY|> hello <|NEUTRAL|> world"),
            " hello  world"
        );
        assert_eq!(strip_markup("a <|b"), "a <|b");
    }

    /// Wire format: Python forwards absent unit lists as explicit `null`.
    #[test]
    fn the_wire_admits_null_unit_lists_as_absent() -> Result<(), serde_json::Error> {
        let segment: FunasrSegmentWire = serde_json::from_value(json!({
            "text": "你好",
            "timestamp": [[0, 100], [100, 200]],
            "words": null,
            "raw_text": "你 好",
        }))?;
        assert_eq!(segment.words, None);
        assert_eq!(segment.raw_text.as_deref(), Some("你 好"));
        assert_eq!(segment.timestamp.len(), 2);
        Ok(())
    }
}
