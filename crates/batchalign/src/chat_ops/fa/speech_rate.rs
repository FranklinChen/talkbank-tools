//! How fast speech would have to be for some words to fit some audio.
//!
//! # Why this is a type and not an `f64` beside a threshold
//!
//! "These words cannot fit this span" is a physical fact about human speech,
//! and it is the fact that decides whether a piece of audio can be aligned at
//! all. It was known in exactly one place: a private `classify_bullet` in
//! `rescue_narrow_bullets`, over a private `CATASTROPHIC_DENSITY_THRESHOLD_WPS`.
//!
//! The place that needed it most could not see it. `estimate_untimed_boundaries`
//! distributes utterances that carry no bullet across whatever audio is left,
//! in proportion to word count, and asks nothing about whether the result is
//! possible. On a real session it placed **175 words into 0.291 seconds**, a
//! rate of 601 words per second, and handed the resulting window to an aligner.
//! Wave2Vec correctly refused it ("targets length is too long for CTC"); the
//! pipeline read that as an engine limitation and retried on Whisper, which pads
//! its input to a fixed 30 second window, cannot refuse, and duly reported
//! tokens across the padding. That is how 28.2 seconds of speech got timings in
//! a recording that had already ended.
//!
//! The threshold is unchanged. What changes is that both callers now ask the
//! same question of the same type, and a rate is a rate rather than a bare
//! float that happens to be compared against the right constant.

use std::fmt;

use super::coordinates::Ms;

/// Above this, it is not speech.
///
/// Normal English conversation is 2-7 words per second; rapid news delivery
/// reaches about 9; auctioneer and rap peak in the 12-15 range. Anything faster
/// is not a fast speaker, it is a broken span.
const HUMAN_CEILING_WPS: f64 = 15.0;

/// A speaking rate.
///
/// Not `Eq`, because it is a measurement in floating point and two rates being
/// bit-identical is not a question worth asking. Comparison against the ceiling
/// is what it is for.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct WordsPerSecond(f64);

impl WordsPerSecond {
    /// The rate itself, for logging and for reporting a refusal.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl fmt::Display for WordsPerSecond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} words/sec", self.0)
    }
}

/// What a span implies about the words it is supposed to contain.
///
/// A sum rather than a rate plus a caller-side comparison: the ceiling belongs
/// with the measurement, so a caller cannot obtain a rate and forget to ask
/// whether it is possible. That is the whole failure being prevented, one level
/// up from the arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeechRate {
    /// A rate a human could produce. Alignment can be attempted.
    Plausible(WordsPerSecond),

    /// Faster than any human speech, so the span cannot contain these words.
    ///
    /// Not a slow case or a hard case: an impossible one. No aligner can place
    /// these words in this audio, and the engines differ only in whether they
    /// say so. Handing this to one that cannot say so is what produces invented
    /// timings.
    Impossible {
        /// The rate the span would demand.
        rate: WordsPerSecond,
        /// How many words were to be placed.
        words: usize,
        /// The audio available for them.
        span: Ms,
    },

    /// There are words but no audio at all.
    ///
    /// Distinct from `Impossible` because there is no rate to report: division
    /// by a zero span has no answer, and reporting an infinity would be a
    /// fabricated measurement.
    NoAudio {
        /// How many words were to be placed.
        words: usize,
    },

    /// There are no words, so the span is unconstrained.
    NoWords,
}

impl SpeechRate {
    /// What `words` in `span` would demand of a speaker.
    pub fn of(words: usize, span: Ms) -> Self {
        match (words, span) {
            (0, _) => Self::NoWords,
            (words, Ms(0)) => Self::NoAudio { words },
            (words, Ms(ms)) => {
                let rate = WordsPerSecond(words as f64 / (ms as f64 / 1000.0));
                match rate.0 > HUMAN_CEILING_WPS {
                    true => Self::Impossible { rate, words, span },
                    false => Self::Plausible(rate),
                }
            }
        }
    }

    /// Whether a span this dense can hold its words at all.
    ///
    /// Named for the question a caller is actually asking, so the answer does
    /// not depend on remembering which way the comparison runs.
    pub fn is_possible(&self) -> bool {
        match self {
            Self::Plausible(_) | Self::NoWords => true,
            Self::Impossible { .. } | Self::NoAudio { .. } => false,
        }
    }
}

impl fmt::Display for SpeechRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plausible(rate) => write!(f, "{rate}"),
            Self::Impossible { rate, words, span } => {
                write!(f, "{words} words in {span} demands {rate}")
            }
            Self::NoAudio { words } => write!(f, "{words} words with no audio"),
            Self::NoWords => f.write_str("no words"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_conversation_is_plausible() {
        // Five words in two seconds: 2.5 wps, unremarkable.
        assert!(SpeechRate::of(5, Ms(2_000)).is_possible());
    }

    #[test]
    fn the_real_failure_is_refused() {
        // The measured case: 175 untimed words distributed across the 291 ms of
        // audio left at the end of a real session. 601 words per second.
        let rate = SpeechRate::of(175, Ms(291));
        assert!(!rate.is_possible());
        let SpeechRate::Impossible { rate, words, span } = rate else {
            panic!("601 words/sec is not plausible speech")
        };
        assert_eq!(words, 175);
        assert_eq!(span, Ms(291));
        assert!(rate.get() > 600.0, "got {rate}");
    }

    #[test]
    fn no_audio_is_distinct_from_an_impossible_rate() {
        // There is no rate to report when the span is empty, and reporting an
        // infinity would be a fabricated measurement.
        assert_eq!(SpeechRate::of(3, Ms(0)), SpeechRate::NoAudio { words: 3 });
        assert!(!SpeechRate::of(3, Ms(0)).is_possible());
    }

    #[test]
    fn no_words_constrains_nothing() {
        assert_eq!(SpeechRate::of(0, Ms(0)), SpeechRate::NoWords);
        assert!(SpeechRate::of(0, Ms(0)).is_possible());
    }

    #[test]
    fn buffering_rescues_a_tight_run_but_not_an_impossible_one() {
        // The estimator measures against the audio the aligner will receive,
        // which is the raw gap widened by 2000 ms at each end. Both halves of
        // that matter, so both are pinned here.
        //
        // A single untimed word immediately before a timed utterance has a
        // zero-width raw gap and a four-second real window: placeable.
        const BUFFERED_ZERO_GAP: u64 = 4_000;
        assert!(SpeechRate::of(1, Ms(BUFFERED_ZERO_GAP)).is_possible());
        // The measured failure survives buffering: 175 words, 291 ms raw,
        // 4291 ms buffered, still about 41 words per second.
        assert!(!SpeechRate::of(175, Ms(291 + 4_000)).is_possible());
    }

    #[test]
    fn the_ceiling_sits_between_rap_and_nonsense() {
        // 15 wps is the documented ceiling: fast human delivery is at or below
        // it, and the boundary is inclusive so a real auctioneer is not refused.
        assert!(SpeechRate::of(15, Ms(1_000)).is_possible());
        assert!(!SpeechRate::of(16, Ms(1_000)).is_possible());
    }
}
