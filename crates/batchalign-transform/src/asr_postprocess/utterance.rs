use std::borrow::Cow;

use super::han_script::NormalizationChangedLength;
use super::{
    AsrNormalizedText, AsrOutput, AsrPipelineSnapshot, AsrTextLanguage, AsrWord, ENDING_PUNCT,
    MOR_PUNCT, MonologueWords, PreparedMonologueChunk, RTL_PUNCT, SpeakerIndex, Utterance, cleanup,
    finalize_words_to_chunks_with_snapshot, prepare_words_pre_expansion_with_snapshot,
    write_word_forms,
};

/// Run the full ASR post-processing pipeline on raw ASR output.
///
/// Applies compound merging, timing conversion, Han-script normalization,
/// multi-word splitting, number expansion, long turn splitting,
/// punctuation-based retokenization, disfluency replacement, and n-gram retrace
/// detection. Returns speaker-attributed utterances ready for CHAT assembly via
/// `build_chat()`.
///
/// Fails only when Han-script normalization changed a monologue's character
/// count; see [`super::AlignedNormalization`].
pub fn process_raw_asr<'a>(
    output: &AsrOutput,
    language: impl Into<AsrTextLanguage<'a>>,
) -> Result<Vec<Utterance>, NormalizationChangedLength> {
    let language = language.into();
    let mut all_utterances = utterances_from_prepared_chunks(prepare_asr_chunks(output, language)?);
    finalize_utterances(&mut all_utterances, language.rules());
    Ok(all_utterances)
}

/// Normalize raw ASR monologues into pre-CHAT chunks while preserving speaker
/// boundaries.
///
/// The two steps, [`prepare_monologue_words`] then [`chunk_monologues`], with
/// nothing between them. A caller that runs a whole-monologue pass (word
/// segmentation, which needs a worker) calls the two steps itself.
pub fn prepare_asr_chunks<'a>(
    output: &AsrOutput,
    language: impl Into<AsrTextLanguage<'a>>,
) -> Result<Vec<PreparedMonologueChunk>, NormalizationChangedLength> {
    let language = language.into();
    Ok(chunk_monologues(
        prepare_monologue_words(output, language)?,
        language.rules(),
    ))
}

/// Stages 1-4 for every monologue: each monologue's words in their final CHAT
/// word forms, not yet chunked.
pub fn prepare_monologue_words<'a>(
    output: &AsrOutput,
    language: impl Into<AsrTextLanguage<'a>>,
) -> Result<Vec<MonologueWords>, NormalizationChangedLength> {
    prepare_monologue_words_with_snapshot(output, language.into(), None)
}

/// Snapshot-aware variant of [`prepare_monologue_words`].
///
/// When `snapshot` is `Some`, fills `raw_elements` and the stage 1-4 fields,
/// concatenating the monologues' stage outputs in order.
pub fn prepare_monologue_words_with_snapshot(
    output: &AsrOutput,
    language: AsrTextLanguage<'_>,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Result<Vec<MonologueWords>, NormalizationChangedLength> {
    let mut monologues = Vec::with_capacity(output.monologues.len());
    for monologue in &output.monologues {
        // The preparation step SETS its snapshot fields, so each monologue
        // records into its own and is appended here.
        let mut sub = snapshot.is_some().then(AsrPipelineSnapshot::default);
        let words =
            prepare_words_pre_expansion_with_snapshot(&monologue.elements, language, sub.as_mut())?;
        // Stage 4: CHAT word forms (number words, alphanumeric linkages,
        // letter strings)
        let words = write_word_forms(words, language);
        if let (Some(s), Some(sub)) = (snapshot.as_deref_mut(), sub) {
            s.raw_elements.extend_from_slice(&monologue.elements);
            s.after_compound_merge.extend(sub.after_compound_merge);
            s.after_timing_extract.extend(sub.after_timing_extract);
            if let Some(normalized) = sub.after_han_script_norm {
                s.after_han_script_norm
                    .get_or_insert_with(Vec::new)
                    .extend(normalized);
            }
            s.after_multiword_split.extend(sub.after_multiword_split);
            s.after_number_expand.extend_from_slice(&words);
        }
        monologues.push(MonologueWords {
            speaker: monologue.speaker,
            words,
        });
    }
    Ok(monologues)
}

/// Stages 5-5b for every monologue: long-turn and pause splitting into
/// chunks, in monologue order.
pub fn chunk_monologues(
    monologues: Vec<MonologueWords>,
    lang: &str,
) -> Vec<PreparedMonologueChunk> {
    chunk_monologues_with_snapshot(monologues, lang, None)
}

/// Snapshot-aware variant of [`chunk_monologues`]: fills
/// `after_long_turn_split` when `snapshot` is `Some`.
pub fn chunk_monologues_with_snapshot(
    monologues: Vec<MonologueWords>,
    lang: &str,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Vec<PreparedMonologueChunk> {
    let mut prepared = Vec::new();
    for MonologueWords { speaker, words } in monologues {
        // Chunking SETS `after_long_turn_split`, so each monologue records
        // into its own snapshot and is appended here.
        let mut sub = snapshot.is_some().then(AsrPipelineSnapshot::default);
        prepared.extend(finalize_words_to_chunks_with_snapshot(
            words,
            speaker,
            lang,
            sub.as_mut(),
        ));
        if let (Some(s), Some(sub)) = (snapshot.as_deref_mut(), sub) {
            s.after_long_turn_split.extend(sub.after_long_turn_split);
        }
    }
    prepared
}

/// Retokenize prepared chunks into utterances using punctuation boundaries.
pub fn utterances_from_prepared_chunks(
    chunks: Vec<super::PreparedMonologueChunk>,
) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    for chunk in chunks {
        utterances.extend(retokenize(chunk.speaker, chunk.words));
    }
    utterances
}

/// Apply the post-retokenization cleanup passes shared by all ASR paths.
pub fn finalize_utterances(utterances: &mut Vec<Utterance>, lang: &str) {
    // Matches BA2's DisfluencyReplacementEngine which ran after ASR on all utterances.
    cleanup::apply_disfluency_replacements(utterances, lang);

    // Matches BA2's NgramRetraceEngine which ran after disfluency on all utterances.
    cleanup::apply_retrace_detection(utterances, lang);

    // 2026-04-23 transcribe-pipeline correction: utterance-initial
    // cap. The two per-word rules (I-cap, title-period strip) have
    // already run pre-retokenize in `finalize_words_to_chunks`;
    // this post-retokenize hook handles the per-utterance rule that
    // needs to see utterance boundaries. English-gated.
    cleanup::apply_english_transcribe_rules_post_retokenize(utterances, lang);

    // Final pass: strip CHAT-illegal characters (engine-emitted
    // noise like Whisper's bare `:` or exotic Unicode) that would
    // otherwise tank the transcript at the
    // `transcript_from_asr_utterances` gate. Runs AFTER number
    // expansion so `$12` etc. are already in word form by now.
    cleanup::sanitize_chat_illegal_chars_in_utterances(utterances);
}

/// Whether retokenization ends an utterance after this word: the word is, or
/// ends with, a sentence terminator, after the `。` normalization
/// [`retokenize`] applies. The one statement of that rule for code that must
/// agree with retokenization, such as where a long turn is cut.
pub(super) fn ends_utterance(word: &AsrWord) -> bool {
    let text = retokenize_text(word.text.as_str());
    is_ending_punct(&text) || ends_with_ending_punct(&text)
}

/// The text [`retokenize`] judges a word by: Japanese `。` as `.`, and the
/// inverted Spanish marks as spaces. One function, so [`ends_utterance`]
/// cannot drift from what retokenization does.
fn retokenize_text(text: &str) -> Cow<'_, str> {
    if text.contains(['。', '¿', '¡']) {
        Cow::Owned(text.replace('。', ".").replace(['¿', '¡'], " "))
    } else {
        Cow::Borrowed(text)
    }
}

/// Check if a word is or ends with a sentence-ending punctuation mark.
fn is_ending_punct(word: &str) -> bool {
    if ENDING_PUNCT.contains(&word) {
        return true;
    }
    // Check RTL punctuation
    for (rtl, _) in RTL_PUNCT {
        if word == *rtl {
            return true;
        }
    }
    false
}

/// Check if a word ends with ending punctuation (last char).
fn ends_with_ending_punct(word: &str) -> bool {
    match word.chars().last() {
        Some(c) => {
            let mut buf = [0u8; 4];
            is_ending_punct(c.encode_utf8(&mut buf))
        }
        None => false,
    }
}

/// Normalize RTL punctuation to ASCII equivalent.
fn normalize_punct(word: &str) -> String {
    for (rtl, ascii) in RTL_PUNCT {
        if word == *rtl {
            return ascii.to_string();
        }
    }
    word.to_string()
}

/// Split a word list into utterances based on punctuation boundaries.
///
/// This is the simple punctuation-based retokenizer (no BERT model).
///
/// # Panic safety
///
/// The `unwrap()` calls on `buf.last()`, `buf.last_mut()`, and `buf.pop()` are
/// all immediately preceded by `buf.push(word)` within the same loop iteration,
/// so `buf` is guaranteed non-empty at each call site.
#[allow(clippy::unwrap_used)]
pub(super) fn retokenize(speaker: SpeakerIndex, words: Vec<AsrWord>) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    let mut buf: Vec<AsrWord> = Vec::new();

    for word in words {
        // Normalize Japanese period and remove inverted punctuation
        let word = AsrWord {
            text: word.text.map(|t| retokenize_text(t).into_owned()),
            ..word
        };

        buf.push(word);

        // `buf.push(word)` immediately above guarantees `buf` is
        // non-empty at this point.
        #[allow(clippy::unwrap_used)]
        let last_text = buf.last().unwrap().text.as_str();

        if is_ending_punct(last_text) {
            // Whole word is ending punct, flush utterance
            let punct = normalize_punct(last_text);
            // Same `buf.push` guarantee above.
            #[allow(clippy::unwrap_used)]
            {
                buf.last_mut().unwrap().text = AsrNormalizedText::new(punct);
            }
            utterances.push(Utterance {
                speaker,
                words: std::mem::take(&mut buf),
                lang: None,
            });
        } else if ends_with_ending_punct(last_text) {
            // Last character is ending punct, split the word.
            // `buf.push` above guarantees pop() returns Some.
            #[allow(clippy::unwrap_used)]
            let text = buf.pop().unwrap();
            let s = text.text.as_str();
            let last_char_boundary = s.char_indices().next_back().map(|(i, _)| i).unwrap_or(0);
            let word_part = &s[..last_char_boundary];
            let punct_part = &s[last_char_boundary..];

            if !word_part.is_empty() {
                buf.push(AsrWord::new(word_part, text.start_ms, text.end_ms));
            }
            buf.push(AsrWord::new(normalize_punct(punct_part), None, None));
            utterances.push(Utterance {
                speaker,
                words: std::mem::take(&mut buf),
                lang: None,
            });
        }
    }

    // Flush remaining words
    if !buf.is_empty() {
        // Remove trailing MOR_PUNCT
        while buf
            .last()
            .is_some_and(|w| MOR_PUNCT.contains(&w.text.as_str()))
        {
            buf.pop();
        }
        if !buf.is_empty() {
            // Append terminator
            buf.push(AsrWord::new(".", None, None));
            utterances.push(Utterance {
                speaker,
                words: buf,
                lang: None,
            });
        }
    }

    utterances
}
