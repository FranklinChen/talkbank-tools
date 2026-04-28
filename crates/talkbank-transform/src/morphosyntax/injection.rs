//! Result injection, clearing, and alignment validation.

use talkbank_model::SpeakerCode;
use talkbank_model::model::{LanguageCode, Line};

use super::{
    BatchItemWithPosition, MappingContext, MisalignmentClass, MisalignmentDiagnostic, MorOutcome,
    MorOutcomeKind, MwtDict, TokenizationMode, UdId, UdResponse, apply_grammatical_invariants,
    map_ud_sentence, map_ud_sentence_expanded,
};
use crate::decisions::{DecisionRecord, DecisionStrategy, MorphosyntaxStrategy};

/// Context supplied by the caller to [`enrich_diagnostic`] so the
/// suspected misalignment class can be inferred more precisely.
enum RetokenizationContext {
    /// Non-retokenize path — Stanza was told to realign to CHAT
    /// boundaries. A mismatch here means realignment did not hold.
    Preserve,
    /// `StanzaRetokenize` mode was used; CHAT was rewritten to match
    /// Stanza's own tokenization, so a mismatch suggests a retokenize
    /// or MWT-expansion bug rather than a realignment issue.
    StanzaRetokenize,
}

/// Enrich an inner [`MisalignmentDiagnostic`] with caller-held context
/// that the inner validator didn't have access to: the Stanza tokens
/// that were sent, and a best-effort suspected-class classification.
fn enrich_diagnostic(
    mut diag: MisalignmentDiagnostic,
    stanza_tokens: &[String],
    context: RetokenizationContext,
) -> MisalignmentDiagnostic {
    if diag.stanza_tokens_after_mapping.is_empty() && !stanza_tokens.is_empty() {
        diag.stanza_tokens_after_mapping = stanza_tokens.to_vec();
    }

    // Infer the suspected class when the inner validator left it
    // `Unknown`. This is a heuristic — the real diagnosis requires a
    // developer looking at the logs — but it points them at the right
    // stage to investigate first.
    if matches!(diag.suspected_class, MisalignmentClass::Unknown) {
        diag.suspected_class = match context {
            RetokenizationContext::StanzaRetokenize => {
                // In retokenize mode the main tier is rewritten to match
                // Stanza's tokenization; a mismatch here usually means
                // the rebuild step dropped or duplicated tokens.
                MisalignmentClass::MwtReassemblyBug
            }
            RetokenizationContext::Preserve => {
                // Non-retokenize: Stanza was told to realign. A mismatch
                // implies either the realignment context wasn't set for
                // this call (RealignmentSkipped), a terminator-filter
                // regression (TerminatorFilterBug), or an MWT-reassembly
                // bug. Without more signal, Unknown is honest.
                MisalignmentClass::Unknown
            }
        };
    }

    diag
}

/// Result of morphosyntax injection: traces + provenance decisions.
#[derive(Debug, Clone)]
pub struct InjectionResult {
    /// Per-utterance retokenization traces for debugging.
    pub retokenization_traces: Vec<RetokenizationInfo>,
    /// Decision records for utterances that were skipped or degraded.
    pub decisions: Vec<DecisionRecord>,
}

/// Retokenization info collected during injection, for trace visualization.
#[derive(Debug, Clone)]
pub struct RetokenizationInfo {
    /// Utterance ordinal (0-based, among processed utterances).
    pub utterance_ordinal: usize,
    /// Original CHAT words.
    pub original_words: Vec<String>,
    /// Stanza tokens after retokenization.
    pub stanza_tokens: Vec<String>,
    /// Word→token index mapping: `mapping[word_idx]` = list of token indices.
    pub mapping: Vec<Vec<usize>>,
    /// Whether the fallback (length-proportional) mapping was used.
    pub used_fallback: bool,
}

// ---------------------------------------------------------------------------
// Result injection (from NLP callback)
// ---------------------------------------------------------------------------

/// Inject UD NLP results back into utterances.
///
/// Applies special form overrides (@c -> c|, @s -> L2|xxx) and
/// optionally retokenizes the main tier based on the [`TokenizationMode`].
///
/// # Errors
///
/// Returns `Err` if a `line_idx` no longer points to an utterance, or if
/// retokenization or morphosyntax injection fails for any utterance.
pub fn inject_results(
    parser: &talkbank_parser::TreeSitterParser,
    chat_file: &mut talkbank_model::model::ChatFile,
    batch_items: Vec<BatchItemWithPosition>,
    responses: Vec<UdResponse>,
    lang: &LanguageCode,
    tokenization_mode: TokenizationMode,
    mwt: &MwtDict,
) -> Result<InjectionResult, String> {
    use talkbank_model::model::FormType;
    use talkbank_model::model::dependent_tier::mor::PosCategory;

    let mut retokenization_traces: Vec<RetokenizationInfo> = Vec::new();
    let mut decisions: Vec<DecisionRecord> = Vec::new();

    for (ud_resp, (line_idx, utt_ordinal, item, words)) in responses.into_iter().zip(batch_items) {
        if let Some(ud_sentence) = ud_resp.sentences.first() {
            let utt = match &mut chat_file.lines[line_idx] {
                Line::Utterance(u) => u,
                _ => {
                    return Err(format!(
                        "Line at index {line_idx} is no longer an utterance"
                    ));
                }
            };

            let ctx = MappingContext { lang: lang.clone() };

            // Apply grammatical-invariant rewrites to correct known
            // Stanza defects (e.g., English copula 's + progressive
            // misanalyzed as possessive-gerund). The rewrite returns
            // the input unchanged when no rule fires, so this is safe
            // to call unconditionally. See
            // `crate::morphosyntax` for the current rule set and
            // `book/src/reference/stanza-limitations.md` for the
            // versioned defect registry.
            let ud_sentence_rescued = apply_grammatical_invariants(ud_sentence, &ctx);

            // Choose mapping strategy based on tokenization mode:
            //
            // - Preserve: merge Range token components into one clitic MOR
            //   (verb|go~part|to) so MOR count == CHAT word count.
            // - StanzaRetokenize: produce one MOR per component word so
            //   MOR count == Stanza token count (after filtering Range parents).
            //   The retokenize path rewrites the main tier with expanded tokens.
            let (mut mors, mut gra_relations) = {
                let map_result = if tokenization_mode == TokenizationMode::StanzaRetokenize {
                    map_ud_sentence_expanded(&ud_sentence_rescued, &ctx)
                } else {
                    map_ud_sentence(&ud_sentence_rescued, &ctx)
                };
                match map_result {
                    Ok(result) => result,
                    Err(e) => {
                        // Same policy as injection failures below: record and
                        // continue. Stanza occasionally returns structurally
                        // invalid UD (e.g., multiple heads=0) for specific
                        // utterances; we log those loudly and proceed.
                        decisions.push(DecisionRecord::new_and_trace(
                            line_idx,
                            utt.main.speaker.as_str().to_string(),
                            DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::MappingFailed),
                            format!("ud_to_chat_error={e}"),
                            true,
                        ));
                        continue;
                    }
                }
            };

            // Apply special form markers
            for (mor, (form_type, resolved_lang)) in mors.iter_mut().zip(item.special_forms.iter())
            {
                if resolved_lang.is_some() {
                    mor.main.pos = PosCategory::new("L2");
                    mor.main.lemma =
                        talkbank_model::model::dependent_tier::mor::MorStem::new("xxx");
                    continue;
                }

                if let Some(ft) = form_type {
                    let pos_tag = match ft {
                        FormType::A => "a",
                        FormType::B => "b",
                        FormType::C => "c",
                        FormType::D => "d",
                        FormType::F => "f",
                        FormType::FP => "fp",
                        FormType::G => "g",
                        FormType::I => "i",
                        FormType::K => "k",
                        FormType::L => "l",
                        FormType::LS => "ls",
                        FormType::N => "n",
                        FormType::O => "o",
                        FormType::P => "p",
                        FormType::Q => "q",
                        FormType::SAS => "sas",
                        FormType::SI => "si",
                        FormType::SL => "sl",
                        FormType::T => "t",
                        FormType::U => "u",
                        FormType::WP => "wp",
                        FormType::X => "x",
                        FormType::UserDefined(name) => {
                            tracing::warn!(form_type = %name, "user-defined form type not mapped to CHAT POS, skipping override");
                            continue;
                        }
                    };

                    mor.main.pos = PosCategory::new(pos_tag);
                }
            }

            if tokenization_mode == TokenizationMode::StanzaRetokenize {
                // Exclude Range parent tokens: including them alongside their
                // components would overcount tokens and break MOR alignment.
                let mut tokens: Vec<String> = ud_sentence
                    .words
                    .iter()
                    .filter(|w| !matches!(&w.id, UdId::Range(_, _)))
                    .map(|w| {
                        if w.text.contains(char::is_whitespace) {
                            w.text.chars().filter(|c| !c.is_whitespace()).collect()
                        } else {
                            w.text.clone()
                        }
                    })
                    .collect();

                // Apply MWT lexicon overrides: when a Stanza token matches an
                // MWT entry, splice in the expansion tokens (and duplicate the
                // corresponding Mor/GRA items so counts stay aligned).
                if !mwt.is_empty() {
                    let mut expanded_tokens = Vec::with_capacity(tokens.len());
                    let mut expanded_mors = Vec::with_capacity(mors.len());
                    let mut expanded_gra = Vec::with_capacity(gra_relations.len());

                    for (tok_idx, tok) in tokens.iter().enumerate() {
                        let tok_lower = tok.to_lowercase();
                        if let Some(expansion) =
                            mwt.get(&tok_lower).or_else(|| mwt.get(tok.as_str()))
                        {
                            // Replace this token with the expansion tokens.
                            expanded_tokens.extend(expansion.iter().cloned());

                            // For the first expansion token, keep the original
                            // Mor/GRA. For subsequent tokens, duplicate so the
                            // alignment stays correct.
                            if tok_idx < mors.len() {
                                expanded_mors.push(mors[tok_idx].clone());
                                for _ in 1..expansion.len() {
                                    expanded_mors.push(mors[tok_idx].clone());
                                }
                            }
                            if tok_idx < gra_relations.len() {
                                expanded_gra.push(gra_relations[tok_idx].clone());
                                for _ in 1..expansion.len() {
                                    expanded_gra.push(gra_relations[tok_idx].clone());
                                }
                            }
                        } else {
                            expanded_tokens.push(tok.clone());
                            if tok_idx < mors.len() {
                                expanded_mors.push(mors[tok_idx].clone());
                            }
                            if tok_idx < gra_relations.len() {
                                expanded_gra.push(gra_relations[tok_idx].clone());
                            }
                        }
                    }

                    tokens = expanded_tokens;
                    mors = expanded_mors;
                    gra_relations = expanded_gra;
                }

                // Collect retokenization trace info before modifying the AST.
                {
                    use crate::retokenize::{
                        build_word_token_mapping, try_deterministic_word_token_mapping,
                    };
                    let mapping = build_word_token_mapping(&words, &tokens);
                    let used_fallback =
                        try_deterministic_word_token_mapping(&words, &tokens).is_none();
                    retokenization_traces.push(RetokenizationInfo {
                        utterance_ordinal: utt_ordinal,
                        original_words: words.iter().map(|w| w.text.as_str().to_string()).collect(),
                        stanza_tokens: tokens.clone(),
                        mapping: (0..words.len())
                            .map(|i| mapping.tokens_for_word(i).to_vec())
                            .collect(),
                        used_fallback,
                    });
                }

                if let Err(diag) = crate::retokenize::retokenize_utterance(
                    parser,
                    utt,
                    &words,
                    &tokens,
                    mors,
                    Some(item.terminator.as_str().to_string()),
                    gra_relations,
                ) {
                    // File-level absorption: convert the typed diagnostic
                    // to a MorOutcome::MisalignmentBug, emit its
                    // DecisionRecord, continue. The diagnostic is loud
                    // because it surfaces through `%xalign`; never silent.
                    let enriched =
                        enrich_diagnostic(diag, &tokens, RetokenizationContext::StanzaRetokenize);
                    let outcome = MorOutcome {
                        line_idx,
                        speaker: SpeakerCode::new(utt.main.speaker.as_str()),
                        kind: MorOutcomeKind::MisalignmentBug(enriched),
                    };
                    if let Some(mut record) = outcome.to_decision_record() {
                        // Retokenization-path failures share the
                        // MisalignmentBug outcome class but have their
                        // own strategy label for %xalign output.
                        record.strategy = DecisionStrategy::Morphosyntax(
                            MorphosyntaxStrategy::RetokenizationFailed,
                        );
                        record.trace();
                        decisions.push(record);
                    }
                    continue;
                }
            } else if let Err(diag) = crate::inject::inject_morphosyntax(
                utt,
                mors,
                Some(item.terminator.as_str().to_string()),
                gra_relations,
            ) {
                // Per-utterance injection failure: the 1-to-1 invariant
                // (CHAT alignable-word count == Mor count after mapping)
                // was violated. This is always a bug — the pipeline was
                // supposed to produce exactly as many Mors as there were
                // CHAT words, and did not. File-level absorption: log
                // loudly via DecisionRecord, continue with the next
                // utterance rather than killing the whole file. Every
                // mismatch produces a record, so a systemic regression
                // (e.g. 2026-04-17 comma-drop) is visible as a corpus-wide
                // warning spike instead of a silent quality drop.
                let enriched = enrich_diagnostic(diag, &[], RetokenizationContext::Preserve);
                let outcome = MorOutcome {
                    line_idx,
                    speaker: SpeakerCode::new(utt.main.speaker.as_str()),
                    kind: MorOutcomeKind::MisalignmentBug(enriched),
                };
                if let Some(record) = outcome.to_decision_record() {
                    record.trace();
                    decisions.push(record);
                }
                continue;
            }
        } else {
            if let Line::Utterance(utt) = &chat_file.lines[line_idx] {
                decisions.push(DecisionRecord::new_and_trace(
                    line_idx,
                    utt.main.speaker.as_str().to_string(),
                    DecisionStrategy::Morphosyntax(MorphosyntaxStrategy::NlpNoSentences),
                    "stanza_returned_empty_response".into(),
                    true,
                ));
            }
        }
    }

    Ok(InjectionResult {
        retokenization_traces,
        decisions,
    })
}
