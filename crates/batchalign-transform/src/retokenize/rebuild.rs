//! AST content rebuilding: walks old content, replacing/splicing words to
//! match NLP tokenization.

// Wildcard matches over closed enums are denied here, following chatter's
// per-file ratchet.
//
// This file walks utterance content in lockstep with a `word_counter` that
// indexes the word list SENT to Stanza, so which variants it descends into is
// not a formatting choice: a variant whose words are counted on one side and
// not the other desynchronizes every later word onto the wrong `%mor` entry.
//
// This used to claim the two sides "agree today ... by coincidence". They did
// not, and the coincidence was the reason. Enumerating the VARIANTS is only
// half the agreement; the other half is the `[e]` exclusion GATE, which the
// extractor applied and this walk did not, so `dog [e] cat .` extracted one
// word and rebuilt two. A deny-lint cannot see that, because the divergence
// was in a gate rather than in a variant list. `excluded_from_mor` is now the
// one place this side states the rule, and
// `word_counter_matches_the_extractor_on_every_exclusion_shape` measures the
// two sides against each other rather than asserting they match.
//
// The variant lists themselves are still hand-maintained here, and chatter
// 0.16.0 exports `UtteranceContent::container_mut` / `BracketedItem::
// container_mut` precisely to end that. Collapsing the twelve container arms
// onto it is the next change; it is deliberately not this one, which is a
// correctness fix with a measured red.
#![deny(clippy::wildcard_enum_match_arm)]
// Test code is exempt, matching this crate's existing treatment of the panic
// lints: `other => panic!("unexpected {other:?}")` is how a test says a variant
// should be unreachable, and denying it there would push tests toward asserting
// less rather than more.
#![cfg_attr(test, allow(clippy::wildcard_enum_match_arm))]

use std::collections::HashSet;

use talkbank_model::alignment::helpers::{
    TierDomain, annotations_have_alignment_ignore, counts_for_tier, is_tag_marker_separator,
};
use talkbank_model::model::ContentAnnotation;
use talkbank_model::model::content::BracketedItems;
use talkbank_model::model::{BracketedItem, Mor, UtteranceContent, Word};

use crate::extract::ExtractedWord;

use super::{
    WordTokenMapping, handle_ending_punct_skip, is_tag_marker_text, resolve_token_text,
    try_parse_token_as_bracketed_item, try_parse_token_as_utterance_content,
    try_parse_token_as_word,
};

/// Mutable state threaded through the retokenize AST walk.
pub(super) struct RetokenizeContext<'a> {
    /// Tree-sitter parser for validating tokens as CHAT words.
    pub parser: &'a talkbank_parser::TreeSitterParser,
    /// Maps original word index to token indices.
    pub mapping: &'a WordTokenMapping,
    /// NLP tokenized output.
    pub stanza_tokens: &'a [String],
    /// Original words extracted from the utterance.
    pub original_words: &'a [ExtractedWord],
    /// Parsed morphosyntax items from the NLP pipeline.
    pub mors: &'a [Mor],
    /// Expected utterance terminator for parse validation.
    pub expected_terminator: Option<&'a str>,
    /// Current position in the original word list.
    pub word_counter: usize,
    /// Current position in the MOR list.
    pub mor_cursor: usize,
    /// Warnings accumulated during retokenization.
    pub diagnostics: Vec<String>,
    /// Tracks which token indices have already produced a word node.
    pub emitted_tokens: HashSet<usize>,
}

/// Whether `[e]`-annotated material is dropped from the `%mor` word list.
///
/// THE gate, stated once, because stating it per-site is what broke.
/// `should_retokenize` is `counts_for_tier`, whose contract is that it depends
/// on the word and "nothing about the containers the word sits inside".
/// chatter's extractor, which builds the very list `word_counter` indexes
/// into, additionally drops alignment-ignored material under `Mor`: see
/// `collect_alignable_word` and `collect_replaced_word` in
/// `talkbank-transform/src/extract.rs`, and `descent`'s container verdict.
///
/// Measured before this gate existed: `dog [e] cat .` extracted 1 word and
/// rebuilt 2; `<dog bone> [e] cat .` extracted 1 and rebuilt 3. Every `%mor`
/// entry after the divergence landed on the wrong word. Pinned by
/// `word_counter_matches_the_extractor_on_an_excluded_word` below.
fn excluded_from_mor(annotations: &[ContentAnnotation]) -> bool {
    annotations_have_alignment_ignore(annotations)
}

fn should_retokenize(word: &Word) -> bool {
    counts_for_tier(word, TierDomain::Mor)
}

/// Rebuild content vector, replacing alignable words with retokenized versions.
pub(super) fn rebuild_content(
    old_content: Vec<UtteranceContent>,
    ctx: &mut RetokenizeContext<'_>,
    new_content: &mut Vec<UtteranceContent>,
) {
    for item in old_content {
        match item {
            UtteranceContent::Word(word) => {
                if should_retokenize(&word) {
                    handle_word_retokenize(*word, ctx, new_content);
                } else {
                    new_content.push(UtteranceContent::Word(word));
                }
            }
            UtteranceContent::AnnotatedWord(mut annotated) => {
                if !excluded_from_mor(&annotated.scoped_annotations)
                    && should_retokenize(&annotated.inner)
                {
                    handle_annotated_word_retokenize(&mut annotated.inner, ctx);
                }
                new_content.push(UtteranceContent::AnnotatedWord(annotated));
            }
            UtteranceContent::ReplacedWord(mut replaced) => {
                if excluded_from_mor(replaced.scoped_annotations.as_slice()) {
                    new_content.push(UtteranceContent::ReplacedWord(replaced));
                } else if replaced.replacement.words.is_empty() {
                    if should_retokenize(&replaced.word) {
                        handle_annotated_word_retokenize(&mut replaced.word, ctx);
                    }
                    new_content.push(UtteranceContent::ReplacedWord(replaced));
                } else {
                    for word in &mut replaced.replacement.words {
                        if should_retokenize(word) {
                            handle_annotated_word_retokenize(word, ctx);
                        }
                    }
                    new_content.push(UtteranceContent::ReplacedWord(replaced));
                }
            }
            UtteranceContent::Group(mut group) => {
                let old_bracketed = group.content.content.take();
                let mut new_bracketed = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed, ctx, &mut new_bracketed);
                group.content.content = BracketedItems::new(new_bracketed);
                new_content.push(UtteranceContent::Group(group));
            }
            UtteranceContent::AnnotatedGroup(mut annotated) => {
                // An excluded container contributed NO words to the list sent
                // to Stanza, so descending would count words that are not
                // there. chatter's `descent` returns `Excluded` for exactly
                // this case.
                if !excluded_from_mor(&annotated.scoped_annotations) {
                    let old_bracketed = annotated.inner.content.content.take();
                    let mut new_bracketed = Vec::with_capacity(old_bracketed.len());
                    rebuild_bracketed_content(old_bracketed, ctx, &mut new_bracketed);
                    annotated.inner.content.content = BracketedItems::new(new_bracketed);
                }
                new_content.push(UtteranceContent::AnnotatedGroup(annotated));
            }
            UtteranceContent::PhoGroup(mut pho) => {
                let old_bracketed = pho.content.content.take();
                let mut new_bracketed = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed, ctx, &mut new_bracketed);
                pho.content.content = BracketedItems::new(new_bracketed);
                new_content.push(UtteranceContent::PhoGroup(pho));
            }
            UtteranceContent::SinGroup(mut sin) => {
                let old_bracketed = sin.content.content.take();
                let mut new_bracketed = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed, ctx, &mut new_bracketed);
                sin.content.content = BracketedItems::new(new_bracketed);
                new_content.push(UtteranceContent::SinGroup(sin));
            }
            UtteranceContent::Quotation(mut quot) => {
                let old_bracketed = quot.content.content.take();
                let mut new_bracketed = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed, ctx, &mut new_bracketed);
                quot.content.content = BracketedItems::new(new_bracketed);
                new_content.push(UtteranceContent::Quotation(quot));
            }
            // A CONTAINER, not a passthrough, and the distinction is
            // load-bearing: chatter's `walk_words` descends into
            // `AnnotatedQuotation` under exactly the rule it uses for
            // `Quotation`, and that walk is what built the word list sent to
            // Stanza. Treating it as a leaf here would leave `word_counter`
            // short by the quotation's word count and land every subsequent
            // `%mor` entry on the wrong word.
            UtteranceContent::AnnotatedQuotation(mut annotated) => {
                // An excluded container contributed NO words to the list sent
                // to Stanza, so descending would count words that are not
                // there. chatter's `descent` returns `Excluded` for exactly
                // this case.
                if !excluded_from_mor(&annotated.scoped_annotations) {
                    let old_bracketed = annotated.inner.content.content.take();
                    let mut new_bracketed = Vec::with_capacity(old_bracketed.len());
                    rebuild_bracketed_content(old_bracketed, ctx, &mut new_bracketed);
                    annotated.inner.content.content = BracketedItems::new(new_bracketed);
                }
                new_content.push(UtteranceContent::AnnotatedQuotation(annotated));
            }
            UtteranceContent::Separator(ref sep) if is_tag_marker_separator(sep) => {
                ctx.word_counter += 1;
                new_content.push(item);
            }
            // Enumerated, not `_`. This arm is a PASSTHROUGH that does not
            // advance `ctx.word_counter`, so membership here is a claim that
            // the variant contributes no words to the list sent to Stanza.
            // That claim is true for `Retrace` and `AnnotatedRetrace` only
            // because chatter's `walk_words` skips retrace content under
            // `TierDomain::Mor`, which is the same walk that built that list;
            // if that rule ever changes, every word after a retrace lands on
            // the wrong `%mor` entry. Listing the variants is what makes the
            // compiler notice a new one instead of silently assuming it too.
            UtteranceContent::Event(_)
            | UtteranceContent::AnnotatedEvent(_)
            | UtteranceContent::Pause(_)
            | UtteranceContent::Retrace(_)
            | UtteranceContent::AnnotatedRetrace(_)
            | UtteranceContent::Action(_)
            | UtteranceContent::AnnotatedAction(_)
            | UtteranceContent::Freecode(_)
            | UtteranceContent::Separator(_)
            | UtteranceContent::OverlapPoint(_)
            | UtteranceContent::InternalBullet(_)
            | UtteranceContent::LongFeatureBegin(_)
            | UtteranceContent::LongFeatureEnd(_)
            | UtteranceContent::UnderlineBegin(_)
            | UtteranceContent::UnderlineEnd(_)
            | UtteranceContent::NonvocalBegin(_)
            | UtteranceContent::NonvocalEnd(_)
            | UtteranceContent::NonvocalSimple(_)
            | UtteranceContent::OtherSpokenEvent(_) => new_content.push(item),
        }
    }
}

fn handle_word_retokenize(
    word: Word,
    ctx: &mut RetokenizeContext<'_>,
    new_content: &mut Vec<UtteranceContent>,
) {
    let orig_idx = ctx.word_counter;
    ctx.word_counter += 1;

    let token_indices = match ctx.mapping.get_nonempty(orig_idx) {
        Some(indices) => indices.to_vec(),
        None => {
            ctx.diagnostics.push(format!(
                "word {orig_idx} has no character-level match in Stanza tokens; keeping original"
            ));
            if ctx.mor_cursor < ctx.mors.len() {
                ctx.mor_cursor += 1;
            }
            new_content.push(UtteranceContent::Word(Box::new(word)));
            return;
        }
    };

    if token_indices.is_empty() {
        if ctx.mor_cursor < ctx.mors.len() {
            ctx.mor_cursor += 1;
        }
        new_content.push(UtteranceContent::Word(Box::new(word)));
        return;
    }

    if token_indices
        .iter()
        .all(|ti| ctx.emitted_tokens.contains(ti))
    {
        return;
    }

    for &ti in &token_indices {
        let token_text = resolve_token_text(&ctx.stanza_tokens[ti], orig_idx, ctx.original_words);
        if token_indices.len() == 1 && word.cleaned_text() == token_text {
            ctx.mor_cursor += 1;
            ctx.emitted_tokens.insert(ti);
            new_content.push(UtteranceContent::Word(Box::new(word)));
            return;
        }
        ctx.mor_cursor += 1;
        ctx.emitted_tokens.insert(ti);
        match try_parse_token_as_utterance_content(
            ctx.parser,
            &token_text,
            ctx.expected_terminator,
            &mut ctx.diagnostics,
        ) {
            Some(content) => new_content.push(content),
            None => {
                new_content.push(UtteranceContent::Word(Box::new(word)));
                #[allow(clippy::unwrap_used)]
                let pos = token_indices.iter().position(|&x| x == ti).unwrap();
                for &remaining_ti in &token_indices[(pos + 1)..] {
                    ctx.emitted_tokens.insert(remaining_ti);
                    ctx.mor_cursor += 1;
                }
                return;
            }
        }
    }
}

fn handle_annotated_word_retokenize(word: &mut Word, ctx: &mut RetokenizeContext<'_>) {
    let orig_idx = ctx.word_counter;
    ctx.word_counter += 1;

    let token_indices = match ctx.mapping.get_nonempty(orig_idx) {
        Some(indices) => indices.to_vec(),
        None => {
            ctx.diagnostics.push(format!(
                "word {orig_idx} has no character-level match in Stanza tokens; keeping original"
            ));
            if ctx.mor_cursor < ctx.mors.len() {
                ctx.mor_cursor += 1;
            }
            return;
        }
    };

    if token_indices.is_empty() {
        if ctx.mor_cursor < ctx.mors.len() {
            ctx.mor_cursor += 1;
        }
        return;
    }

    if token_indices
        .iter()
        .all(|ti| ctx.emitted_tokens.contains(ti))
    {
        return;
    }

    let ti = token_indices[0];
    let token_text = resolve_token_text(&ctx.stanza_tokens[ti], orig_idx, ctx.original_words);
    if word.cleaned_text() != token_text
        && !is_tag_marker_text(&token_text)
        && !handle_ending_punct_skip(&token_text, ctx.expected_terminator, &mut ctx.diagnostics)
        && let Some(parsed) = try_parse_token_as_word(ctx.parser, &token_text, &mut ctx.diagnostics)
    {
        *word = parsed;
    }

    for &ti in &token_indices {
        ctx.emitted_tokens.insert(ti);
    }
    ctx.mor_cursor += token_indices.len();
}

fn rebuild_bracketed_content(
    old_items: Vec<BracketedItem>,
    ctx: &mut RetokenizeContext<'_>,
    new_items: &mut Vec<BracketedItem>,
) {
    for item in old_items {
        match item {
            BracketedItem::Word(word) => {
                if should_retokenize(&word) {
                    handle_bracketed_word_retokenize(*word, ctx, new_items);
                } else {
                    new_items.push(BracketedItem::Word(word));
                }
            }
            BracketedItem::AnnotatedWord(mut annotated) => {
                if !excluded_from_mor(&annotated.scoped_annotations)
                    && should_retokenize(&annotated.inner)
                {
                    handle_annotated_word_retokenize(&mut annotated.inner, ctx);
                }
                new_items.push(BracketedItem::AnnotatedWord(annotated));
            }
            BracketedItem::ReplacedWord(mut replaced) => {
                if excluded_from_mor(replaced.scoped_annotations.as_slice()) {
                    // Nothing to count; fall through to the shared push below.
                } else if replaced.replacement.words.is_empty() {
                    if should_retokenize(&replaced.word) {
                        handle_annotated_word_retokenize(&mut replaced.word, ctx);
                    }
                } else {
                    for word in &mut replaced.replacement.words {
                        if should_retokenize(word) {
                            handle_annotated_word_retokenize(word, ctx);
                        }
                    }
                }
                new_items.push(BracketedItem::ReplacedWord(replaced));
            }
            BracketedItem::Group(mut group) => {
                let old_bracketed = BracketedItems::new(group.content.content.take());
                let mut sub_items = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed.into_vec(), ctx, &mut sub_items);
                group.content.content = BracketedItems::new(sub_items);
                new_items.push(BracketedItem::Group(group));
            }
            BracketedItem::AnnotatedGroup(mut annotated) => {
                // Excluded: see the utterance-level twin above.
                if !excluded_from_mor(&annotated.scoped_annotations) {
                    let old_bracketed = annotated.inner.content.content.take();
                    let mut sub_items = Vec::with_capacity(old_bracketed.len());
                    rebuild_bracketed_content(old_bracketed, ctx, &mut sub_items);
                    annotated.inner.content.content = BracketedItems::new(sub_items);
                }
                new_items.push(BracketedItem::AnnotatedGroup(annotated));
            }
            BracketedItem::PhoGroup(mut pho) => {
                let old_bracketed = BracketedItems::new(pho.content.content.take());
                let mut sub_items = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed.into_vec(), ctx, &mut sub_items);
                pho.content.content = BracketedItems::new(sub_items);
                new_items.push(BracketedItem::PhoGroup(pho));
            }
            BracketedItem::SinGroup(mut sin) => {
                let old_bracketed = BracketedItems::new(sin.content.content.take());
                let mut sub_items = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed.into_vec(), ctx, &mut sub_items);
                sin.content.content = BracketedItems::new(sub_items);
                new_items.push(BracketedItem::SinGroup(sin));
            }
            BracketedItem::Quotation(mut quot) => {
                let old_bracketed = BracketedItems::new(quot.content.content.take());
                let mut sub_items = Vec::with_capacity(old_bracketed.len());
                rebuild_bracketed_content(old_bracketed.into_vec(), ctx, &mut sub_items);
                quot.content.content = BracketedItems::new(sub_items);
                new_items.push(BracketedItem::Quotation(quot));
            }
            // Containers, both of them, for the same reason the utterance-level
            // `AnnotatedQuotation` arm above is: chatter's bracketed walk lists
            // `Group` and `AnnotatedQuotation` in its container arm alongside
            // `AnnotatedGroup` and `Quotation`, so a passthrough here would
            // desynchronise `word_counter` from the list sent to Stanza.
            BracketedItem::AnnotatedQuotation(mut annotated) => {
                // Excluded: see the utterance-level twin above.
                if !excluded_from_mor(&annotated.scoped_annotations) {
                    let old_bracketed = annotated.inner.content.content.take();
                    let mut sub_items = Vec::with_capacity(old_bracketed.len());
                    rebuild_bracketed_content(old_bracketed, ctx, &mut sub_items);
                    annotated.inner.content.content = BracketedItems::new(sub_items);
                }
                new_items.push(BracketedItem::AnnotatedQuotation(annotated));
            }
            BracketedItem::Separator(ref sep) if is_tag_marker_separator(sep) => {
                ctx.word_counter += 1;
                new_items.push(item);
            }
            // The bracketed twin of the passthrough above; same claim, same reason.
            BracketedItem::Event(_)
            | BracketedItem::AnnotatedEvent(_)
            | BracketedItem::Pause(_)
            | BracketedItem::Action(_)
            | BracketedItem::AnnotatedAction(_)
            | BracketedItem::Retrace(_)
            | BracketedItem::AnnotatedRetrace(_)
            | BracketedItem::OverlapPoint(_)
            | BracketedItem::Separator(_)
            | BracketedItem::InternalBullet(_)
            | BracketedItem::Freecode(_)
            | BracketedItem::LongFeatureBegin(_)
            | BracketedItem::LongFeatureEnd(_)
            | BracketedItem::UnderlineBegin(_)
            | BracketedItem::UnderlineEnd(_)
            | BracketedItem::NonvocalBegin(_)
            | BracketedItem::NonvocalEnd(_)
            | BracketedItem::NonvocalSimple(_)
            | BracketedItem::OtherSpokenEvent(_) => new_items.push(item),
        }
    }
}

fn handle_bracketed_word_retokenize(
    word: Word,
    ctx: &mut RetokenizeContext<'_>,
    new_items: &mut Vec<BracketedItem>,
) {
    let orig_idx = ctx.word_counter;
    ctx.word_counter += 1;

    let token_indices = match ctx.mapping.get_nonempty(orig_idx) {
        Some(indices) => indices.to_vec(),
        None => {
            ctx.diagnostics.push(format!(
                "word {orig_idx} has no character-level match in Stanza tokens; keeping original"
            ));
            if ctx.mor_cursor < ctx.mors.len() {
                ctx.mor_cursor += 1;
            }
            new_items.push(BracketedItem::Word(Box::new(word)));
            return;
        }
    };

    if token_indices.is_empty() {
        if ctx.mor_cursor < ctx.mors.len() {
            ctx.mor_cursor += 1;
        }
        new_items.push(BracketedItem::Word(Box::new(word)));
        return;
    }

    if token_indices
        .iter()
        .all(|ti| ctx.emitted_tokens.contains(ti))
    {
        return;
    }

    for &ti in &token_indices {
        let token_text = resolve_token_text(&ctx.stanza_tokens[ti], orig_idx, ctx.original_words);
        if token_indices.len() == 1 && word.cleaned_text() == token_text {
            ctx.mor_cursor += 1;
            ctx.emitted_tokens.insert(ti);
            new_items.push(BracketedItem::Word(Box::new(word)));
            return;
        }
        ctx.mor_cursor += 1;
        ctx.emitted_tokens.insert(ti);
        match try_parse_token_as_bracketed_item(
            ctx.parser,
            &token_text,
            ctx.expected_terminator,
            &mut ctx.diagnostics,
        ) {
            Some(item) => new_items.push(item),
            None => {
                new_items.push(BracketedItem::Word(Box::new(word)));
                #[allow(clippy::unwrap_used)]
                let pos = token_indices.iter().position(|&x| x == ti).unwrap();
                for &remaining_ti in &token_indices[(pos + 1)..] {
                    ctx.emitted_tokens.insert(remaining_ti);
                    ctx.mor_cursor += 1;
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Test code: the panic family is relaxed by policy here, as at every
    // other test site in this workspace.
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::extract::extract_words;
    use talkbank_model::alignment::helpers::PositionalDomain;

    const HEADER: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
                          @ID:\teng|corpus|CHI|2;||||Target_Child|||\n";

    /// Rebuild's `word_counter` must count exactly the words the EXTRACTOR
    /// produced, because that list is what was sent to Stanza and the counter
    /// is what indexes back into it.
    ///
    /// This is the invariant this whole file rests on, and until now it was
    /// asserted only in prose. `should_retokenize` is `counts_for_tier(word,
    /// Mor)`, whose own contract is that it depends on the word and "nothing
    /// about the containers the word sits inside", while chatter's extractor
    /// additionally drops `[e]`-annotated material under `Mor`
    /// (`excluded_by_annotations`, and `extract.rs` applies
    /// `annotations_have_alignment_ignore` at two sites). So the two walks
    /// disagreed on exactly the `[e]` cases, and every `%mor` entry after one
    /// landed on the wrong word.
    fn assert_lockstep(body: &str) {
        let source = format!("{HEADER}*CHI:\t{body}\n@End\n");
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
        let chat_file = match crate::parse_and_validate_with_parser(
            &parser,
            &source,
            talkbank_model::ParseValidateOptions::default(),
        ) {
            Ok(f) => f,
            Err(e) => panic!("fixture {body:?} must parse: {e:?}"),
        };

        let extracted = extract_words(&chat_file, PositionalDomain::Mor);
        let expected = extracted.first().map_or(0, |u| u.words.len());

        let mut utterance = chat_file
            .lines
            .iter()
            .find_map(|line| match line {
                talkbank_model::model::Line::Utterance(u) => Some(u.clone()),
                _ => None,
            })
            .expect("one utterance");

        let mapping = crate::retokenize::build_word_token_mapping(&[], &[]);
        let mut ctx = RetokenizeContext {
            parser: &parser,
            mapping: &mapping,
            stanza_tokens: &[],
            original_words: &[],
            mors: &[],
            expected_terminator: None,
            word_counter: 0,
            mor_cursor: 0,
            diagnostics: Vec::new(),
            emitted_tokens: std::collections::HashSet::new(),
        };
        let old = utterance.main.content.content.take();
        let mut new = Vec::with_capacity(old.len());
        rebuild_content(old, &mut ctx, &mut new);

        assert_eq!(
            ctx.word_counter, expected,
            "{body:?}: the extractor produced {expected} word(s) for Stanza but the \
             rebuild counted {}. Every %mor entry after the divergence lands on the \
             wrong word.",
            ctx.word_counter
        );
    }

    #[test]
    fn word_counter_matches_the_extractor_on_plain_words() {
        assert_lockstep("dog cat .");
    }

    #[test]
    fn word_counter_matches_the_extractor_on_an_excluded_word() {
        assert_lockstep("dog [e] cat .");
    }

    #[test]
    fn word_counter_matches_the_extractor_on_an_excluded_group() {
        assert_lockstep("<dog bone> [e] cat .");
    }

    /// The shapes the fix was NOT written against, which is the point.
    ///
    /// Every case in the two tests above is one the fix was built from; these
    /// are the ones I went looking for afterwards. Each was checked against
    /// `chatter validate` first, because a fixture that is not valid CHAT
    /// tests the parser's refusal, not this walk.
    ///
    /// What is deliberately absent, having been measured rather than assumed:
    /// `[e]` does not attach to material INSIDE a group (`<dog [e] bone>` is
    /// refused), and groups do not nest (`<a <b c> d>` is refused). So the
    /// bracketed twins of the annotated-word and replaced-word gates are
    /// defensive rather than reachable from valid CHAT today. They are kept
    /// because the two levels must state the same rule, and the next
    /// construct that becomes legal must not silently reintroduce the desync.
    #[test]
    fn word_counter_matches_the_extractor_on_every_exclusion_shape() {
        for body in [
            // A replaced word carrying the exclusion, and one not carrying it,
            // so the gate cannot degenerate into "skip all replaced words".
            "dog [: cat] [e] bird .",
            "dog [: cat] bird .",
            // Quotations, bare and inside a group, which are the constructs
            // the 0.16.0 bump added arms for.
            "\u{201c}dog bone\u{201d} [e] cat .",
            "<\u{201c}dog\u{201d}> [e] cat .",
            "<\u{201c}dog\u{201d} bone> [e] cat .",
            // Exclusion last, so a desync cannot be masked by running out of
            // words before it matters.
            "cat dog [e] .",
            // Several exclusions in one utterance.
            "a [e] b c [e] d .",
        ] {
            assert_lockstep(body);
        }
    }
}
