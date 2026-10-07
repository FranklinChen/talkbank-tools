//! Complete, source-bound expansion of scoped words and replacement targets.

use talkbank_model::model::annotation::ReplacementWords;
use talkbank_model::model::{
    Annotated, BracketedContent, BracketedItem, Group, ReplacedWord, UtteranceContent, Word,
    WordContent,
};

use super::{RetokenizeContext, excluded_from_mor, should_retokenize};
use crate::retokenize::{MappingBasis, resolve_token_text, try_parse_token_as_word};

#[derive(Debug, thiserror::Error)]
enum ExpansionRefusal {
    #[error("no character-level mapping for scoped word")]
    Unmapped,
    #[error("the mapped source word does not match this scoped word")]
    WrongSource,
    #[error("length-based mapping cannot establish annotation ownership")]
    LengthMapping,
    #[error("a model token crosses a scoped source-word boundary")]
    CrossesScope,
    #[error("a scoped model token was already emitted")]
    AlreadyEmitted,
    #[error("mapped model token {0} is absent")]
    MissingToken(usize),
    #[error("model token is not a CHAT word")]
    NotAWord,
    #[error("rewriting this decorated word cannot preserve its word-specific evidence")]
    DecoratedWord,
}

/// Nonempty by construction; consumers cannot select only the first token.
struct ExpandedWords {
    first: Word,
    rest: Vec<Word>,
}

impl ExpandedWords {
    fn retained(word: Word) -> Self {
        Self {
            first: word,
            rest: Vec::new(),
        }
    }

    fn into_replacement_words(self) -> ReplacementWords {
        let mut words = ReplacementWords::from_word(self.first);
        for word in self.rest {
            words.push(word);
        }
        words
    }
}

/// Owns the complete parsed expansion and borrows the very context that
/// admitted it. Commitment takes no independently supplied indices or cursor.
struct AdmittedExpansion<'ctx, 'source> {
    ctx: &'ctx mut RetokenizeContext<'source>,
    token_indices: Vec<usize>,
    words: ExpandedWords,
}

impl<'ctx, 'source> AdmittedExpansion<'ctx, 'source> {
    fn admit(
        word: &Word,
        ctx: &'ctx mut RetokenizeContext<'source>,
    ) -> Result<Self, ExpansionRefusal> {
        let original_index = ctx.word_counter;
        ctx.word_counter += 1;
        let indices = ctx
            .mapping
            .get_nonempty(original_index)
            .ok_or(ExpansionRefusal::Unmapped)?
            .to_vec();
        let original = ctx
            .original_words
            .get(original_index)
            .ok_or(ExpansionRefusal::WrongSource)?;
        if original.text.as_str() != word.cleaned_text() {
            return Err(ExpansionRefusal::WrongSource);
        }
        if ctx.mapping.basis() != MappingBasis::Text {
            return Err(ExpansionRefusal::LengthMapping);
        }
        for index in &indices {
            if ctx.emitted_tokens.contains(index) {
                return Err(ExpansionRefusal::AlreadyEmitted);
            }
            if (0..ctx.mapping.word_count()).any(|other| {
                other != original_index && ctx.mapping.tokens_for_word(other).contains(index)
            }) {
                return Err(ExpansionRefusal::CrossesScope);
            }
        }

        let mut words = Vec::with_capacity(indices.len());
        for index in &indices {
            let token = ctx
                .stanza_tokens
                .get(*index)
                .ok_or(ExpansionRefusal::MissingToken(*index))?;
            let text = resolve_token_text(token, original_index, ctx.original_words);
            if indices.len() == 1 && word.cleaned_text() == text {
                words.push(word.clone());
            } else {
                // Do not drop suffixes, content markings or timing in order
                // to make an expansion fit. Plain contractions are splittable.
                if word.word_id.is_some()
                    || word.category.is_some()
                    || word.form_type.is_some()
                    || word.lang.is_some()
                    || word.part_of_speech.is_some()
                    || word.inline_bullet.is_some()
                    || word.content().len() != 1
                    || !word
                        .content()
                        .iter()
                        .all(|part| matches!(part, WordContent::Text(_)))
                {
                    return Err(ExpansionRefusal::DecoratedWord);
                }
                words.push(
                    try_parse_token_as_word(ctx.parser, &text, &mut ctx.diagnostics)
                        .ok_or(ExpansionRefusal::NotAWord)?,
                );
            }
        }
        let mut words = words.into_iter();
        let first = words.next().ok_or(ExpansionRefusal::Unmapped)?;
        Ok(Self {
            ctx,
            token_indices: indices,
            words: ExpandedWords {
                first,
                rest: words.collect(),
            },
        })
    }

    fn commit(self) -> ExpandedWords {
        self.ctx.mor_cursor += self.token_indices.len();
        self.ctx.emitted_tokens.extend(self.token_indices);
        self.words
    }
}

fn expand_word(word: Word, ctx: &mut RetokenizeContext<'_>) -> ExpandedWords {
    if !should_retokenize(&word) {
        return ExpandedWords::retained(word);
    }
    match AdmittedExpansion::admit(&word, ctx) {
        Ok(admitted) => admitted.commit(),
        Err(refused) => {
            ctx.diagnostics.push(format!(
                "retokenize scoped word: {refused}; keeping original"
            ));
            ExpandedWords::retained(word)
        }
    }
}

pub(super) enum RebuiltScopedWord {
    Word(Box<Annotated<Word>>),
    Group(Annotated<Group>),
}

impl RebuiltScopedWord {
    pub(super) fn into_utterance_content(self) -> UtteranceContent {
        match self {
            Self::Word(word) => UtteranceContent::AnnotatedWord(word),
            Self::Group(group) => UtteranceContent::AnnotatedGroup(group),
        }
    }

    pub(super) fn into_bracketed_item(self) -> BracketedItem {
        match self {
            Self::Word(word) => BracketedItem::AnnotatedWord(word),
            Self::Group(group) => BracketedItem::AnnotatedGroup(group),
        }
    }
}

pub(super) fn rebuild_annotated_word(
    annotated: Annotated<Word>,
    ctx: &mut RetokenizeContext<'_>,
) -> RebuiltScopedWord {
    if excluded_from_mor(&annotated.scoped_annotations) {
        return RebuiltScopedWord::Word(Box::new(annotated));
    }
    let Annotated {
        inner,
        scoped_annotations,
        span,
    } = annotated;
    let expansion = expand_word(inner, ctx);
    if expansion.rest.is_empty() {
        RebuiltScopedWord::Word(Box::new(
            Annotated::new(expansion.first, scoped_annotations).with_span(span),
        ))
    } else {
        let items = std::iter::once(expansion.first)
            .chain(expansion.rest)
            .map(|word| BracketedItem::Word(Box::new(word)))
            .collect();
        let group = Group::new(BracketedContent::new(items)).with_span(span);
        RebuiltScopedWord::Group(Annotated::new(group, scoped_annotations).with_span(span))
    }
}

pub(super) fn rebuild_replaced_word(
    mut replaced: ReplacedWord,
    ctx: &mut RetokenizeContext<'_>,
) -> ReplacedWord {
    if excluded_from_mor(replaced.scoped_annotations.as_slice()) {
        return replaced;
    }
    let mut original = replaced.replacement.words.clone().into_vec().into_iter();
    let Some(first) = original.next() else {
        ctx.diagnostics
            .push("retokenize: replacement has no admitted words; keeping original".into());
        return replaced;
    };
    let mut rewritten = expand_word(first, ctx).into_replacement_words();
    for word in original {
        for expanded in expand_word(word, ctx).into_replacement_words() {
            rewritten.push(expanded);
        }
    }
    replaced.replacement.words = rewritten;
    replaced
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_context(check: impl FnOnce(Word, RetokenizeContext<'_>)) {
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
        let source = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
                      @ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tcan't [= cannot] .\n@End\n";
        let chat = crate::parse_and_validate_with_parser(
            &parser,
            source,
            talkbank_model::ParseValidateOptions::default(),
        )
        .expect("valid source");
        let word = chat
            .lines
            .iter()
            .find_map(|line| match line {
                talkbank_model::model::Line::Utterance(utterance) => {
                    match &utterance.main.content.content[0] {
                        UtteranceContent::AnnotatedWord(annotated) => Some(annotated.inner.clone()),
                        _ => None,
                    }
                }
                _ => None,
            })
            .expect("scoped source word");
        let extracted = crate::extract::extract_words(
            &chat,
            talkbank_model::alignment::helpers::PositionalDomain::Mor,
        );
        let original = &extracted[0].words;
        let tokens = vec!["ca".to_owned(), "n't".to_owned()];
        let mapping = crate::retokenize::build_word_token_mapping(original, &tokens);
        check(
            word,
            RetokenizeContext {
                parser: &parser,
                mapping: &mapping,
                stanza_tokens: &tokens,
                original_words: original,
                mors: &[],
                expected_terminator: Some("."),
                word_counter: 0,
                mor_cursor: 0,
                diagnostics: Vec::new(),
                emitted_tokens: std::collections::HashSet::new(),
            },
        );
    }

    #[test]
    fn an_admitted_expansion_commits_all_tokens_to_its_producing_context() {
        with_context(|word, mut ctx| {
            let expansion = AdmittedExpansion::admit(&word, &mut ctx)
                .unwrap_or_else(|error| panic!("complete expansion: {error}"));
            let words = expansion.commit().into_replacement_words();
            assert_eq!(
                words.iter().map(Word::cleaned_text).collect::<Vec<_>>(),
                ["ca", "n't"]
            );
            assert_eq!(ctx.mor_cursor, 2);
            assert_eq!(ctx.emitted_tokens, std::collections::HashSet::from([0, 1]));
        });
    }

    #[test]
    fn dropping_a_plan_does_not_commit_any_model_positions() {
        with_context(|word, mut ctx| {
            drop(
                AdmittedExpansion::admit(&word, &mut ctx)
                    .unwrap_or_else(|error| panic!("admitted: {error}")),
            );
            assert_eq!(ctx.mor_cursor, 0);
            assert!(ctx.emitted_tokens.is_empty());
        });
    }

    #[test]
    fn a_missing_later_token_refuses_before_committing_the_first() {
        with_context(|word, mut ctx| {
            ctx.stanza_tokens = &ctx.stanza_tokens[..1];
            let refusal = AdmittedExpansion::admit(&word, &mut ctx)
                .err()
                .expect("refusal");
            assert!(matches!(refusal, ExpansionRefusal::MissingToken(1)));
            assert_eq!(ctx.mor_cursor, 0);
            assert!(ctx.emitted_tokens.is_empty());
        });
    }

    #[test]
    fn a_different_source_word_cannot_acquire_the_expansion() {
        with_context(|_word, mut ctx| {
            let refusal = AdmittedExpansion::admit(&Word::simple("dog"), &mut ctx)
                .err()
                .expect("refusal");
            assert!(matches!(refusal, ExpansionRefusal::WrongSource));
            assert_eq!(ctx.mor_cursor, 0);
            assert!(ctx.emitted_tokens.is_empty());
        });
    }
}
