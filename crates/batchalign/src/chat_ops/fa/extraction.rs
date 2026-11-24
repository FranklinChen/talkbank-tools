//! Word extraction for forced alignment (Wor domain).

use talkbank_model::alignment::helpers::{TierDomain, WordItem, counts_for_tier, walk_words};
use talkbank_model::model::{UtteranceContent, Word};

/// Collect alignable word texts from utterance content for forced alignment.
///
/// Uses the `Wor` alignment domain to decide which words are alignable.
/// Extracted texts are the cleaned (CHAT-marker-free) forms.
///
/// Compound fillers (`&-you_know`, `&-sort_of`) are split at underscores
/// into separate words for the FA engine, because ASR models (Whisper,
/// wav2vec) return them as multiple tokens that the DP aligner must
/// match individually.
///
/// * `content` - The top-level content items of an utterance.
/// * `out` - Accumulator that word texts are pushed into.
pub fn collect_fa_words(content: &[UtteranceContent], out: &mut Vec<String>) {
    walk_wor_units(content, &mut |word| push_fa_word(word, out));
}

/// Call `f` for each word that occupies a `%wor` slot, in main-tier order:
/// the one walk every FA pass that asks "which words does FA align" uses.
///
/// # Why this exists
///
/// `walk_words`'s domain argument decides only which CONTAINERS the walk
/// enters; it filters no leaf. Whether a word occupies a `%wor` slot is
/// `counts_for_tier(word, Wor)`, which excludes untranscribed words (`xxx`),
/// phonological fragments (`&+fr`), nonwords (`&~`) and omissions. Passing
/// `Some(TierDomain::Wor)` and reading every leaf as alignable looks right
/// and is wrong: an utterance-initial `xxx` became the "first alignable
/// word" and hid a leading filler from the edge-filler pass. Owning the walk
/// and the predicate together here removes that choice from every caller.
///
/// A replaced word's FA unit is the word the transcriber wrote, not its
/// replacement; separators are never FA units. Containers are walked with
/// no domain, as extraction always has: FA aligns retraced words too.
pub fn walk_wor_units<'a>(content: &'a [UtteranceContent], f: &mut impl FnMut(&'a Word)) {
    walk_words(content, None, &mut |leaf| {
        let word = match leaf {
            WordItem::Word(word) => word,
            WordItem::ReplacedWord(replaced) => &replaced.word,
            WordItem::Separator(_) => return,
        };
        if counts_for_tier(word, TierDomain::Wor) {
            f(word);
        }
    });
}

/// Push a word's cleaned text to the FA word list, splitting compound fillers
/// at underscores so each part aligns separately against ASR output.
fn push_fa_word(word: &Word, out: &mut Vec<String>) {
    for part in super::split_compound_filler(word) {
        out.push(part);
    }
}
