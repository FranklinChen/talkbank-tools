//! Pre-FA utterance bullet expansion for edge fillers.
//!
//! When a filler (`&-you_know`, `&-um`) sits at the start or end of an
//! utterance, the UTR-assigned bullet may be too narrow, ASR engines
//! typically don't transcribe fillers, so UTR never "sees" them. The
//! filler's audio lives in the gap between utterances, outside the bullet.
//!
//! This pass expands utterance bullets into adjacent gaps so the FA engine
//! receives audio covering the full utterance including edge fillers.
//! Runs after UTR but before FA grouping.

use talkbank_model::model::{ChatFile, Line, UtteranceContent, WordCategory};

/// Maximum expansion (ms) into a gap for a single edge filler.
///
/// Fillers in natural speech typically last 200-600 ms. We cap at 1500 ms
/// to avoid consuming too much inter-utterance silence even when large gaps
/// exist (e.g., long speaker turn transitions). Separate from the
/// group-level `TRAILING_GAP_EXTENSION_MS` in `grouping.rs`, this
/// adjusts individual utterance bullets before grouping.
const MAX_FILLER_EXPANSION_MS: u64 = 1500;

/// What one edge word is.
///
/// A named pair of cases rather than a `bool`, because the two arms drive
/// different behaviour and a bare `true` at a call site says nothing about
/// which question it answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// The word is a filler, so its audio is in the adjacent gap and the
    /// bullet should be expanded to reclaim it.
    Filler,
    /// The word is ordinary, so the bullet already covers it.
    Ordinary,
}

/// The alignable words at an utterance's two edges.
///
/// A sum rather than `(bool, bool)`, because "this utterance has no alignable
/// words at all" is a third state the pair cannot express. It used to be
/// written as `(false, false)` by `unwrap_or(false)`, which asserts that both
/// edges are ordinary words about an utterance that HAS no edges: an invented
/// fact in the same shape as an observed one, which is the defect class this
/// module's neighbours exist to make unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeFillers {
    /// Nothing in this utterance reaches the Wor tier, so there is no first or
    /// last word to classify and nothing to expand around.
    NoAlignableWords,
    /// The utterance has at least one alignable word. With exactly one, both
    /// edges name that same word, which is correct: it is both the first and
    /// the last.
    Edges {
        /// The first alignable word.
        leading: Edge,
        /// The last alignable word.
        trailing: Edge,
    },
}

/// One utterance's summary used by the mutation pass.
///
/// A struct rather than the `(Option<(u64, u64)>, bool, bool)` it used to be:
/// the two `bool`s were indistinguishable at every use, and the tuple's own
/// meaning lived in a comment above the alias.
#[derive(Debug, Clone, Copy)]
struct UtteranceFillerSummary {
    /// The utterance's bullet bounds, if it has a bullet.
    bullet: Option<(u64, u64)>,
    /// What sits at its edges.
    edges: EdgeFillers,
}

/// Expand utterance bullets to cover edge fillers in inter-utterance gaps.
///
/// ASR engines don't transcribe fillers, so UTR-assigned bullets stop at
/// the last recognized word. Fillers at utterance edges are audible in the
/// gap but outside the bullet. This expansion reclaims that audio for FA.
pub fn expand_bullets_for_edge_fillers(chat_file: &mut ChatFile) {
    // Single pass: collect boundaries + edge filler status together.
    let utterance_data: Vec<UtteranceFillerSummary> = chat_file
        .lines
        .iter()
        .filter_map(|line| match line {
            Line::Utterance(u) => {
                let boundary = u
                    .main
                    .content
                    .bullet
                    .as_ref()
                    .and_then(talkbank_model::BulletState::as_parsed)
                    .map(|b| (b.timing.start_ms, b.timing.end_ms));
                Some(UtteranceFillerSummary {
                    bullet: boundary,
                    edges: detect_edge_fillers(&u.main.content.content),
                })
            }
            _ => None,
        })
        .collect();

    // Mutation pass: expand bullets using collected neighbor data.
    let mut utt_idx = 0;
    for line in &mut chat_file.lines {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let bullet = match utt
            .main
            .content
            .bullet
            .as_mut()
            .and_then(talkbank_model::BulletState::as_parsed_mut)
        {
            Some(b) => b,
            None => {
                utt_idx += 1;
                continue;
            }
        };

        // An utterance with no alignable words has no edge to expand around.
        // Written as its own arm rather than folded in with "neither edge is a
        // filler", because they are different facts that happen to call for the
        // same action today.
        let (leading, trailing) = match utterance_data[utt_idx].edges {
            EdgeFillers::NoAlignableWords => {
                utt_idx += 1;
                continue;
            }
            EdgeFillers::Edges { leading, trailing } => (leading, trailing),
        };

        // Expand start backward: reclaim gap before this utterance for a leading filler.
        match leading {
            Edge::Ordinary => {}
            Edge::Filler => {
                // The first utterance has no predecessor, so the gap it can
                // reclaim runs back to the start of the file.
                let prev_end = match utt_idx.checked_sub(1) {
                    Some(previous) => utterance_data[previous].bullet.map(|(_, end)| end),
                    None => Some(0),
                };
                if let Some(prev_end) = prev_end
                    && bullet.timing.start_ms > prev_end
                {
                    let gap = bullet.timing.start_ms - prev_end;
                    // expansion <= gap/2, so start_ms -= expansion cannot underflow.
                    let expansion = (gap / 2).min(MAX_FILLER_EXPANSION_MS);
                    bullet.timing.start_ms -= expansion;
                }
            }
        }

        // Expand end forward: reclaim gap after this utterance for a trailing filler.
        match trailing {
            Edge::Ordinary => {}
            Edge::Filler => {
                // The last utterance has no successor, so there is no measured
                // boundary to expand toward and nothing is reclaimed.
                let next_start = utterance_data
                    .get(utt_idx + 1)
                    .and_then(|next| next.bullet.map(|(start, _)| start));
                if let Some(next_start) = next_start
                    && next_start > bullet.timing.end_ms
                {
                    let gap = next_start - bullet.timing.end_ms;
                    let expansion = (gap / 2).min(MAX_FILLER_EXPANSION_MS);
                    bullet.timing.end_ms += expansion;
                }
            }
        }

        utt_idx += 1;
    }
}

/// Detect whether the first and last alignable words are fillers.
///
/// "Alignable" is the `%wor` occupant set of
/// [`walk_wor_units`](super::extraction::walk_wor_units), so an `xxx`,
/// fragment or nonword at an edge never stands in for the word FA aligns.
pub fn detect_edge_fillers(content: &[UtteranceContent]) -> EdgeFillers {
    let mut first: Option<Edge> = None;
    let mut last: Option<Edge> = None;

    super::extraction::walk_wor_units(content, &mut |word| {
        let edge = match word.category {
            Some(WordCategory::Filler) => Edge::Filler,
            _ => Edge::Ordinary,
        };
        // `get_or_insert` rather than an `is_none` test: the first alignable
        // word is whichever one gets here first, and asking twice invites the
        // two questions to disagree.
        first.get_or_insert(edge);
        last = Some(edge);
    });

    // `first` and `last` are set together on every visit, so one being present
    // means both are. The pair is matched rather than unwrapped so that the
    // "no alignable words" case is answered by its own variant instead of by a
    // fabricated `(false, false)`.
    match (first, last) {
        (Some(leading), Some(trailing)) => EdgeFillers::Edges { leading, trailing },
        _ => EdgeFillers::NoAlignableWords,
    }
}

#[cfg(test)]
mod tests {
    use super::{Edge, EdgeFillers, detect_edge_fillers};
    use talkbank_model::model::Line;
    use talkbank_parser_tests::test_error::ParseProductTestExt;

    /// The edge classification of the one utterance in `main_tier`.
    fn edges_of(main_tier: &str) -> EdgeFillers {
        let chat = talkbank_parser::TreeSitterParser::new()
            .expect("parser")
            .parse_chat_file(&format!(
                "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
                 @ID:\teng|test|CHI|||||Child|||\n*CHI:\t{main_tier}\n@End\n"
            ))
            .expect_built();
        let utterance = chat
            .lines
            .iter()
            .find_map(|line| match line {
                Line::Utterance(utterance) => Some(utterance),
                _ => None,
            })
            .expect("one utterance");
        detect_edge_fillers(&utterance.main.content.content)
    }

    /// Guards: The edge-filler pass classifies the first and last words that occupy a `%wor` slot, so an untranscribed word, a phonological fragment or a nonword at an utterance edge never hides a filler beside it from bullet expansion.
    /// Kind: policy
    /// Not a type: `walk_wor_units` is now the only walk the pass can use and owns the `%wor` predicate, so no caller can read a domain-gated walk as filtered; which word categories occupy `%wor` is chatter's `counts_for_tier` policy, and only classifying real parsed words shows the pass applies it.
    /// State: None. Main tiers parsed in memory by chatter's parser; no audio, worker or file.
    /// Concurrency: None. Each main tier is parsed into its own file value and its edges read once; nothing is shared between rows.
    #[test]
    fn non_wor_words_at_an_edge_do_not_hide_a_filler() {
        for (main_tier, expected) in [
            (
                "xxx &-um hello .",
                EdgeFillers::Edges {
                    leading: Edge::Filler,
                    trailing: Edge::Ordinary,
                },
            ),
            (
                "hello &-um &+fr .",
                EdgeFillers::Edges {
                    leading: Edge::Ordinary,
                    trailing: Edge::Filler,
                },
            ),
            (
                "&~ah hello .",
                EdgeFillers::Edges {
                    leading: Edge::Ordinary,
                    trailing: Edge::Ordinary,
                },
            ),
            ("xxx .", EdgeFillers::NoAlignableWords),
        ] {
            assert_eq!(edges_of(main_tier), expected, "{main_tier}");
        }
    }
}
