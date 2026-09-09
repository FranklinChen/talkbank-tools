use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::PositionalDomain;
use talkbank_model::model::{ChatFile, DependentTier, Line};

use crate::dp_align::{self, AlignResult, MatchMode};
use crate::extract::{self, ExtractedUtterance};
use crate::wer_conform;

use super::metrics::MetricAccumulator;
use super::model::{
    CompareStatus, CompareToken, ComparisonBundle, GoldCoverage, GoldWordMatch, UtteranceComparison,
};

/// Where one token of the concatenated gold came from.
///
/// Named rather than a `(usize, usize)`: two same-typed indices in positional
/// order is the shape that lets a silent swap survive review, which the
/// workspace charter rules out at domain seams.
#[derive(Debug, Clone, Copy)]
struct GoldSource {
    /// Gold utterance the token belongs to.
    utterance: usize,
    /// Index into the flattened gold word list.
    word: usize,
}

/// One alignment: a main utterance, the gold utterances that mapped to it, or
/// either one alone.
///
/// Phase 2 aligns units, not utterances, so that the three situations it has to
/// cover are one situation with different inputs:
///
/// - both sides present: the ordinary stitch;
/// - main alone (`gold` empty): a main utterance nothing mapped to, which
///   `dp_align` renders as all-insertions against an empty reference;
/// - gold alone (`main` `None`): a gold utterance phase 1 could not place,
///   which `dp_align` renders as all-deletions against an empty hypothesis.
///
/// Writing the last two as their own emission loops is what let them drift
/// apart from the first, and from each other, on the anchor rule and on the
/// `cwer` bookkeeping.
#[derive(Debug, Clone)]
struct AlignmentUnit {
    /// Main utterance being aligned, or `None` for unplaced gold.
    main: Option<usize>,
    /// Gold utterances concatenated into this alignment, in gold order.
    gold: Vec<usize>,
}

#[derive(Debug, Clone)]
struct FlattenedWordInfo {
    utterance_index: usize,
    word_position: usize,
    compare_position: usize,
    pos: Option<String>,
}

/// Punctuation and fillers to exclude from comparison (matching BA2 behavior).
///
/// Terminators are recognized via the typed `Terminator` enum so the set
/// stays in lockstep with the grammar. Separators (`,`, `‡`, `„`) are
/// additionally excluded because BA2's compare skipped them too.
pub(in crate::compare) fn is_punct_or_filler(word: &str) -> bool {
    static FILLERS: &[&str] = &["um", "uhm", "em", "mhm", "uhhm", "eh", "uh", "hm"];

    let w = word.trim();
    talkbank_model::model::content::Terminator::is_chat_terminator(w)
        || matches!(w, "," | "‡" | "„")
        || FILLERS.contains(&w.to_lowercase().as_str())
}

fn is_punct_pos(pos: Option<&str>) -> bool {
    pos.is_some_and(|value| value.eq_ignore_ascii_case("PUNCT"))
}

/// Apply conform_words per word, returning expanded tokens and an index
/// mapping back to the original word list.
///
/// `mapping[j]` = index into the original `words` list that `conformed[j]`
/// originated from.
pub(in crate::compare) fn conform_with_mapping(words: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut conformed = Vec::new();
    let mut mapping = Vec::new();
    for (idx, word) in words.iter().enumerate() {
        let expanded = wer_conform::conform_words(std::slice::from_ref(word));
        for token in expanded {
            conformed.push(token);
            mapping.push(idx);
        }
    }
    (conformed, mapping)
}

/// Find the best local main-token window for one gold utterance.
///
/// This follows the BA2 compare engine's rough-pass strategy:
/// - compare contiguous windows using bag-of-words overlap
/// - only consider windows near the gold utterance length
/// - prefer better overlap, then more aligner matches (so order is respected
///   and cross-utterance fragments that happen to be dense don't beat
///   in-utterance matches), then fewer wasted tokens, then the latest window
///
/// Tiebreaker order matches BA2-master 86230ef (2026-04-17,
/// "fix part 2 of compare").
///
/// `main_utts[i]` is the utterance index that `main_tokens[i]` belongs to.
/// BA2 (compare.py:200-249) projects each candidate window to its majority
/// utterance by trimming non-majority tokens from both ends before scoring,
/// preventing cross-utterance bag-of-words inflation. The projection is a
/// no-op for windows whose tokens all share one utterance index.
pub(in crate::compare) fn find_best_segment(
    gold_tokens: &[String],
    main_tokens: &[String],
    main_utts: &[usize],
) -> (usize, usize) {
    debug_assert_eq!(
        main_tokens.len(),
        main_utts.len(),
        "main_tokens and main_utts must be parallel arrays",
    );
    if gold_tokens.is_empty() || main_tokens.is_empty() {
        return (0, 0);
    }

    let gold_len = gold_tokens.len();
    let main_len = main_tokens.len();
    let min_window = std::cmp::max(1, gold_len.saturating_sub(2));
    let max_window = std::cmp::min(main_len, gold_len + 2);
    let gold_counts = token_counts(gold_tokens);

    // Comparing `overlap` is equivalent to BA2's float `score = overlap /
    // gold_len` because `gold_len` is constant within the call, and lets us
    // collapse all four tiebreaker axes into one tuple comparison. `Reverse`
    // flips the lower-is-better waste axis.
    let mut best_window = (0usize, std::cmp::min(main_len, gold_len));
    let mut best_key: Option<(usize, usize, Reverse<usize>, usize)> = None;

    for span in min_window..=max_window {
        for start in 0..=(main_len - span) {
            let end = start + span;
            // Majority-project the candidate window before scoring (BA2
            // compare.py:200-249). If trimming non-majority tokens from
            // both ends empties the window, BA2 `continue`s the loop.
            let (ts, te) = match majority_project(main_utts, start, end) {
                Some(window) => window,
                None => continue,
            };
            let projected = &main_tokens[ts..te];
            let projected_len = te - ts;
            let overlap = token_overlap(projected, &gold_counts);
            let waste = projected_len.saturating_sub(overlap);
            let align_matches = count_alignment_matches(projected, gold_tokens);

            let key = (overlap, align_matches, Reverse(waste), te);
            if best_key.is_none_or(|best| key > best) {
                best_window = (ts, te);
                best_key = Some(key);
            }
        }
    }

    // No tokens overlap at all → return an empty window so the caller doesn't
    // consume main tokens that belong to a later gold utterance.
    if best_key.is_none_or(|(overlap, ..)| overlap == 0) {
        return (0, 0);
    }

    best_window
}

/// Trim a candidate window down to its majority-utterance subrange.
///
/// Mirrors BA2's `Counter(window_utts).most_common(1)` followed by
/// leading/trailing non-majority-token trim. Returns `None` if the projected
/// window is empty.
fn majority_project(main_utts: &[usize], start: usize, end: usize) -> Option<(usize, usize)> {
    if start >= end {
        return None;
    }

    let majority = majority_utterance(&main_utts[start..end])?;

    let mut ts = start;
    while ts < end && main_utts[ts] != majority {
        ts += 1;
    }
    let mut te = end;
    while te > ts && main_utts[te - 1] != majority {
        te -= 1;
    }
    if te <= ts {
        return None;
    }
    Some((ts, te))
}

/// The utterance index holding the most tokens in `utts`, first-seen on ties.
///
/// The single owner of BA2's majority rule, used both by [`majority_project`]
/// (which then trims to that utterance's subrange) and by phase 1 (which turns
/// a chosen window into the one main utterance a gold utterance maps to).
///
/// Ties resolve to **first-seen** because BA2's `Counter.most_common` is
/// insertion-ordered; `Iterator::max_by_key` would pick last-seen, the
/// opposite. Stated once, here, so the two callers cannot drift apart on it.
fn majority_utterance(utts: &[usize]) -> Option<usize> {
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for &utt in utts {
        match counts.iter_mut().find(|(idx, _)| *idx == utt) {
            Some(entry) => entry.1 += 1,
            None => counts.push((utt, 1)),
        }
    }
    counts
        .iter()
        .fold(None::<(usize, usize)>, |best, &cur| match best {
            Some(b) if b.1 >= cur.1 => Some(b),
            _ => Some(cur),
        })
        .map(|(utt, _)| utt)
}

fn count_alignment_matches(window: &[String], gold_tokens: &[String]) -> usize {
    dp_align::align(window, gold_tokens, MatchMode::CaseInsensitive)
        .into_iter()
        .filter(|item| matches!(item, AlignResult::Match { .. }))
        .count()
}

fn token_counts(tokens: &[String]) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.as_str()).or_insert(0) += 1;
    }
    counts
}

fn token_overlap(window: &[String], gold_counts: &HashMap<&str, usize>) -> usize {
    let mut window_counts = HashMap::new();
    for token in window {
        *window_counts.entry(token.as_str()).or_insert(0) += 1;
    }

    window_counts
        .iter()
        .map(|(token, count)| std::cmp::min(*count, *gold_counts.get(token).unwrap_or(&0)))
        .sum()
}

/// Compare a main transcript against a gold-standard reference.
///
/// Both inputs are parsed CHAT files. Words are extracted from the Mor
/// domain (excluding punctuation and fillers), normalized via
/// `conform_words`, then aligned with the Hirschberg DP aligner.
///
/// Returns per-utterance comparison annotations and aggregate metrics.
pub fn compare(
    main_file: &ChatFile,
    gold_file: &ChatFile,
    gold_coverage: GoldCoverage,
) -> ComparisonBundle {
    // 1. Extract words from both files
    let main_utts = extract::extract_words(main_file, PositionalDomain::Mor);
    let gold_utts = extract::extract_words(gold_file, PositionalDomain::Mor);

    // 2. Flatten words, filtering punctuation and fillers
    let (main_words, main_info) = flatten_words(main_file, &main_utts);
    let (gold_words, gold_info) = flatten_words(gold_file, &gold_utts);

    // 3. Apply conform with index mapping
    let (conformed_main, main_map) = conform_with_mapping(&main_words);
    let (conformed_gold, gold_map) = conform_with_mapping(&gold_words);

    // Per-conformed-token utterance index, parallel to `conformed_main`.
    // `find_best_segment` needs this for BA2's majority-projection step
    // (compare.py:200-249), which trims cross-utterance leaders/trailers
    // before scoring each candidate window.
    let conformed_main_utts: Vec<usize> = main_map
        .iter()
        .map(|&orig_idx| main_info[orig_idx].utterance_index)
        .collect();

    // 4. Partition conformed gold tokens by utterance so compare can work
    // sequentially, one gold utterance at a time.
    let mut gold_utt_tokens: Vec<Vec<String>> = vec![Vec::new(); gold_utts.len()];
    let mut gold_utt_maps: Vec<Vec<usize>> = vec![Vec::new(); gold_utts.len()];
    for (conformed_idx, token) in conformed_gold.iter().enumerate() {
        let orig_gold_idx = gold_map[conformed_idx];
        let gold_utt_idx = gold_info[orig_gold_idx].utterance_index;
        gold_utt_tokens[gold_utt_idx].push(token.clone());
        gold_utt_maps[gold_utt_idx].push(orig_gold_idx);
    }

    // 5. PHASE 1, the mapping pass.
    //
    // Run the window search for each gold utterance, but use its answer ONLY
    // to decide which main utterance that gold utterance corresponds to. The
    // window is a bag-of-words heuristic for locating material; it is not a
    // decision about what deserves to be scored.
    //
    // This is the half that used to do everything. It aligned inside the
    // chosen window and advanced past it, so any main token the window did not
    // select was never emitted in any status: not a match, not an insertion,
    // not anything. Reported WER was therefore systematically lower than the
    // truth by however many hypothesis words the windows happened to miss.
    let mut gold_to_main: Vec<Option<usize>> = vec![None; gold_utts.len()];
    let mut search_start = 0usize;

    for gold_utt_idx in 0..gold_utts.len() {
        let g_tokens = &gold_utt_tokens[gold_utt_idx];
        if g_tokens.is_empty() {
            continue;
        }

        let remaining_main = &conformed_main[search_start..];
        let remaining_main_utts = &conformed_main_utts[search_start..];
        let (win_start, win_end) = find_best_segment(g_tokens, remaining_main, remaining_main_utts);
        let abs_start = search_start + win_start;
        let abs_end = search_start + win_end;

        if abs_end > abs_start {
            gold_to_main[gold_utt_idx] =
                majority_utterance(&conformed_main_utts[abs_start..abs_end]);
        }

        // Advance past the window so a later gold utterance cannot re-consume
        // main tokens an earlier one already claimed. The cursor still matters
        // for MAPPING even though it no longer bounds what gets aligned.
        search_start = abs_end;
    }

    // 6. PHASE 2, the stitch.
    //
    // One alignment per main utterance, over that utterance's FULL conformed
    // token span, against the concatenated gold tokens of every gold utterance
    // that mapped to it. Every main token now sits inside exactly one such
    // span, so a token the window missed surfaces as an insertion instead of
    // disappearing.
    //
    // No rotation here. Rotation existed to re-phase a window that had been
    // cut at an arbitrary offset; aligning a whole utterance has no such
    // offset, and rotating one would scramble real token order.
    let mut main_to_gold: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (gold_utt_idx, mapped) in gold_to_main.iter().enumerate() {
        if let Some(main_idx) = mapped {
            main_to_gold
                .entry(*main_idx)
                .or_default()
                .push(gold_utt_idx);
        }
    }

    // Each main utterance's conformed tokens as a contiguous span.
    //
    // `flatten_words` walks utterances in order and `conform_with_mapping`
    // expands each word in place, so `conformed_main_utts` is non-decreasing
    // and one utterance's tokens are always adjacent. That makes a span of two
    // indices enough, where a per-utterance index vector would be a heap
    // allocation per utterance holding what `start..end` already says. The
    // invariant is asserted rather than assumed, in the style
    // `find_best_segment` uses for its own parallel-array precondition.
    let mut main_utt_spans: Vec<Range<usize>> = vec![0..0; main_utts.len()];
    for (conformed_idx, &utt_idx) in conformed_main_utts.iter().enumerate() {
        let Some(span) = main_utt_spans.get_mut(utt_idx) else {
            continue;
        };
        if span.start == span.end {
            *span = conformed_idx..conformed_idx + 1;
        } else {
            debug_assert_eq!(
                span.end, conformed_idx,
                "conformed main tokens of one utterance must be contiguous",
            );
            span.end = conformed_idx + 1;
        }
    }

    let mut main_positioned: Vec<Vec<(f64, CompareToken)>> = vec![Vec::new(); main_utts.len()];
    let mut gold_positioned: Vec<Vec<(f64, CompareToken)>> = vec![Vec::new(); gold_utts.len()];
    let mut gold_word_matches = Vec::new();
    let mut metrics = MetricAccumulator::default();
    let mut last_global_main_anchor: Option<(usize, usize)> = None;

    // Every alignment the file needs, in one list.
    //
    // Main utterances in index order (each with the gold that mapped to it,
    // possibly none), then the gold utterances nothing could place. Under
    // `GoldCoverage::Partial` a main utterance with no gold is dropped here,
    // which is the whole of what `Partial` means: a filter on the unit list
    // rather than a branch wrapped around a duplicated emission block.
    let mut units: Vec<AlignmentUnit> = Vec::new();
    for main_idx in 0..main_utts.len() {
        let gold = main_to_gold.get(&main_idx).cloned().unwrap_or_default();
        if gold.is_empty() && gold_coverage == GoldCoverage::Partial {
            continue;
        }
        units.push(AlignmentUnit {
            main: Some(main_idx),
            gold,
        });
    }
    units.extend(
        gold_to_main
            .iter()
            .enumerate()
            .filter(|(_, mapped)| mapped.is_none())
            .map(|(gold_utt_idx, _)| AlignmentUnit {
                main: None,
                gold: vec![gold_utt_idx],
            }),
    );

    for unit in &units {
        let main_span = unit
            .main
            .map_or(0..0, |main_idx| main_utt_spans[main_idx].clone());
        let main_tokens = &conformed_main[main_span.clone()];

        // Concatenated gold, with a parallel map back to the gold utterance and
        // word each token came from, so an aligned gold token still knows its
        // origin once several utterances have been joined. Parallel rather than
        // one vector of pairs because `dp_align::align` wants a `&[String]`.
        let mut gold_tokens: Vec<String> = Vec::new();
        let mut gold_sources: Vec<GoldSource> = Vec::new();
        for &gold_utt_idx in &unit.gold {
            for (within, token) in gold_utt_tokens[gold_utt_idx].iter().enumerate() {
                gold_tokens.push(token.clone());
                gold_sources.push(GoldSource {
                    utterance: gold_utt_idx,
                    word: gold_utt_maps[gold_utt_idx][within],
                });
            }
        }

        // `Some` for any unit with a main utterance; `None` for a gold-only
        // unit, which is why the anchor chain below still needs its last arm.
        let default_main_anchor = main_tokens.first().map(|_| {
            let info = &main_info[main_map[main_span.start]];
            (info.utterance_index, info.word_position)
        });

        let alignment = dp_align::align(main_tokens, &gold_tokens, MatchMode::CaseInsensitive);
        let mut local_main_cursor = 0usize;
        let mut local_gold_cursor = 0usize;
        let mut last_gold_word_position: Option<usize> = None;
        let mut local_main_anchor: Option<(usize, usize)> = None;

        for item in alignment {
            match item {
                AlignResult::Match { key, .. } => {
                    let orig_main_idx = main_map[main_span.start + local_main_cursor];
                    let main_word = &main_info[orig_main_idx];
                    let GoldSource {
                        utterance: gold_utt_idx,
                        word: orig_gold_idx,
                    } = gold_sources[local_gold_cursor];
                    let gold_word = &gold_info[orig_gold_idx];

                    // BA2 (compare.py:540-550) attributes the gold form's
                    // POS to every Match, the gold standard is what the
                    // reviewer needs to see, not the transcriber's tag.
                    let token = CompareToken {
                        text: key,
                        pos: gold_word.pos.clone(),
                        status: CompareStatus::Match,
                    };
                    metrics.record(&token);
                    main_positioned[main_word.utterance_index]
                        .push((main_word.word_position as f64, token.clone()));
                    gold_positioned[gold_utt_idx].push((gold_word.word_position as f64, token));

                    let structural_match = GoldWordMatch {
                        gold_utterance_index: gold_utt_idx,
                        gold_word_position: gold_word.compare_position,
                        main_utterance_index: main_word.utterance_index,
                        main_word_position: main_word.compare_position,
                    };
                    if gold_word_matches.last() != Some(&structural_match) {
                        gold_word_matches.push(structural_match);
                    }

                    local_main_anchor = Some((main_word.utterance_index, main_word.word_position));
                    last_global_main_anchor = local_main_anchor;
                    last_gold_word_position = Some(gold_word.word_position);
                    local_main_cursor += 1;
                    local_gold_cursor += 1;
                }
                AlignResult::ExtraPayload { key, .. } => {
                    let orig_main_idx = main_map[main_span.start + local_main_cursor];
                    let main_word = &main_info[orig_main_idx];

                    let token = CompareToken {
                        text: key,
                        pos: main_word.pos.clone(),
                        status: CompareStatus::ExtraMain,
                    };
                    metrics.record(&token);
                    main_positioned[main_word.utterance_index]
                        .push((main_word.word_position as f64, token.clone()));
                    // Attribute the insertion to the gold utterance the
                    // alignment had reached, falling back to the first one
                    // mapped here when it precedes every gold token.
                    let owning_gold = gold_sources
                        .get(local_gold_cursor)
                        .or_else(|| gold_sources.last())
                        .map(|source| source.utterance);
                    if let Some(gold_utt_idx) = owning_gold {
                        gold_positioned[gold_utt_idx].push((
                            last_gold_word_position.map_or(-0.5, |pos| pos as f64 + 0.5),
                            token,
                        ));
                    }

                    local_main_anchor = Some((main_word.utterance_index, main_word.word_position));
                    last_global_main_anchor = local_main_anchor;
                    local_main_cursor += 1;
                }
                AlignResult::ExtraReference { key, .. } => {
                    let GoldSource {
                        utterance: gold_utt_idx,
                        word: orig_gold_idx,
                    } = gold_sources[local_gold_cursor];
                    let gold_word = &gold_info[orig_gold_idx];

                    let token = CompareToken {
                        text: key,
                        pos: gold_word.pos.clone(),
                        status: CompareStatus::ExtraGold,
                    };
                    metrics.record(&token);
                    gold_positioned[gold_utt_idx]
                        .push((gold_word.word_position as f64, token.clone()));

                    if let Some((target_utt, target_word_pos)) = local_main_anchor
                        .or(default_main_anchor)
                        .or(last_global_main_anchor)
                        && let Some(target_tokens) = main_positioned.get_mut(target_utt)
                    {
                        target_tokens.push((target_word_pos as f64 + 0.5, token));
                    }

                    last_gold_word_position = Some(gold_word.word_position);
                    local_gold_cursor += 1;
                }
            }
        }

        metrics.finish_utterance();
    }

    // 7. Append the gold utterance terminator as a PUNCT token so gold-projected
    // `%xsrep` / `%xsmor` lines match batchalign2-master output shape.
    for (gold_utt_idx, terminator) in collect_utterance_terminators(gold_file)
        .into_iter()
        .enumerate()
    {
        let Some(terminator) = terminator else {
            continue;
        };
        gold_positioned[gold_utt_idx].push((
            gold_utt_tokens[gold_utt_idx].len() as f64,
            CompareToken {
                text: terminator,
                pos: Some("PUNCT".to_string()),
                status: CompareStatus::Match,
            },
        ));
    }

    // 8. Stabilize per-utterance token order.
    for tokens in &mut main_positioned {
        tokens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    for tokens in &mut gold_positioned {
        tokens.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    let main_utterances = build_utterance_comparisons(&main_utts, main_positioned);
    let gold_utterances = build_utterance_comparisons(&gold_utts, gold_positioned);

    ComparisonBundle {
        main_utterances,
        gold_utterances,
        gold_word_matches,
        metrics: metrics.finish(),
    }
}

fn build_utterance_comparisons(
    utterances: &[ExtractedUtterance],
    positioned: Vec<Vec<(f64, CompareToken)>>,
) -> Vec<UtteranceComparison> {
    utterances
        .iter()
        .enumerate()
        .map(|(utt_idx, utt)| UtteranceComparison {
            utterance_index: utt_idx,
            speaker: utt.speaker.as_str().to_string(),
            tokens: positioned[utt_idx]
                .iter()
                .map(|(_, token)| token.clone())
                .collect(),
        })
        .collect()
}

/// Flatten extracted utterances into a word list and info vector.
///
/// Returns:
/// - `words`: cleaned text for each non-punct/non-filler word
/// - `info`: word position and `%mor`-derived metadata for each word
fn flatten_words(
    chat_file: &ChatFile,
    utts: &[ExtractedUtterance],
) -> (Vec<String>, Vec<FlattenedWordInfo>) {
    let mut words = Vec::new();
    let mut info = Vec::new();
    let mor_positions = collect_mor_pos_labels(chat_file);

    for utt in utts {
        let mut compare_position = 0usize;
        for extracted in &utt.words {
            let text = extracted.text.as_str();
            let pos = mor_positions
                .get(utt.utterance_index.raw())
                .and_then(|positions| positions.get(extracted.utterance_word_index.raw()))
                .cloned()
                .flatten();
            if is_punct_or_filler(text) || is_punct_pos(pos.as_deref()) {
                continue;
            }
            words.push(text.to_string());
            info.push(FlattenedWordInfo {
                utterance_index: utt.utterance_index.raw(),
                word_position: extracted.utterance_word_index.raw(),
                compare_position,
                pos,
            });
            compare_position += 1;
        }
    }

    (words, info)
}

fn collect_mor_pos_labels(chat_file: &ChatFile) -> Vec<Vec<Option<String>>> {
    let mut utterance_positions = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            let mor_positions = utt
                .dependent_tiers
                .iter()
                .find_map(|tier| match &tier.tier {
                    DependentTier::Mor(mor) => Some(
                        mor.items()
                            .iter()
                            .map(|item| Some(item.main.pos.to_string().to_uppercase()))
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default();
            utterance_positions.push(mor_positions);
        }
    }
    utterance_positions
}

pub(in crate::compare) fn collect_utterance_terminators(
    chat_file: &ChatFile,
) -> Vec<Option<String>> {
    let mut terminators = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            terminators.push(
                utt.main
                    .content
                    .terminator
                    .as_ref()
                    .map(|term| term.to_chat_string()),
            );
        }
    }
    terminators
}
