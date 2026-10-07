use crate::chat_ops::fa::utr::AsrTimingToken;
use batchalign_transform::{AdmittedSourceChat, parse_source_with_parser};
use talkbank_model::model::TranscriptName;
use talkbank_parser::TreeSitterParser;

/// Ground-truth audio window for an utterance, recovered from the synthetic
/// cadence used by [`build_scenario`]. Drift detection compares the UTR-
/// assigned bullet against this envelope.
#[derive(Debug, Clone, Copy)]
pub(super) struct ExpectedWindow {
    pub(super) utt_index: usize,
    pub(super) expected_start_ms: u64,
    pub(super) expected_end_ms: u64,
    pub(super) surviving_asr_words: usize,
}

/// Complete source admission, synthetic acoustic tokens and expected windows
/// remain one owned scenario until the algorithm consumes them together.
pub(super) struct AdmittedDriftScenario {
    pub(super) source: AdmittedSourceChat<'static>,
    pub(super) tokens: Vec<AsrTimingToken>,
    pub(super) expected: Vec<ExpectedWindow>,
}

/// Which overlap convention the synthesized CHAT document uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OverlapConvention {
    /// `⌈ … ⌉` top-overlap + `⌊ … ⌋` bottom-overlap, the CA bracket pair.
    CaBracket,
    /// `+<` lazy-overlap-precedes linker.
    LazyPrecedes,
    /// `&*SPK:word` inline backchannel tokens embedded in the main utterance.
    InlineBackchannel,
    /// All three conventions interleaved.
    Mixed,
    /// No overlap markers at all.
    None,
}

/// Knobs that control how a drift scenario's CHAT+ASR pair is built.
#[derive(Debug, Clone, Copy)]
pub(super) struct DriftParams {
    /// Number of main-speaker utterances to emit.
    pub(super) n_utts: usize,
    /// Overlap convention used when an utterance carries a marker.
    pub(super) convention: OverlapConvention,
    /// Fraction of utterances (0.0-1.0) that carry an overlap marker.
    pub(super) overlap_density: f64,
    /// Fraction of transcript words (0.0-1.0) that are DROPPED from the
    /// synthesized ASR stream, to emulate missed recognition.
    pub(super) asr_missing_rate: f64,
    /// If true, every transcript word is a short stopword (< 4 chars), which
    /// tests the anchor-sparse matching case.
    pub(super) stopword_only: bool,
}

/// Shared-vocabulary high-frequency tokens. All speakers draw from this pool so
/// that vocabulary alone never disambiguates a match, the DP must rely on
/// temporal position, which is precisely what breaks under overlap.
const AMBIGUOUS_WORDS: &[&str] = &[
    "the", "and", "it", "is", "of", "to", "a", "that", "we", "he", "she", "they", "this", "was",
    "had",
];

/// All < 4 characters: stresses the anchor-sparse fallback pathway.
const STOPWORD_VOCAB: &[&str] = &["the", "a", "an", "it", "is", "of", "to", "in", "on"];

/// Anchor density: one anchor token (unique, > 4 chars) per this many words
/// across the document. This fixes the fixture's sparse unique-word cadence.
const ANCHOR_EVERY_N_WORDS: usize = 60;

/// Words per main-speaker utterance.
const WORDS_PER_UTT: usize = 6;

/// Words per backchannel utterance (InlineBackchannel only).
const BACKCHANNEL_WORDS: usize = 2;

/// Nominal duration of a main utterance in the synthetic timeline.
const UTT_DURATION_MS: u64 = 3000;

// Standard LCG multipliers used for deterministic pseudo-random draws in
// this test module. Different constants give statistically independent
// streams for density sampling vs. word selection vs. drop masking
// matters because we want the overlap-pick, word-selection, and drop-mask
// decisions to be uncorrelated even though they share an input seed family.
const LCG_MULT_GLIBC: u64 = 1103515245; // glibc srand48 multiplier
const LCG_INC_GLIBC: u64 = 12345; // glibc srand48 increment
const LCG_MULT_KNUTH: u64 = 6364136223846793005; // Knuth MMIX multiplier
const LCG_MULT_FIB: u64 = 2654435761; // Fibonacci hashing constant
const LCG_MULT_PARK_MILLER: u64 = 48271; // Park-Miller minstd multiplier

/// How far into the host utterance a backchannel nest starts (relative to the
/// host's `start_ms`). Backchannels span ~400 ms inside a 3000 ms host.
const BACKCHANNEL_INSET_MS: u64 = 1200;
const BACKCHANNEL_SPAN_MS: u64 = 400;

/// Role a single utterance plays in the overlap layout. Drives both CHAT
/// markup and the monotonicity exclusion downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UttRole {
    /// Standalone utterance, no overlap markup.
    Solo,
    /// First member of a CA-bracket pair. Wraps head word in `⌈ … ⌉`.
    CaTop,
    /// Second member of a CA-bracket pair. Wraps head word in `⌊ … ⌋`.
    /// Counts as overlap-continuation for the monotonicity checker.
    CaBottom,
    /// First member of a lazy-overlap pair (no `+<` marker).
    LazyLeading,
    /// Second member of a lazy-overlap pair (carries `+<` linker).
    /// Counts as overlap-continuation for the monotonicity checker.
    LazyContinuation,
    /// Short backchannel nested inside a host utterance's time range.
    Backchannel,
}

/// A fully-laid-out utterance: a time range in the synthetic "audio" timeline,
/// the words that will be spoken, and the role it plays in the overlap layout.
///
/// Document order in the produced CHAT matches the order of this Vec. The
/// temporal order (ASR stream) is recovered by flattening per-word timings
/// across ALL layouts and re-sorting by `start_ms`, that re-sort is what
/// produces the interleaving that GlobalUtr's document-order DP cannot
/// reconcile.
#[derive(Debug, Clone)]
struct UttLayout {
    utt_index: usize,
    speaker: &'static str,
    role: UttRole,
    start_ms: u64,
    end_ms: u64,
    words: Vec<String>,
}

/// Geometry for a pair of adjacent overlapping utterances (CaBracket /
/// LazyPrecedes): the second utterance starts halfway through the first
/// and extends half a unit past its end, producing a 1500 ms intersection
/// for `UTT_DURATION_MS = 3000`.
///
/// Returns `(s0, e0, s1, e1)` where `s0 < s1 < e0 < e1`. Callers still decide
/// which speaker (MAI/OTH) and `UttRole` goes into each slot.
fn pair_overlap_ranges(cursor: u64) -> (u64, u64, u64, u64) {
    let s0 = cursor;
    let e0 = s0 + UTT_DURATION_MS * 3 / 2; // +4500
    let s1 = s0 + UTT_DURATION_MS / 2; // +1500
    let e1 = s1 + UTT_DURATION_MS * 3 / 2; // +6000 == s0 + 2*UTT_DURATION_MS
    (s0, e0, s1, e1)
}

/// Geometry for an inline backchannel: a short utterance whose time range
/// sits strictly inside a longer host utterance's range.
///
/// Returns `(host_s, host_e, bc_s, bc_e)` where
/// `host_s < bc_s < bc_e < host_e`. Uses the module-level
/// `BACKCHANNEL_INSET_MS` / `BACKCHANNEL_SPAN_MS` knobs so all tuning stays
/// in one place.
fn backchannel_ranges(cursor: u64) -> (u64, u64, u64, u64) {
    let host_s = cursor;
    let host_e = host_s + UTT_DURATION_MS;
    let bc_s = host_s + BACKCHANNEL_INSET_MS;
    let bc_e = bc_s + BACKCHANNEL_SPAN_MS;
    (host_s, host_e, bc_s, bc_e)
}

/// Step 1: place every utterance in time + assign a role. This is the heart of
/// the redesign: overlap conventions produce intersecting time ranges here,
/// and document-order CHAT emission downstream preserves the ordering that
/// GlobalUtr sees while the ASR re-sort breaks it.
fn layout_utterances_in_time(params: DriftParams) -> Vec<UttLayout> {
    // Deterministic Bernoulli: returns true with frequency ~= p, seeded by k.
    let overlap_picked = |k: usize, p: f64| -> bool {
        ((k as u64).wrapping_mul(LCG_MULT_PARK_MILLER) % 1000) < (p * 1000.0) as u64
    };

    let vocab: &[&str] = if params.stopword_only {
        STOPWORD_VOCAB
    } else {
        AMBIGUOUS_WORDS
    };

    // Utterance-index counter. Distinct from the main-timeline index `k`
    // because backchannels also consume utt indices while not advancing the
    // main timeline.
    let mut next_utt_idx: usize = 0;
    // Running anchor-slot counter, used to decide whether the j-th word is an
    // anchor (every Nth word across the document).
    let mut global_word_counter: usize = 0;
    // Running anchor-number counter, used to mint unique anchor tokens.
    let mut next_anchor_num: usize = 0;

    // Word-picker: closes over the anchor counters so anchors are globally
    // unique AND placed at the right sparse cadence. Stopword-only mode
    // disables anchors entirely.
    let mut mint_words = |seed: u64, count: usize| -> Vec<String> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let make_anchor = !params.stopword_only
                && global_word_counter > 0
                && global_word_counter.is_multiple_of(ANCHOR_EVERY_N_WORDS);
            global_word_counter += 1;
            if make_anchor {
                let mut w = String::from("anchor");
                let mut ordinal = next_anchor_num;
                // Base-26 alphabetic suffixes preserve globally unique rare
                // tokens without illegal English digits. Cadence is unchanged.
                loop {
                    w.push((b'a' + (ordinal % 26) as u8) as char);
                    ordinal /= 26;
                    if ordinal == 0 {
                        break;
                    }
                }
                next_anchor_num += 1;
                out.push(w);
            } else {
                let mix = seed.wrapping_add((i as u64).wrapping_mul(LCG_MULT_KNUTH)) as usize;
                out.push(vocab[mix % vocab.len()].to_string());
            }
        }
        out
    };

    let mut layouts: Vec<UttLayout> = Vec::new();
    // Main-timeline cursor advances one `UTT_DURATION_MS` per main utterance.
    // Overlap pairs under CaBracket/LazyPrecedes still consume 2 main slots of
    // timeline each, just with intersecting intervals; backchannels nest
    // inside their host's range and do NOT advance the cursor.
    let mut cursor_ms: u64 = 0;
    let mut k: usize = 0;
    while k < params.n_utts {
        let seed = (k as u64).wrapping_mul(LCG_MULT_FIB);
        let remaining = params.n_utts - k;

        // Resolve a "pair-style" convention (two overlapping utts) for this k.
        // Mixed rotates through the three overlap styles.
        let effective_conv = match params.convention {
            OverlapConvention::Mixed => match k % 3 {
                0 => OverlapConvention::CaBracket,
                1 => OverlapConvention::LazyPrecedes,
                _ => OverlapConvention::InlineBackchannel,
            },
            other => other,
        };
        // Overall overlap density applies uniformly; Mixed rotates through
        // style at each k. (If we wanted style-specific densities later we
        // could branch here.)
        let wants_overlap_here = overlap_picked(k, params.overlap_density);

        match (effective_conv, wants_overlap_here, remaining >= 2) {
            (OverlapConvention::CaBracket, true, true) => {
                // Pair: two utts whose ranges intersect by 1500 ms.
                let (s0, e0, s1, e1) = pair_overlap_ranges(cursor_ms);
                let words0 = mint_words(seed, WORDS_PER_UTT);
                let words1 = mint_words(seed ^ 0xdead_beef, WORDS_PER_UTT);
                layouts.push(UttLayout {
                    utt_index: next_utt_idx,
                    speaker: "MAI",
                    role: UttRole::CaTop,
                    start_ms: s0,
                    end_ms: e0,
                    words: words0,
                });
                next_utt_idx += 1;
                layouts.push(UttLayout {
                    utt_index: next_utt_idx,
                    speaker: "OTH",
                    role: UttRole::CaBottom,
                    start_ms: s1,
                    end_ms: e1,
                    words: words1,
                });
                next_utt_idx += 1;
                cursor_ms = s0 + 2 * UTT_DURATION_MS;
                k += 2;
            }
            (OverlapConvention::LazyPrecedes, true, true) => {
                // Same temporal geometry as CA bracket; CHAT emission differs.
                let (s0, e0, s1, e1) = pair_overlap_ranges(cursor_ms);
                let words0 = mint_words(seed, WORDS_PER_UTT);
                let words1 = mint_words(seed ^ 0xcafe_babe, WORDS_PER_UTT);
                layouts.push(UttLayout {
                    utt_index: next_utt_idx,
                    speaker: "MAI",
                    role: UttRole::LazyLeading,
                    start_ms: s0,
                    end_ms: e0,
                    words: words0,
                });
                next_utt_idx += 1;
                layouts.push(UttLayout {
                    utt_index: next_utt_idx,
                    speaker: "OTH",
                    role: UttRole::LazyContinuation,
                    start_ms: s1,
                    end_ms: e1,
                    words: words1,
                });
                next_utt_idx += 1;
                cursor_ms = s0 + 2 * UTT_DURATION_MS;
                k += 2;
            }
            (OverlapConvention::InlineBackchannel, true, _) => {
                // Solo main utt with a short backchannel nested INSIDE it.
                let (s0, e0, bc_s, bc_e) = backchannel_ranges(cursor_ms);
                let words0 = mint_words(seed, WORDS_PER_UTT);
                let bc_words = mint_words(seed ^ 0xfeed_face, BACKCHANNEL_WORDS);
                layouts.push(UttLayout {
                    utt_index: next_utt_idx,
                    speaker: "MAI",
                    role: UttRole::Solo,
                    start_ms: s0,
                    end_ms: e0,
                    words: words0,
                });
                next_utt_idx += 1;
                layouts.push(UttLayout {
                    utt_index: next_utt_idx,
                    speaker: "OTH",
                    role: UttRole::Backchannel,
                    start_ms: bc_s,
                    end_ms: bc_e,
                    words: bc_words,
                });
                next_utt_idx += 1;
                cursor_ms = e0;
                // InlineBackchannel consumes only 1 main-timeline slot per
                // pair (the backchannel nests inside). Advance k by 2 so the
                // pair still counts against `n_utts`.
                k += 2;
            }
            _ => {
                // Solo / no-overlap / pair not possible: one main utt.
                let s0 = cursor_ms;
                let e0 = s0 + UTT_DURATION_MS;
                let words0 = mint_words(seed, WORDS_PER_UTT);
                layouts.push(UttLayout {
                    utt_index: next_utt_idx,
                    speaker: "MAI",
                    role: UttRole::Solo,
                    start_ms: s0,
                    end_ms: e0,
                    words: words0,
                });
                next_utt_idx += 1;
                cursor_ms = e0;
                k += 1;
            }
        }
    }

    layouts
}

/// Step 2: place each word uniformly within its utterance's time range and
/// flatten into one document-wide Vec of timed tokens.
#[derive(Debug, Clone)]
struct TimedWord {
    utterance_index: usize,
    text: String,
    start_ms: u64,
    end_ms: u64,
    /// Stable tie-breaker for the temporal sort: preserves layout emission
    /// order when two words have identical start_ms.
    insertion_order: usize,
}

fn place_words_in_time(layouts: &[UttLayout]) -> Vec<TimedWord> {
    let mut out: Vec<TimedWord> = Vec::new();
    let mut insertion: usize = 0;
    for utt in layouts {
        let span = utt.end_ms.saturating_sub(utt.start_ms).max(1);
        let step = span / (utt.words.len().max(1) as u64);
        for (i, w) in utt.words.iter().enumerate() {
            let s = utt.start_ms + (i as u64) * step;
            let e = if i + 1 == utt.words.len() {
                utt.end_ms
            } else {
                s + step
            };
            out.push(TimedWord {
                utterance_index: utt.utt_index,
                text: w.clone(),
                start_ms: s,
                end_ms: e,
                insertion_order: insertion,
            });
            insertion += 1;
        }
    }
    out
}

/// Step 3: produce the ASR stream in TEMPORAL order by sorting across all
/// utterances. This is the key step that breaks document-order DP: under
/// overlap, utts A and B produce interleaved ASR tokens (A1, B1, A2, B2, ...)
/// even though CHAT lists all of A's words before any of B's.
fn interleave_asr(mut words: Vec<TimedWord>) -> Vec<TimedWord> {
    words.sort_by(|a, b| {
        a.start_ms
            .cmp(&b.start_ms)
            .then_with(|| a.end_ms.cmp(&b.end_ms))
            .then_with(|| a.insertion_order.cmp(&b.insertion_order))
    });
    words
}

/// Step 4: drop tokens with probability `rate`, deterministic on a stable
/// per-word seed (insertion_order). We cannot use array position because the
/// temporal re-sort has already happened; we need the drop mask to be
/// independent of sort order so the same word is always dropped across runs.
fn apply_asr_drop(words: Vec<TimedWord>, rate: f64) -> Vec<TimedWord> {
    words
        .into_iter()
        .filter(|word| {
            let roll = (word.insertion_order as u64).wrapping_mul(LCG_MULT_GLIBC) ^ LCG_INC_GLIBC;
            let normalized = (roll % 1000) as f64 / 1000.0;
            normalized >= rate
        })
        .collect()
}

/// Step 5: emit the CHAT document in DOCUMENT order (layouts' order). Overlap
/// markup appears exactly where it belongs but the document never reveals the
/// temporal interleaving, which is what makes GlobalUtr's document-order DP
/// mis-align against the temporally re-sorted ASR.
fn emit_chat_source(layouts: &[UttLayout]) -> String {
    let mut chat = String::new();
    chat.push_str("@UTF8\n@Begin\n");
    chat.push_str("@Languages:\teng\n");
    chat.push_str("@Participants:\tMAI Participant, OTH Participant\n");
    chat.push_str("@ID:\teng|test|MAI|||||Participant|||\n");
    chat.push_str("@ID:\teng|test|OTH|||||Participant|||\n");
    // This test has synthetic acoustic tokens rather than a file to resolve.
    // Its untimed source is valid; real linked-media admission is tested at FA.
    chat.push_str("@Media:\tsynthetic, audio, unlinked\n");

    for utt in layouts {
        let words_joined = utt.words.join(" ");
        let line = match utt.role {
            UttRole::CaTop => format!(
                "*{}:\t⌈ {} ⌉ {} .\n",
                utt.speaker,
                utt.words[0],
                utt.words[1..].join(" "),
            ),
            UttRole::CaBottom => format!(
                "*{}:\t⌊ {} ⌋ {} .\n",
                utt.speaker,
                utt.words[0],
                utt.words[1..].join(" "),
            ),
            UttRole::LazyContinuation => {
                format!("*{}:\t+< {} .\n", utt.speaker, words_joined)
            }
            UttRole::Solo | UttRole::LazyLeading | UttRole::Backchannel => {
                format!("*{}:\t{} .\n", utt.speaker, words_joined)
            }
        };
        chat.push_str(&line);
    }

    chat.push_str("@End\n");
    chat
}

/// Build a CHAT document + corresponding ASR token stream for a scenario.
///
/// # Algorithm
///
/// 1. [`layout_utterances_in_time`]: assign each utterance a `(start_ms,
///    end_ms)` range and a `UttRole`. Overlap-bearing pair conventions
///    (`CaBracket`, `LazyPrecedes`) produce two adjacent utts with
///    intersecting ranges. `InlineBackchannel` places short backchannel utts
///    nested inside a host's range.
/// 2. [`place_words_in_time`]: distribute each utt's words uniformly across
///    its range, producing a flat Vec of timed tokens tagged with an
///    insertion_order for stable sort / drop-mask seeding.
/// 3. [`interleave_asr`]: sort the tokens by `start_ms` to produce the
///    temporal-order ASR stream. Overlapping utts now have interleaved words
///    in the ASR stream by construction.
/// 4. [`apply_asr_drop`]: probabilistically drop tokens at `asr_missing_rate`.
/// 5. [`emit_chat_source`]: serialize the CHAT in DOCUMENT order (layouts'
///    order), with convention-appropriate markup.
///
/// Drift arises from the impedance mismatch between document-order CHAT and
/// temporal-order ASR. A monotone DP walking CHAT order over the temporally
/// re-sorted ASR cannot reconcile the two when words interleave; that is the
/// synthetic interleaving mechanism this fixture tests. It is not acoustic gold.
pub(super) fn build_scenario(params: DriftParams) -> AdmittedDriftScenario {
    let layouts = layout_utterances_in_time(params);
    let timed = place_words_in_time(&layouts);
    let temporal = interleave_asr(timed);
    let surviving = apply_asr_drop(temporal, params.asr_missing_rate);
    let mut surviving_counts = vec![0; layouts.len()];
    for word in &surviving {
        surviving_counts[word.utterance_index] += 1;
    }
    // Oracle ownership remains fixture-only; production receives the same
    // text/timing tokens as before, without a synthetic speaker attribution.
    let asr_tokens = surviving
        .into_iter()
        .map(|word| AsrTimingToken {
            text: word.text,
            start_ms: word.start_ms,
            end_ms: word.end_ms,
        })
        .collect();

    let expected_windows: Vec<ExpectedWindow> = layouts
        .iter()
        .map(|u| ExpectedWindow {
            utt_index: u.utt_index,
            expected_start_ms: u.start_ms,
            expected_end_ms: u.end_ms,
            surviving_asr_words: surviving_counts[u.utt_index],
        })
        .collect();

    let chat_text = emit_chat_source(&layouts);
    // Grammar construction and complete CHAT admission are distinct. Neither
    // parse success nor an empty parser diagnostic list establishes validity.
    #[allow(clippy::expect_used)]
    let parser = TreeSitterParser::new().expect("construct TreeSitterParser");
    let errors = talkbank_model::ErrorCollector::new();
    let source = parse_source_with_parser(&parser, &chat_text)
        .admit(TranscriptName::Anonymous, &errors)
        .unwrap_or_else(|error| panic!("synthesized CHAT failed complete admission: {error}"))
        .into_owned();
    AdmittedDriftScenario {
        source,
        tokens: asr_tokens,
        expected: expected_windows,
    }
}
