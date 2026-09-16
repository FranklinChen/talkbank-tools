//! Frozen baseline for ASR number expansion.
//!
//! # What this protects
//!
//! Number expansion is a pure function from `(token, language)` to text, but
//! it is spread across several detectors (English ordinals and decades,
//! currency, dash ranges, digit-leading hyphen compounds, CJK numerals, the
//! per-language `NUM2LANG` tables and their decomposition) and it interacts
//! with the tokenizer that runs before it. A change aimed at one language or
//! one token shape can silently alter another. The committed fixture
//! `data/number_expansion_baseline.json` records, for every language in the
//! fixture's `languages` list crossed with every token in its `inputs` list:
//!
//! - `expand_number`: what [`expand_number`] returns for the bare token (the
//!   per-word entry the transcribe pipeline calls), and
//! - `pipeline_tokens`: the word texts [`prepare_asr_chunks`] produces when
//!   the token arrives as one timed ASR element (tokenizer, separator
//!   handling and expansion together).
//!
//! [`number_expansion_matches_frozen_baseline`] asserts that the current code
//! reproduces every row exactly, and that the language list still covers
//! every `NUM2LANG` table language plus the CJK numeral languages, so a new
//! table language cannot slip in without a baseline.
//!
//! # Regenerating deliberately
//!
//! A baseline row may only change because of an intended behaviour change.
//! To accept one:
//!
//! 1. If the input set itself should change, edit the `languages` or
//!    `inputs` arrays in the fixture (the fixture is the single owner of
//!    both lists).
//! 2. Run the ignored regeneration test with its opt-in variable:
//!    `BATCHALIGN_REGENERATE_NUMBER_BASELINE=1 cargo test -p batchalign-transform --lib regenerate_number_expansion_baseline -- --ignored`
//! 3. Review the fixture diff row by row (one row per line) and name every
//!    changed row, with the reason, in the commit message.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::num2text::{EM_DASH, EN_DASH, NUM2LANG};
use super::{
    AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs,
    SpeakerIndex, expand_number, prepare_asr_chunks,
};

/// Fixture location relative to the crate manifest directory.
const FIXTURE_RELATIVE_PATH: &str = "data/number_expansion_baseline.json";

/// The fixture as compiled into the verifying test.
const FIXTURE_TEXT: &str = include_str!("../../data/number_expansion_baseline.json");

/// Opt-in variable for the regeneration test, so that running every ignored
/// test at once cannot rewrite the baseline by accident.
const REGENERATE_ENV: &str = "BATCHALIGN_REGENERATE_NUMBER_BASELINE";

/// Languages expanded by `num2chinese` that have no `NUM2LANG` table.
/// (`jpn` also routes through `num2chinese` but already has a table entry,
/// so the table key set covers it.)
const CJK_WITHOUT_TABLE: [&str; 3] = ["cmn", "yue", "zho"];

/// The whole fixture: the two input axes and one row per cell of their cross
/// product, in `languages` then `inputs` order.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NumberExpansionBaseline {
    languages: Vec<String>,
    inputs: Vec<String>,
    rows: Vec<BaselineRow>,
}

/// One observed `(language, input)` cell.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BaselineRow {
    lang: String,
    input: String,
    expand_number: String,
    pipeline_tokens: Vec<String>,
}

impl BaselineRow {
    /// Run the current converter on one cell.
    ///
    /// The pipeline column feeds the token as a single element timed at
    /// 0.0-1.0 s, the shape a timed-word ASR provider returns.
    fn observe(lang: &str, input: &str) -> Self {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![AsrElement {
                    value: AsrRawText::new(input),
                    ts: AsrTimestampSecs::Observed(0.0),
                    end_ts: AsrTimestampSecs::Observed(1.0),
                    kind: AsrElementKind::Text,
                }],
            }],
        };
        let pipeline_tokens = prepare_asr_chunks(&output, lang)
            .expect("test: ASR post-processing must not refuse this input")
            .into_iter()
            .flat_map(|chunk| chunk.words)
            .map(|word| word.text.as_str().to_owned())
            .collect();
        Self {
            lang: lang.to_owned(),
            input: input.to_owned(),
            expand_number: expand_number(input, lang),
            pipeline_tokens,
        }
    }
}

impl NumberExpansionBaseline {
    fn parse(text: &str) -> Self {
        serde_json::from_str(text).expect("number expansion baseline fixture must deserialize")
    }

    /// Observe every cell of the cross product, in fixture order.
    fn observed_rows(&self) -> impl Iterator<Item = BaselineRow> + '_ {
        self.languages.iter().flat_map(move |lang| {
            self.inputs
                .iter()
                .map(move |input| BaselineRow::observe(lang, input))
        })
    }

    /// Serialize with one row per line, so a behaviour change shows up as a
    /// reviewable one-line diff per affected cell.
    ///
    /// Em and en dashes are written as JSON `\u` escapes so the committed
    /// file never carries the raw characters.
    fn to_fixture_text(&self) -> String {
        fn to_json<T: Serialize>(value: &T) -> String {
            serde_json::to_string(value).expect("baseline values serialize")
        }
        let mut out = String::from("{\n");
        out.push_str(&format!("  \"languages\": {},\n", to_json(&self.languages)));
        out.push_str(&format!("  \"inputs\": {},\n", to_json(&self.inputs)));
        out.push_str("  \"rows\": [\n");
        let last = self.rows.len().saturating_sub(1);
        for (index, row) in self.rows.iter().enumerate() {
            let separator = if index == last { "" } else { "," };
            out.push_str(&format!("    {}{separator}\n", to_json(row)));
        }
        out.push_str("  ]\n}\n");
        out.replace(EM_DASH, "\\u2014").replace(EN_DASH, "\\u2013")
    }
}

/// Every name in `values` appears once.
fn assert_unique(axis: &str, values: &[String]) {
    let unique: BTreeSet<&str> = values.iter().map(String::as_str).collect();
    assert_eq!(
        unique.len(),
        values.len(),
        "baseline `{axis}` list contains duplicates"
    );
}

#[test]
fn number_expansion_matches_frozen_baseline() {
    let baseline = NumberExpansionBaseline::parse(FIXTURE_TEXT);
    assert_unique("languages", &baseline.languages);
    assert_unique("inputs", &baseline.inputs);

    let declared: BTreeSet<&str> = baseline.languages.iter().map(String::as_str).collect();
    let uncovered: Vec<&str> = NUM2LANG
        .keys()
        .map(String::as_str)
        .chain(CJK_WITHOUT_TABLE)
        .filter(|lang| !declared.contains(lang))
        .collect();
    assert!(
        uncovered.is_empty(),
        "languages with an expander but no baseline rows: {uncovered:?}; add them to \
         {FIXTURE_RELATIVE_PATH} and regenerate"
    );

    let expected_rows = baseline.languages.len() * baseline.inputs.len();
    assert_eq!(
        baseline.rows.len(),
        expected_rows,
        "baseline must hold exactly one row per (language, input) pair"
    );

    let mismatches: Vec<String> = baseline
        .rows
        .iter()
        .zip(baseline.observed_rows())
        .filter(|(frozen, observed)| *frozen != observed)
        .map(|(frozen, observed)| format!("  frozen:   {frozen:?}\n  observed: {observed:?}"))
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} of {expected_rows} number expansion rows differ from {FIXTURE_RELATIVE_PATH} \
         (first 20 shown):\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
#[ignore = "rewrites the committed baseline fixture; run only to accept a deliberate change"]
fn regenerate_number_expansion_baseline() {
    assert!(
        std::env::var_os(REGENERATE_ENV).is_some(),
        "set {REGENERATE_ENV}=1 to rewrite {FIXTURE_RELATIVE_PATH}"
    );
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_RELATIVE_PATH);
    // Read from disk rather than the compiled-in copy, so axis edits made
    // just before this run are honoured.
    let current = std::fs::read_to_string(&path).expect("read baseline fixture");
    let mut baseline = NumberExpansionBaseline::parse(&current);
    let rows: Vec<BaselineRow> = baseline.observed_rows().collect();
    baseline.rows = rows;
    std::fs::write(&path, baseline.to_fixture_text()).expect("write baseline fixture");
}
