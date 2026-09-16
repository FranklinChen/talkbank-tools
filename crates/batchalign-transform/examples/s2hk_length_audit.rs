//! Measure whether Cantonese normalization can ever change a character count.
//!
//! `AlignedNormalization` hands each ASR unit back exactly the characters it
//! contributed, which is only sound while the normalization preserves the
//! character count of the run. Its constructor refuses otherwise. This audit
//! answers the other half of that question: is the refusal reachable at all
//! with the tables this build embeds?
//!
//! Two input classes are measured:
//!
//! 1. **Every Han code point** the pipeline treats as an ideograph, converted
//!    one at a time. This settles the single-character dictionaries
//!    exhaustively, with no external data.
//! 2. **Every key and every value of the OpenCC dictionaries that the `s2hk`
//!    chain uses**, when their text files are passed as arguments. This is how
//!    the multi-character (phrase) entries get measured, because the embedded
//!    tables are not enumerable through `ferrous-opencc`'s public API.
//!
//! The dictionary files are OpenCC's own, in its `data/dictionary` directory
//! (`STPhrases.txt`, `STCharacters.txt`, `HKVariantsPhrases.txt`,
//! `HKVariants.txt`, `CJK_Compatibility_Ideographs.txt`). Each line is
//! `key<TAB>value[ value...]`, and `#` starts a comment.
//!
//! ```text
//! cargo run -p batchalign-transform --example s2hk_length_audit -- <dictionary>.txt ...
//! ```
//!
//! Exit status is non-zero when any input normalized to a different character
//! count, and the first few offenders are printed with both counts.

use std::time::Instant;

use batchalign_transform::asr_postprocess::cantonese::AlignedNormalization;
use batchalign_transform::asr_postprocess::is_cjk_ideograph;

/// How many offending strings to print before summarizing the rest.
const REPORTED_OFFENDERS: usize = 20;

/// What one input class measured.
#[derive(Default)]
struct Audit {
    /// Strings normalized.
    checked: usize,
    /// Strings whose normalization changed the character count.
    changed: Vec<String>,
}

impl Audit {
    /// Normalize one string through the one owner and record the outcome.
    fn check(&mut self, source: &str) {
        if source.is_empty() {
            return;
        }
        self.checked += 1;
        if let Err(refusal) = AlignedNormalization::admit([source]) {
            self.changed
                .push(format!("{source} ({} to {})", refusal.before, refusal.after));
        }
    }

    /// Print what this class measured.
    fn report(&self, label: &str) {
        println!(
            "{label}: {} strings checked, {} changed length",
            self.checked,
            self.changed.len()
        );
        for offender in self.changed.iter().take(REPORTED_OFFENDERS) {
            println!("  {offender}");
        }
        if self.changed.len() > REPORTED_OFFENDERS {
            println!("  ... and {} more", self.changed.len() - REPORTED_OFFENDERS);
        }
    }
}

/// Sweep every code point in the ideograph ranges the pipeline recognizes.
fn audit_han_code_points() -> Audit {
    let mut audit = Audit::default();
    for code_point in 0x3400u32..=0x2FA1Fu32 {
        let Some(character) = char::from_u32(code_point) else {
            continue;
        };
        if is_cjk_ideograph(character) {
            audit.check(&character.to_string());
        }
    }
    audit
}

/// Sweep one OpenCC dictionary text file: every key and every value variant.
fn audit_dictionary(path: &str) -> std::io::Result<Audit> {
    let text = std::fs::read_to_string(path)?;
    let mut audit = Audit::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split('\t');
        let Some(key) = fields.next() else {
            continue;
        };
        audit.check(key.trim());
        for values in fields {
            for value in values.split_whitespace() {
                audit.check(value);
            }
        }
    }
    Ok(audit)
}

fn main() -> std::io::Result<()> {
    let started = Instant::now();
    let mut changed = 0usize;

    let code_points = audit_han_code_points();
    code_points.report("Han code points");
    changed += code_points.changed.len();

    for path in std::env::args().skip(1) {
        let audit = audit_dictionary(&path)?;
        audit.report(&path);
        changed += audit.changed.len();
    }

    println!("elapsed: {:.1?}", started.elapsed());
    if changed > 0 {
        eprintln!(
            "{changed} input(s) changed character count: the aligned-normalization refusal is \
             reachable and the book pages that say otherwise are stale"
        );
        std::process::exit(1);
    }
    println!("no input changed its character count");
    Ok(())
}
