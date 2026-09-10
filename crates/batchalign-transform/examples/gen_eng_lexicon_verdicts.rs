//! Regenerate `data/eng_lexicon_verdicts.json` from a checkout of `TalkBank/mor`.
//!
//! Usage:
//!     cargo run -p batchalign-transform --example gen_eng_lexicon_verdicts -- \
//!         <mor-checkout>/eng/eng/lex <mor-commit-sha> crates/batchalign-transform/data/eng_lexicon_verdicts.json
//!
//! Every `.cut` file in the directory is parsed (failing closed on a malformed
//! line, naming the file), the derivation table is checked against the
//! lexicon's own affix entries, the verdicts are derived, validated through
//! the same constructor the runtime uses, and written sorted so the diff of a
//! regeneration is readable.

use std::error::Error;
use std::path::PathBuf;

use batchalign_transform::morphosyntax::lexicon::{
    LexiconVerdicts, MorEntry, VerdictSource, derive_verdicts, parse_cut,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(lex_dir), Some(commit), Some(out_path)) = (args.next(), args.next(), args.next())
    else {
        return Err("usage: gen_eng_lexicon_verdicts <lex-dir> <mor-commit> <out.json>".into());
    };
    let lex_dir = PathBuf::from(lex_dir);
    let commit = commit.to_string_lossy().into_owned();

    let mut cut_files: Vec<PathBuf> = std::fs::read_dir(&lex_dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    cut_files.retain(|p| p.extension().is_some_and(|ext| ext == "cut"));
    cut_files.sort();
    if cut_files.is_empty() {
        return Err(format!("no .cut files under {}", lex_dir.display()).into());
    }

    let mut entries: Vec<MorEntry> = Vec::new();
    for path in &cut_files {
        let text = std::fs::read_to_string(path)?;
        let parsed = parse_cut(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        entries.extend(parsed);
    }

    let source = VerdictSource {
        repo: "TalkBank/mor".to_string(),
        commit,
        path: "eng/eng/lex".to_string(),
        generator: "cargo run -p batchalign-transform --example gen_eng_lexicon_verdicts"
            .to_string(),
    };
    let raw = derive_verdicts(entries, source)?;
    let json = serde_json::to_string_pretty(&raw)?;
    // Prove the file the runtime will embed passes the runtime's own admission.
    let validated = LexiconVerdicts::try_from(raw)?;
    std::fs::write(&out_path, format!("{json}\n"))?;
    eprintln!(
        "wrote {}: {} .cut files, {} communicator verdicts, {} noun verdicts",
        PathBuf::from(&out_path).display(),
        cut_files.len(),
        validated.communicator_count(),
        validated.noun_count()
    );
    Ok(())
}
