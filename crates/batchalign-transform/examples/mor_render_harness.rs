//! Render morphotag output from saved worker analyses, without a server.
//!
//! Why this exists: a change to the UD-to-`%mor` mapping (the feature
//! renderer above all) must be provable output-identical, or its differences
//! listed, on real Stanza analyses, before anything is built for a host. This
//! splits morphotag at the worker boundary so the expensive half (Stanza) runs
//! once and the cheap half (mapping) runs on each version of the code:
//!
//! ```text
//! cargo run -p batchalign-transform --example mor_render_harness -- \
//!     collect <corpus-root> <items.jsonl> <file.cha>...
//! python scripts/analyze_mor_harness_items.py <items.jsonl> <analyses.jsonl>
//! cargo run -p batchalign-transform --example mor_render_harness -- \
//!     render <corpus-root> <analyses.jsonl> <out-dir> <file.cha>...
//! diff -r <out-dir of one version> <out-dir of another>
//! ```
//!
//! Both halves use BA3's own functions, in the order `morphotag` uses them:
//! `clear_morphosyntax`, `declared_languages`, `collect_payloads`
//! (`MultilingualPolicy::ProcessAll`), then, for the analyses,
//! `parse_raw_stanza_output` and `inject_results` (`TokenizationMode::Preserve`,
//! an empty MWT dictionary). `render` re-collects the items rather than
//! reading them back, so it fails loudly if the corpus changed in between.
//! Not covered: the job layer around injection (incremental reuse, L2
//! dispatch for `@s` words, POS hints); this harness is for the mapping.
//!
//! File arguments are paths relative to `<corpus-root>`; they name the files
//! in the JSONL and in `<out-dir>`.
//!
//! A file whose worker item failed is not written, as the server would fail
//! it.

use std::collections::BTreeMap;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use batchalign_transform::morphosyntax::{
    MorphosyntaxBatchItem, MultilingualPolicy, MwtDict, TokenizationMode, UdResponse,
    clear_morphosyntax, collect_payloads, declared_languages, inject_results,
    parse_raw_stanza_output,
};
use talkbank_model::model::{LanguageCode, WriteChat};
use talkbank_model::{ChatFile, ErrorCollector};
use talkbank_parser::TreeSitterParser;

type Error = Box<dyn std::error::Error>;

/// Where one worker item's utterance sits: its file (relative to the corpus
/// root) and its line in the parsed transcript. The one key every record and
/// lookup uses.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
struct ItemKey {
    file: String,
    line_idx: usize,
}

/// One worker item, keyed by where its utterance sits.
#[derive(serde::Serialize)]
struct ItemRecord {
    #[serde(flatten)]
    key: ItemKey,
    item: MorphosyntaxBatchItem,
}

/// The worker's outcome for one item, as `analyze_mor_harness_items.py` saves
/// it: exactly one of the three kinds the server distinguishes.
#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Analysis {
    Analyzed {
        raw_sentences: Vec<serde_json::Value>,
    },
    NoWords,
    Failed {
        error: String,
    },
}

#[derive(serde::Deserialize)]
struct AnalysisRecord {
    #[serde(flatten)]
    key: ItemKey,
    analysis: Analysis,
}

/// What `render` did with one file.
enum FileOutcome {
    /// Written.
    Rendered,
    /// Not written, as the server would fail it: a worker item failed.
    Skipped(String),
}

/// The harness's two commands, parsed from the arguments once.
enum Command {
    Collect {
        root: PathBuf,
        items: PathBuf,
        files: Vec<String>,
    },
    Render {
        root: PathBuf,
        analyses: PathBuf,
        out_dir: PathBuf,
        files: Vec<String>,
    },
}

impl Command {
    fn parse(args: &[String]) -> Result<Self, Error> {
        match args {
            [cmd, root, items, files @ ..] if cmd == "collect" && !files.is_empty() => {
                Ok(Self::Collect {
                    root: root.into(),
                    items: items.into(),
                    files: files.to_vec(),
                })
            }
            [cmd, root, analyses, out_dir, files @ ..] if cmd == "render" && !files.is_empty() => {
                Ok(Self::Render {
                    root: root.into(),
                    analyses: analyses.into(),
                    out_dir: out_dir.into(),
                    files: files.to_vec(),
                })
            }
            _ => Err(
                "usage: mor_render_harness collect <root> <items.jsonl> <file>...\n       \
                      mor_render_harness render <root> <analyses.jsonl> <out-dir> <file>..."
                    .into(),
            ),
        }
    }
}

/// A transcript parsed with no diagnostics, its `%mor`/`%gra` cleared as
/// `morphotag` clears them.
fn read_cleared(parser: &TreeSitterParser, path: &Path) -> Result<ChatFile, Error> {
    let text = std::fs::read_to_string(path)?;
    let errors = ErrorCollector::new();
    let mut chat = parser.parse_chat_file_streaming(&text, &errors);
    let diagnostics = errors.into_vec();
    if !diagnostics.is_empty() {
        return Err(format!(
            "{}: {} parse diagnostics",
            path.display(),
            diagnostics.len()
        )
        .into());
    }
    clear_morphosyntax(&mut chat);
    Ok(chat)
}

fn primary_language(chat: &ChatFile, path: &Path) -> Result<LanguageCode, Error> {
    chat.languages
        .first()
        .cloned()
        .ok_or_else(|| format!("{}: no @Languages", path.display()).into())
}

fn collect(root: &Path, out: &Path, files: &[String]) -> Result<(), Error> {
    let parser = TreeSitterParser::new()?;
    let mut writer = BufWriter::new(std::fs::File::create(out)?);
    let mut count = 0usize;
    for file in files {
        let path = root.join(file);
        let chat = read_cleared(&parser, &path)?;
        let primary = primary_language(&chat, &path)?;
        let langs = declared_languages(&chat, &primary);
        let payloads = collect_payloads(&chat, &primary, &langs, MultilingualPolicy::ProcessAll);
        for utterance in payloads.batch_items {
            let key = ItemKey {
                file: file.clone(),
                line_idx: utterance.line().raw(),
            };
            let item = utterance.into_item();
            serde_json::to_writer(&mut writer, &ItemRecord { key, item })?;
            writer.write_all(b"\n")?;
            count += 1;
        }
    }
    writer.flush()?;
    eprintln!(
        "collected {count} items from {} files into {}",
        files.len(),
        out.display()
    );
    Ok(())
}

/// Map, count and write one file from its saved analyses.
fn render_file(
    parser: &TreeSitterParser,
    root: &Path,
    file: &str,
    analyses: &mut BTreeMap<ItemKey, Analysis>,
    out_dir: &Path,
) -> Result<FileOutcome, Error> {
    let path = root.join(file);
    let mut chat = read_cleared(parser, &path)?;
    let primary = primary_language(&chat, &path)?;
    let langs = declared_languages(&chat, &primary);
    let payloads = collect_payloads(&chat, &primary, &langs, MultilingualPolicy::ProcessAll);

    let mut responses: Vec<UdResponse> = Vec::with_capacity(payloads.batch_items.len());
    for utterance in &payloads.batch_items {
        let key = ItemKey {
            file: file.to_string(),
            line_idx: utterance.line().raw(),
        };
        let response = match analyses.remove(&key) {
            Some(Analysis::Analyzed { raw_sentences }) => parse_raw_stanza_output(&raw_sentences)?,
            Some(Analysis::NoWords) => UdResponse {
                sentences: Vec::new(),
            },
            // The server fails the whole file on a failed item; so does this.
            Some(Analysis::Failed { error }) => {
                return Ok(FileOutcome::Skipped(format!(
                    "line {}: {error}",
                    key.line_idx
                )));
            }
            None => return Err(format!("{file}:{}: no analysis saved", key.line_idx).into()),
        };
        responses.push(response);
    }
    // The decisions and traces are not part of the rendered output.
    inject_results(
        parser,
        &mut chat,
        payloads.batch_items,
        responses,
        &primary,
        TokenizationMode::Preserve,
        &MwtDict::new(),
    )?;
    let target = out_dir.join(file);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&target, chat.to_chat_string())?;
    Ok(FileOutcome::Rendered)
}

fn render(
    root: &Path,
    analyses_path: &Path,
    out_dir: &Path,
    files: &[String],
) -> Result<(), Error> {
    let mut analyses: BTreeMap<ItemKey, Analysis> = BTreeMap::new();
    for line in std::io::BufReader::new(std::fs::File::open(analyses_path)?).lines() {
        let record: AnalysisRecord = serde_json::from_str(&line?)?;
        let key = record.key.clone();
        if analyses.insert(record.key, record.analysis).is_some() {
            return Err(format!("{}:{}: two analyses", key.file, key.line_idx).into());
        }
    }

    let parser = TreeSitterParser::new()?;
    let mut rendered = 0usize;
    for file in files {
        match render_file(&parser, root, file, &mut analyses, out_dir)? {
            FileOutcome::Rendered => {
                rendered += 1;
            }
            FileOutcome::Skipped(reason) => {
                eprintln!("skipped {file} (a worker item failed): {reason}");
            }
        }
    }
    if let Some((key, _)) = analyses.into_iter().next() {
        return Err(format!(
            "{}:{}: saved analysis matches no collected item",
            key.file, key.line_idx
        )
        .into());
    }

    eprintln!("rendered {rendered} files into {}", out_dir.display());
    Ok(())
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match Command::parse(&args)? {
        Command::Collect { root, items, files } => collect(&root, &items, &files),
        Command::Render {
            root,
            analyses,
            out_dir,
            files,
        } => render(&root, &analyses, &out_dir, &files),
    }
}
