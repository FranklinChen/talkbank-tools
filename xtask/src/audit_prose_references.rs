//! Layer 1 CI gate: prose references to deleted crates, moved book paths, and
//! documents that do not exist in this repository.
//!
//! Two checks:
//!
//! 1. High-severity patterns from `audit_docs::FLAG_PATTERNS` over every
//!    markdown file. Catalog-independent: it does not touch the `audit.db`
//!    that lives outside this repo, so it runs cleanly in CI.
//! 2. Every local markdown path named in prose, under either doc root, must
//!    RESOLVE. HTTP(S) URL paths are external link targets, not references to
//!    this checkout. A URL does not exempt other paths on its line. This check
//!    reads Rust doc comments as well as markdown, which check 1 does not.
//!
//! Why check 2 reads Rust. On 2026-07-30 this repository carried 38 doc-comment
//! references to documents that exist only in the maintainer's private
//! workspace: investigations, postmortems, an architecture series, panic-audit
//! snapshots. They were correct when written, because the code was developed
//! beside those documents before it became a public repository, and they came
//! along with it. A public reader got a path they could not open, and several
//! leaked dated incident titles. Nothing noticed, because this gate read only
//! markdown while rustdoc is prose too, and because the gate itself was in no
//! CI target despite the line above calling it one. Both are now fixed.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use regex::Regex;
use walkdir::WalkDir;

use crate::audit_docs::FLAG_PATTERNS;

/// One historical-content allow-list entry. A hit on `pattern_name`
/// inside `path` is treated as expected and does not fail the gate.
///
/// Only add an entry here when the doc's subject is *the removed
/// dependency itself* (e.g. a "What was removed" section, a release
/// changelog, a panic-audit snapshot). Never allow-list a current-state
/// doc that happens to lag a rename, fix that doc instead.
struct AllowEntry {
    /// Repo-relative path to a single markdown file, forward slashes.
    path: &'static str,
    /// Pattern name (matches `FLAG_PATTERNS.name`).
    pattern_name: &'static str,
    /// Short justification recorded inline for future readers.
    #[allow(dead_code)]
    rationale: &'static str,
}

/// Historical surfaces that legitimately name deleted crates / moved
/// paths. Reviewed 2026-05-11 against `flag-staleness` output after the
/// `overview.md` Crate Dependency Graph fix.
const ALLOW_LIST: &[AllowEntry] = &[AllowEntry {
    path: "book/src/batchalign/developer/maturin-pyo3-surface.md",
    pattern_name: "deleted-crate-batchalign-revai",
    rationale: "The '### What was removed' table documents \
                    dependencies dropped from the slim PyO3 surface; \
                    naming `batchalign-revai` is the subject of the table.",
}];

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Violation {
    pub(crate) path: String,
    pub(crate) line: usize,
    pub(crate) pattern_name: &'static str,
    pub(crate) excerpt: String,
}

/// A prose reference to a document, and where it was written.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DocReference {
    /// Repo-relative file the reference was written in.
    source: String,
    /// 1-based line it sits on.
    line: usize,
    /// The repo-relative document path the prose names.
    target: LocalDocPath,
}

/// A document reference admitted to this repository's resolution domain.
/// Web URLs belong to link checking, not local filesystem resolution.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LocalDocPath(String);

impl LocalDocPath {
    fn from_match(line: &str, matched: regex::Match<'_>) -> Option<Self> {
        let prefix = line[..matched.start()]
            .rsplit(|ch: char| {
                ch.is_whitespace()
                    || matches!(
                        ch,
                        '`' | '(' | ')' | '<' | '>' | '[' | ']' | '\'' | '"' | '|'
                    )
            })
            .next()
            .unwrap_or("");
        if prefix.starts_with("https://") || prefix.starts_with("http://") {
            return None;
        }
        Some(Self(matched.as_str().to_owned()))
    }
}

/// Doc paths named in prose, in both markdown and Rust doc comments.
///
/// Rust is included deliberately: a `//!` block is prose that ships to rustdoc,
/// and it was the format in which 38 references to a private doc tree survived
/// a repository split unnoticed.
fn collect_doc_references(repo_root: &Path) -> Vec<DocReference> {
    let pattern = match Regex::new(r"(?:docs|book/src)/[A-Za-z0-9._/-]+\.md") {
        Ok(pattern) => pattern,
        // A malformed literal here is a bug in this file, not in the tree.
        Err(_) => return Vec::new(),
    };

    let mut found = Vec::new();
    for entry in WalkDir::new(repo_root)
        .into_iter()
        .filter_entry(|e| !is_excluded(e.file_name().to_string_lossy().as_ref()))
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "rs" && ext != "md" {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let source = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        for (index, line) in text.lines().enumerate() {
            // Rust: comments only. A path inside a string literal is usually a
            // runtime path rather than a citation, and this gate is about prose.
            if ext == "rs" && !line.trim_start().starts_with("//") {
                continue;
            }
            for m in pattern.find_iter(line) {
                let Some(target) = LocalDocPath::from_match(line, m) else {
                    continue;
                };
                found.push(DocReference {
                    source: source.clone(),
                    line: index + 1,
                    target,
                });
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

/// Every referenced document that does not exist in this repository.
fn unresolvable_doc_references(repo_root: &Path) -> Vec<DocReference> {
    collect_doc_references(repo_root)
        .into_iter()
        .filter(|reference| !repo_root.join(&reference.target.0).is_file())
        .collect()
}

pub fn run(repo_root: &Path) -> crate::Result<()> {
    let docs = collect_markdown(repo_root);
    let total_files = docs.len();
    let violations = scan_docs(&docs)?;

    let dangling = unresolvable_doc_references(repo_root);
    if !dangling.is_empty() {
        eprintln!(
            "xtask audit-prose-references: {} reference(s) to a document that does \
             not exist in this repository.\n  A public reader cannot open it, and if \
             it lives in a private tree the path itself leaks. State the substance \
             inline instead of pointing at it.",
            dangling.len()
        );
        for reference in &dangling {
            eprintln!(
                "  {}:{}: {}",
                reference.source, reference.line, reference.target.0
            );
        }
        return Err("prose references a document that does not exist here".into());
    }

    if violations.is_empty() {
        println!(
            "xtask audit-prose-references: 0 violations across {} markdown files \
             ({} allow-listed historical surfaces respected); every doc path in \
             markdown and Rust doc comments resolves locally (external URLs excluded)",
            total_files,
            ALLOW_LIST.len()
        );
        return Ok(());
    }

    eprintln!(
        "xtask audit-prose-references: {} violation(s): prose names a deleted crate or moved book path",
        violations.len()
    );
    for v in &violations {
        eprintln!(
            "  {}:{}  [{}]  {}",
            v.path, v.line, v.pattern_name, v.excerpt
        );
    }
    eprintln!();
    eprintln!(
        "Fix the prose to match current code, or, if the doc legitimately \
         describes historical state: add an entry to ALLOW_LIST in \
         xtask/src/audit_prose_references.rs with a rationale."
    );

    Err(format!("audit-prose-references: {} violation(s)", violations.len()).into())
}

/// Scan a set of `(rel_path, content)` pairs for high-severity prose-
/// reference hits. Deduplicated by `(path, line, pattern)` so a single
/// line referenced by overlapping section ranges is reported once.
pub(crate) fn scan_docs(docs: &[(String, String)]) -> crate::Result<Vec<Violation>> {
    let high_severity: Vec<(&'static str, Regex)> = FLAG_PATTERNS
        .iter()
        .filter(|p| p.severity == "high")
        .map(|p| Regex::new(p.pattern).map(|re| (p.name, re)))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| -> crate::DynError { Box::new(e) })?;

    let mut violations: Vec<Violation> = Vec::new();
    let mut seen: BTreeSet<(String, usize, &'static str)> = BTreeSet::new();

    for (rel, content) in docs {
        for (line_idx, line) in content.lines().enumerate() {
            let line_no = line_idx + 1;
            for (name, regex) in &high_severity {
                if let Some(m) = regex.find(line) {
                    let key = (rel.clone(), line_no, *name);
                    if !seen.insert(key) {
                        continue;
                    }
                    if is_allow_listed(rel, name) {
                        continue;
                    }
                    violations.push(Violation {
                        path: rel.clone(),
                        line: line_no,
                        pattern_name: name,
                        excerpt: m.as_str().to_owned(),
                    });
                }
            }
        }
    }

    violations.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then(a.line.cmp(&b.line))
            .then(a.pattern_name.cmp(b.pattern_name))
    });
    Ok(violations)
}

fn is_allow_listed(path: &str, pattern_name: &str) -> bool {
    ALLOW_LIST
        .iter()
        .any(|entry| entry.path == path && entry.pattern_name == pattern_name)
}

fn collect_markdown(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_excluded(e.file_name().to_string_lossy().as_ref()))
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let rel: PathBuf = match path.strip_prefix(root) {
            Ok(p) => p.to_path_buf(),
            Err(_) => continue,
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        out.push((rel_str, content));
    }
    out
}

fn is_excluded(name: &str) -> bool {
    matches!(name, "target" | "node_modules" | "build" | ".git") || name.starts_with('.')
}

#[cfg(test)]
mod tests {
    use super::{LocalDocPath, Violation, scan_docs};

    #[test]
    fn external_web_paths_are_not_local_documents() -> Result<(), Box<dyn std::error::Error>> {
        let pattern = regex::Regex::new(r"(?:docs|book/src)/[A-Za-z0-9._/-]+\.md")?;
        for line in [
            "[header](https://example.org/project/blob/main/book/src/headers.md)",
            "<http://example.org/docs/guide.md>",
            "//! See https://example.org/docs/guide.md for details.",
            "[source]: https://example.org/book/src/guide.md",
        ] {
            let targets: Vec<_> = pattern
                .find_iter(line)
                .filter_map(|matched| LocalDocPath::from_match(line, matched))
                .collect();
            assert!(targets.is_empty(), "external URL became local: {line}");
        }
        Ok(())
    }

    #[test]
    fn external_link_does_not_hide_local_references() -> Result<(), Box<dyn std::error::Error>> {
        let pattern = regex::Regex::new(r"(?:docs|book/src)/[A-Za-z0-9._/-]+\.md")?;
        let line =
            "[docs/missing.md](https://example.org/docs/remote.md), then `book/src/missing.md`";
        let targets: Vec<_> = pattern
            .find_iter(line)
            .filter_map(|matched| LocalDocPath::from_match(line, matched))
            .map(|target| target.0)
            .collect();
        assert_eq!(targets, ["docs/missing.md", "book/src/missing.md"]);
        Ok(())
    }

    fn doc(path: &str, body: &str) -> (String, String) {
        (path.to_owned(), body.to_owned())
    }

    #[test]
    fn clean_tree_passes() -> Result<(), Box<dyn std::error::Error>> {
        let docs = vec![
            doc("README.md", "# Hello\n\nNothing stale here.\n"),
            doc("book/src/foo.md", "Some prose about batchalign.\n"),
        ];
        let v = scan_docs(&docs)?;
        assert!(v.is_empty(), "expected no violations, got {v:?}");
        Ok(())
    }

    #[test]
    fn deleted_crate_in_current_doc_is_flagged() -> Result<(), Box<dyn std::error::Error>> {
        let docs = vec![doc(
            "book/src/arch.md",
            "We use `batchalign-app` for the server.\n",
        )];
        let v = scan_docs(&docs)?;
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].pattern_name, "deleted-crate-batchalign-app");
        assert_eq!(v[0].line, 1);
        assert_eq!(v[0].path, "book/src/arch.md");
        Ok(())
    }

    #[test]
    fn allow_listed_surface_is_silent() -> Result<(), Box<dyn std::error::Error>> {
        let docs = vec![doc(
            "book/src/batchalign/developer/maturin-pyo3-surface.md",
            "| `batchalign-revai` | Dead code, server uses Rev.AI directly |\n",
        )];
        let v = scan_docs(&docs)?;
        assert!(v.is_empty(), "expected allow-list to suppress, got {v:?}");
        Ok(())
    }

    #[test]
    fn allow_list_is_scoped_to_specific_pattern() -> Result<(), Box<dyn std::error::Error>> {
        // The maturin-pyo3-surface allow-list entry covers batchalign-revai
        // ONLY: a hit for batchalign-app on the same path must still fail.
        let docs = vec![doc(
            "book/src/batchalign/developer/maturin-pyo3-surface.md",
            "Used to depend on `batchalign-app`.\n",
        )];
        let v = scan_docs(&docs)?;
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].pattern_name, "deleted-crate-batchalign-app");
        Ok(())
    }

    #[test]
    fn same_line_counted_once() -> Result<(), Box<dyn std::error::Error>> {
        let docs = vec![doc(
            "book/src/x.md",
            "## Parent\n\nbatchalign-app exists here\n\n### Child\n\nUnrelated\n",
        )];
        let v: Vec<Violation> = scan_docs(&docs)?;
        assert_eq!(v.len(), 1, "got {v:?}");
        assert_eq!(v[0].line, 3);
        Ok(())
    }

    #[test]
    fn multiple_files_report_in_sorted_order() -> Result<(), Box<dyn std::error::Error>> {
        let docs = vec![
            doc("z/last.md", "batchalign-app\n"),
            doc("a/first.md", "batchalign-revai\n"),
        ];
        let v = scan_docs(&docs)?;
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].path, "a/first.md");
        assert_eq!(v[1].path, "z/last.md");
        Ok(())
    }
}
