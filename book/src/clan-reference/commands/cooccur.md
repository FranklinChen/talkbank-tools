# COOCCUR — Word Co-occurrence (Bigram) Counting

**Status:** Current
**Last updated:** 2026-05-22 09:45 EDT

## Purpose

Counts adjacent word pairs (bigrams) across utterances. For each utterance, every pair of consecutive countable words is recorded as a directed bigram. Pairs are directional: ("put", "the") and ("the", "put") are counted separately.

COOCCUR is part of the FREQ family of commands and is useful for studying word collocations and sequential patterns in speech.

## Usage

```bash
chatter clan cooccur file.cha
chatter clan cooccur file.cha --speaker CHI
```

## Options (chatter-native)

| Option | CLAN flag | Description |
|--------|-----------|-------------|
| `--speaker <CODE>` | `+t*CHI` (or `+tCHI`) | Include speaker |
| `--exclude-speaker <CODE>` | `-t*CHI` (or `-tCHI`) | Exclude speaker |
| `--gem <LABEL>` | `+g"label"` | Restrict to gem segment |
| `--range <START-END>` | `+z25-125` | Utterance range |
| `--id-filter <PATTERN>` | `+t@ID="..."` | Filter by @ID pattern |
| `--include-retracings` | `+r6` | Include retraced words in counting |
| `--format <FMT>` | -- | Output format: clan (default), text, json, csv |

## CLAN `+`-flag coverage audit

Authoritative enumeration of every CLAN `cooccur` flag. Sources:

* `OSX-CLAN/src/clan/cooccur.cpp` — `usage()`.
* `OSX-CLAN/src/clan/cutt.cpp` — `mainusage()` COOCCUR branches.
* `crates/talkbank-clan/src/clan_args.rs` — chatter's rewriter.
* `crates/talkbank-cli/src/cli/args/clan_commands.rs::Cooccur` plus
  `clan_common.rs::CommonAnalysisArgs`.

(Status legend: same as [FREQ](./freq.md#status-legend).)

### COOCCUR-specific `+`-flags (from `cooccur.cpp::usage`)

| CLAN flag | Meaning | Chatter | Status | Notes |
|---|---|---|---|---|
| `+b` / `-b` | Match `+s` words only at beginning / end of cluster | — | Missing | Boundary-sensitive matching. |
| `+nN` | Cluster size of N words (default 2) | `--cluster-size N` | Done | Landed 2026-05-23. Rewriter `+nN` → `--cluster-size N`. `CooccurConfig` gains `cluster_size: u8` (default 2). `WordPair` generalized to `WordCluster(Vec<NormalizedWord>)`; `PairDisplay` flattened to `displays: Vec<String>`; `CooccurPair` JSON shape replaced with `CooccurCluster { words, displays, count }`. `process_utterance` uses `windows(cluster_size)` instead of `windows(2)`; utterances shorter than N produce no clusters. Pinned by `cooccur_cluster_size_three_emits_trigrams`, `cooccur_cluster_size_larger_than_utterance_is_skipped`, and the rewriter test `cooccur_cluster_size`. End-to-end smoke verified for N=2/3/4. JSON schema breaking change: `pairs/unique_pairs/total_pair_instances` renamed to `clusters/unique_clusters/total_cluster_instances`; per-row fields `word1/word2/display1/display2` collapsed into `words: Vec<String>` and `displays: Vec<String>`. |
| `+o` | Sort output by descending frequency | (default) | Done | |
| `+d` | Strip frequency counts from output | `--no-frequency-counts` | Done | Rewriter intercepts bare `+d`. |

### Audit summary

| Bucket | Count |
|---|---|
| Done | 8 |
| Partial | 1 |
| Rewriter only | 3 |
| Missing | 3 |

COOCCUR's largest practical gap is `+nN`: chatter is fixed-bigram
while CLAN allows arbitrary N-gram clusters. A `--cluster-size N`
field would close this.

## Display Modes (`+d`)

Bare `+d` (no following digits) is intercepted by the rewriter and
mapped to `--no-frequency-counts`. The CLAN manual §7.8.1
(`Unique Options`, COOCUR) describes it as: "Strip the numbers from
the output data that indicate how often a certain cluster occurred."
The chatter implementation drops the leading count column from
CLAN-format output and is captured in
`CooccurConfig::no_frequency_counts`.

No `+d1`/`+d2`/... forms are documented for COOCUR.

## Output

- Table of adjacent word pairs with co-occurrence counts
- Default sort: by frequency descending, then alphabetically
- CLAN output: sorted alphabetically by pair display form
- Summary: unique pair count, total pair instances, total utterances

## Differences from CLAN

- Word identification uses AST-based `is_countable_word()` instead of CLAN's string-prefix matching
- Bigram extraction operates on parsed AST content rather than raw text
- Output supports text, JSON, and CSV formats (CLAN produces text only)
- Deterministic output ordering via sorted collections
- **Golden test parity**: Verified against CLAN C binary output
