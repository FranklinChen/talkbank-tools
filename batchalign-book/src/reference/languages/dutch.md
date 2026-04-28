# Dutch

**Status:** Current
**Last updated:** 2026-04-22 00:30 EDT

## Scope

Dutch is a Germanic language with a relatively simple MWT profile:

* Possessive `'s` suffixes on proper names: `Claus's`, `Maria's`,
  `Jan's`
* Idiomatic time expressions with leading `'s`: `'s-avonds` ("in the
  evening"), `'s-morgens` ("in the morning"), `'s-nachts`
* Short apostrophe contractions: `'t` (= `het`), `'n` (= `een`)
* Occasional reduced-vowel contractions such as `het's` (dialectal)

Dutch carries no per-language BA3 MWT-override rules as of
2026-04-21. The single BA2-ported rule was audited and removed —
see [History](#history).

## What Stanza handles natively

Paired probes with Stanza 1.11.1 (free-tokenize vs our
postprocessor) on 13 `'s`-bearing Dutch constructions produce
identical output on both paths:

| Pattern | Example | Stanza output |
|---------|---------|---------------|
| Possessive `'s` on proper name | `Claus's`, `Maria's`, `Jan's` | 1 UD word, no MWT expansion |
| Pseudo-contraction `het's` / `er's` | `het's koud` | 1 UD word per whitespace-separated token |
| Leading `'s` time idiom | `'s-avonds`, `'s-morgens` | 1 UD word (hyphen-joined form preserved) |
| Short contraction | `'t is koud`, `'n huis` | 1 UD word per token |
| Plain noun | `huis` | 1 UD word (control case) |

Probes in
`batchalign/tests/investigations/_cases/dutch.py` (typed
`ProbeCase` fixtures consumed by the matrix harness at
`test_stanza_mwt_probe_matrix.py`).

## Known Stanza limitations

No Dutch-specific Stanza defects are tracked as of 2026-04-21.

## History

### Rule that existed and was removed (2026-04-21)

| Rule | What it did | Audit finding |
|------|-------------|---------------|
| `EndsWith("'s") → SuppressMwt` | For any Dutch token ending in `'s`, flip the MWT hint to False to prevent expansion | **Dormant.** Modern Stanza Dutch does not emit `'s`-suffix tokens with MWT hints today — the 13-case probe showed identical output with and without the rule active. The rule was also worryingly broad: it would fire on any `'s`-suffix form (possessives, contractions, idioms), with no per-pattern scoping. |

Removed in the 2026-04-21 audit. Dutch now runs through the default
tokenizer-postprocessor path with no per-language overrides.

### Why the DP alignment is the load-bearing piece

As with the Romance languages, the character-level Hirschberg
realignment in `align_tokens` is the unconditional rescue when
Stanza's Dutch tokenizer occasionally over-splits. Dutch
inherits this behavior for free.

## Tests

* **Probe matrix cases:**
  `batchalign/tests/investigations/_cases/dutch.py` — 14 typed
  `ProbeCase` fixtures covering possessives, contractions, time
  idioms, and control nouns.
* **Matrix harness:**
  `batchalign/tests/investigations/test_stanza_mwt_probe_matrix.py`
  runs every case through paired pipelines. Invoke with
  `uv run pytest batchalign/tests/investigations/ -m golden`.
* **Behavior-table renderer:**
  `scripts/analysis/render_probe_matrix_table.py --lang nld`.

## References

* [Morphotag Invariants](../../architecture/morphotag-invariants.md)
* 2026-04-21 investigation (private meta-repo):
  `docs/investigations/2026-04-21-stanza-chat-input-behavior.md`
