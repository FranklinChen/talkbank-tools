# utseg — Developer Reference

**Status:** Current
**Last updated:** 2026-04-14 17:35 EDT

Implementation guide for the `utseg` command. For user-facing documentation,
see [User Guide: utseg](../../user-guide/commands/utseg.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign-cli/src/args/commands.rs` — `UtsegArgs` | lang, num-speakers |
| Command definition | `crates/batchalign-app/src/commands/utseg.rs` | `CommandDefinition` impl |
| Utseg orchestration | `crates/batchalign-app/src/utseg.rs` | Cross-file batching, cache, boundary application |
| Batch dispatch | `crates/batchalign-app/src/runner/dispatch/infer_batched.rs` | Shared with morphotag and translate |
| Worker IPC | `batchalign/inference/utseg.py` — `batch_infer_utseg()` | Loads Stanza constituency, returns raw parse trees |
| Boundary application | `crates/batchalign-chat-ops/src/utseg.rs` | Maps predicted boundaries back to CHAT utterance structure |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. See [Submission Modes](../../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse).

---

## Cache key structure

Utseg cache keys (BLAKE3 hash of):
- Word sequence
- Language code

Boundaries are purely a function of word sequence and language — no other
context affects the result.

---

## Worker IPC: utseg task

```
batch_infer request:
{
  "task": "utseg",
  "items": [
    { "words": ["hello", "world", "how", "are", "you"], "lang": "eng" },
    ...
  ]
}

batch_infer response:
[
  { "boundaries": [2, 5], "parse_tree": "..." },
  ...
]
```

`boundaries` is a list of word indices where utterance breaks are inserted.
The Rust `utseg.rs` library maps these back to CHAT utterance splits/merges.

---

## Stanza constituency availability

About 11 languages have Stanza constituency models. Languages without
constituency support fall back to punctuation-based boundary detection. The
available processors are queried at worker startup via
`batchalign/worker/_stanza_capabilities.py` — never hardcoded.

---

## Pre-validation gate

`utseg` requires CHAT Level 1 (parseable + valid headers). Gate in
`crates/batchalign-app/src/utseg.rs`. Implemented via
`validate_to_level(chat, ValidationLevel::Headers)`.

---

## Testing

```bash
make test
cargo nextest run -p batchalign-app -E 'test(utseg::)'
# ML golden tests — only on net
cargo nextest run --profile ml -E 'test(utseg::golden)'
```

---

## Related developer documentation

- [Command Flowcharts: utseg](../architecture/command-flowcharts.md#utseg)
- [Utterance Segmentation](../reference/utterance-segmentation.md)
- [Stanza Capability Registry](../architecture/stanza-capability-registry.md)
