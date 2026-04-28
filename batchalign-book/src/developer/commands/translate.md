# translate — Developer Reference

**Status:** Current
**Last updated:** 2026-04-14 17:35 EDT

Implementation guide for the `translate` command. For user-facing
documentation, see [User Guide: translate](../../user-guide/commands/translate.md).

---

## Implementation map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| CLI args | `crates/batchalign-cli/src/args/commands.rs` — `TranslateArgs` | lang override |
| Command definition | `crates/batchalign-app/src/commands/translate.rs` | `CommandDefinition` impl |
| Translate orchestration | `crates/batchalign-app/src/translate.rs` | Cross-file batching, cache, `%xtra` injection |
| Batch dispatch | `crates/batchalign-app/src/runner/dispatch/infer_batched.rs` | Shared with morphotag and utseg |
| Injection | `crates/batchalign-chat-ops/src/translate.rs` | Writes `%xtra:` tiers from translation strings |
| Worker IPC | `batchalign/inference/translate.py` — `batch_infer_translate()` | Google Translate or Seamless M4T, returns translated text |

Local submissions (auto-daemon or loopback `--server`) use `paths_mode=true`
as of 2026-04-14: the CLI posts source/output path lists instead of CHAT
bytes. See [Submission Modes](../../reference/command-io.md#submission-modes-paths_modetrue-vs-paths_modefalse).

---

## Cache key structure

Translation cache keys (BLAKE3 hash of):
- Normalized utterance text
- Source language code
- Target language code (always `eng`)

---

## Worker IPC: translate task

```
batch_infer request:
{
  "task": "translate",
  "items": [
    { "text": "Bonjour le monde.", "src_lang": "fra", "tgt_lang": "eng" },
    ...
  ]
}

batch_infer response:
[ "Hello world.", ... ]
```

---

## Pre-validation gate

`translate` requires CHAT Level 1.

---

## Testing

```bash
make test
cargo nextest run -p batchalign-app -E 'test(translate::)'
```

---

## Related developer documentation

- [Command Flowcharts: translate](../architecture/command-flowcharts.md#translate)
