# CHAINS -- Clause Chain Analysis

**Status:** Current
## Purpose

Analyzes sequences of interactional codes on a single specified coding tier. In the legacy manual, `CHAINS` is described as tracking hand-entered codes on one coding tier, with `%cod`, `%spa`, and `%sit` given as examples, and with dependent-tier periods used to mark correspondence to `[^c]` clause units on the main tier.

In `talkbank-clan`, a "chain" is a run of consecutive utterances by the same speaker that all contain a given code item on the selected coding tier. When the code disappears or the speaker changes, the chain is flushed and its length is recorded.

Reports chain count, average/min/max length, and standard deviation per code and speaker.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409147) for the original CHAINS command specification.

## Usage

```bash
chatter clan chains file.cha
chatter clan chains --speaker CHI file.cha
chatter clan chains --tier cod file.cha
```

## Options

| Option | Description |
|--------|-------------|
| `--speaker <CODE>` | Include speaker |
| `--tier <NAME>` | Tier label to read codes from (default: `cod`) |
| `--format <FMT>` | Output format: text, json, csv, clan |

## Output

Per code and speaker:

- Number of separate chains
- Average chain length
- Standard deviation of chain lengths (sample, N-1)
- Minimum chain length
- Maximum chain length

## Differences from CLAN

- **Manual intent**: The legacy manual frames `CHAINS` as a coding-tier command, not a generic dependent-tier tokenizer.
- **Speaker-change detection**: Flushes all open chains for the previous speaker, matching CLAN's behavior of treating chains as speaker-scoped.
- **Coding-tier semantics**: `CHAINS` operates on `%cod`-style coding items, not `%gra`.
- **Code extraction**: For `%cod`, chains now read a clan-local semantic `%cod` item stream derived from the parsed AST rather than flattening the tier back to whitespace text. For other tiers passed via `--tier`, extraction remains content-token based.
- **Manual constraint not yet fully enforced**: The manual examples all use coding tiers. Non-`%cod` `--tier` fallback still tokenizes tier text; this should eventually be restricted to tiers with explicit coding semantics.
- **Standard deviation**: Uses the sample (N-1) formula rather than population.
- **Golden test parity**: Verified against CLAN C binary output.
