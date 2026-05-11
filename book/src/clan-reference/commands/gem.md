# GEM -- Extract Gem Segments

**Status:** Current
## Purpose

Extracts material within gem boundaries. The legacy manual gives `GEM` a dedicated section; in `talkbank-clan`, it extracts utterances and their dependent tiers that fall within `@Bg`/`@Eg` gem boundaries, producing a new CHAT file containing only the gem-scoped content.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409206) for the original GEM command specification.

## Usage

```bash
chatter clan gem file.cha
chatter clan gem --gem story file.cha
```

## CLAN Equivalence

| CLAN command                    | Rust equivalent                            |
|---------------------------------|--------------------------------------------|
| `gem file.cha`                  | `chatter clan gem file.cha`                |
| `gem +g"story" file.cha`        | `chatter clan gem --gem story file.cha`    |

## Options

| Option | CLAN Flag | Description |
|--------|-----------|-------------|
| `--gem <LABEL>` | `+g"label"` | Extract only gem segments matching this label |

Without `--gem`, all gem segments in the file are extracted.

## Behavior

The transform scans for `@Bg:` (begin gem) and `@Eg:` (end gem) header boundaries. All utterances between a matching `@Bg`/`@Eg` pair are included in the output, along with their dependent tiers. The gem boundary headers themselves are preserved. File-level headers and participant metadata are carried through unchanged.

## Differences from CLAN

- Gem boundary detection operates on parsed `Header` variants from the AST rather than raw text line matching for `@BG:`/`@EG:`.
- Handles both `@Bg:`/`@Eg:` (mixed case) and `@BG:`/`@EG:` (uppercase).
- Without `--gem` filter, extracts all gem segments. With `--gem`, extracts only matching labels.
