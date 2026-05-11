# CHSTRING -- String Replacement Using a Changes File

**Status:** Current
## Purpose

Reimplements CLAN's `chstring` command, which reads a changes file containing find/replace pairs (alternating lines) and applies text substitutions to main-tier words. Replacements are applied to all word nodes, including words inside annotated groups, replacement forms, and bracketed groups.

See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409309) for the original command documentation.

## Usage

```bash
chatter clan chstring --changes changes.cut file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--changes` | path | `changes.cut` | Path to the changes file containing find/replace pairs |

## Changes File Format

The changes file contains alternating lines of find and replace strings:

```text
find_text1
replace_text1
find_text2
replace_text2
```

The file must have an even number of non-empty lines. By default, CLAN looks for `changes.cut` in the current directory.

## Behavior

For each utterance in the file, the transform walks all word nodes on the main tier -- including words inside annotated groups, replacement forms, and bracketed groups -- and applies find/replace substitutions from the changes file.

## Differences from CLAN

- Operates on the parsed AST rather than raw text, ensuring structural integrity of the CHAT file after substitution.
- Does not support CLAN's regex-based pattern matching in the changes file.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
- **Golden test parity**: Verified against CLAN C binary output.
