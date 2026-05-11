# CHECK — CHAT File Validation

**Status:** Current
CHECK validates CHAT files for structural correctness, checking headers, tier
formatting, bracket matching, bullet consistency, speaker declarations, and more.

## Usage

```bash
chatter clan check file.cha
chatter clan check +c0 file.cha          # Full bullet check
chatter clan check +e file.cha           # List all error numbers
chatter clan check +e6 file.cha          # Only report error 6
chatter clan check -e6 file.cha          # Exclude error 6
chatter clan check +g2 file.cha          # Check CHI has Target_Child
chatter clan check +g5 file.cha          # Check for unused speakers
chatter clan check +u file.cha           # Check UD features on %mor
```

## Options

| CLAN flag | Modern flag | Description |
|-----------|-------------|-------------|
| `+c0` | `--bullets 0` | Full bullet consistency check |
| `+c1` | `--bullets 1` | Check for missing bullets only |
| `+e` | `--list-errors` | List all 161 error numbers and exit |
| `+eN` | `--error N` | Only report error number N (repeatable) |
| `-eN` | `--exclude-error N` | Exclude error number N (repeatable) |
| `+g1` | *(no-op)* | Prosodic delimiters (always recognized) |
| `+g2` | `--check-target` | Verify CHI has Target_Child role |
| `+g3` | *(partial)* | Word detail checks (via parser) |
| `+g4` | `--check-id` | Check for missing @ID tiers (on by default) |
| `+g5` | `--check-unused` | Check for unused speakers |
| `+u` | `--check-ud` | Validate UD features on %mor tier |

## Output Format

CHECK output matches CLAN's format:

```text
*** File "sample.cha": line 12.
*CHI:	doggy wanna play .
[E501] Illegal word character in 'wanna' (47)
```

Each error shows the file path and line number, the offending tier text, and the
error message with CHECK's numbered error code in parentheses.

Errors that don't map to a CHECK number show our internal code in brackets instead:

```text
*** File "sample.cha": line 5.
@Participants:	CHI Child
Missing role for CHI — expected format: CODE Name Role [E312]
```

## CHECK vs `chatter validate`

See [CHECK vs chatter validate](../divergences/check-vs-validate.md) for a
detailed comparison of these two validation tools.

## Differences from CLAN

- **Parsing**: Uses tree-sitter grammar instead of CLAN's character-by-character
  parser. More rigorous and consistent; catches structural errors that CHECK
  sometimes misses.
- **Error numbering**: CLAN CHECK uses flat numbers 1-161. We map our typed
  error codes to CHECK numbers where correspondence exists; unmapped errors
  get number 0.
- **Two-pass vs single-pass**: CLAN runs `check_OverAll` then `check_CheckRest`.
  Our parser combines both into a single streaming parse+validate pipeline.
- **depfile.cut**: CLAN reads `depfile.cut` for tier/code templates. We validate
  against the CHAT specification directly.
- **Bug fixes**: Several CHECK errors in the original are unreachable or
  duplicate (e.g., errors 51, 96 are commented out). We skip those.
- **`+g1`**: Always a no-op — our parser recognizes prosodic delimiters by default.
- **`+g3`**: Partially implemented through existing word validation.
