# Scoped Find

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

VS Code's built-in Find (`Cmd+F`) searches the entire file as raw text. Scoped Find adds CHAT-aware filtering: search within specific tiers, specific speakers, or both. This replaces the tedious workflow of visually scanning tier lines or using fragile regex patterns that try to match CHAT structure.

## How to Use

1. Open a `.cha` file.
2. Invoke the command using any of these methods:
   - **Context menu:** right-click in the editor, then select **Find in Tier...**
   - **Command Palette:** `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Windows/Linux), then type **TalkBank: Find in Tier**
3. A picker appears with two filter dimensions:
   - **Tier type:** main tier (`*SPK:`), `%mor`, `%gra`, `%pho`, `%sin`, `%act`, `%cod`, `%com`, `%exp`, or all tiers.
   - **Speaker:** optionally restrict the search to utterance blocks belonging to one or more speakers.
4. Enter your search query. Supports both plain text and regex (prefix with `/` for regex mode).
5. Results appear in a QuickPick list with clickable matches that navigate directly to the matching location in the file.

## Search Modes

### Plain Text

Type a word or phrase to find exact matches within the selected tier scope. The search is case-insensitive by default.

**Example:** searching for `cookie` in the `%mor` tier finds all morphological items containing "cookie" (e.g., `n|cookie`, `n|cookie-PL`).

### Regex

Prefix your query with `/` to use regular expression matching.

**Example:** searching for `/v\|want` in the `%mor` tier finds all verb entries for "want" (matching `v|want`, `v|want-PAST`, etc.).

## Use Cases

- **Find a word in one speaker's speech:** select the main tier, filter to `CHI`, and search for `cookie`. This finds only the child's productions of that word, ignoring adult uses.
- **Search %mor for POS tags:** select the `%mor` tier and search for `adj|` to find all adjective uses across the file.
- **Find specific dependency relations:** select the `%gra` tier and search for `SUBJ` to find all subject relations.
- **Search annotations:** select `%cod` or `%act` to search within coding or speech act tiers only.

## How It Works

The scoped find uses the parsed `ChatFile` model to determine tier boundaries. It identifies which spans of the document belong to each tier type and each speaker, then searches raw text only within those spans. This means it never produces false matches from other tiers or headers -- unlike a raw regex search that might accidentally match a `%mor` pattern appearing in a comment or header.

Results appear in a VS Code QuickPick list. Selecting a result navigates the cursor to the exact location of the match.

## Comparison with Legacy CLAN

Legacy CLAN has no equivalent to scoped find. Users must use the generic find-and-replace dialog without any tier or speaker awareness, relying on manual inspection to filter irrelevant matches. The scoped find feature brings the semantic awareness of CLAN's `KWAL` and `COMBO` search commands into the interactive editor experience.

## Related Chapters

- [Speaker Filtering](speaker-filtering.md) -- create a filtered view of selected speakers' utterances
- [Cross-Tier Alignment](alignment.md) -- hover and highlighting to explore tier relationships
- [CLAN Analysis Commands](../analysis/command-reference.md) -- `kwal` and `combo` for batch search across files
- [Keyboard Shortcuts](../configuration/keyboard-shortcuts.md) -- bind scoped find to a custom shortcut
