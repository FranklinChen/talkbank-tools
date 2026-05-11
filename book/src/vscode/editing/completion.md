# Code Completion & Snippets

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

The extension provides context-aware autocompletion and snippet templates to speed up CHAT editing. Completions are triggered by specific characters at the start of a line or within an utterance, and snippets expand common CHAT structures with a single Tab press.

## Code Completion

Completion suggestions appear automatically as you type certain trigger characters. You can also invoke the completion menu manually with **Ctrl+Space**.

### Speaker Codes (trigger: `*`)

When you type `*` at the start of a line, the extension offers all speakers declared in the file's `@Participants` header. Each suggestion includes the speaker's role and auto-inserts the colon and tab separator:

```text
*|  <-- cursor here

Suggestions:
  CHI   target child     --> inserts "*CHI:\t"
  MOT   mother           --> inserts "*MOT:\t"
  FAT   father           --> inserts "*FAT:\t"
```

This ensures you never mistype a speaker code or use a code that is not declared in the file.

### Tier Types (trigger: `%`)

When you type `%` at the start of a line after a main tier, the extension offers all standard dependent tier types:

```rust,ignore
%|  <-- cursor here

Suggestions:
  mor   Morphological analysis       --> inserts "%mor:\t"
  gra   Grammatical relations        --> inserts "%gra:\t"
  pho   Phonological transcription   --> inserts "%pho:\t"
  mod   Model phonology              --> inserts "%mod:\t"
  sin   Gesture/sign annotations     --> inserts "%sin:\t"
  act   Action coding                --> inserts "%act:\t"
  cod   General coding               --> inserts "%cod:\t"
  com   Comments                     --> inserts "%com:\t"
  ...and more
```

### Headers (trigger: `@`)

When you type `@` at the start of a line, the extension offers all 28 standard CHAT headers with descriptions. Each suggestion auto-inserts the header name, colon, and tab separator:

> **(SCREENSHOT: Completion menu showing @ header options)**
> *Capture this: in a .cha file, place cursor at the beginning of a blank line in the header section and type @. The completion menu should appear showing headers like @Languages, @Participants, @ID, @Media, @Date, @Location, etc.*

Standard headers include `@Begin`, `@End`, `@UTF8`, `@Languages`, `@Participants`, `@ID`, `@Media`, `@Date`, `@Location`, `@Situation`, `@Activities`, `@Comment`, `@Bg`, `@Eg`, `@New Episode`, `@Birth of`, `@Birthplace of`, `@Education of`, `@Group of`, `@SES of`, `@Sex of`, `@Warning`, `@Transcriber`, `@Coder`, `@Options`, `@Font`, `@ColorWords`, and `@Window`.

### Brackets (trigger: `[`)

When you type `[` inside an utterance, the extension offers 17 standard bracket annotations with descriptions:

- `[//]` -- retracing with correction
- `[/]` -- retracing without correction
- `[//?]` -- uncertain retracing
- `[*]` -- error marking
- `[= text]` -- explanation
- `[: alt]` -- replacement
- `[=! text]` -- paralinguistic material
- `[+ post]` -- postcode
- `[- pre]` -- precode
- `[%act: text]` -- inline action
- `[%com: text]` -- inline comment
- And others

This eliminates the need to memorize all bracket codes and ensures correct formatting.

### Postcodes (trigger: `+`)

When you type `+` in an utterance context, the extension offers valid CHAT postcodes:

```text
+|  <-- cursor here

Suggestions:
  +"     Quotation follows
  +,.    Self-completion
  +/.    Interruption
  +/?    Interruption question
  +//.   Self-interruption
  +...   Trailing off
```

## Code Lens

Above the `@Participants` header, a code lens annotation displays the utterance count for each speaker in the file. For example:

```text
CHI: 42 utterances | MOT: 38 utterances
@Participants:	CHI Target_Child, MOT Mother
```

This provides an at-a-glance summary of speaker activity without running an analysis command. The counts update automatically as you edit the document.

## Code Folding

The extension provides CHAT-aware code folding via the language server:

- **Utterance blocks**: Each main tier line (`*SPEAKER:`) together with its dependent tiers (`%mor:`, `%gra:`, etc.) folds as a single unit. Collapsing an utterance hides all its annotation tiers while keeping the main tier visible.
- **Header block**: The header section (`@Begin` through the first utterance) folds as one unit, letting you hide metadata to focus on the transcript body.

Use the fold/unfold controls in the gutter, or the keyboard shortcuts:
- **Fold**: Cmd+Shift+[ (macOS) / Ctrl+Shift+[ (Windows/Linux)
- **Unfold**: Cmd+Shift+] (macOS) / Ctrl+Shift+] (Windows/Linux)
- **Fold All**: Cmd+K Cmd+0
- **Unfold All**: Cmd+K Cmd+J

## Snippets

The extension includes 8 CHAT snippets for common file structures. Type the prefix and press **Tab** to expand. Tab stops (shown as `$1`, `$2`, etc. in the template) let you fill in values sequentially by pressing Tab.

| Prefix | Name | Description |
|--------|------|-------------|
| `@UTF8` / `header` / `newfile` | Header block | Complete file skeleton: `@UTF8`, `@Begin`, `@Languages`, `@Participants`, `@ID` lines, first utterance, `@End` |
| `@Participants` / `participant` | Participants | `@Participants` header with two speaker placeholders |
| `@ID` / `id` | ID header | Single `@ID` line with all 10 pipe-delimited fields as tab stops |
| `*` / `utterance` | Main tier | `*CHI:` utterance line with terminator |
| `%mor` | Mor tier | `%mor:` dependent tier line |
| `%gra` | Gra tier | `%gra:` dependent tier line |
| `@Comment` / `comment` | Comment | `@Comment:` header with content placeholder |
| `@Bg` / `@Eg` / `gem` | Gem block | `@Bg`/`@Eg` pair wrapping a cursor position for gem segments |

### Example: Using the Header Block Snippet

1. Open a new `.cha` file
2. Type `newfile` and press Tab
3. The full file skeleton expands with cursor positioned at the language code
4. Press Tab to move through each placeholder: language, participant names, ID fields, first utterance content

Snippets are defined in `snippets/chat.json` and are available in any `.cha` file.

## Related Chapters

- [Quick Fixes](quick-fixes.md) -- automatic corrections for errors
- [Syntax Highlighting](syntax-highlighting.md) -- visual feedback while editing
- [Special Characters](special-characters.md) -- inserting CA and CHAT Unicode symbols
- [Participant Editor](participant-editor.md) -- visual table editor for `@ID` headers
