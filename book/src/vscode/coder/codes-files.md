# Codes Files (.cut)

**Status:** Current
**Last updated:** 2026-03-30 13:40 EDT

Codes files (`.cut`) define the hierarchical coding scheme used by Coder Mode. Each file describes the valid codes for a particular research task -- speech act annotation, error coding, conversational strategy labeling, and so on.

## File Format

A `.cut` file is a plain text file with the following structure:

1. **Header lines** -- metadata and CLAN configuration options (lines starting with `#`, `\`, `@`, or the magic first line)
2. **Target tier declaration** -- a `%xxx:` line specifying which dependent tier receives the codes (e.g., `%spa:`, `%err:`, `%cod:`)
3. **Code hierarchy** -- indented lines defining the valid codes and their tree structure

### Header Section

The file begins with CLAN-specific metadata that the VS Code extension ignores:

```text
lXs Special Text file saves all fonts LxS
@Font:	CAfont:13:0
@UTF8
# coder mode commands:
# ESC-e  = toggle between coders and text editing modes
# ...
\ +b50 +d +l1 +s1 +fCAfont
```

Lines starting with `#` are comments. The `\` line contains CLAN coder options (auto-save interval, font selection, reorder behavior). These are specific to CLAN's modal editor and have no effect in the VS Code implementation.

### Target Tier

A line like `%spa:` or `%err:` declares the dependent tier that will receive the codes. When you select a code in the QuickPick, it is inserted as this tier type on the current utterance.

### Code Hierarchy

Codes are organized in an indented tree. Each level of indentation represents a level in the hierarchy. Codes typically follow the pattern `$CATEGORY:subcategory`:

```text
%spa:
 " $POS
  :QUE
   :NV
   :VE
  :ANS
   :NV
   :VE
  :COM
   :NV
   :VE
  :ACK
   :VE
```

In this example:
- `$POS` is a top-level category (positive speech acts)
- `:QUE`, `:ANS`, `:COM`, `:ACK` are subcategories under `$POS`
- `:NV` and `:VE` are leaf codes under each subcategory (non-verbal and verbal)

The `"` prefix marks speaker scope boundaries (e.g., `" $MOT` means the following codes apply when coding the mother's utterances).

## Example: codes-basic.cut

The simplest standard code file, `codes-basic.cut`, defines a basic speech act scheme for mother-child interaction:

```text
%spa:
 " $MOT
  :POS
   :Que
   :Res
  :NEG
 " $CHI
```

This creates a hierarchy where:
- Mother (`$MOT`) utterances can be coded as positive (`:POS`) with subtypes question (`:Que`) or response (`:Res`), or as negative (`:NEG`)
- Child (`$CHI`) utterances use the same top-level scheme

## Example: codeserr.cut

The error coding file defines categories for lexical errors on the `%err` tier:

```text
%err:
 " $LEX
  :INC     incomplete
  :UNC     uncertain
  :ACC     accented
  :ADD     added
  :LOS     lost/omitted
  :HAP     haplology
  :BLE     blend
  :EX1     example type 1
  :EX2     example type 2
  :SH1     shift type 1
  :SH2     shift type 2
```

## Standard Code Files

The following `.cut` files are distributed with TalkBank in `clan-info/lib/coder/`:

| File | Lines | Target Tier | Description |
|------|-------|-------------|-------------|
| `codes-basic.cut` | 24 | `%spa` | Minimal speech act scheme (POS/NEG, Que/Res) |
| `codes1.cut` | 35 | `%spa` | Extended speech acts (QUE, ANS, COM, ACK with NV/VE subtypes) |
| `codeserr.cut` | 114 | `%err` | Lexical error classification (INC, UNC, ACC, ADD, LOS, etc.) |
| `codeshar.cut` | 109 | `%spa` | Conversational strategies (NIA: acknowledgment, adaptation, clarification, giving, greeting, etc.) |

## Creating Custom .cut Files

To create a custom coding scheme:

1. **Start from a template.** Copy one of the standard `.cut` files and modify it, or start from scratch with the minimal structure below.

2. **Minimal structure:**

   ```text
   @UTF8
   %cod:
    " $TOP
     :sub1
     :sub2
      :leaf1
      :leaf2
   ```

3. **Choose your target tier.** Use `%cod:` for general-purpose coding, `%spa:` for speech acts, `%err:` for error coding, or any valid CHAT dependent tier prefix.

4. **Define the hierarchy.** Use single-space indentation for each level. Category codes start with `$`, subcodes start with `:`.

5. **Keep it flat enough to navigate.** The VS Code QuickPick displays the hierarchy with indentation. Deep hierarchies (more than 4 levels) become difficult to scan in the picker.

## Tips for Organizing Code Hierarchies

- **Use meaningful category prefixes.** `$PRA` for pragmatic functions, `$ERR` for errors, `$NIA` for conversational acts. The prefix makes codes identifiable when reading coded tiers in the transcript.

- **Balance breadth and depth.** A flat list of 50 codes is hard to navigate; a deep tree with 6 levels is also hard. Aim for 3-4 levels with 5-10 items per level.

- **Document your codes.** Add `#` comment lines in the `.cut` file explaining what each category means. Both CLAN and the VS Code extension ignore comment lines.

- **Use speaker scoping sparingly.** The `"` prefix scopes codes to specific speakers. This is useful when different participants have different valid codes (e.g., adult vs. child speech acts), but adds complexity.

- **Test with a small file first.** Open the `.cut` file in Coder Mode with a short transcript to verify the hierarchy displays correctly in the QuickPick before coding a large corpus.

## Related Chapters

- [Coder Mode Overview](overview.md) -- what coding is and why to use it
- [Coding Workflow](workflow.md) -- step-by-step usage in VS Code
