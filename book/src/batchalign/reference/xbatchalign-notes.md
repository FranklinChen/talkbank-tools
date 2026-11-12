# `%xbatchalign` notes

**Status:** Current
**Last updated:** 2026-11-12

A localized problem never fails a whole file. When Batchalign cannot finish
its work on one utterance, it leaves that utterance as it was written, goes on
with the rest of the file, and says where and why in two places:

- a warning in the job record (the file is reported as written with
  diagnostics), and
- a `%xbatchalign` dependent tier on that utterance, so the spot is visible in
  the CHAT itself, in CLAN or any editor, without the job record.

```text
*CHI:	hello good world .
%xbatchalign:	morphotag: not analysed; the worker could not analyse this utterance (token realignment: native token 0 does not fit word 0)
```

## Format

`%xbatchalign` is a user-defined (`%x`) tier: CHAT treats its content as
unparsed text, so it never affects validation or analysis by CLAN.

- One tier label for every command. The text begins with the command that
  wrote it (`morphotag:`, `align:`), then what happened, then the producer's
  reason in parentheses.
- One line. A reason that contained line breaks is written with spaces, and a
  note longer than 500 characters is cut with `…`; the job record keeps the
  bounded detail too.
- At most one note per command per utterance.

## Reruns

Each run of a command replaces the notes that command wrote before, and leaves
every other command's notes alone. A rerun that has nothing to note removes
that command's old notes, so a note never describes an earlier run. To remove
all notes by hand, delete the `%xbatchalign` lines.

## What writes a note

| Command | Note | Meaning |
|---|---|---|
| `morphotag` | `not analysed; <cause> (<detail>)` | The utterance was declined: it has no new `%mor` or `%gra`. The causes are the job record's `PrimaryMorphologyDeclineCause`s. |

## Not the same as decision tiers

Older versions wrote `%xalign` and `%xrev` tiers recording routine pipeline
decisions. Those are provenance; Batchalign now keeps them in run evidence and
strips them from output. `%xbatchalign` marks only an utterance whose work
could not be done.

## Implementation

`batchalign_transform::notes::replace_notes` is the one writer: it takes the
command and every note for this run, removes that command's earlier notes, and
writes the new ones through the typed CHAT model (Chatter's
`DependentTier::UserDefined` with a checked label). For morphotag,
`PrimaryReport::annotate` builds the notes from the same declined-utterance
list that becomes the job record's `PrimaryMorphologyIncomplete` shortfall, so
the note and the warning cannot disagree; it runs on every morphotag output,
after every write and before output admission.
