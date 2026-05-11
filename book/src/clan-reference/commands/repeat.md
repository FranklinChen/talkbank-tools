# REPEAT -- Mark Utterances Containing Revisions

**Status:** Current
## Purpose

Reimplements CLAN's `repeat` command, which adds a `[+ rep]` postcode to utterances from a target speaker that contain revision markers. Only utterances that do not already have `[+ rep]` are modified.

## Usage

```bash
chatter clan repeat --speaker CHI file.cha
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--speaker` | speaker code | *(required)* | Target speaker to process. Only utterances from this speaker are checked. |

## Revision Markers Detected

- `[//]` -- retracing (exact repetition with correction)
- `[///]` -- multiple retracing
- `[/-]` -- reformulation (false start)
- `[/?]` -- uncertain retracing

Note: Simple repetitions (`[/]`) do **not** trigger the `[+ rep]` marker. Only revisions and reformulations do.

## Behavior

For each utterance from the target speaker, the transform checks whether the main-tier content contains any revision markers. If revision markers are found and the utterance does not already have a `[+ rep]` postcode, one is appended.

Utterances from other speakers are left unchanged.

## Differences from CLAN

- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
