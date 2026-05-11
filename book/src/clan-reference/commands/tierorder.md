# TIERORDER -- Reorder Dependent Tiers to Canonical Order

**Status:** Current
## Purpose

Reorders dependent tiers into a consistent order. The legacy manual describes `TIERORDER` as putting dependent tiers into a consistent alphabetical order, with `/lib/fixes/tierorder.cut` able to control the order.

`talkbank-clan` sorts dependent tiers on each utterance according to its built-in canonical ordering.

## Usage

```bash
chatter clan tierorder file.cha
```

## Options

This command has no configurable options.

## Behavior

Dependent tiers are sorted into the following canonical order, grouped by function:

1. **Linguistic analysis tiers** (highest priority):
   `%mor` -> `%gra` -> `%pho` -> `%mod` -> `%wor` -> `%sin`

2. **Behavioral/descriptive tiers**:
   `%act` -> `%cod` -> `%com` -> `%spa` -> `%gpx` -> `%sit` -> `%exp` -> `%int` -> `%add`

3. **Simple text tiers**:
   `%alt` -> `%coh` -> `%def` -> `%eng` -> `%err` -> `%fac` -> `%flo` -> `%gls` -> `%ort` -> `%par` -> `%tim`

4. **User-defined tiers** (lowest priority):
   `%x*` (sorted last)

Utterances with zero or one dependent tier are left unchanged.

## Differences from CLAN

- **Manual configurability not yet mirrored**: The legacy manual describes `tierorder.cut` as controlling tier order. The current implementation uses a built-in ordering instead.
- Operates on AST rather than raw text.
- Uses the framework transform pipeline (parse -> transform -> serialize -> write).
