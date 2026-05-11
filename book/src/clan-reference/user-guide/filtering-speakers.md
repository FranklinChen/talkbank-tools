# Speaker Filtering

**Status:** Current
Speaker filters restrict which speakers' utterances are analyzed. This is one of the most frequently used filters — most CLAN analyses target a specific participant (e.g., the child in a child language study).

## Include speakers

Analyze only specific speakers:

```bash
chatter clan freq --speaker CHI file.cha
chatter clan mlu --speaker CHI --speaker MOT file.cha
```

CLAN equivalent: `+t*CHI`, `+t*CHI +t*MOT`

Multiple `--speaker` flags use OR logic: utterances from *any* listed speaker are included.

## Exclude speakers

Remove specific speakers from analysis:

```bash
chatter clan freq --exclude-speaker INV file.cha
```

CLAN equivalent: `-t*INV`

## @ID filtering

Filter speakers by metadata fields in the `@ID` header (language, corpus, role, age, sex, education, group, custom):

```bash
# All children (by role)
chatter clan freq --id-filter "|||Target_Child" file.cha

# English speakers only
chatter clan freq --id-filter "eng|" file.cha
```

CLAN equivalent: `+t@ID="eng|"`

The `@ID` filter matches against the pipe-delimited fields: `language|corpus|code|age|sex|group|education|role|custom|`.

## Interaction with other filters

- Speaker filtering is applied first, before range or word filters
- Include and exclude can be combined; excludes are applied after includes
- Speaker codes are case-sensitive and must match the `@Participants` header exactly (e.g., `CHI`, not `chi`)
