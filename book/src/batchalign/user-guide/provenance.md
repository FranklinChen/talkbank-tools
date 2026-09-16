# Processing Provenance

**Status:** Current
**Last updated:** 2026-09-16 10:19 EDT

## What is provenance?

Every time batchalign3 processes a CHAT file, it records what it did in a
`@Comment` header. This creates a machine-readable processing history
inside the file itself.

## Format

Batchalign3 provenance comments use a structured format inside square
brackets:

```chat
@Comment:	[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-09-15T18:30:00-04:00]
```

The format is: `[fc-ba3 <command> | <key>=<value> ; ... | <timestamp>]`

- **`fc-ba3`**: identifies this as a batchalign3 provenance comment
- **command**: which operation was performed
- **key=value pairs**: engine identities and options that affect output
- **timestamp**: ISO 8601 with timezone, when processing occurred

Values never contain `|`, `;`, `]` or a line break, and never begin or end
with whitespace, so a comment always reads back the way it was written. A
comment is always one line, however long it is. If a checkpoint you select for
transcribe (for example a custom model name) breaks that rule, the job is
refused when you submit it, before any work runs, rather than writing a
comment that would read back differently.

### Older files: `[ba3 ...]`

Files processed before 2026-09-15 carry the same format under the name `ba3`
(`[ba3 morphotag | ... | ...]`). A file keeps whichever name wrote it.
Batchalign3 treats both names as its own: re-running a command replaces the
older stamp, and reading a file's processing history (for example in the
dashboard) shows both.

If a comment starts like one of these stamps but is damaged (for example the
closing `]` is missing, or a field has no `=`), reading that file's processing
history reports an error naming the file instead of quietly leaving the entry
out.

## Example: Multiple Commands

When you run morphotag, then align on the same file, both comments
accumulate:

```chat
@UTF8
@Begin
@Languages:	eng
@Participants:	CHI Target_Child
@ID:	eng|test|CHI|2;0.||||Target_Child|||
@Comment:	[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | 2026-09-15T18:30:00-04:00]
@Comment:	[fc-ba3 align | fa=whisper-fa-large-v2 ; lang=eng | 2026-09-15T19:15:00-04:00]
*CHI:	the dog is running . 0_4500
%mor:	det|the-Def-Art noun|dog aux|be-Fin-Ind-Pres-S3 verb|run-Part-Pres-S .
%gra:	1|2|DET 2|4|NSUBJ 3|4|AUX 4|0|ROOT 5|4|PUNCT
@End
```

## Re-running a command

If you re-run morphotag on a file that already has a morphotag provenance
comment, the old comment is **replaced**: not duplicated. Comments from
other commands (align, transcribe, etc.) are preserved.

Whether the file is rewritten depends on what changed:

- **Not rewritten** when the only difference is the comment's timestamp or the
  older `ba3` name. The same holds for the transcribe warning below when only
  the build it names, or its older wording, differs.
- **Rewritten** when any recorded value differs: a different engine, a
  different spelling of the engine, a language, or a flag such as
  `retokenize`. The comment then states what produced the file now.
- **Not rewritten** for a damaged comment that belongs to a different command,
  or that is too damaged to say which command wrote it, when it is the same in
  both. Only the command that wrote a comment replaces it, so rewriting the
  file would repair nothing. A damaged comment belonging to the command you are
  running IS replaced, and comes back well formed.

A command that runs other stages writes their comments too: `transcribe`
writes `transcribe`, `utseg` and `morphotag` comments. Every comment the
command writes is compared this way, so re-running `transcribe` with a newer
batchalign3 does not rewrite a file when the transcript and every recorded
value come out the same.

Practical consequence: files stamped by older builds with a different
`engine=` spelling (for example `engine=stanza-1.11.1` or
`engine=stanza-1.11.1:eng`) are rewritten the next time you run morphotag over
them, even if `%mor` and `%gra` come out the same. Expect that one-time churn
when re-running over an existing corpus.

## What each command records

### morphotag

```text
[fc-ba3 morphotag | engine=stanza-1.11.1:eng:standard ; lang=eng | ...]
```

| Key | Meaning |
|-----|---------|
| `engine` | Each Stanza model that analyzed the file, as `stanza-<version>:<language>:<pipeline>`, joined with `+` in text order when more than one ran (for example a secondary language) |
| `lang` | Language code |
| `retokenize` | Present if CJK retokenization was applied |
| `incremental` | Present if `--before` incremental mode was used |
| `ud_repairs` | How many dependency relations had to be repaired. Present only when at least one was |

The pipeline names which analysis ran: `standard`, `mandarin_retokenize`
(Mandarin with retokenization), or `cantonese_pycantonese_pos` (Cantonese with
PyCantonese part-of-speech tagging).

The engine is the one the worker that ran the analysis reported, so a
morphotag stage inside `transcribe` names Stanza rather than the ASR engine.
When no model analyzed anything (for example, only utterances without words),
`engine` is absent rather than guessed.

### Repaired relations

Stanza does not promise that the dependency relations it produces are
Universal Dependencies relations. When one is not, morphotag repairs it rather
than writing it into `%gra`: a padding label (`<PAD>`) or an unrecognised
label becomes `dep`, a relation in the wrong case is lowercased, and a known
non-UD spelling is replaced by the UD relation it means (`iob` becomes
`iobj`). Subtypes are never touched: `nmod:poss` and `acl:relcl` are
legitimate and language-specific.

Those repairs are counted in the comment:

```text
[fc-ba3 morphotag | engine=stanza-1.11.1:ita:standard ; lang=ita ; ud_repairs=3 | ...]
```

`ud_repairs=3` means three relations in this file were repaired. The key is
absent when nothing was repaired, so there is no `ud_repairs=0` to read: no
key means no repairs. A file tagged by a build from before this key existed
says nothing either way.

A high count is worth looking at. It means the model for that language is
producing labels outside UD often, which is a fact about the model rather than
about the transcript, and the transcript now records it instead of leaving it
in a worker log nobody keeps. With `--before`, the count covers the whole file:
the repairs behind the tiers kept from the `--before` file, plus this run's.

With `--before`, morphotag reanalyzes only the utterances you changed and keeps
the other `%mor` and `%gra` tiers from the `--before` file. The comment then
carries `incremental=true`, and `engine` names every model behind the file's
tiers: the models the `--before` file's morphotag comment named, together with
the models used for the reanalysis, each once, in text order. If no model ran
(nothing needed reanalysis, or the changed utterances had no words or were in
an unsupported language), the file's existing morphotag comment is kept as it
is.

The command fails, rather than writing a comment a later run could not read
back, when the `--before` file's morphotag comment is damaged or names an
engine that cannot be written back, or when a model used now has `+` in its
name.

### align

```text
[fc-ba3 align | fa=whisper-fa-large-v2 ; lang=eng ; utr=rev | ...]
```

| Key | Meaning |
|-----|---------|
| `fa` | Forced alignment engine the worker reported |
| `lang` | Language code |
| `utr` | Timing recovery engine (`rev`, `whisper`, `tencent`), present only if a recovery pass ran |
| `wor` | Present if %wor tier was written |
| `incremental` | Present if `--before` incremental mode was used |

### transcribe

```text
[fc-ba3 transcribe | asr=rev ; lang=eng | ...]
```

| Key | Meaning |
|-----|---------|
| `asr` | ASR engine (rev, whisper, tencent, aliyun, funaudio) |
| `asr_model` | The models that produced the transcript, each at the revision it was loaded at: `<id>@<revision>`, with `+<role>:<id>@<revision>` for any helper model (a forced aligner, a voice-activity or punctuation model). Written for every engine. A cloud service records the service and the model type it was called with, never an account credential |
| `lang` | Language code |
| `diarize` | Present if speaker diarization was enabled |
| `wor` | Present if %wor tier was written |

Transcribe also writes a human-readable warning:

```chat
@Comment:	fc-ba3 <build identity>, ASR engine rev. Unchecked output of ASR model, DO NOT USE.
```

The build identity names the exact build that produced the file. Re-running
transcribe replaces this warning, including the older form
`Batchalign <version>, ASR Engine <engine>. Unchecked output of ASR model.`,
but leaves any other comment alone.

When the engine name is followed by a parenthetical, it names the models that
produced the text, in the same form the `asr_model=` field uses, for example
`ASR engine funaudio (FunAudioLLM/SenseVoiceSmall@<commit>+vad:funasr/fsmn-vad@<commit>)`.
The two always agree, because both are rendered from the same value.

A file whose timings were rebuilt by replaying older evidence carries no
parenthetical: nothing in that run reported which models it loaded, and the
warning says less rather than repeating what was requested as though it had
been checked.

### utseg

```text
[fc-ba3 utseg | engine=stanza-constituency ; lang=eng | ...]
```

`engine` names the source of the boundaries that were applied: a boundary
model as `<model id>@<revision>` (or its id alone when the worker exposed no
revision), or `stanza-constituency`. Several sources are joined with `+` in
text order. A source that does not name itself is not given a placeholder
name: the file gets no utseg comment, and the job's per-file record says why.

Files segmented by a build before 2026-09-15 carry no utseg comment, because
the standalone `utseg` command wrote none. Re-running `utseg` over them adds
one.

### translate

```text
[fc-ba3 translate | engine=googletrans-v1 ; lang=spa | ...]
```

`engine` names the engines that produced the translations applied, as each
translation reported them, joined with `+` in text order. A file where nothing
was translated gets no comment.

Files translated by a build before 2026-09-15 carry no translate comment,
because the batch path wrote none. Re-running `translate` over them adds one,
and also re-translates from what was spoken (see
[translate](commands/translate.md)).

### coref

```text
[fc-ba3 coref | engine=stanza-1.11.1/ontonotes-singletons_roberta-large-lora ; lang=eng | ...]
```

`engine` names the model that produced the chains (the Stanza release and the
coreference package), as the result reported it.
A file with nothing resolved gets no comment, and neither does a non-English
file, which passes through untouched.

Files processed by a build before 2026-09-15 carry no coref comment, because
the batch path wrote none. Re-running `coref` over them adds one.

## Parsing provenance programmatically

The stamp names make provenance comments easy to extract. Match both names so
older files are included:

```bash
# Find all provenance comments in a file
grep -E '\[(fc-)?ba3 ' file.cha

# Find all files that were morphotagged
grep -rlE '\[(fc-)?ba3 morphotag' corpus/
```

In Python:

```python
import re

PROVENANCE_RE = re.compile(
    r'^\[(?:fc-)?ba3 (\w+) \| (.*?) \| (\S+)\]$'
)

with open('file.cha') as f:
    for line in f:
        if line.startswith('@Comment:'):
            content = line.split('\t', 1)[1].strip()
            m = PROVENANCE_RE.match(content)
            if m:
                command = m.group(1)     # "morphotag"
                fields = m.group(2)      # "engine=stanza-1.11.1:eng:standard ; lang=eng"
                timestamp = m.group(3)   # "2026-09-15T18:30:00-04:00"
```

From the server, the job results endpoints return each file's provenance
already parsed, as one of three states:

```json
{"kind": "parsed", "entries": [{"command": "morphotag", "fields": {"lang": "eng"}, "timestamp": "..."}]}
{"kind": "unparseable", "reason": "provenance stamp \"[fc-ba3 align]\" does not have a command and a timestamp separated by ` | `"}
{"kind": "not_read"}
```

`not_read` is non-CHAT output or a file that failed. A damaged stamp is
reported for that file only; the other files in the job are served normally.

## When a file carries no comment

A command that writes per-file provenance records what it decided on the file's
own status, so "no comment" is an answer with a reason rather than an absence
you have to interpret:

```json
{"kind": "stamped", "command": "translate"}
{"kind": "not_stamped", "command": "translate", "reason": "no engine produced anything that was applied to this file"}
{"kind": "unrecorded"}
```

`unrecorded` means no stamp decision was recorded: a command that writes no
per-file comment, or a file whose status was rebuilt from the job database
after a server restart (the decision is not persisted).

## What is NOT recorded

Runtime options that don't affect output are omitted:
- `--workers` (concurrency)
- `--timeout` (inference timeout)
- `--server` (where processing happened)
- `--verbose` (logging)
- `--tui` / `--no-tui` (display)

These are operational, not semantic.
