"""Strip neural-LM control tokens from Stanza Document output.

Upstream Stanza pipelines occasionally leak character language model
control tokens (``<SOS>``, ``<EOS>``, ``<UNK>``, ``<PAD>``, ``<s>``,
``</s>``) into the public ``Document`` API's ``text`` and ``lemma``
fields. The known case is Finnish MWT splitting of ``tollei`` in a
3+-word context on Stanza 1.11.1 — see
``batchalign/tests/pipelines/morphosyntax/test_stanza_fi_mwt_sos_leak.py``
for the minimal reproducer and
``book/src/reference/stanza-limitations.md`` for the defect-registry
entry (look up by the slug ``stanza-fi-mwt-sos-leak``; the registry's
numeric "Defect N" labels can renumber as entries are retired).

The workaround follows the project's documented upstream-defect
policy (``book/src/developer/upstream-defect-policy.md``): detect the
known upstream defect, rewrite to produce coherent output, log the
rewrite so it's visible in ops monitoring, and let ``chatter
validate`` catch anything we missed. Specifically: strip the
control-token substring from ``text`` and ``lemma`` fields, return
the stripped leaks so callers can log them with filename/language
context.

This is not silent swallowing — the workaround is registered, logged,
and has permanent regression tests. If Stanza fixes the leak upstream,
the detector's preconditions never fire and we retire the workaround.

A parallel-but-currently-unused validator
``UdWord._sanitize_pad_deprel`` in ``morphosyntax.py`` handles the
same defect class for the ``deprel`` field via Pydantic. That
validator is only exercised by tests today; the production
``batch_infer_morphosyntax`` does not pipe Stanza output through
``UdWord``. If a future refactor wires ``UdWord`` into the
production path, consolidate the two sanitizers there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Control-token vocabulary
# ---------------------------------------------------------------------------

#: Regex matching the neural-LM control tokens we strip.
#:
#: Covers the canonical set used by character language models and
#: BERT-family tokenizers: start-of-sequence (``<SOS>``, ``<s>``),
#: end-of-sequence (``<EOS>``, ``</s>``), unknown (``<UNK>``), and
#: padding (``<PAD>``). Case-insensitive so both the tokenizer-stage
#: ``<SOS>`` and the lemmatizer-lowercased ``<sos>`` are caught.
#:
#: The pattern requires exact tag shape ``<NAME>``; it does NOT strip
#: arbitrary angle-bracketed content, so legitimate CHAT/UD text that
#: happens to contain ``<some-word>`` would be preserved.
CONTROL_TOKEN_RE: re.Pattern[str] = re.compile(
    r"<(?:SOS|EOS|UNK|PAD|BOS|CLS|SEP|MASK|s|/s)>",
    re.IGNORECASE,
)


def strip_control_tokens(text: str) -> str:
    """Remove every control-token substring from ``text``.

    Returns the original string if no control tokens are present;
    otherwise returns the text with every match of [`CONTROL_TOKEN_RE`]
    removed. Never panics; never returns ``None``.
    """
    return CONTROL_TOKEN_RE.sub("", text)


# ---------------------------------------------------------------------------
# Leak reporting record
# ---------------------------------------------------------------------------


class LeakField(str, Enum):
    """Which field on a Stanza token dict carried the leaked value.

    ``str`` mixin so the value flows through structured logging
    transparently and existing string-comparison test code keeps
    working without explicit conversion.
    """

    TEXT = "text"
    LEMMA = "lemma"


@dataclass(frozen=True)
class ControlTokenLeak:
    """One Stanza token field that contained a neural-LM control token
    and was rewritten.

    Returned by [`strip_control_tokens_in_sentence`] so the caller can
    emit one tracing warn per leak with filename/language context.
    """

    #: Stanza token id. ``int`` for normal words, ``list[int]`` for
    #: MWT parent tokens. Matches the shape of ``tok["id"]`` from
    #: ``stanza.Document.to_dict()``; carried as a passthrough record
    #: for log messages, never destructured.
    token_id: int | list[int]
    field: LeakField
    #: The raw value Stanza emitted, unsanitized, preserved for logs.
    value: str
    #: What was written into the token dict in place of ``value``.
    stripped: str


# ---------------------------------------------------------------------------
# Stanza sentence workaround
# ---------------------------------------------------------------------------


def strip_control_tokens_in_sentence(
    sent: list[dict[str, Any]],
) -> list[ControlTokenLeak]:
    """Strip control tokens from every token dict in one Stanza sentence.

    Mutates ``sent`` in place: for every token whose ``text`` or
    ``lemma`` contains a control token, the field is replaced with
    its sanitized form.

    Returns the list of leaks that were stripped, in iteration order.
    Empty list means the sentence was already clean. Callers should
    log the returned leaks so the workaround is visible in ops logs
    (same pattern as the typed UD invariants rewrites in
    ``batchalign-chat-ops/src/nlp/invariants/``).

    Operates on the shape produced by ``stanza.Document.to_dict()``:
    a list of token dicts with keys including ``id``, ``text``,
    ``lemma``, ``upos``. Handles MWT parent tokens (``id`` is a list,
    no ``lemma``/``upos``) and regular words uniformly — it only
    touches the ``text`` and ``lemma`` keys when they are strings.
    """
    leaks: list[ControlTokenLeak] = []
    for tok in sent:
        token_id = tok.get("id", -1)
        for field in (LeakField.TEXT, LeakField.LEMMA):
            value = tok.get(field.value)
            if isinstance(value, str) and _contains_control_token(value):
                stripped = strip_control_tokens(value)
                leaks.append(
                    ControlTokenLeak(
                        token_id=token_id,
                        field=field,
                        value=value,
                        stripped=stripped,
                    )
                )
                tok[field.value] = stripped
    return leaks


def _contains_control_token(text: str) -> bool:
    """Fast prefilter before running the regex.

    Control tokens all start with ``<``; skipping the regex on strings
    that have no ``<`` avoids the majority of the per-word cost in the
    common case (well-formed Stanza output). The regex pass confirms
    the token is from our known vocabulary (so legitimate content like
    ``<foo>`` passes through unchanged).
    """
    return "<" in text and bool(CONTROL_TOKEN_RE.search(text))
