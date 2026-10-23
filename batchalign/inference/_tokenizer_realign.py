"""Tokenizer realignment for tokenize_no_ssplit mode.

When Stanza's neural tokenizer runs (``tokenize_no_ssplit=True``), it may
split compound words like "ice-cream" into multiple tokens ("ice", "-",
"cream").  This module provides a ``tokenize_postprocessor`` callback that
merges such spurious splits back, preserving the 1:1 mapping between CHAT
words and Stanza tokens.

MWT hints (True/False tuples)
------------------------------
Stanza's ``tokenize_postprocessor`` uses a tuple convention::

    (text, True): MWT: let the MWT processor expand (e.g. "don't" → do + n't)
    (text, False), NOT an MWT: suppress expansion (e.g. merged "ice-cream")
    plain string: retain the token; this wrapper restores unchanged native hints
    (text, components): retain these checked native components within one word

The realigner distinguishes spelling boundaries from analysis boundaries:

* **Default (all languages)**: merged spurious splits → ``(text, False)``
  Prevents a merge like "ice-cream" from being expanded again by the MWT model.
* **English contractions**: merged text that contains ``'``, *unless* the
  prefix before the first ``'`` is ``"o"`` (e.g. o'clock, o'er) → ``(text, True)``
  Allows "don't", "Claus'" etc. to be handled by Stanza's MWT model. The rule
  is Rust's (``RealignLanguage`` in ``batchalign-transform``), applied by
  ``align_tokens``.
* **French native elisions**: retain the tokenizer's components as an explicit
  expansion only after Rust binds their exact concatenation to one input word.
  No component spelling, lemma or POS is guessed by the realigner.

Thread safety: one pipeline is never called by two threads at once (every
call goes through its entry's guard, ``LoadedPipeline.calling``), and the
context is set inside that guard, so the context a call reads is the one its
own thread set. :class:`TokenizerContext` additionally keeps
``original_words`` in ``threading.local()``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeAlias

from batchalign.inference._domain_types import StanzaCode
from batchalign.inference._italian_mwt import (
    ItalianMwtPolicy,
    MwtExpansion,
    could_be_enclisis,
    decide_expansion,
)
from batchalign_core import RealignLanguage, align_tokens

L = logging.getLogger("batchalign")


TokenizerToken: TypeAlias = str | tuple[str, bool | list[str]]


class TokenizerContext:
    """Context shared between one pipeline's batch callback and its
    postprocessor: the pipeline's realignment language, and the words of the
    call in progress.

    The language is read from the pipeline's Stanza code once, here, by
    ``RealignLanguage.of_stanza_code`` (Rust's mapping, the only one), which
    refuses an empty code. A context therefore cannot exist without the
    language its realignment rules depend on.

    Every call on a pipeline goes through that pipeline's own guard, so one
    call runs on it at a time and the words a batch callback sets are the
    ones its postprocessor reads. ``threading.local()`` keeps each thread's
    ``original_words`` separate as well, so that stays true without relying
    on the guard.
    """

    def __init__(self, stanza_code: StanzaCode) -> None:
        self.language = RealignLanguage.of_stanza_code(stanza_code)
        self._local = threading.local()

    @property
    def original_words(self) -> list[list[str]]:
        return getattr(self._local, "original_words", [])

    @original_words.setter
    def original_words(self, value: list[list[str]]) -> None:
        self._local.original_words = value


def make_tokenizer_postprocessor(
    ctx: TokenizerContext,
    italian_policy: ItalianMwtPolicy | None = None,
) -> Callable[[list[list[TokenizerToken]]], list[list[TokenizerToken]]]:
    """Create a ``tokenize_postprocessor`` callback for ``stanza.Pipeline``.

    The returned callable has the signature Stanza expects::

        postprocessor(tokenized_batch: list[list]) -> list[list]

    where each inner list is the tokens for one sentence (paragraph).

    Parameters
    ----------
    ctx:
        The pipeline's context: its realignment language, which decides how
        merged tokens are hinted, and the original CHAT words of the batch
        in progress, updated before each ``nlp()`` call.
    italian_policy:
        Inputs for the Italian single-word multi-word-token decision. Applied
        only when the context's language is Italian; ``None`` leaves Stanza's
        own behavior untouched.
    """

    def postprocessor(
        tokenized_batch: list[list[TokenizerToken]],
    ) -> list[list[TokenizerToken]]:
        if not ctx.original_words:
            return tokenized_batch
        if len(tokenized_batch) != len(ctx.original_words):
            raise ValueError(
                "token realignment: native sentence count differs from bound input"
            )

        result: list[list[TokenizerToken]] = []
        for sent_idx, sent_tokens in enumerate(tokenized_batch):
            original = ctx.original_words[sent_idx]
            result.append(_realign_sentence(sent_tokens, original, ctx.language))

        # Applied to the WHOLE batch, not per sentence: the decision needs a
        # second Stanza pass over the candidate words, and doing that once per
        # batch instead of once per utterance is the difference between one
        # extra model call and several hundred.
        if ctx.language == RealignLanguage.Italian and italian_policy is not None:
            result = _apply_italian_mwt_policy(result, italian_policy)
        return result

    return postprocessor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _conform(token: TokenizerToken) -> str:
    """Extract text from a Stanza token (string or tuple)."""
    if isinstance(token, tuple):
        return str(token[0])
    return str(token)


def _realign_sentence(
    stanza_tokens: list[TokenizerToken],
    original_words: list[str],
    language: RealignLanguage,
) -> list[TokenizerToken]:
    """Merge Stanza tokens that map to the same original CHAT word.

    Delegates to ``batchalign_core.align_tokens()`` (Rust) for checked
    character-position grouping. French native elision components survive
    as Stanza's explicit-component tuple, retaining one CHAT word without
    suppressing the article/pronoun analysis inside it.

    Stanza may return tokens with embedded spaces (rare edge case). These
    are flattened before passing to Rust so the character sequences match.

    Returned items may be plain strings or ``(text, bool)`` tuples, matching
    Stanza's postprocessor contract for MWT expansion hints.

    MWT hint preservation
    ---------------------
    Stanza's tokenizer natively emits ``(text, True)`` tuples for tokens
    it wants its MWT processor to expand (English contractions, French
    elisions, etc.). The Rust ``align_tokens`` function operates on raw
    strings for its char-DP algorithm, so tuple hints are temporarily
    erased when we flatten. When no merging actually happened (1:1
    mapping between Stanza tokens and the aligner's output), we overlay
    Stanza's original tuples back onto the aligned output so MWT
    expansion still fires downstream. Without this overlay, Stanza's MWT
    processor sees only plain strings and silently skips expansion, the
    direct cause of the 2026-04-13 Preserve-mode regression.
    """
    # Flatten tokens that Stanza may have returned with embedded spaces
    flat_tokens: list[TokenizerToken] = []
    for tok in stanza_tokens:
        text = _conform(tok)
        parts = text.split(" ")
        flat_tokens.extend(parts if len(parts) > 1 else [tok])

    merged = align_tokens(
        original_words, [_conform(tok) for tok in flat_tokens], language
    )

    # Restore Stanza's own MWT hint tuples wherever a merged/aligned token
    # still corresponds 1:1 to one original Stanza token. This matters even
    # when another token elsewhere in the sentence was merged (e.g.
    # "mm" + "-" + "hmm" -> "mm-hmm"): the unaffected English contraction
    # later in the sentence still needs its native (text, True) hint so the
    # MWT processor expands it.
    restored: list[TokenizerToken] = []
    orig_idx = 0
    for new in merged:
        target = _conform(new)
        start = orig_idx
        buf = ""
        while orig_idx < len(flat_tokens) and len(buf) < len(target):
            buf += _conform(flat_tokens[orig_idx])
            orig_idx += 1
            if buf == target:
                break

        if buf != target:
            raise RuntimeError("checked realignment lost its native token binding")

        if isinstance(new, tuple):
            restored.append(new)
            continue

        if (
            orig_idx - start == 1
            and isinstance(flat_tokens[start], tuple)
            and _conform(flat_tokens[start]) == target
        ):
            restored.append(flat_tokens[start])
        else:
            restored.append(new)

    if orig_idx != len(flat_tokens):
        raise RuntimeError("checked realignment omitted native tokens")

    return restored


def _apply_italian_mwt_policy(
    batch: list[list[TokenizerToken]],
    policy: ItalianMwtPolicy,
) -> list[list[TokenizerToken]]:
    """Suppress Italian MWT expansion where Stanza's proposed split is not real.

    Stanza's Italian MWT over-splits, inventing verbs. Context-free it turns
    ``cavallo`` into *cava* + *lo* and ``attenzione`` into *attenzi* + *ne*; in
    full sentences it turns the article ``la`` into *il* + *i* and ``mozzarella``
    into *mozzar* + *la*. Both are the same defect, so both are judged here by
    the same rule. What counts as a genuine Italian multi-word token, and the
    measurements behind it, live in ``batchalign/inference/_italian_mwt.py``.

    Candidates are exactly the tokens Stanza itself marked ``(text, True)``,
    which is its documented way of saying "the MWT processor should expand
    this". Reading that marker rather than guessing which words might be
    multi-word tokens is what makes the pass both correct in every context and
    cheap: a typical sentence marks two tokens out of eleven, and the rest are
    never probed.

    Emitting ``(text, False)`` is the same protocol in reverse: do not expand.
    """
    lexicon = policy.lexicon()
    if lexicon is None:
        # A check that did not run must not change the analysis.
        L.debug("Italian MWT policy inactive: lexicon unavailable")
        return batch

    # Two kinds of candidate, judged by the same rule:
    #  - tokens Stanza already marked `(text, True)`, which may be over-splits;
    #  - tokens it left alone that look lexically like enclisis, which may be
    #    the opposite defect, where it declines to split a real imperative and
    #    invents a verb for the whole surface (`aprilo` -> `verb|aprilare`).
    candidates: dict[tuple[int, int], str] = {}
    for sent_idx, tokens in enumerate(batch):
        for tok_idx, token in enumerate(tokens):
            text = _conform(token)
            already_marked = isinstance(token, tuple) and token[1]
            if already_marked or could_be_enclisis(text, lexicon):
                candidates[(sent_idx, tok_idx)] = text

    if not candidates:
        return batch

    # One probe call for the whole batch. It gets whole utterances, not the
    # candidate words alone, because Stanza's MWT is context-sensitive: `hai`
    # in isolation is left whole but becomes *ha* + *i* mid-sentence, so an
    # isolated probe would report nothing to judge and the over-split would
    # pass straight through.
    #
    # Every candidate is force-hinted. That costs nothing for the ones Stanza
    # was already going to expand (the hint is what it emitted itself), and it
    # is the only way to see the split it declines to make.
    utterances = [[_conform(token) for token in tokens] for tokens in batch]
    per_sentence = policy.propose_splits(utterances, frozenset(candidates.values()))
    if per_sentence is None:
        L.debug("Italian MWT policy inactive: split probe unavailable")
        return batch

    adjusted = [list(tokens) for tokens in batch]
    for (sent_idx, tok_idx), text in candidates.items():
        if sent_idx >= len(per_sentence):
            continue
        split = per_sentence[sent_idx].get(text)
        if split is None or len(split) < 2:
            # Nothing to judge: leave the token exactly as Stanza left it rather
            # than asserting an expansion that has no proposed shape.
            continue
        if decide_expansion(text, split, lexicon) is MwtExpansion.ALLOW:
            adjusted[sent_idx][tok_idx] = (text, True)
        else:
            adjusted[sent_idx][tok_idx] = (text, False)
    return adjusted
