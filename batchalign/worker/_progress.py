"""User-visible phase events for time-transparency UX.

Time transparency is a project-wide UX principle: any operation that takes
more than ~1 second must surface to every UI surface batchalign3 exposes
(CLI, TUI, desktop app, web dashboard). Silent waits are UX bugs, the user
must always know what BA3 is doing and roughly how long it'll take.

This module emits PHASE events: a model loading from disk, a catalog check,
a notice. Each is a ``PhaseV2`` measure (its edge, ``BEGIN`` or ``COMPLETE``)
on the phase's stable stage name, which carries no edge suffix. A model
DOWNLOAD is announced only by ``_download_liveness.download_liveness``, when
bytes arrive, with its growth reported. The stages here are a closed set
(``PhaseStage``) none of which spells ``downloading_``, so no code outside
liveness can announce a download, in advance or at all; an audit with
exemption comments used to hold that rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

from batchalign.worker._protocol import write_load_progress
from batchalign.worker._types_v2 import PhaseEdgeV2, PhaseV2

L = logging.getLogger(__name__)


class Phase(Enum):
    """A fixed phase stage, by its wire name."""

    REFRESHING_STANZA_CATALOG = "refreshing_stanza_catalog"
    LOADING_FUNAUDIO_ASR = "loading_funaudio_asr"
    LOADING_QWEN_ASR = "loading_qwen_asr"
    UTSEG_FALLBACK_OPT_IN = "utseg_unsupported_language_fallback"


@dataclass(frozen=True, slots=True)
class LoadingTask:
    """The bootstrap's load of one task's models for one language."""

    task: str
    lang: str


PhaseStage = Phase | LoadingTask
"""Every phase stage there is. None is a download."""


def _stage_name(stage: PhaseStage) -> str:
    match stage:
        case Phase() as phase:
            return phase.value
        case LoadingTask(task=task, lang=lang):
            return f"loading_task_{task}_{lang}"
        case _ as unreachable:
            assert_never(unreachable)


def emit_phase_event(stage: PhaseStage, edge: PhaseEdgeV2, user_message: str) -> None:
    """Surface one phase's beginning or end to every UI, via ``progress_v2``.

    Writes ``{"op": "progress_v2", "event": {...}}`` to stdout as load
    progress (bound to no request), and logs ``user_message`` so it reaches
    the daemon log even before the runner's forwarder is wired. The supervisor
    re-arms a starting worker's stall window the first time it sees each
    (stage, edge), so a startup loading several tasks is not killed between
    them, and a repeated phase extends nothing.
    """
    name = _stage_name(stage)
    L.info("phase event: %s %s: %s", name, edge.value, user_message)
    write_load_progress(name, PhaseV2(edge=edge))
