"""One Stanza pipeline is never called by two threads at once."""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any

from batchalign.inference.morphosyntax import batch_infer_morphosyntax
from batchalign.providers import BatchInferRequest
from batchalign.worker._pipeline_cache import static_pipelines


class _Doc:
    def __init__(self, rows: list[list[dict[str, Any]]]) -> None:
        self._rows = rows

    def to_dict(self) -> list[list[dict[str, Any]]]:
        return deepcopy(self._rows)


class _OverlapDetectingNlp:
    """A pipeline double that records whether two calls were ever inside it.

    The first call waits inside until the second thread has started its
    batch (bounded), so a missing guard is observed deterministically rather
    than by a lucky interleaving; with the guard, the second call cannot enter
    until the first has left, and no overlap can be recorded.
    """

    def __init__(self, second_started: threading.Event) -> None:
        self._second_started = second_started
        self._count_lock = threading.Lock()
        self._inside = 0
        self.max_inside = 0
        self.calls = 0

    def __call__(self, text: str) -> _Doc:
        with self._count_lock:
            self._inside += 1
            self.calls += 1
            first = self.calls == 1
            self.max_inside = max(self.max_inside, self._inside)
        if first:
            self._second_started.wait(timeout=2)
            # Give the second thread time to reach the pipeline.
            threading.Event().wait(0.2)
        with self._count_lock:
            self._inside -= 1
        return _Doc(
            [
                [
                    {
                        "id": 1,
                        "text": "hello",
                        "lemma": "hello",
                        "upos": "INTJ",
                        "head": 0,
                        "deprel": "root",
                    },
                    {
                        "id": 2,
                        "text": ".",
                        "lemma": ".",
                        "upos": "PUNCT",
                        "head": 1,
                        "deprel": "punct",
                    },
                ]
            ]
        )


def _request() -> BatchInferRequest:
    return BatchInferRequest(
        task="morphosyntax",
        lang="eng",
        items=[
            {
                "words": ["hello"],
                "terminator": ".",
                "special_forms": [],
                "runs": [1],
                "lang": "eng",
            }
        ],
    )


def test_one_pipeline_is_never_called_by_two_threads_at_once() -> None:
    """RED FIRST: on free-threaded Python the batch skipped the lock.

    Stanza does not promise that one pipeline may be called concurrently, and
    its unreleased lemma results cache is an unsynchronized ``OrderedDict``.
    A pipeline is now callable only through its own guard, in every
    interpreter mode.
    """
    second_started = threading.Event()
    nlp = _OverlapDetectingNlp(second_started)
    pipelines = static_pipelines({"eng": nlp})

    def run() -> None:
        batch_infer_morphosyntax(_request(), pipelines)

    first = threading.Thread(target=run)
    first.start()
    second = threading.Thread(target=lambda: (second_started.set(), run()))
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert nlp.calls == 2
    assert nlp.max_inside == 1, "two threads were inside one Stanza pipeline at once"
