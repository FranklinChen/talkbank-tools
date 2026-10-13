"""Owned UD graph admission and dependency-preserving cue projection."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from batchalign.inference.morphosyntax import (
        JSONObject,
        RelationRepair,
        RepairedSentence,
    )


class ProjectionFailure(StrEnum):
    INVALID_ID = "invalid word id"
    DUPLICATE_ID = "duplicate word id"
    MISSING_HEAD = "missing dependency target"
    CYCLE = "dependency cycle"
    MWT = "malformed multiword token"
    BOUNDARIES = "realigned boundaries do not match model output"
    CUE = "final realigned boundary is not a single cue word"
    AMBIGUOUS_ROOT = "ambiguous lexical root after cue removal"
    POS_ALIGNMENT = "POS override does not preserve word identity"


class DependencyProjectionError(ValueError):
    """A named admission/projection failure, serialized at the item boundary."""

    def __init__(self, kind: ProjectionFailure, detail: str) -> None:
        self.kind = kind
        super().__init__(f"{kind.value}: {detail}")


@dataclass(frozen=True, slots=True)
class _Word:
    head: int


class DependencyGraph:
    """Copied rows and their admitted ID/head index belong to one sentence.

    External row shapes and relation aliases were admitted by RepairedSentence.
    This constructor admits the relationships: no missing target, duplicate ID,
    cycle, or detached MWT component can reach projection or emission.
    """

    def __init__(self, sentence: RepairedSentence) -> None:
        self._rows = sentence.words
        self._repairs = sentence.repairs
        self._repair_rows = sentence.repair_rows
        words: dict[int, _Word] = {}
        tokens: list[int] = []
        components: tuple[int, ...] = ()
        for at, row in enumerate(self._rows):
            identifier = row["id"]
            if isinstance(identifier, float):
                # UD empty nodes are not syntactic words or token boundaries.
                if components:
                    raise DependencyProjectionError(
                        ProjectionFailure.MWT, str(identifier)
                    )
                continue
            if isinstance(identifier, list):
                if components:
                    raise DependencyProjectionError(
                        ProjectionFailure.MWT, str(identifier)
                    )
                match identifier:
                    case [int(start), int(end)] if 0 < start <= end:
                        pass
                    case _:
                        raise DependencyProjectionError(
                            ProjectionFailure.MWT, str(identifier)
                        )
                # Bound allocation by the actual rows, not an untrusted UD id.
                if end - start + 1 > len(self._rows) - at - 1:
                    raise DependencyProjectionError(
                        ProjectionFailure.MWT, str(identifier)
                    )
                components = tuple(range(start, end + 1))
                tokens.append(at)
                continue
            if not isinstance(identifier, int) or identifier <= 0:
                raise DependencyProjectionError(
                    ProjectionFailure.INVALID_ID, str(identifier)
                )
            if identifier in words:
                raise DependencyProjectionError(
                    ProjectionFailure.DUPLICATE_ID, str(identifier)
                )
            head = row["head"]
            if not isinstance(head, int) or head < 0:
                raise DependencyProjectionError(
                    ProjectionFailure.MISSING_HEAD, str(head)
                )
            words[identifier] = _Word(head)
            if components:
                if identifier != components[0]:
                    raise DependencyProjectionError(
                        ProjectionFailure.MWT, str(identifier)
                    )
                components = components[1:]
            else:
                tokens.append(at)
        if components:
            raise DependencyProjectionError(ProjectionFailure.MWT, str(components))
        for identifier, word in words.items():
            if word.head != 0 and word.head not in words:
                raise DependencyProjectionError(
                    ProjectionFailure.MISSING_HEAD,
                    f"word {identifier} points to {word.head}",
                )
        # Each edge is visited once; the finished set is shared across chains.
        finished: set[int] = set()
        for identifier in words:
            chain: set[int] = set()
            cursor = identifier
            while cursor != 0 and cursor not in finished:
                if cursor in chain:
                    raise DependencyProjectionError(
                        ProjectionFailure.CYCLE, str(cursor)
                    )
                chain.add(cursor)
                cursor = words[cursor].head
            finished.update(chain)
        self._words = words
        self._tokens = tuple(tokens)

    def preserve(self) -> AnalyzableSentence:
        """Free tokenization owns no positional cue and deletes no row."""
        return AnalyzableSentence._owned(self._rows, self._repairs)

    def preserve_boundaries(self, boundaries: int) -> AnalyzableSentence:
        """Admit a lexical-only model result without deleting any token."""
        if len(self._tokens) != boundaries:
            raise DependencyProjectionError(
                ProjectionFailure.BOUNDARIES,
                f"expected {boundaries}, got {len(self._tokens)}",
            )
        return self.preserve()

    def project_cue(self, boundaries: int) -> AnalyzableSentence:
        """Contract the source-owned final cue without inventing a head/root.

        An attached cue's parent is an unambiguous replacement target. A root
        cue has a replacement only when it has exactly one lexical child and
        no other root exists. Contracting either admitted edge preserves the
        graph's closure and acyclicity, so emission needs no second audit.
        """
        if len(self._tokens) != boundaries or not self._tokens:
            raise DependencyProjectionError(
                ProjectionFailure.BOUNDARIES,
                f"expected {boundaries}, got {len(self._tokens)}",
            )
        cue_at = self._tokens[-1]
        cue = self._rows[cue_at]["id"]
        if not isinstance(cue, int):
            raise DependencyProjectionError(ProjectionFailure.CUE, str(cue))
        cue_word = self._words[cue]
        children = [
            identifier for identifier, word in self._words.items() if word.head == cue
        ]
        promoted: int | None = None
        if cue_word.head == 0:
            other_roots = [
                identifier
                for identifier, word in self._words.items()
                if word.head == 0 and identifier != cue
            ]
            if len(children) != 1 or other_roots:
                raise DependencyProjectionError(
                    ProjectionFailure.AMBIGUOUS_ROOT,
                    f"cue {cue}, children {children}, other roots {other_roots}",
                )
            promoted = children[0]
            target = promoted
        else:
            target = cue_word.head
        rows = []
        repairs = [
            entry.repair for entry in self._repair_rows if entry.position != cue_at
        ]
        for at, original in enumerate(self._rows):
            if at == cue_at:
                continue
            row = deepcopy(original)
            if row["id"] == promoted:
                row["head"] = 0
                if row["deprel"] != "root":
                    # Runtime import avoids a module-initialization cycle:
                    # the row/repair owner constructs this validated record.
                    from batchalign.inference.morphosyntax import (
                        RelationRepair,
                        RelationRepairKind,
                    )

                    repairs.append(
                        RelationRepair(
                            kind=RelationRepairKind.CUE_ROOT_PROMOTION,
                            word=cast(str, row["text"]),
                            from_relation=cast(str, row["deprel"]),
                            to_relation="root",
                        )
                    )
                row["deprel"] = "root"
            elif row["head"] == cue:
                row["head"] = target
            rows.append(row)
        return AnalyzableSentence._owned(tuple(rows), tuple(repairs))


@dataclass(frozen=True, slots=True, init=False)
class AnalyzableSentence:
    """Only the owning graph can issue an immutable emission capability."""

    _rows: tuple[JSONObject, ...]
    repairs: tuple[RelationRepair, ...]

    def __init__(self) -> None:
        raise TypeError("analyzable sentences are issued by DependencyGraph")

    @classmethod
    def _owned(
        cls,
        rows: tuple[JSONObject, ...],
        repairs: tuple[RelationRepair, ...],
    ) -> AnalyzableSentence:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_rows", rows)
        object.__setattr__(instance, "repairs", repairs)
        return instance

    def wire_rows(self) -> list[JSONObject]:
        """JSON needs mutable containers; never expose the owned graph rows."""
        return deepcopy(list(self._rows))

    def with_pos(
        self,
        tagger: Callable[[list[JSONObject]], list[JSONObject]],
    ) -> AnalyzableSentence:
        """Apply only POS values; a tagger cannot replace admitted heads/IDs.

        Runs on the projected lexical rows, as the established POS procedure
        does. The returned JSON is external evidence, never a replacement graph.
        """
        tagged = tagger(self.wire_rows())
        if len(tagged) != len(self._rows):
            raise DependencyProjectionError(
                ProjectionFailure.POS_ALIGNMENT, "row count"
            )
        rows = []
        for original, candidate in zip(self._rows, tagged, strict=True):
            if candidate.get("id") != original["id"] or not isinstance(
                candidate.get("upos"), str
            ):
                raise DependencyProjectionError(
                    ProjectionFailure.POS_ALIGNMENT, "ID or tag"
                )
            rows.append({**deepcopy(original), "upos": candidate["upos"]})
        return self._owned(tuple(rows), self.repairs)
