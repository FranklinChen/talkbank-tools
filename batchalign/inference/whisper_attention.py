"""Request-bound SQLite capture of Whisper FA numerics, never timing policy."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from batchalign.worker._types_v2 import WhisperAttentionCaptureRequestV2

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _CaptureContext(BaseModel):
    request_id: str
    capture: WhisperAttentionCaptureRequestV2


class WhisperAttentionCapture:
    """One request owns one admitted observation and one exclusive-create file."""

    def __init__(
        self, context_json: str, audio: NDArray[np.float32], text: str, sample_rate: int
    ) -> None:
        self._context = _CaptureContext.model_validate_json(context_json)
        self._path = Path(self._context.capture.destination)
        if not self._path.is_absolute() or audio.ndim != 1 or sample_rate <= 0:
            raise ValueError("invalid attention capture destination or audio geometry")
        self._audio_sha256 = hashlib.sha256(
            np.asarray(audio, dtype="<f4").tobytes()
        ).hexdigest()
        self._audio_samples = len(audio)
        self._sample_rate = sample_rate
        self._text = text
        self._recorded = False

    def record(
        self,
        *,
        token_ids: list[int],
        token_texts: list[str],
        prefix_length: int,
        heads: list[tuple[int, int]],
        raw_attention: NDArray[np.float32],
        costs: NDArray[np.float32],
        row_entries: NDArray[np.int64],
        hop_length: int,
        median_width: int,
        model_id: str,
        model_revision: str,
    ) -> None:
        """Validate and persist the tensors actually used by this inference."""
        rows = len(token_ids) - prefix_length
        if (
            self._recorded
            or not heads
            or not model_id
            or not model_revision
            or not 0 <= prefix_length < len(token_ids)
            or len(token_ids) != len(token_texts)
            or hop_length <= 0
            or median_width <= 0
            or median_width % 2 != 1
            or raw_attention.ndim != 3
            or costs.ndim != 2
            or raw_attention.shape[:2] != (len(heads), rows)
            or costs.shape != raw_attention.shape[1:]
            or costs.shape[1] != self._audio_samples // hop_length // 2
            or costs.shape[1] == 0
            or row_entries.shape != (rows,)
            or np.any(row_entries < 0)
            or np.any(row_entries >= costs.shape[1])
            or np.any(np.diff(row_entries) < 0)
            or not np.isfinite(raw_attention).all()
            or not np.isfinite(costs).all()
        ):
            raise ValueError(
                "attention capture does not match its admitted labels/audio"
            )
        from batchalign.worker._protocol import write_progress_event

        write_progress_event(
            self._context.request_id, 0, len(heads) + 1, "attention_capture"
        )
        # Never replace a previous capture or follow an existing symlink. A
        # failed write remains inspectable but cannot produce a success receipt.
        fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        connection = sqlite3.connect(self._path)
        try:
            connection.executescript("""
                CREATE TABLE capture(
                  schema_version INTEGER NOT NULL, request_id TEXT NOT NULL,
                  semantic_key TEXT NOT NULL, audio_sha256 TEXT NOT NULL,
                  audio_samples INTEGER NOT NULL, sample_rate INTEGER NOT NULL,
                  text TEXT NOT NULL, model_id TEXT NOT NULL, model_revision TEXT NOT NULL,
                  prefix_length INTEGER NOT NULL, hop_length INTEGER NOT NULL,
                  median_width INTEGER NOT NULL, frame_count INTEGER NOT NULL,
                  alignment_revision TEXT NOT NULL
                ) STRICT;
                CREATE TABLE labels(ordinal INTEGER PRIMARY KEY, token_id INTEGER NOT NULL,
                  text TEXT NOT NULL, role TEXT NOT NULL, row_entry_frame INTEGER) STRICT;
                CREATE TABLE heads(ordinal INTEGER PRIMARY KEY, layer INTEGER NOT NULL,
                  head INTEGER NOT NULL) STRICT;
                CREATE TABLE raw_attention(head_ordinal INTEGER NOT NULL,
                  label_row INTEGER NOT NULL, frame INTEGER NOT NULL, value REAL NOT NULL,
                  PRIMARY KEY(head_ordinal,label_row,frame)) WITHOUT ROWID, STRICT;
                CREATE TABLE costs(label_row INTEGER NOT NULL, frame INTEGER NOT NULL,
                  value REAL NOT NULL, PRIMARY KEY(label_row,frame)) WITHOUT ROWID, STRICT;
            """)
            with connection:
                connection.execute(
                    "INSERT INTO capture VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        1,
                        self._context.request_id,
                        self._context.capture.semantic_key,
                        self._audio_sha256,
                        self._audio_samples,
                        self._sample_rate,
                        self._text,
                        model_id,
                        model_revision,
                        prefix_length,
                        hop_length,
                        median_width,
                        costs.shape[1],
                        "open-begin-fixed-end-v1",
                    ),
                )
                connection.executemany(
                    "INSERT INTO labels VALUES(?,?,?,?,?)",
                    (
                        (
                            i,
                            token,
                            token_texts[i],
                            "conditioning"
                            if i < prefix_length
                            else "terminal"
                            if i == len(token_ids) - 1
                            else "lexical",
                            None
                            if i < prefix_length
                            else int(row_entries[i - prefix_length]),
                        )
                        for i, token in enumerate(token_ids)
                    ),
                )
                connection.executemany(
                    "INSERT INTO heads VALUES(?,?,?)",
                    ((i, layer, head) for i, (layer, head) in enumerate(heads)),
                )
                for h in range(len(heads)):
                    connection.executemany(
                        "INSERT INTO raw_attention VALUES(?,?,?,?)",
                        (
                            (h, r, f, float(raw_attention[h, r, f]))
                            for r in range(rows)
                            for f in range(costs.shape[1])
                        ),
                    )
                    write_progress_event(
                        self._context.request_id,
                        h + 1,
                        len(heads) + 1,
                        "attention_capture",
                    )
                connection.executemany(
                    "INSERT INTO costs VALUES(?,?,?)",
                    (
                        (r, f, float(costs[r, f]))
                        for r in range(rows)
                        for f in range(costs.shape[1])
                    ),
                )
        finally:
            connection.close()
        self._recorded = True
        write_progress_event(
            self._context.request_id,
            len(heads) + 1,
            len(heads) + 1,
            "attention_capture",
        )

    def require_recorded(self) -> None:
        """A callback cannot claim success without having recorded its tensor."""
        if not self._recorded:
            raise ValueError("Whisper attention capture was requested but not recorded")
