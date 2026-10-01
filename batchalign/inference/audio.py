"""Audio I/O and lazy-loading container for ASR/FA inference.

Provides:
- ``ASRAudioFile``: lazy/eager audio container with LRU chunk cache
- ``load_audio``, ``save_audio``, ``audio_info``: soundfile-backed I/O
- Whisper token-timestamp helpers used to bind an instance-local extractor
  instead of globally patching the model class
"""

from __future__ import annotations

import inspect
import json
import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Protocol

import numpy as np
import soundfile as sf
import torch

from batchalign.inference._domain_types import SampleRate

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateOutput

L = logging.getLogger("batchalign")

LAZY_AUDIO_ENABLED = True


def set_lazy_audio_enabled(enabled: bool) -> None:
    """Toggle the global lazy-audio feature flag."""
    global LAZY_AUDIO_ENABLED
    LAZY_AUDIO_ENABLED = bool(enabled)


class AudioLoaderFn(Protocol):
    """Callable signature for chunk/file audio loading.

    The lazy audio container keeps this dependency injectable so tests can use
    explicit fake loaders instead of monkeypatching the module-level
    `load_audio()` function.
    """

    def __call__(
        self,
        filepath: str | Path,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, int]: ...


# ---------------------------------------------------------------------------
# Audio I/O (soundfile-backed, replaces torchaudio)
# ---------------------------------------------------------------------------


@dataclass
class AudioInfo:
    """Audio file metadata, compatible with torchaudio.info() output."""

    sample_rate: SampleRate
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def load_audio(
    filepath: str | Path,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """Load audio from a file (drop-in for torchaudio.load)."""
    filepath = str(filepath)
    file_info = sf.info(filepath)
    total_frames = file_info.frames
    sample_rate = file_info.samplerate

    start = max(frame_offset, 0)
    if start >= total_frames:
        return torch.zeros(file_info.channels, 0, dtype=dtype), sample_rate

    frames_to_read = num_frames if num_frames >= 0 else -1

    audio_np, sr = sf.read(
        filepath,
        start=start,
        frames=frames_to_read if frames_to_read > 0 else -1,
        dtype="float32" if normalize else "int16",
        always_2d=True,
    )

    audio = torch.from_numpy(audio_np.T)
    if dtype != torch.float32:
        audio = audio.to(dtype)
    return audio, sr


def save_audio(
    filepath: str | Path,
    audio: torch.Tensor,
    sample_rate: SampleRate,
    bits_per_sample: int = 16,
) -> None:
    """Save audio to a file (drop-in for torchaudio.save)."""
    filepath = str(filepath)
    audio_np = audio.numpy() if audio.dim() == 1 else audio.T.numpy()

    subtype_map = {8: "PCM_S8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
    subtype = subtype_map.get(bits_per_sample, "PCM_16")

    format_map = {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG"}
    file_format = format_map.get(Path(filepath).suffix.lower())

    sf.write(filepath, audio_np, sample_rate, subtype=subtype, format=file_format)


def audio_info(filepath: str | Path) -> AudioInfo:
    """Get audio file metadata (drop-in for torchaudio.info)."""
    filepath = str(filepath)
    file_info = sf.info(filepath)
    subtype = file_info.subtype
    if "PCM_16" in subtype or "16" in subtype:
        bits = 16
    elif "PCM_24" in subtype or "24" in subtype:
        bits = 24
    elif "PCM_32" in subtype or "32" in subtype or "FLOAT" in subtype:
        bits = 32
    elif "PCM_S8" in subtype or "PCM_U8" in subtype or "8" in subtype:
        bits = 8
    else:
        bits = 16
    return AudioInfo(
        sample_rate=file_info.samplerate,
        num_frames=file_info.frames,
        num_channels=file_info.channels,
        bits_per_sample=bits,
        encoding=subtype,
    )


# ---------------------------------------------------------------------------
# Lazy-loading audio container
# ---------------------------------------------------------------------------


@dataclass
class ASRAudioFile:
    """Container for a mono audio waveform consumed by ASR/FA engines.

    Supports eager (pre-loaded tensor) and lazy (on-demand with LRU cache) modes.
    """

    file: str
    tensor: torch.Tensor
    rate: SampleRate
    _lazy: bool = False
    _cache: dict[tuple[int, int], torch.Tensor] | None = None
    _cache_order: list[tuple[int, int]] | None = None
    _cache_limit: int = 16
    _file_id: str | None = None
    _load_audio_fn: AudioLoaderFn = field(default=load_audio, repr=False, compare=False)

    def _init_cache(self) -> None:
        if self._cache is None:
            self._cache = {}
            self._cache_order = []

    @classmethod
    def lazy(
        cls,
        file_path: str,
        rate: int,
        cache_limit: int = 16,
        load_audio_fn: AudioLoaderFn = load_audio,
    ) -> ASRAudioFile:
        """Create a lazy-loading audio file handle.

        The loader dependency remains injectable so focused unit tests can
        provide a fake chunk reader without patching module globals.
        """
        if not LAZY_AUDIO_ENABLED:
            raise RuntimeError("Lazy audio disabled")
        return cls(
            file_path,
            torch.empty(0),
            rate,
            _lazy=True,
            _cache_limit=cache_limit,
            _load_audio_fn=load_audio_fn,
        )

    def _read_frames(self, frame_offset: int, num_frames: int) -> torch.Tensor:
        """Read raw frames, resample if needed, mix to mono."""
        if num_frames < 0:
            audio_arr, rate = self._load_audio_fn(self.file)
        else:
            audio_arr, rate = self._load_audio_fn(
                self.file, frame_offset=frame_offset, num_frames=num_frames
            )
        if rate != self.rate:
            from torchaudio import transforms as T

            audio_arr = T.Resample(rate, self.rate)(audio_arr)
        return torch.mean(audio_arr.transpose(0, 1), dim=1)

    def chunk(self, begin_ms: int | float, end_ms: int | float) -> torch.Tensor:
        """Get a chunk of the audio in the given millisecond range."""
        begin_frame = int(round((begin_ms / 1000) * self.rate))
        end_frame = int(round((end_ms / 1000) * self.rate))
        if end_frame <= begin_frame:
            return torch.zeros(0)

        if not self._lazy:
            return self.tensor[begin_frame:end_frame]

        self._init_cache()
        assert self._cache is not None
        assert self._cache_order is not None
        key = (begin_frame, end_frame)
        if key in self._cache:
            return self._cache[key]
        try:
            data = self._read_frames(begin_frame, end_frame - begin_frame)
        except Exception:
            L.debug("Chunk read failed, falling back to full load", exc_info=True)
            if self.tensor is None or self.tensor.numel() == 0:
                self.tensor = self._read_frames(0, -1)
                self._lazy = False
            data = self.tensor[begin_frame:end_frame]
        self._cache[key] = data
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_limit:
            old_key = self._cache_order.pop(0)
            self._cache.pop(old_key, None)
        return data

    def file_identity(self) -> str:
        """Stable identity hash: SHA256(resolved_path | file_size)."""
        if self._file_id is None:
            import hashlib
            import os

            real = os.path.realpath(self.file)
            size = os.path.getsize(real)
            self._file_id = hashlib.sha256(f"{real}|{size}".encode()).hexdigest()
        return self._file_id

    def hash_chunk(self, begin_ms: int | float, end_ms: int | float) -> str:
        """SHA256 fingerprint of a chunk (midpoint sampling)."""
        import hashlib

        data = self.chunk(begin_ms, end_ms)
        num_samples = data.numel()
        if num_samples > 100:
            mid = num_samples // 2
            samples = data[mid - 50 : mid + 50]
        else:
            samples = data
        header = f"{num_samples}|".encode()
        return hashlib.sha256(header + samples.cpu().numpy().tobytes()).hexdigest()

    def hash_all(self) -> str:
        """SHA256 fingerprint of the entire waveform."""
        import hashlib

        data = self.all()
        num_samples = data.numel()
        if num_samples > 100:
            mid = num_samples // 2
            samples = data[mid - 50 : mid + 50]
        else:
            samples = data
        header = f"{num_samples}|".encode()
        return hashlib.sha256(header + samples.cpu().numpy().tobytes()).hexdigest()

    def all(self) -> torch.Tensor:
        """Return the complete mono waveform."""
        if not self._lazy:
            return self.tensor
        self._init_cache()
        if self.tensor is None or self.tensor.numel() == 0:
            try:
                self.tensor = self._read_frames(0, -1)
                self._lazy = False
            except Exception:
                L.warning("Failed to load audio tensor", exc_info=True)
                return torch.zeros(0)
        return self.tensor


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------


def load_audio_file(
    path: str,
    target_sample_rate: SampleRate = 16000,
) -> ASRAudioFile:
    """Load an audio file with lazy/eager fallback.

    Tries lazy loading first; falls back to eager if sample rate
    differs or soundfile can't handle the format.
    """
    try:
        info = audio_info(path)
        sample_rate = info.sample_rate
        lazy = ASRAudioFile.lazy(path, sample_rate)
    except Exception:
        L.debug(
            "Lazy audio load failed for %s, falling back to eager load",
            path,
            exc_info=True,
        )
        audio_arr, rate = load_audio(path)
        if audio_arr.dim() > 1:
            audio_arr = torch.mean(audio_arr.transpose(0, 1), dim=1)
        if rate != target_sample_rate:
            from torchaudio import transforms as T

            audio_arr = T.Resample(rate, target_sample_rate)(audio_arr)
        return ASRAudioFile(path, audio_arr, target_sample_rate)

    if sample_rate != target_sample_rate:
        audio_arr, rate = load_audio(path)
        if audio_arr.dim() > 1:
            audio_arr = torch.mean(audio_arr.transpose(0, 1), dim=1)
        from torchaudio import transforms as T

        audio_arr = T.Resample(rate, target_sample_rate)(audio_arr)
        return ASRAudioFile(path, audio_arr, target_sample_rate)

    return lazy


# ---------------------------------------------------------------------------
# Whisper token-timestamp monkey-patch
# ---------------------------------------------------------------------------


def bind_whisper_token_timestamp_extractor(model: torch.nn.Module) -> None:
    """Install the variable-batch timestamp extractor on one Whisper instance.

    The old implementation mutated ``WhisperForConditionalGeneration`` at the
    class level. That made the override process-global and harder to reason
    about in tests. The current boundary keeps the same workaround but narrows
    it to the concrete model instance that batchalign loaded.
    """
    model._extract_token_timestamps = MethodType(_extract_token_timestamps, model)  # type: ignore[attr-defined]


def _extract_token_timestamps(
    self: torch.nn.Module,  # Whisper model instance with an injected bound method
    generate_outputs: GenerateOutput,
    alignment_heads: list[list[int]],
    time_precision: float = 0.02,
    num_frames: int | list[int] | np.ndarray | torch.Tensor | None = None,
    num_input_ids: int | None = None,
) -> torch.Tensor:
    """Instance-local Whisper token-timestamp extractor for variable-length batches.

    Batchalign binds this helper onto the loaded Whisper model instance with
    :func:`bind_whisper_token_timestamp_extractor` so the workaround stays
    local to that instance instead of mutating the third-party class globally.

    Type errors in this body are suppressed per-line because:
    - ``generate_outputs`` is a 4-variant union; we always receive the
      encoder-decoder variant but mypy cannot narrow it.
    - ``self`` is ``nn.Module`` whose ``__getattr__`` returns ``Tensor | Module``,
      hiding the real ``WhisperConfig`` attributes.
    """
    from transformers.models.whisper.generation_whisper import (
        _dynamic_time_warping,
        _median_filter,
    )

    config = self.config

    cross_attentions = []
    for i in range(config.decoder_layers):
        cross_attentions.append(
            torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2)
        )

    weights = torch.stack(
        [cross_attentions[layer][:, head] for layer, head in alignment_heads]
    )
    weights = weights.permute([1, 0, 2, 3])

    weight_length = None

    if "beam_indices" in generate_outputs:
        weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
        weight_length = (
            weight_length if num_input_ids is None else weight_length + num_input_ids
        )
        beam_indices = torch.zeros_like(
            generate_outputs.beam_indices[:, :weight_length],
            dtype=generate_outputs.beam_indices.dtype,
        )
        beam_indices[:, num_input_ids:] = generate_outputs.beam_indices[
            :, : weight_length - num_input_ids
        ]
        weights = weights[:, :, :weight_length]
        beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)
        weights = torch.stack(
            [
                torch.index_select(weights[:, :, i, :], dim=0, index=beam_indices[:, i])
                for i in range(beam_indices.shape[1])
            ],
            dim=2,
        )

    input_length = weight_length or cross_attentions[0].shape[2]
    batch_size = generate_outputs.sequences.shape[0]
    timestamps = torch.zeros(
        (batch_size, input_length + 1),
        dtype=torch.float32,
        device=generate_outputs.sequences.device,
    )

    if num_frames is not None:
        if isinstance(num_frames, int):
            weights = weights[..., : num_frames // 2]
        elif (
            isinstance(num_frames, (list, tuple, np.ndarray))
            and len(np.unique(num_frames)) == 1
        ):
            weights = weights[..., : num_frames[0] // 2]
        elif (
            isinstance(num_frames, torch.Tensor) and len(torch.unique(num_frames)) == 1
        ):
            weights = weights[..., : num_frames[0] // 2]
        else:
            repeat_time = (
                batch_size
                if isinstance(num_frames, int)
                else batch_size // len(num_frames)
            )
            num_frames = (
                num_frames.cpu() if isinstance(num_frames, torch.Tensor) else num_frames
            )
            num_frames = np.repeat(num_frames, repeat_time)

    if num_frames is None or isinstance(num_frames, int):
        std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
        mean = torch.mean(weights, dim=-2, keepdim=True)
        weights = (weights - mean) / std
        weights = _median_filter(weights, config.median_filter_width)
        weights = weights.mean(dim=1)

    for batch_idx in range(batch_size):
        if num_frames is not None and isinstance(
            num_frames, (tuple, list, np.ndarray, torch.Tensor)
        ):
            matrix = weights[batch_idx, ..., : num_frames[batch_idx] // 2]
            std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
            mean = torch.mean(matrix, dim=-2, keepdim=True)
            matrix = (matrix - mean) / std
            matrix = _median_filter(matrix, config.median_filter_width)
            matrix = matrix.mean(dim=0)
        else:
            matrix = weights[batch_idx]

        text_indices, time_indices = _dynamic_time_warping(
            -matrix.cpu().double().numpy()
        )
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * time_precision
        timestamps[batch_idx, 1:] = torch.tensor(jump_times)

    return timestamps


# ---------------------------------------------------------------------------
# Routing torchaudio's decode entry point away from torchcodec
# ---------------------------------------------------------------------------


#: The start of pyannote's import-time notice when torchcodec is absent
#: (`pyannote/audio/core/io.py`). Matched as a regex against the message start.
_PYANNOTE_TORCHCODEC_NOTICE = r"\s*torchcodec is not installed correctly"


@contextmanager
def pyannote_import_without_torchcodec_notice() -> Iterator[None]:
    """Import pyannote.audio without its warning that decoding "will fail".

    pyannote warns at import time when torchcodec is missing. torchcodec is
    excluded from Batchalign on purpose (the reasons are in `pyproject.toml`),
    and Batchalign always hands pyannote in-memory `{"waveform",
    "sample_rate"}` audio, the path that never decodes. So the notice is false
    for us, and it lands in every worker log and test run that loads speaker
    models. Only that one message is silenced, and only around the import;
    every other warning still reaches the caller.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_PYANNOTE_TORCHCODEC_NOTICE, category=UserWarning
        )
        yield


class SoundfileBackendInstall(StrEnum):
    """What a call to `install_soundfile_backend` actually DID.

    Returned rather than `None` so a caller can report it and a test can assert
    it, instead of trusting that a side effect occurred.
    """

    #: This call rebound `torchaudio.load`.
    INSTALLED = "installed"
    #: A previous call had already done it; this one changed nothing. Repeated
    #: worker bootstraps therefore report honestly instead of implying work.
    ALREADY_INSTALLED = "already-installed"


def install_soundfile_backend() -> SoundfileBackendInstall:
    """Make `torchaudio.load` use soundfile, so torchcodec is never loaded.

    # Why this exists

    `torchaudio.load` delegates to `torchcodec`, which ships shared libraries
    for a FIXED set of FFmpeg majors, each linked `@rpath/libavutil.NN.dylib`
    against the UNVERSIONED Homebrew prefix. When Homebrew moves that prefix to
    a major torchcodec has no library for, EVERY call raises, and it raises at
    job time rather than at startup. That took out transcription on every
    machine in one Homebrew upgrade.

    There is no configuration escape. As of torchaudio 2.11 the `backend`
    argument is ignored: `backend="soundfile"` and `backend="ffmpeg"` both go
    through torchcodec and both fail identically. Verified, not assumed.

    So the only durable fix is not to call it. Batchalign's own audio I/O
    already moved to soundfile; what remained was that DEPENDENCIES still reach
    `torchaudio.load`, and there is nothing we can do at our own call sites
    about that. Binding our drop-in over the entry point covers every caller,
    ours and theirs.

    # It is NOT load-bearing, and that is deliberate

    Measured, not assumed: the only production caller of `torchaudio.load` in
    the installed dependency set is FunASR, and it wraps the call in its own
    `except`, falling back first to `soundfile.read` and then to an ffmpeg
    subprocess. The Cantonese path therefore works whether or not this shim is
    installed. The other callers are torchaudio's own dataset helpers and
    `torch_audiomentations`, which is pyannote TRAINING code we never reach.

    So this is defence for a caller that does not exist yet, which is why its
    installation is allowed to FAIL OPEN at worker startup: a worker that could
    not install it is not thereby broken. If it were load-bearing, failing open
    would be indefensible and the worker would have to refuse to start.

    The honest end state is deletion, and the condition for it has been MET:
    a Cantonese FunASR run, with this shim absent, produced real timed words
    (2026-08-13). Deletion is a change to worker startup and was left as a
    deliberate decision rather than taken with the evidence.

    The standing half of that evidence is
    `tests/models/test_audio_decode_independence.py
    ::test_funasr_still_loads_audio_when_torchaudio_load_raises`, which makes
    `torchaudio.load` raise and asserts FunASR still returns audio. That is the
    assumption a FunASR upgrade could break silently, so it fails loudly here
    instead of being a fact somebody once observed.
    """
    import torchaudio

    existing = getattr(torchaudio, "load", None)
    if getattr(existing, "_batchalign_soundfile_backend", False):
        return SoundfileBackendInstall.ALREADY_INSTALLED

    # Derived from `load_audio`'s signature, so a new parameter there cannot
    # be silently dropped here. Computed once per install, not per call.
    load_audio_keywords = frozenset(inspect.signature(load_audio).parameters) - {
        "filepath"
    }

    def _load(filepath, *args, **kwargs):  # type: ignore[no-untyped-def]
        # `torchaudio.load` accepts keywords ours does not model
        # (`channels_first`, `format`, `buffer_size`, `backend`). Dropping
        # them is safe for the shapes we serve and keeps the seam narrow.
        accepted = {
            key: value for key, value in kwargs.items() if key in load_audio_keywords
        }
        # No fallback: `original` routes to torchcodec, which is absent on
        # purpose, so falling back would report that instead of the real
        # soundfile error.
        return load_audio(filepath, *args, **accepted)

    _load._batchalign_soundfile_backend = True  # type: ignore[attr-defined]
    torchaudio.load = _load  # type: ignore[assignment]
    return SoundfileBackendInstall.INSTALLED


@dataclass(frozen=True)
class DecodeCapability:
    """Proof that this interpreter can decode audio, with what it measured.

    THE one definition of "can this machine decode", so that the callers who
    need the answer stop each carrying their own copy of the question.

    Before this there were three: an embedded Python string in the Rust
    `batchalign3 doctor`, a second embedded string in the fleet deploy's host
    runtime, and a third inline in the test suite. They had already diverged at
    birth: one printed three fields and handled a missing module as a distinct
    outcome, the other printed four and had no exception handling at all, and
    each end parsed a positional string with its own hand-written splitter.

    `frames` and `sample_rate` are MEASURED. There is deliberately no field
    naming the backend: it would be a constant supplied by this function's own
    source, travelling beside real measurements and reading like one.
    """

    frames: int
    sample_rate: SampleRate

    def to_json(self) -> str:
        """One line, for a caller in another language or another process.

        `asdict` rather than a hand-written mapping: spelling the field names a
        second time is how a new field gets silently dropped from the wire,
        which is the same defect this module fixed eighty lines above by
        deriving the keyword filter from `load_audio`'s signature.
        """
        return json.dumps(asdict(self))


def verify_decode_capability() -> DecodeCapability:
    """Decode a generated waveform through the configured backend, or raise.

    Generated rather than read from a fixture so the check has no data
    dependency and cannot pass by finding a stale file.

    It exercises `load_audio`, which is what Batchalign actually decodes with.
    Earlier versions called `torchaudio.load`, so they certified a
    MONKEYPATCHED THIRD-PARTY FUNCTION that none of this codebase's own paths
    call, and would have reported a healthy machine unhealthy if the patch were
    ever removed. The question worth answering is "can Batchalign decode", not
    "is somebody else's entry point currently rebound".
    """
    import tempfile
    import wave

    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "probe.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 16000)

        tensor, rate = load_audio(path)
    return DecodeCapability(frames=int(tensor.shape[-1]), sample_rate=int(rate))


if __name__ == "__main__":  # pragma: no cover - the out-of-process entry point
    # `python -m batchalign.inference.audio` prints one JSON line, which is the
    # contract both out-of-process callers (the Rust doctor and the fleet
    # deploy) consume instead of embedding their own probe.
    print(verify_decode_capability().to_json())
