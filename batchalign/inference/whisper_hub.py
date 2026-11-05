"""HuggingFace Whisper fine-tune ASR backend (engine: ``whisper_hub``).

Loads a Malayalam (or any other language) Whisper fine-tune from the
snapshot the worker resolved for it. Which fine-tune a language gets is
pinned by the control plane; an unpinned worker names it from a
per-language default table (``resolve_whisper_hub_model_id``) or an explicit
user override via
``--engine-overrides '{"asr":"whisper_hub","model_id":"owner/model"}'``,
and the worker resolves that id before loading.

Why a separate engine rather than overloading ``whisper``:

- Stock OpenAI Whisper checkpoints and community fine-tunes have
  different generation_config semantics. Fine-tunes pin
  ``language`` / ``task`` inside their generation_config; passing
  ``language=...`` / ``task="transcribe"`` on a fine-tune produces
  gibberish. Stock Whisper requires the opposite, it *needs* the
  language hint. Merging the two loaders into one engine would make
  every line of dispatch code branch on the checkpoint's shape.
- Opt-out is surgical: ``--asr-engine whisper`` and
  ``--asr-engine whisper_hub`` are two different knobs.
- Per-engine language-support / quality metadata is a per-variant
  concern and composes cleanly when the variants are separate.

Mechanics:

- Model_id resolution is pure: ``resolve_whisper_hub_model_id`` takes
  the language + engine_overrides and returns either a concrete HF
  model_id or raises ``WhisperHubModelNotFoundError``.
- Model loading takes the resolved ``TransformersSnapshot`` only and
  delegates through ``batchalign.inference.asr.load_whisper_asr``
  with ``CheckpointPinnedLanguage`` decoding, so ``gen_kwargs`` never forces
  a task or language (which breaks fine-tunes). The returned
  handle is the same ``WhisperASRHandle`` used by stock Whisper, so
  the V2 inference boundary
  (``batchalign.inference.asr.infer_whisper_prepared_audio``) works
  without modification.
- Inference reuses the existing Whisper prepared-audio path. This
  module contributes no new inference function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batchalign.inference._domain_types import LanguageCode
from batchalign.models.resolve import resolve

if TYPE_CHECKING:
    from batchalign.device import DevicePolicy
    from batchalign.inference.types import WhisperASRHandle
    from batchalign.worker._model_loading.pinned_hub import TransformersSnapshot


# Key under which an explicit user override for the fine-tune model_id
# travels inside the ``engine_overrides`` dictionary. Owning this as a
# named constant rather than a bare string literal lets future callers
# grep the codebase to find every reader/writer.
MODEL_ID_OVERRIDE_KEY = "model_id"


class WhisperHubModelNotFoundError(RuntimeError):
    """Raised when no default model_id is seeded for the requested language
    and the caller did not pass an explicit ``model_id`` override.

    Surfaces the gap loudly so the user knows to either
    (1) pick a model and pass it via ``--engine-overrides``, or
    (2) request that the team seed a default for this language after
    empirical evaluation. See
    ``book/src/reference/whisper-hub-asr.md`` for the escalation path.
    """


def resolve_whisper_hub_model_id(
    lang: LanguageCode,
    engine_overrides: dict[str, str] | None,
) -> str:
    """Return the concrete HF model_id to load for this (lang, overrides) pair.

    Precedence:

    1. Explicit ``engine_overrides[MODEL_ID_OVERRIDE_KEY]``: user-chosen
       model_id wins unconditionally.
    2. Per-language default from ``resolve("whisper_hub", lang)``.

    Raises ``WhisperHubModelNotFoundError`` when both are absent. No
    silent fallback to a stock OpenAI Whisper checkpoint; that would
    be exactly the foot-gun this engine variant exists to prevent.
    """
    overrides = engine_overrides or {}
    explicit = overrides.get(MODEL_ID_OVERRIDE_KEY)
    if explicit:
        return explicit
    resolved = resolve("whisper_hub", lang)
    if resolved is not None:
        return resolved
    raise WhisperHubModelNotFoundError(
        f"whisper_hub has no default model_id for language '{lang}'. "
        f"Either add a seed entry in batchalign/models/resolve.py (after "
        f"empirical evaluation: see book/src/reference/whisper-hub-asr.md) "
        f"or pass an explicit model_id via "
        f'--engine-overrides \'{{"asr":"whisper_hub","model_id":"<owner>/<model>"}}\'.'
    )


def load_whisper_hub_asr(
    snapshot: TransformersSnapshot,
    *,
    device_policy: DevicePolicy | None = None,
) -> WhisperASRHandle:
    """Load an HF Whisper fine-tune and return the shared ``WhisperASRHandle``.

    ``snapshot`` is the fine-tune the worker already resolved at the revision
    the control plane pinned (Rust owns which fine-tune a language gets,
    `model_manifest`; an unpinned worker resolves the table below first).
    It used to take an optional path and fall back to resolving the table
    here, which downloaded inside the load when the path was missing.

    The returned handle is the same type used by stock Whisper. Downstream
    V2 inference (``infer_whisper_prepared_audio``) works without
    modification because the handle carries the pipeline and metadata
    uniformly.

    The handle decodes with ``CheckpointPinnedLanguage``: fine-tunes pin
    language and task inside their own ``generation_config``, and re-forcing
    them produces gibberish, so ``gen_kwargs`` never names either.
    """
    # Import lazily so the resolver path and error types are usable
    # without pulling in the heavy ML stack (transformers / torch) just
    # for a configuration lookup.
    from batchalign.inference._domain_types import WhisperTimestampMode
    from batchalign.inference.asr import load_whisper_asr
    from batchalign.inference.types import CheckpointPinnedLanguage

    return load_whisper_asr(
        snapshot,
        decoding=CheckpointPinnedLanguage(),
        device_policy=device_policy,
        timestamp_mode=WhisperTimestampMode.SEGMENT,
    )


__all__ = [
    "MODEL_ID_OVERRIDE_KEY",
    "WhisperHubModelNotFoundError",
    "load_whisper_hub_asr",
    "resolve_whisper_hub_model_id",
]
