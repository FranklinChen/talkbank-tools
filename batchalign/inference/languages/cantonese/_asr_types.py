"""Typed ASR payload models shared across built-in HK/Cantonese engines."""

from __future__ import annotations

from typing import Final, Literal, TypedDict

from pydantic import TypeAdapter, ValidationError

from batchalign.inference._domain_types import LanguageCode, TimestampMs
from batchalign.inference.asr import AsrElement as AsrElementModel
from batchalign.inference.asr import AsrMonologue as AsrMonologueModel
from batchalign.inference.asr import MonologueAsrResponse, SpeakerAttribution


class AsrElement(TypedDict):
    """One ASR token entry for process_generation_to_chat payloads."""

    type: Literal["text"]
    ts: float | None
    end_ts: float | None
    value: str


class AttributedSpeakerPayload(TypedDict):
    """The provider named a speaker; this is its own label for them."""

    kind: Literal["attributed"]
    label: str


class UndiarizedSpeakerPayload(TypedDict):
    """The provider separates no speakers and named none."""

    kind: Literal["undiarized"]


SpeakerAttributionPayload = AttributedSpeakerPayload | UndiarizedSpeakerPayload
"""Who a provider attributed one monologue to, in the shared payload shape.

Tagged rather than an ``int``, because an engine that separates nobody used to
have to write ``0`` here, which is indistinguishable downstream from a
provider's real first speaker.
"""


def undiarized_speaker() -> UndiarizedSpeakerPayload:
    """The payload spelling of "this engine separates nobody and named none".

    A named constructor rather than a mapping literal at each producer, so the
    tag is written where the type is defined instead of once per engine that
    has to say it.
    """
    return UndiarizedSpeakerPayload(kind="undiarized")


class AsrMonologue(TypedDict):
    """One speaker segment for process_generation_to_chat payloads."""

    elements: list[AsrElement]
    speaker: SpeakerAttributionPayload


class AsrGenerationPayload(TypedDict):
    """Top-level ASR payload consumed by batchalign ASR post-processing."""

    monologues: list[AsrMonologue]


_SPEAKER_ADMISSION: Final = TypeAdapter(SpeakerAttribution)
"""The one parser from a payload's speaker mapping into the tagged union."""


class ProviderPayloadRefused(ValueError):
    """A provider payload carried a speaker this boundary cannot admit."""


def admit_provider_monologues(
    payload: AsrGenerationPayload, lang: LanguageCode
) -> MonologueAsrResponse:
    """Admit one provider payload into the typed ASR response.

    THE conversion from the payload shape the extension module returns (and
    that ``_qwen_common`` builds in Python) into the models this boundary hands
    back to Rust. Each built-in HK engine used to write this inline and pass
    ``monologue["speaker"]``, a plain mapping, straight into a pydantic field:
    three silent coercions of one value, so a single meaning had two
    representations joined by nothing that had a name. Here the speaker is
    PARSED, once, and a tag this boundary does not know is refused against the
    monologue it came from rather than surfacing as a validation error raised
    from inside a model construction three frames down.

    Blank elements are dropped here for the same reason: the filter was the same
    three lines in each engine, and an engine that forgot it sent whitespace
    tokens downstream.
    """
    monologues: list[AsrMonologueModel] = []
    for index, monologue in enumerate(payload["monologues"]):
        try:
            speaker = _SPEAKER_ADMISSION.validate_python(monologue["speaker"])
        except ValidationError as error:
            raise ProviderPayloadRefused(
                f"monologue {index} does not say who spoke: {error}"
            ) from error
        monologues.append(
            AsrMonologueModel(
                speaker=speaker,
                elements=[
                    AsrElementModel(
                        value=element["value"],
                        ts=element["ts"],
                        end_ts=element["end_ts"],
                        type=element["type"],
                    )
                    for element in monologue["elements"]
                    if element["value"].strip()
                ],
            )
        )
    return MonologueAsrResponse(lang=lang, monologues=monologues)


class TimedWord(TypedDict):
    """Word timing payload consumed by ParsedChat.add_utterance_timing."""

    word: str
    start_ms: TimestampMs
    end_ms: TimestampMs


class AliyunSentenceWord(TypedDict, total=False):
    """Aliyun per-word result item received from websocket payloads.

    Deprecated: prefer AliyunWord (Pydantic) for new code.
    """

    text: str
    startTime: int | float | str
    endTime: int | float | str
