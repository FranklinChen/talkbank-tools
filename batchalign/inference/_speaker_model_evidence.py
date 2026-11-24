"""Path-free evidence from a local speaker model load."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class SpeakerLoadedModelGraph(BaseModel):
    """Digests retained from verification of the actual loader inputs."""

    model_config = ConfigDict(frozen=True)

    pipeline_config_sha256: Sha256Hex
    segmentation_sha256: Sha256Hex
    embedding_sha256: Sha256Hex
