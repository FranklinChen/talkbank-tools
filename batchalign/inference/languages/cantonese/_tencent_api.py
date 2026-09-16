"""Tencent Cloud ASR client helpers for built-in HK/Cantonese engines."""

from __future__ import annotations

import configparser
import json
import logging
import pathlib
import time
import uuid
from typing import Any

from batchalign.inference._domain_types import AudioPath, LanguageCode
from batchalign.worker._types_v2 import (
    IntegratedDiarizationV2,
    NotRequestedDiarizationV2,
    ProviderDiarizationV2,
)

from ._asr_types import AsrGenerationPayload, TimedWord
from ._common import read_asr_config

_MAX_POLL_SECONDS = 600  # 10-minute safety timeout for ASR task polling

L = logging.getLogger("batchalign.hk.tencent")


class TencentRecognizer:
    """Thin wrapper around Tencent ASR transport plus Rust-owned projection."""

    def __init__(
        self,
        lang: LanguageCode,
        poll_interval_s: float = 10.0,
        *,
        engine_model_type: str,
        config: configparser.ConfigParser | None = None,
    ) -> None:
        """Load credentials and initialize Tencent ASR/COS clients."""
        cfg = read_asr_config(
            (
                "engine.tencent.id",
                "engine.tencent.key",
                "engine.tencent.region",
                "engine.tencent.bucket",
            ),
            engine="Tencent",
            config=config,
        )

        try:
            from qcloud_cos import CosConfig, CosS3Client
            from tencentcloud.asr.v20190614.asr_client import AsrClient
            from tencentcloud.common.credential import Credential
        except Exception as exc:
            raise ImportError(
                "Tencent engine dependencies are missing from this "
                "environment. Reinstall batchalign3 or install the Tencent "
                "SDK packages."
            ) from exc

        secret_id = cfg["engine.tencent.id"]
        secret_key = cfg["engine.tencent.key"]
        region = cfg["engine.tencent.region"]

        self.lang_code = lang
        # Chosen by the Rust control plane, which owns the one ISO 639-3 to
        # 639-1 conversion; this class no longer derives it from the language.
        self.engine_model_type = engine_model_type
        self._poll_interval_s = max(1.0, poll_interval_s)
        self._bucket_name = cfg["engine.tencent.bucket"]
        self._region = region

        self._asr_client = AsrClient(Credential(secret_id, secret_key), region)
        self._bucket = CosS3Client(
            CosConfig(
                Region=region,
                SecretId=secret_id,
                SecretKey=secret_key,
                Token=None,
                Scheme="https",
            )
        )

    def transcribe(
        self, source_path: AudioPath, diarization: ProviderDiarizationV2
    ) -> list[Any]:
        """Upload media, submit ASR task, poll for completion, return ResultDetail.

        ``diarization`` carries no default: whether to separate speakers is the
        caller's question, and the ``num_speakers = 0`` this parameter used to
        default to was a spelling of "do not separate" that only this file and
        the Rust bridge knew.
        """
        try:
            from tencentcloud.asr.v20190614 import models
        except Exception as exc:
            raise ImportError(
                "Tencent engine dependencies are missing from this "
                "environment. Reinstall batchalign3 or install the Tencent "
                "SDK packages."
            ) from exc

        upload_key = f"{uuid.uuid4()}{pathlib.Path(source_path).suffix}"

        L.info("Tencent uploading '%s'", pathlib.Path(source_path).name)
        self._bucket.upload_file(
            Bucket=self._bucket_name,
            LocalFilePath=source_path,
            Key=upload_key,
            PartSize=1,
            MAXThread=10,
            EnableMD5=False,
        )

        media_url = (
            f"https://{self._bucket_name}.cos.{self._region}.myqcloud.com/{upload_key}"
        )

        create_req = models.CreateRecTaskRequest()
        create_req.EngineModelType = self.engine_model_type
        create_req.ResTextFormat = 1
        # Separation is requested only when the control plane asked for it, and
        # the request says which in a type rather than in a number: `Integrated`
        # carries a count that is at least two by construction, `NotRequested`
        # carries none at all. Before the typed request this asked Tencent to
        # diarize every recording, including ones the job declared to have a
        # single speaker, and then attributed whatever came back.
        match diarization:
            case IntegratedDiarizationV2():
                create_req.SpeakerDiarization = 1
                create_req.SpeakerNumber = diarization.speakers
            case NotRequestedDiarizationV2():
                create_req.SpeakerDiarization = 0
            case _:
                # Unreachable through the union above, and it raises rather than
                # picking a reasonable-looking default, because every default
                # here is a claim about a recording that nobody made.
                raise TypeError(f"unknown provider diarization state: {diarization!r}")
        create_req.ChannelNum = 1
        create_req.Url = media_url
        create_req.SourceType = 0

        try:
            create_resp = self._asr_client.CreateRecTask(create_req)
            task_id = int(create_resp.Data.TaskId)

            status_req = models.DescribeTaskStatusRequest()
            status_req.TaskId = task_id

            deadline = time.monotonic() + _MAX_POLL_SECONDS
            while True:
                status_resp = self._asr_client.DescribeTaskStatus(status_req)
                status = int(getattr(status_resp.Data, "Status", 0))
                if status in (2, 3):
                    if status == 3:
                        error_msg = str(
                            getattr(
                                status_resp.Data, "ErrorMsg", "unknown Tencent error"
                            )
                        )
                        raise RuntimeError(f"Tencent ASR failed: {error_msg}")
                    result_detail = getattr(status_resp.Data, "ResultDetail", None)
                    return list(result_detail or [])
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Tencent ASR task {task_id} timed out after {_MAX_POLL_SECONDS}s"
                    )
                time.sleep(self._poll_interval_s)
        finally:
            try:
                self._bucket.delete_object(Bucket=self._bucket_name, Key=upload_key)
            except Exception:
                L.debug("Tencent cleanup failed for key=%s", upload_key, exc_info=True)

    def monologues(self, result_detail: list[Any]) -> AsrGenerationPayload:
        """Convert Tencent `ResultDetail` into shared ASR monologues via Rust."""
        return {"monologues": self._projection(result_detail)["monologues"]}

    def timed_words(self, result_detail: list[Any]) -> list[TimedWord]:
        """Extract timed words for `ParsedChat.add_utterance_timing()` via Rust."""
        return self._projection(result_detail)["timed_words"]

    def _projection(self, result_detail: list[Any]) -> dict[str, Any]:
        """Delegate Tencent result-detail projection to the shared Rust helper."""
        import batchalign_core

        return json.loads(batchalign_core.tencent_result_detail_to_asr(result_detail))
