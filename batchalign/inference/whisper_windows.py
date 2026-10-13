"""Decode each generated window before upstream cross-window text matching.

Imported only by the model loader: importing the worker does not load torch.
Generation, batching and timestamp extraction are unchanged. Rust owns the
reconciliation of these independently decoded observations.
"""

from typing import Any, Literal

from transformers import AutomaticSpeechRecognitionPipeline

from batchalign.worker._types_v2 import (
    WhisperChunkSpanV2,
    WhisperWindowsV2,
    WhisperWindowV2,
)


class WindowOwnedWhisperPipeline(AutomaticSpeechRecognitionPipeline):
    """Preserve the sample ownership supplied by the pipeline preprocessor."""

    def _sanitize_parameters(
        self,
        *args: Any,
        observation_mode: Literal["windows", "legacy"] = "windows",
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        # Transformers exposes this model-host hook without annotations.
        preprocess, forward, postprocess = super()._sanitize_parameters(*args, **kwargs)  # type: ignore[no-untyped-call]
        if observation_mode not in ("windows", "legacy"):
            raise ValueError("unknown Whisper observation mode")
        postprocess["observation_mode"] = observation_mode
        return preprocess, forward, postprocess

    def postprocess(
        self,
        model_outputs: Any,
        decoder_kwargs: dict[str, Any] | None = None,
        return_timestamps: Any = None,
        return_language: Any = None,
        observation_mode: Any = "windows",
        **postprocess_parameters: Any,
    ) -> dict[str, Any]:
        if observation_mode == "legacy":
            return dict(
                super().postprocess(
                    model_outputs,
                    decoder_kwargs=decoder_kwargs,
                    return_timestamps=return_timestamps,
                    return_language=return_language,
                    **postprocess_parameters,
                )
            )
        windows = []
        owned_end = 0
        for output in model_outputs:
            if "stride" not in output:
                raise ValueError(
                    "windowed Whisper output lost its source sample stride"
                )
            count, left, right = output["stride"]
            start = owned_end - left
            # A shallow copy protects the original tensors and sample metadata.
            # Removing stride prevents _decode_asr from clipping this independent
            # observation as though it had already been joined to its neighbors.
            independent = {
                key: value for key, value in output.items() if key != "stride"
            }
            decoded = super().postprocess(
                [independent],
                decoder_kwargs=decoder_kwargs,
                return_timestamps=return_timestamps,
                return_language=return_language,
                **postprocess_parameters,
            )
            chunks = decoded.get("chunks", [])
            if not chunks and decoded.get("text"):
                chunks = [{"text": decoded["text"], "timestamp": (None, None)}]
            spans = []
            for chunk in chunks:
                begin, end = chunk.get("timestamp") or (None, None)
                spans.append(
                    WhisperChunkSpanV2(text=chunk["text"], start_s=begin, end_s=end)
                )
            windows.append(
                WhisperWindowV2(
                    start_sample=start,
                    sample_count=count,
                    left_context=left,
                    right_context=right,
                    spans=spans,
                )
            )
            owned_end = start + count - right
        if self.feature_extractor is None:
            raise ValueError("windowed Whisper has no audio feature extractor")
        return {
            "window_observations": WhisperWindowsV2(
                sample_rate_hz=self.feature_extractor.sampling_rate,
                windows=windows,
            )
        }
