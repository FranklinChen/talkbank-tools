"""Declared numerical fixtures: no downloads and no host-selected kernel drift."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from batchalign.inference.qwen_forced_alignment import (
    LoadedQwenFaHost,
    QwenForcedAligner,
    resolve_qwen_language,
)


@contextmanager
def slow_cpu_convolutions() -> Iterator[None]:
    """Scope both execution and projection to the source-qualified slow kernels.

    Automatic oneDNN/NNPACK selection differs across hosts. This controls the
    actual synthetic execution route; memory admission is never disabled.
    Unqualified-native-layout refusal has its own boundary control.
    """
    import torch

    with (
        torch.backends.mkldnn.flags(enabled=False),
        torch.backends.nnpack.flags(enabled=False),
    ):
        yield


def native_qwen_host() -> LoadedQwenFaHost:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        Qwen2TokenizerFast,
        Qwen3ASRConfig,
        Qwen3ASRFeatureExtractor,
        Qwen3ASRForTokenClassification,
        Qwen3ASRProcessor,
    )

    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<audio>": 2,
        "<timestamp>": 3,
        "<a>": 4,
        "</a>": 5,
        "hi": 6,
        "there": 7,
        "<unk>": 8,
    }
    core = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        extra_special_tokens={
            "audio_token": "<audio>",
            "audio_bos_token": "<a>",
            "audio_eos_token": "</a>",
        },
        additional_special_tokens=["<timestamp>"],
    )
    template = "{% for m in messages %}{% for c in m['content'] %}{% if c['type'] == 'audio' %}<a><audio></a>{% elif c['type'] == 'text' %}<timestamp> {{ c['text'] }} <timestamp>{% endif %}{% endfor %}{% endfor %}"
    processor = Qwen3ASRProcessor(
        feature_extractor=Qwen3ASRFeatureExtractor(
            feature_size=4, n_window=50, min_length=0
        ),
        tokenizer=tokenizer,
        chat_template=template,
    )
    config = Qwen3ASRConfig(
        audio_config={
            "model_type": "qwen3_asr_encoder",
            "num_mel_bins": 4,
            "d_model": 8,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "n_window": 50,
            "n_window_infer": 100,
            "downsample_hidden_size": 1,
            "max_position_embeddings": 13,
            "output_dim": 8,
        },
        text_config={
            "model_type": "qwen3",
            "vocab_size": len(tokenizer),
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 128,
        },
        audio_token_id=tokenizer.audio_token_id,
        timestamp_token_id=3,
        pad_token_id=0,
        eos_token_id=1,
        num_labels=16,
    )
    model = Qwen3ASRForTokenClassification(config).eval()
    model.set_attn_implementation("sdpa")
    return LoadedQwenFaHost(
        aligner=QwenForcedAligner(processor=processor, model=model),
        language=resolve_qwen_language("eng"),
    )
