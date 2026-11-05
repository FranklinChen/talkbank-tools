"""Stateless controls for source/target routing at every loaded engine."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from batchalign.worker._model_loading import translation as loader


@pytest.fixture
def local_models(monkeypatch):
    calls = []

    class Tokenizer:
        src_lang = None

        @classmethod
        def from_pretrained(cls, _model):
            return cls()

        def convert_tokens_to_ids(self, code):
            return {"eng_Latn": 1, "spa_Latn": 2, "fra_Latn": 3}[code]

        def __call__(self, **kwargs):
            return kwargs

        def decode(self, _tokens, **_kwargs):
            return "translated"

    class NllbTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"text": text, "source": self.src_lang, **kwargs}

    class Model:
        @classmethod
        def from_pretrained(cls, _model):
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(tolist=lambda: [[1]])]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=NllbTokenizer,
            AutoModelForSeq2SeqLM=Model,
            AutoProcessor=Tokenizer,
            SeamlessM4TModel=Model,
        ),
    )
    # No hub: the resolve edge names a stand-in snapshot the fakes "load".
    from batchalign.tests._snapshots import fake_resolve_transformers

    monkeypatch.setattr(
        "batchalign.worker._model_loading.pinned_hub.resolve_transformers",
        fake_resolve_transformers,
    )
    return calls


@pytest.mark.parametrize("backend", ["nllb", "seamless"])
def test_local_engine_uses_each_requests_target_not_a_cached_english_target(
    local_models, backend
):
    translation = getattr(loader, f"_load_{backend}_translate")()
    assert translation.translate("hello", "eng", "spa") == "translated"
    assert translation.translate("hello", "eng", "fra") == "translated"
    if backend == "nllb":
        assert [call["forced_bos_token_id"] for call in local_models] == [2, 3]
        assert [call["source"] for call in local_models] == ["eng_Latn", "eng_Latn"]
    else:
        assert [call["tgt_lang"] for call in local_models] == ["spa", "fra"]
        assert [call["src_lang"] for call in local_models] == ["eng", "eng"]


@pytest.mark.parametrize("source,target", [("xyz", "eng"), ("eng", "xyz")])
def test_nllb_unmapped_route_is_refused_before_generation(local_models, source, target):
    translation = loader._load_nllb_translate()
    with pytest.raises(ValueError, match="mapping"):
        translation.translate("hello", source, target)
    assert local_models == []


@pytest.fixture
def tencent(monkeypatch):
    calls = []
    response = {"Response": {"TargetText": "translated"}}

    class Client:
        def __init__(self, service, version, cred, region, profile):
            assert service == "tmt" and version == "2018-03-21"
            assert profile.httpProfile.endpoint == "tmt.tencentcloudapi.com"

        def call_json(self, action, params):
            calls.append((action, params))
            return response

    monkeypatch.setattr("tencentcloud.common.common_client.CommonClient", Client)
    monkeypatch.setattr(
        "batchalign.inference.languages.cantonese._common.read_asr_config",
        lambda *args, **kwargs: {
            "engine.tencent.id": "test-id",
            "engine.tencent.key": "test-key",
            "engine.tencent.region": "test-region",
        },
    )
    return calls, response, loader._load_tencent_translate()


def test_tencent_supported_sdk_client_carries_non_english_target(tencent):
    calls, _, translation = tencent
    assert translation.translate("hello", "eng", "spa") == "translated"
    assert calls == [
        (
            "TextTranslate",
            {"SourceText": "hello", "Source": "en", "Target": "es", "ProjectId": 0},
        )
    ]


def test_tencent_unmapped_target_is_not_sent(tencent):
    calls, _, translation = tencent
    with pytest.raises(ValueError, match="target language"):
        translation.translate("hello", "eng", "xyz")
    assert calls == []


@pytest.mark.parametrize("value", [None, 12, {"text": "hello"}])
def test_tencent_bad_response_is_not_coerced_to_translation(tencent, value):
    _, response, translation = tencent
    response["Response"]["TargetText"] = value
    with pytest.raises(ValidationError):
        translation.translate("hello", "eng", "spa")


def test_legacy_batch_admits_its_english_contract_without_discarding_v2_targets(
    monkeypatch,
):
    from batchalign.inference.translate import LoadedTranslation
    from batchalign.providers import BatchInferRequest
    from batchalign.worker._infer_hosts import build_translate_runner, legacy_translate
    from batchalign.worker._loads import LoadedTranslate

    calls = []

    def translate(text, source, target):
        calls.append((text, source, target))
        return "hello"

    host = LoadedTranslate(
        runner=build_translate_runner(
            LoadedTranslation(engine="test-engine", translate=translate)
        )
    )
    good = legacy_translate(
        host, BatchInferRequest(task="translate", lang="spa", items=[{"text": "hola"}])
    )
    assert calls == [("hola", "spa", "eng")]
    assert good.results[0].error is None
    refused = legacy_translate(
        host, BatchInferRequest(task="translate", lang="", items=[{"text": "hola"}])
    )
    assert refused.results[0].error.startswith("Invalid legacy translation direction")
    assert len(calls) == 1
