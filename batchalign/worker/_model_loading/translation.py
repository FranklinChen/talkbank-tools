"""Translation-engine bootstrap helpers for worker startup.

Four backends are supported:

* ``TranslationBackend.GOOGLE`` — public Google Translate via the
  ``googletrans`` library. Requires reachability to
  ``translate.google.com``; unusable behind the Great Firewall.
* ``TranslationBackend.TENCENT`` — Tencent Cloud TMT (Text Translation).
  Cloud-API engine; CAM credentials in ``~/.batchalign.ini`` `[asr]`
  section (``engine.tencent.id``/``key``/``region``). Free tier
  5M chars/month. Best empirical quality on Mandarin (zh→en); does
  NOT support Cantonese (yue→en) — those routes must use NLLB.
* ``TranslationBackend.SEAMLESS`` — Meta's SeamlessM4T, loaded locally
  from HuggingFace. No outbound network at inference time. Known to
  produce poor CJK quality on short utterances; retained for back-compat.
* ``TranslationBackend.NLLB`` — Meta's NLLB-200-distilled-1.3B,
  text-MT-native, ~5 GB model. No outbound network at inference time.
  Self-hosted fallback that handles Cantonese first-class. Short CJK
  greetings (≤ 5 chars) are a known weakness.

Selection is driven by the same ``engine_overrides`` dict ASR and FA use
(see ``asr.py::resolve_asr_engine``). The Rust control plane decides which
backend a worker pool loads and passes the choice through
``WorkerBootstrapRuntime.engine_overrides``.
"""

from __future__ import annotations

import logging
from typing import NewType

from batchalign.inference._domain_types import LanguageCode, TranslationBackend
from batchalign.worker._types import WorkerBootstrapRuntime, _state

# A FLORES-200 language tag (e.g. ``"spa_Latn"``, ``"zho_Hans"``,
# ``"yue_Hant"``) as accepted by NLLB's tokenizer ``src_lang`` setter
# and ``convert_tokens_to_ids`` for the target language token. Distinct
# from ``LanguageCode`` (ISO-639-3) so a misplaced FLORES tag at an
# ISO-639-3 site won't typecheck.
FloresLanguageTag = NewType("FloresLanguageTag", str)

L = logging.getLogger("batchalign.worker")


def load_translation_engine(bootstrap: WorkerBootstrapRuntime) -> None:
    """Load the translation engine for this worker.

    Dispatches on the resolved ``TranslationBackend`` so adding a new
    variant later forces a missing-arm error rather than silently
    falling through to Google.
    """
    backend = resolve_translate_engine(bootstrap.engine_overrides or None)
    if backend is TranslationBackend.GOOGLE:
        _load_google_translate()
    elif backend is TranslationBackend.SEAMLESS:
        _load_seamless_translate()
    elif backend is TranslationBackend.NLLB:
        _load_nllb_translate()
    elif backend is TranslationBackend.TENCENT:
        _load_tencent_translate()
    else:
        # Exhaustive — if a new TranslationBackend variant is added and
        # not wired in here, we raise rather than leaving translate_fn
        # unset (the batch-infer handler would later fail opaquely).
        raise RuntimeError(f"unhandled translation backend: {backend!r}")


def resolve_translate_engine(
    engine_overrides: dict[str, str] | None,
) -> TranslationBackend:
    """Pick the translation backend from the engine-overrides dict.

    Precedence:

    1. An explicit ``"translate"`` entry selects that backend. Unknown
       values raise ``ValueError`` rather than silently falling back —
       a typo in a per-host config would otherwise produce silently-
       wrong translations.
    2. Default is Google, preserving historical behavior for hosts that
       never set a translate override.
    """
    if not engine_overrides or "translate" not in engine_overrides:
        return TranslationBackend.GOOGLE
    choice = engine_overrides["translate"]
    try:
        return TranslationBackend(choice)
    except ValueError as exc:
        supported = ", ".join(b.value for b in TranslationBackend)
        raise ValueError(
            f"unknown translate engine {choice!r}; expected one of: {supported}"
        ) from exc


def _load_google_translate() -> None:
    """Bind ``_state.translate_fn`` to a googletrans-backed translator."""
    from googletrans import Translator

    async def _do_translate(translator: Translator, text: str) -> str:
        result = await translator.translate(text)
        return str(getattr(result, "text", result))

    def translate_fn(text: str, src_lang: LanguageCode) -> str:
        """Run the async translator behind the worker's synchronous IPC seam."""
        import asyncio

        translator = Translator()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_do_translate(translator, text))
        finally:
            loop.close()

    _state.translate_backend = TranslationBackend.GOOGLE
    _state.translate_fn = translate_fn


def _load_seamless_translate() -> None:
    """Bind ``_state.translate_fn`` to a locally-loaded SeamlessM4T model.

    Model is downloaded from HuggingFace on first load and cached
    thereafter. Operators on hosts where the public HF endpoint is slow
    or blocked can point at a mirror via ``HF_ENDPOINT`` before the
    worker starts.
    """
    from transformers import AutoProcessor, SeamlessM4TModel

    from batchalign.worker._progress import (
        HF_ARTIFACTS_SEAMLESS,
        emit_hf_download_if_missing,
    )

    emit_hf_download_if_missing(
        "facebook/hf-seamless-m4t-medium",
        kind="translation",
        artifacts=HF_ARTIFACTS_SEAMLESS,
    )

    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        "facebook/hf-seamless-m4t-medium"
    )
    model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
    # torch.nn.Module.eval() — sets the module to inference mode,
    # unrelated to Python's builtin eval().
    if hasattr(model, "eval"):
        model.eval()  # type: ignore[no-untyped-call]

    def seamless_fn(text: str, src_lang: LanguageCode) -> str:
        """Translate one text payload through SeamlessM4T."""
        inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
        output = model.generate(**inputs, tgt_lang="eng", generate_speech=False)
        return str(processor.decode(output[0].tolist()[0], skip_special_tokens=True))

    _state.translate_backend = TranslationBackend.SEAMLESS
    _state.translate_fn = seamless_fn


# Only languages empirically validated against NLLB are listed; an
# unmapped source language raises at inference time rather than
# silently producing wrong-language output. FLORES-200 codes per
# Meta's NLLB documentation.
_ISO_639_3_TO_FLORES_200: dict[LanguageCode, FloresLanguageTag] = {
    LanguageCode("eng"): FloresLanguageTag("eng_Latn"),
    LanguageCode("spa"): FloresLanguageTag("spa_Latn"),
    LanguageCode("fra"): FloresLanguageTag("fra_Latn"),
    LanguageCode("deu"): FloresLanguageTag("deu_Latn"),
    LanguageCode("ita"): FloresLanguageTag("ita_Latn"),
    LanguageCode("por"): FloresLanguageTag("por_Latn"),
    LanguageCode("nld"): FloresLanguageTag("nld_Latn"),
    LanguageCode("cmn"): FloresLanguageTag("zho_Hans"),
    LanguageCode("zho"): FloresLanguageTag("zho_Hans"),
    LanguageCode("yue"): FloresLanguageTag("yue_Hant"),
    LanguageCode("jpn"): FloresLanguageTag("jpn_Jpan"),
    LanguageCode("kor"): FloresLanguageTag("kor_Hang"),
    LanguageCode("rus"): FloresLanguageTag("rus_Cyrl"),
}


def _load_nllb_translate() -> None:
    """Bind ``_state.translate_fn`` to a locally-loaded NLLB-200-distilled-1.3B.

    Model downloads from HuggingFace on first load (~5 GB) and is
    cached thereafter. Operators on hosts where the public HF endpoint
    is slow or blocked can point at a mirror via ``HF_ENDPOINT`` before
    the worker starts.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from batchalign.worker._progress import (
        HF_ARTIFACTS_NLLB,
        emit_hf_download_if_missing,
    )

    model_id = "facebook/nllb-200-distilled-1.3B"
    emit_hf_download_if_missing(
        model_id,
        kind="translation",
        artifacts=HF_ARTIFACTS_NLLB,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    # torch.nn.Module.eval() — sets the module to inference mode
    # (disables dropout/BN training behavior). Without this, the 1.3B
    # encoder-decoder stays in training mode and generation is
    # non-deterministic + lower quality.
    if hasattr(model, "eval"):
        model.eval()
    eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    def nllb_fn(text: str, src_lang: LanguageCode) -> str:
        """Translate one text payload through NLLB-200."""
        flores_src = _ISO_639_3_TO_FLORES_200.get(src_lang)
        if flores_src is None:
            raise ValueError(
                f"NLLB backend has no FLORES-200 mapping for source "
                f"language {src_lang!r}; add an entry to "
                f"_ISO_639_3_TO_FLORES_200 in "
                f"batchalign/worker/_model_loading/translation.py "
                f"after validating output quality against NLLB"
            )
        tokenizer.src_lang = flores_src
        inputs = tokenizer(text, return_tensors="pt")
        translated = model.generate(
            **inputs,
            forced_bos_token_id=eng_token_id,
            max_length=256,
        )
        return str(tokenizer.decode(translated[0], skip_special_tokens=True))

    _state.translate_backend = TranslationBackend.NLLB
    _state.translate_fn = nllb_fn


# Mapping from the ISO-639-3 codes BA3 emits per CHAT @Languages header
# to the ISO-639-1 codes Tencent TMT expects (e.g. ``"es"``, ``"zh"``,
# ``"ja"``). Only languages Tencent TMT supports + we have empirically
# validated are listed; an unmapped source language raises ValueError
# at inference time directing the operator to ``--translate-engine
# nllb`` for that language. NOTE: Tencent does NOT support Cantonese
# (``yue``) — verified 2026-05-26, returns InvalidParameterValue
# ("不支持的语种：yue_to_en"). Use NLLB for Cantonese.
_ISO_639_3_TO_TENCENT_LANG: dict[LanguageCode, str] = {
    LanguageCode("eng"): "en",
    LanguageCode("spa"): "es",
    LanguageCode("fra"): "fr",
    LanguageCode("deu"): "de",
    LanguageCode("ita"): "it",
    LanguageCode("por"): "pt",
    LanguageCode("rus"): "ru",
    LanguageCode("cmn"): "zh",
    LanguageCode("zho"): "zh",
    LanguageCode("jpn"): "ja",
    LanguageCode("kor"): "kor",
    LanguageCode("ara"): "ar",
    LanguageCode("tha"): "th",
    LanguageCode("vie"): "vi",
    LanguageCode("tur"): "tr",
    LanguageCode("ind"): "id",
    LanguageCode("msa"): "ms",
}


def _load_tencent_translate() -> None:
    """Bind ``_state.translate_fn`` to a Tencent Cloud TMT translator.

    Credentials are read from ``~/.batchalign.ini`` ``[asr]`` section
    (``engine.tencent.id``, ``engine.tencent.key``,
    ``engine.tencent.region``) — the same credential bundle used by
    the Tencent ASR backend. The ``[asr]`` section name is a
    historical artifact; the CAM SecretId/SecretKey pair authorizes
    both ASR and TMT once the CAM user has ``tmt:TextTranslate``
    permission and the TMT service is "opened" at the Tencent Cloud
    account level (one-time setup per account).

    Free tier: 5M characters/month, ample for typical research-corpus
    workloads. Empty ``SourceText`` is rejected by the Tencent API
    with ``InvalidParameter`` — the inference closure short-circuits
    empty input upstream to avoid spurious failures.
    """
    import configparser
    from pathlib import Path

    from batchalign.inference._domain_types import (
        TencentSecretId,
        TencentSecretKey,
        TencentRegion,
    )

    ini_path = Path.home() / ".batchalign.ini"
    cp = configparser.ConfigParser()
    cp.read(ini_path)
    if "asr" not in cp:
        raise RuntimeError(
            f"Tencent translate backend requires credentials in {ini_path} "
            f"[asr] section (engine.tencent.id, engine.tencent.key, "
            f"engine.tencent.region); the file has no [asr] section"
        )
    asr = cp["asr"]
    secret_id = TencentSecretId(asr.get("engine.tencent.id", "").strip())
    secret_key = TencentSecretKey(asr.get("engine.tencent.key", "").strip())
    region = TencentRegion(asr.get("engine.tencent.region", "").strip())
    if not (secret_id and secret_key and region):
        raise RuntimeError(
            f"Tencent translate backend requires engine.tencent.id, .key, "
            f"and .region in {ini_path} [asr]; at least one is missing"
        )

    from tencentcloud.common import credential
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
        TencentCloudSDKException,
    )
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.tmt.v20180321 import models, tmt_client

    cred = credential.Credential(secret_id, secret_key)
    http_profile = HttpProfile()
    http_profile.endpoint = "tmt.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    client = tmt_client.TmtClient(cred, region, client_profile)

    def tencent_fn(text: str, src_lang: LanguageCode) -> str:
        """Translate one text payload through Tencent TMT."""
        if not text:
            # Tencent rejects empty SourceText with InvalidParameter.
            # Upstream batch infer should already skip empties, but
            # short-circuit here so a slip doesn't surface as a typed
            # SDK exception that looks like a credential failure.
            return ""
        tencent_src = _ISO_639_3_TO_TENCENT_LANG.get(src_lang)
        if tencent_src is None:
            raise ValueError(
                f"Tencent TMT does not support source language "
                f"{src_lang!r}; use --translate-engine nllb for this "
                f"language (Cantonese in particular requires nllb)"
            )
        req = models.TextTranslateRequest()
        req.SourceText = text
        req.Source = tencent_src
        req.Target = "en"
        req.ProjectId = 0
        try:
            resp = client.TextTranslate(req)
        except TencentCloudSDKException as exc:
            raise RuntimeError(
                f"Tencent TMT translation failed: {exc}"
            ) from exc
        return str(resp.TargetText)

    _state.translate_backend = TranslationBackend.TENCENT
    _state.translate_fn = tencent_fn


__all__ = [
    "load_translation_engine",
    "resolve_translate_engine",
]
