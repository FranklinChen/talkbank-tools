"""Stanza capability table builder: single source of truth.

Reads Stanza's ``resources.json`` to discover per-language processor
availability.  Replaces 7 scattered hardcoded tables across Python and
Rust with one authoritative data structure built from the installed
Stanza version.

The table is built once (lazily) and cached for the process lifetime.

On a fresh install where Stanza's resource catalog has never been seeded,
``resources.json`` does not yet exist. In that case the builder bootstraps
the catalog by calling Stanza's own ``download_resources_json()`` (a small,
fast download: the catalog itself, no language packs) and retries.
Language packs are then downloaded lazily by ``stanza.Pipeline()`` on first
use of each language. The end user never has to seed anything by hand.

A real network/disk failure during catalog bootstrap surfaces as the typed
``StanzaCatalogDownloadError`` rather than a silent ``None``, the original
silent-None path masked the bug whose downstream effect was a worker
exit-1 retry storm with multi-GB log spam from a deterministic catalog
miss.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field

from batchalign.worker._progress import emit_download_event

L = logging.getLogger(__name__)

# How long the one-time, per-worker resources.json refresh may block before
# giving up and using the cached manifest. Short so an offline or slow-network
# worker boots promptly instead of hanging on catalog I/O (Rule 11).
_MANIFEST_REFRESH_TIMEOUT_S = 10


class StanzaCatalogDownloadError(RuntimeError):
    """Stanza resource catalog could not be downloaded.

    Raised when batchalign3 attempted to bootstrap Stanza's
    ``resources.json`` on first use, but the download failed (network
    unreachable, upstream returned non-200, disk full, permission denied,
    etc.).

    Distinct from "language not supported" (a domain-level rejection)
    and from "Stanza package not installed" (a deploy-config error). This
    error is recoverable by retrying when network is available.
    """


# ISO-639-3 → Stanza alpha-2 overrides for codes that pycountry
# doesn't map correctly or that Stanza uses non-standard identifiers for.
_ISO3_OVERRIDES: dict[str, str] = {
    "nor": "nb",  # Norwegian → Bokmål (Stanza uses "nb")
    "yue": "zh-hans",  # Cantonese → Chinese (Stanza's zh-hans model)
    "cmn": "zh-hans",  # Mandarin → Chinese
    "zho": "zh-hans",  # Chinese (generic) → zh-hans
    "msa": "ms",  # Malay (ISO-639-3 msa) → Stanza ms
}


@dataclass(frozen=True)
class StanzaLanguageCapability:
    """Per-language processor availability from Stanza resources.json."""

    alpha2: str
    has_tokenize: bool = False
    has_pos: bool = False
    has_lemma: bool = False
    has_depparse: bool = False
    has_mwt: bool = False
    has_constituency: bool = False
    has_coref: bool = False


@dataclass(frozen=True)
class StanzaCapabilityTable:
    """Complete capability registry derived from resources.json.

    ``languages`` is keyed by ISO-639-3 code (e.g. ``"eng"``, ``"nld"``).
    ``iso3_to_alpha2`` maps ISO-639-3 → Stanza alpha-2 for all supported
    languages (derived from pycountry + overrides).

    The table carries no Stanza version. It used to, with an empty-string
    default here and ``"unknown"`` in both builders, and nothing read it;
    the installed version has one accessor, ``_state.stanza_version()``.
    """

    languages: dict[str, StanzaLanguageCapability] = field(default_factory=dict)
    iso3_to_alpha2: dict[str, str] = field(default_factory=dict)

    def supports_morphosyntax(self, iso3: str) -> bool:
        """Whether ``iso3`` has the core Stanza processors morphotag needs."""
        cap = self.languages.get(iso3)
        if cap is None:
            return False
        return all((cap.has_tokenize, cap.has_pos, cap.has_lemma, cap.has_depparse))


def build_stanza_capability_table_from_resources(
    resources: dict[str, object],
) -> StanzaCapabilityTable:
    """Build the capability table from an already-loaded resources mapping."""
    # Build alpha2 → capability mapping from resources.json.
    # Skip non-language keys (like "default") and alias entries.
    alpha2_caps: dict[str, StanzaLanguageCapability] = {}

    # Stanza resources keys are alpha-2 codes (or variants like "zh-hans").
    # We check which processors are listed as top-level keys in the resource entry.
    _SKIP_KEYS = {"default"}
    for alpha2, lang_data in resources.items():
        if alpha2 in _SKIP_KEYS:
            continue
        if not isinstance(lang_data, dict):
            continue
        # A real language entry has processor keys like "tokenize", "pos", etc.
        # Alias entries are strings pointing to another language.
        if "tokenize" not in lang_data:
            continue

        alpha2_caps[alpha2] = StanzaLanguageCapability(
            alpha2=alpha2,
            has_tokenize="tokenize" in lang_data,
            has_pos="pos" in lang_data,
            has_lemma="lemma" in lang_data,
            has_depparse="depparse" in lang_data,
            has_mwt="mwt" in lang_data,
            has_constituency="constituency" in lang_data,
            has_coref="coref" in lang_data,
        )

    # Build ISO-639-3 → alpha-2 mapping using pycountry.
    iso3_map: dict[str, str] = {}

    # First: apply explicit overrides (these take priority).
    for iso3, alpha2 in _ISO3_OVERRIDES.items():
        if alpha2 in alpha2_caps:
            iso3_map[iso3] = alpha2

    # Second: use pycountry for standard mappings.
    try:
        import pycountry

        for lang in pycountry.languages:
            alpha3 = getattr(lang, "alpha_3", None)
            alpha2_code: str | None = getattr(lang, "alpha_2", None)
            if not alpha3 or not alpha2_code:
                continue
            # Don't override explicit overrides.
            if alpha3 in iso3_map:
                continue
            # Check if Stanza has this alpha-2 code.
            if alpha2_code in alpha2_caps:
                iso3_map[alpha3] = alpha2_code
    except ImportError:
        L.warning(
            "pycountry not installed: iso3_to_alpha2 mapping will only "
            "include explicit overrides"
        )

    # Build the final table keyed by ISO-639-3.
    languages: dict[str, StanzaLanguageCapability] = {}
    for iso3, alpha2 in iso3_map.items():
        if alpha2 in alpha2_caps:
            languages[iso3] = alpha2_caps[alpha2]

    L.info(
        "Built Stanza capability table: %d languages, %d with constituency, "
        "%d with mwt",
        len(languages),
        sum(1 for c in languages.values() if c.has_constituency),
        sum(1 for c in languages.values() if c.has_mwt),
    )

    return StanzaCapabilityTable(
        languages=languages,
        iso3_to_alpha2=iso3_map,
    )


def build_stanza_capability_table() -> StanzaCapabilityTable:
    """Build the capability table from Stanza's installed resources.json.

    This is the single source of truth for what Stanza can process per
    language.  Called once at worker startup; the result is cached.
    """
    import stanza.resources.common as src

    return build_stanza_capability_table_from_resources(src.load_resources_json())


def refresh_resources_manifest_if_present() -> None:
    """Re-fetch a present-but-stale Stanza ``resources.json`` from upstream.

    Every worker pipeline is built with ``DownloadMethod.REUSE_RESOURCES``,
    which reuses the cached manifest and never re-fetches it, then verifies any
    freshly-downloaded model against that cached manifest. When upstream
    re-publishes a model under the *same* resources version (Stanford did this
    to four lemma models under 1.12.0 in 2026-06), a worker still holding the
    pre-publish manifest fails Stanza's integrity check on the next download
    (``md5 for ... is X, expected Y``) and the job dies. Refreshing the manifest
    keeps it in lockstep with the models downloaded against it.

    This is a **worker-startup action**, called once from the worker bootstrap
    (``_main.main``), not from ``get_cached_capability_table``, which is also
    invoked by tests and tooling that must not trigger a network fetch or mutate
    the real on-disk cache.

    Guarded for the fleet:

    - Only acts when a manifest is already present. The missing case is the
      fresh-install bootstrap handled by ``_bootstrap_and_retry``; its download
      already fetches the current manifest, so no skew is possible there.
    - Any failure (offline, timeout, HTTP error, malformed body, disk error) is
      logged and swallowed, leaving the cached manifest in place. An air-gapped
      worker therefore behaves exactly as before the fix (plain
      ``REUSE_RESOURCES`` semantics); the refresh must never break a worker.
    """
    import stanza.resources.common as src

    manifest_path = os.path.join(src.DEFAULT_MODEL_DIR, "resources.json")
    if not os.path.exists(manifest_path):
        return

    # Mirror Stanza's own URL construction (download_resources_json):
    # ``<resources_url>/resources_<version>.json``. Reading the module
    # constants keeps us aligned with whatever STANZA_RESOURCES_URL /
    # STANZA_RESOURCES_VERSION the worker environment is configured with.
    url = f"{src.DEFAULT_RESOURCES_URL}/resources_{src.DEFAULT_RESOURCES_VERSION}.json"
    # Rule 11 (time transparency): this is an external blocking call (up to the
    # timeout), so frame it on the progress channel even though it is usually
    # sub-second.
    emit_download_event(
        stage="refreshing_stanza_catalog",
        user_message="Checking for Stanza resource catalog updates (~1 MB)…",
    )
    outcome_message = "Using cached Stanza resource catalog."
    try:
        with urllib.request.urlopen(
            url, timeout=_MANIFEST_REFRESH_TIMEOUT_S
        ) as response:
            data = response.read()
        json.loads(
            data
        )  # validate before installing: never persist a torn body or HTML error page

        # Atomic install so a concurrent reader never observes a partial file.
        tmp_path = manifest_path + ".tmp-refresh"
        with open(tmp_path, "wb") as fout:
            fout.write(data)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp_path, manifest_path)

        outcome_message = "Stanza resource catalog up to date."
        L.info("Refreshed Stanza resources.json from %s", url)
    except Exception as exc:
        # Best-effort, fail-open to the cached manifest: a refresh must never
        # break a worker. Offline / timeout / HTTP / JSON / disk errors all
        # fall back to pre-fix REUSE_RESOURCES behavior.
        L.warning(
            "Stanza resources.json refresh skipped; using cached manifest (%s)",
            exc,
        )
    emit_download_event(
        stage="refreshing_stanza_catalog_complete",
        user_message=outcome_message,
    )


@functools.lru_cache(maxsize=1)
def get_cached_capability_table() -> StanzaCapabilityTable | None:
    """Return the cached capability table, building it on first call.

    Behavior on a fresh install (no ``resources.json`` yet): bootstrap the
    catalog via ``stanza.resources.common.download_resources_json()`` and
    retry. The user is informed of the download via the ``progress_v2``
    event channel: silent waits are a UX bug.

    Returns:
        - Populated ``StanzaCapabilityTable`` on success (cache hit, fresh
          download, or both).
        - ``None`` only when the Stanza Python package itself is not
          installed in the worker venv (a deploy/config error, not a
          recoverable miss).

    Raises:
        StanzaCatalogDownloadError: when the catalog must be downloaded
            but the download fails (network, disk, etc.). The orchestrator
            should classify this as non-retryable at the bootstrap layer:
            retrying 3× immediately won't fix a network outage.
    """
    try:
        return build_stanza_capability_table()
    except ImportError:
        # Distinct, legitimate silent-None path: Stanza package not in venv.
        # This is a deploy-config error, not a recoverable miss.
        L.warning("Stanza not installed: capability table unavailable")
        return None
    except Exception as exc:
        # Recoverable miss: the catalog file ``resources.json`` does not yet
        # exist locally. Stanza itself raises ``ResourcesFileNotFoundError``
        # (a subclass of ``FileNotFoundError``); we bootstrap and retry.
        # Any other exception type is a genuine error and re-raised.
        if not _is_resources_missing(exc):
            L.error(
                "Stanza capability table build failed with unexpected error: %s",
                exc,
            )
            raise

        return _bootstrap_and_retry(exc)


def _is_resources_missing(exc: BaseException) -> bool:
    """Return True iff ``exc`` is Stanza's ResourcesFileNotFoundError.

    Imported lazily because the import chain itself depends on Stanza
    being present, which is the very condition the caller is checking.
    """
    try:
        from stanza.resources.common import ResourcesFileNotFoundError
    except ImportError:
        return False
    return isinstance(exc, ResourcesFileNotFoundError)


def _bootstrap_and_retry(original: BaseException) -> StanzaCapabilityTable:
    """Download Stanza's resource catalog, then rebuild the capability table.

    Pairs of ``progress_v2`` events frame the download so every UI surface
    (CLI, TUI, desktop app, web dashboard) shows the user that the wait is
    a one-time bootstrap, not a hang.
    """
    import stanza.resources.common as src

    L.info(
        "Stanza resource catalog missing (%s); bootstrapping from %s",
        original,
        src.DEFAULT_RESOURCES_URL,
    )
    emit_download_event(
        stage="downloading_stanza_catalog",
        user_message=(
            "Downloading Stanza resource catalog (one-time, ~1 MB; "
            "future runs will be instant)…"
        ),
    )
    try:
        src.download_resources_json()
    except Exception as download_exc:
        emit_download_event(
            stage="downloading_stanza_catalog_failed",
            user_message=(f"Stanza resource catalog download failed: {download_exc}"),
        )
        raise StanzaCatalogDownloadError(
            f"Failed to download Stanza resource catalog from "
            f"{src.DEFAULT_RESOURCES_URL}: {download_exc}"
        ) from download_exc

    emit_download_event(
        stage="downloading_stanza_catalog_complete",
        user_message="Stanza resource catalog ready.",
    )

    # Catalog is now on disk; the second build attempt should succeed. If
    # this still fails, surface the error; it means the downloaded catalog
    # is corrupt or unreadable, which is a real bug worth investigating.
    return build_stanza_capability_table()
