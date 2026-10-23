"""Stanza capability table builder: single source of truth.

Reads Stanza's ``resources.json`` to discover per-language processor
availability, and maps ISO 639-3 codes onto its entries through Rust's ISO
639-1 table. It replaced scattered hardcoded tables across Python and Rust,
the last of them the server's static list of supported codes, with one
data structure built from the installed Stanza version.

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
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import batchalign_core
from batchalign.inference._domain_types import StanzaCode
from batchalign.worker._progress import emit_download_event
from batchalign.worker._types import StanzaProcessorName

_STANZA_PROCESSORS: tuple[StanzaProcessorName, ...] = batchalign_core.STANZA_PROCESSORS
"""Every processor a catalog entry can ship, in Rust's order (its closed set)."""

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


CHINESE_STANZA_CODE = StanzaCode("zh-hans")
"""Stanza's Simplified Chinese model, the one every Chinese language uses."""

# The Stanza code BA3 CHOOSES for languages that neither the ISO table nor the
# catalog resolves. Mandarin (``cmn``) and Cantonese (``yue``) have no ISO
# 639-1 code and no catalog entry or alias of their own; BA3 analyses both with
# Stanza's Simplified Chinese model (Cantonese POS is then overridden by
# PyCantonese). This is a policy, not a fact about codes, which is why it is
# the only hand-written table in the mapping. ``zho`` needs no entry: its ISO
# 639-1 code ``zh`` is a catalog alias of ``zh-hans``. ``nor`` needs none
# either: ``no`` is a catalog alias of ``nb`` (Bokmål).
_STANZA_CODE_POLICY: dict[str, StanzaCode] = {
    "cmn": CHINESE_STANZA_CODE,
    "yue": CHINESE_STANZA_CODE,
}

if TYPE_CHECKING:
    from batchalign_core.batchalign_core import (
        StanzaRequirementName as StanzaRequirementName,
    )
else:
    # Read from Rust's closed table at runtime; mypy reads the extension
    # stub's spelling, held to the same table by
    # ``test_stanza_names_match_rust``.
    StanzaRequirementName = Literal[
        tuple(name for name, _ in batchalign_core.STANZA_REQUIREMENTS)
    ]
"""A Stanza-backed command's requirement, by the name Rust exports it under
(``StanzaRequirement::wire_name``)."""


def processors_argument(processors: Iterable[StanzaProcessorName]) -> str:
    """A ``stanza.Pipeline`` ``processors`` argument naming exactly these, in
    the closed processor set's order."""
    wanted = set(processors)
    return ",".join(p for p in _STANZA_PROCESSORS if p in wanted)


@functools.cache
def requirement_processors(
    requirement: StanzaRequirementName,
) -> frozenset[StanzaProcessorName]:
    """The processors one command's pipeline needs and loads.

    Read from Rust's closed table (``batchalign_core.STANZA_REQUIREMENTS``),
    the one the server admits languages by, so what this worker loads and
    what the server admitted are one list. Its processor names and the
    processor set are both Rust's, so they cannot disagree.
    """
    return frozenset(dict(batchalign_core.STANZA_REQUIREMENTS)[requirement])


@dataclass(frozen=True)
class StanzaLanguageCapability:
    """One catalog entry and the processors it ships, from resources.json."""

    stanza_code: StanzaCode
    processors: frozenset[StanzaProcessorName]

    def has(self, processor: StanzaProcessorName) -> bool:
        """Whether the entry ships ``processor``."""
        return processor in self.processors

    def serves(self, requirement: StanzaRequirementName) -> bool:
        """Whether the entry ships every processor ``requirement`` needs."""
        return requirement_processors(requirement) <= self.processors

    @property
    def supports_morphosyntax(self) -> bool:
        """Whether the entry has every processor morphotag loads."""
        return self.serves("morphosyntax")


@dataclass(frozen=True)
class StanzaNotInstalled:
    """The ``stanza`` package does not import in this worker environment.

    A deploy fact, distinct from a catalog that lacks a language and from a
    failure to build the table (which raises).
    """

    import_error: str


@dataclass(frozen=True)
class StanzaCapabilityTable:
    """Every ISO 639-3 language the installed Stanza catalog can serve.

    ``languages`` is keyed by ISO 639-3 code (``"eng"``, ``"kaz"``); each
    value names the catalog entry that serves it (``stanza_code``) and that
    entry's processors. It is the ONE ISO 639-3 to Stanza mapping in the
    worker: the pipeline loaders read it (``stanza_code_for``), and the
    capabilities report sends it to the server, whose language admission
    reads nothing else.

    The table carries no Stanza version. It used to, with an empty-string
    default here and ``"unknown"`` in both builders, and nothing read it;
    the installed version has one accessor, ``_state.stanza_version()``.
    """

    languages: dict[str, StanzaLanguageCapability] = field(default_factory=dict)

    def supports_morphosyntax(self, iso3: str) -> bool:
        """Whether ``iso3`` has the core Stanza processors morphotag needs."""
        cap = self.languages.get(iso3)
        return cap is not None and cap.supports_morphosyntax


def _is_iso_639_3_shaped(code: str) -> bool:
    """Three lowercase ASCII letters: the shape of an ISO 639-3 code, and the
    only key the server's capability admission accepts."""
    return len(code) == 3 and code.isascii() and code.isalpha() and code.islower()


def build_stanza_capability_table_from_resources(
    resources: dict[str, object],
    iso_639_1_pairs: Iterable[tuple[str, str]],
) -> StanzaCapabilityTable:
    """Build the capability table from a loaded catalog and the ISO table.

    An ISO 639-3 code is served by the catalog entry found by, in order:

    1. ``_STANZA_CODE_POLICY`` (Mandarin and Cantonese);
    2. its ISO 639-1 code (from ``iso_639_1_pairs``, which the worker reads
       from Rust's one table, ``batchalign_core.ISO_639_1_PAIRS``), looked up
       in the catalog directly or through one of the catalog's own aliases
       (``no`` is an alias of ``nb``, ``zh`` of ``zh-hans``);
    3. the code itself, likewise: the catalog keys some languages by their
       three-letter code (``grc``, ``kmr``, ``ckb``) and aliases others
       (``kaz`` to ``kk``).

    A code none of these reaches is absent, never passed through: an absent
    language is one the server refuses with a typed reason.
    """
    entries: dict[str, StanzaLanguageCapability] = {}
    aliases: dict[str, str] = {}
    for code, lang_data in resources.items():
        # Non-language keys (``url``, ``default``) are not dictionaries of
        # processors or aliases.
        if not isinstance(lang_data, dict):
            continue
        target = lang_data.get("alias")
        if isinstance(target, str):
            aliases[code] = target
            continue
        # A real language entry has processor keys like "tokenize", "pos", etc.
        if "tokenize" not in lang_data:
            continue
        entries[code] = StanzaLanguageCapability(
            stanza_code=StanzaCode(code),
            processors=frozenset(p for p in _STANZA_PROCESSORS if p in lang_data),
        )

    def entry_for(code: str) -> StanzaLanguageCapability | None:
        direct = entries.get(code)
        if direct is not None:
            return direct
        target = aliases.get(code)
        return entries.get(target) if target is not None else None

    part1 = dict(iso_639_1_pairs)
    candidates = (
        set(_STANZA_CODE_POLICY)
        | set(part1)
        | {code for code in (*entries, *aliases) if _is_iso_639_3_shaped(code)}
    )

    languages: dict[str, StanzaLanguageCapability] = {}
    for iso3 in sorted(candidates):
        policy = _STANZA_CODE_POLICY.get(iso3)
        two_letter = part1.get(iso3)
        if policy is not None:
            entry = entries.get(policy)
        elif two_letter is not None and (via_part1 := entry_for(two_letter)):
            entry = via_part1
        else:
            entry = entry_for(iso3)
        if entry is not None:
            languages[iso3] = entry

    L.info(
        "Built Stanza capability table: %d languages, %d with morphosyntax, "
        "%d with constituency, %d with mwt",
        len(languages),
        sum(1 for c in languages.values() if c.supports_morphosyntax),
        sum(1 for c in languages.values() if c.has("constituency")),
        sum(1 for c in languages.values() if c.has("mwt")),
    )

    return StanzaCapabilityTable(languages=languages)


def build_stanza_capability_table() -> StanzaCapabilityTable:
    """Build the capability table from Stanza's installed resources.json.

    This is the single source of truth for what Stanza can process per
    language.  Called once at worker startup; the result is cached.
    """
    import stanza.resources.common as src

    from batchalign_core import ISO_639_1_PAIRS

    return build_stanza_capability_table_from_resources(
        src.load_resources_json(), ISO_639_1_PAIRS
    )


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
def get_cached_capability_table() -> StanzaCapabilityTable | StanzaNotInstalled:
    """Return the cached capability table, building it on first call.

    Behavior on a fresh install (no ``resources.json`` yet): bootstrap the
    catalog via ``stanza.resources.common.download_resources_json()`` and
    retry. The user is informed of the download via the ``progress_v2``
    event channel: silent waits are a UX bug.

    Returns:
        - Populated ``StanzaCapabilityTable`` on success (cache hit, fresh
          download, or both).
        - ``StanzaNotInstalled`` only when importing the ``stanza`` package
          itself fails (a deploy/config error, not a recoverable miss).

    Raises:
        StanzaCatalogDownloadError: when the catalog must be downloaded
            but the download fails (network, disk, etc.). The orchestrator
            should classify this as non-retryable at the bootstrap layer:
            retrying 3× immediately won't fix a network outage.
        Exception: any other failure to build the table, including an
            ``ImportError`` from anything but ``stanza`` itself (a stale
            ``batchalign_core`` extension without ``ISO_639_1_PAIRS``). Only
            the ``stanza`` import is "not installed"; every other failure
            propagates with its own type and message.
    """
    try:
        import stanza.resources.common  # noqa: F401  (the installed-or-not probe)
    except ImportError as exc:
        L.warning("Stanza not installed: capability table unavailable (%s)", exc)
        return StanzaNotInstalled(import_error=f"{type(exc).__name__}: {exc}")
    try:
        return build_stanza_capability_table()
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
