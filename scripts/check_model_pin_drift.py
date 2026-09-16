#!/usr/bin/env python3
"""Check every pinned model in the manifest against its upstream.

Pinning exists so that output stops changing underneath us for reasons nobody
chose. A pin with no drift checker, though, freezes the project quietly instead
of prompting reviewed upgrades, which inverts that intent. This script is the
other half: it turns "upstream moved" into a reviewed decision with a date and a
reviewer, the same way the morphosyntax engine is upgraded (find a defect,
report it upstream, upgrade deliberately once it is fixed, rerun the affected
material surgically).

It reads the pins from `crates/batchalign/src/model_manifest.rs` and queries
PUBLIC APIs only. It uses no credentials, writes nothing, downloads no model,
and never consults a local model cache: a cache would only say what one machine
happens to hold, and the question here is what UPSTREAM holds.

Usage:
    python3 scripts/check_model_pin_drift.py
    python3 scripts/check_model_pin_drift.py --manifest <path to a manifest>

Exit codes:
    0  every checkable pin still matches upstream
    1  some upstream has MOVED away from its pin
    2  some check COULD NOT RUN, so the answer is unknown

2 outranks 1 deliberately: an unknown is worse than a known difference, because
a known difference can be reviewed and an unknown cannot.

# The three outcomes, and why "could not check" is never folded into "matches"

Every pin gets one of three outcomes, never two: it matches, it moved, or the
check could not run and says why. A failed lookup that reads as a clean result
is the defect this shape exists to prevent, so a network failure, a rate limit,
a withdrawn repository and a private repository are all "could not check" and
each names which.

Two further verdicts exist because the SOURCES differ and the report must not
pretend otherwise:

* ModelScope exposes tags but no commit behind a tag, so a tag pin can be
  checked only for tag existence. That is reported as its own verdict rather
  than as a match: the tag can be re-pointed upstream and this check would not
  see it.
* Cloud providers expose no revision at all, so drift there is invisible. That
  is reported as "not observable" rather than as "no drift".

Neither is an error, and neither is counted as a match. Both are permanent
properties of the source rather than failures of a run, so they do not raise the
exit code; the summary states how many entries they cover, so a reader can never
mistake the green exit for "every model was confirmed unchanged".

# How a pin's source is decided

The manifest's own "How the pins were resolved" section is the authority:
Hugging Face entries pin a commit, ModelScope entries pin a tag (because that
hub exposes no commit behind one), and cloud providers carry a provider
parameter. This script therefore routes by revision KIND.

That mapping is safe in the direction that matters. If a future Commit pin ever
lands on a non-Hugging-Face hub, this script asks Hugging Face, is refused, and
reports "could not check" with exit 2. It cannot produce a false green.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "crates" / "batchalign" / "src" / "model_manifest.rs"

HUGGING_FACE_API = "https://huggingface.co/api/models/{repo}"
MODELSCOPE_API = "https://modelscope.cn/api/v1/models/{repo}/revisions"

REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "batchalign3-model-pin-drift-check"


# ---------------------------------------------------------------------------
# The pins, read from the manifest
# ---------------------------------------------------------------------------


class PinKind(Enum):
    """How a manifest entry names its revision.

    These are the three `ManifestRevision` variants. A variant in the manifest
    that is not one of these makes the parse fail loudly rather than be skipped.
    """

    COMMIT = "Commit"
    TAG = "Tag"
    PROVIDER_PARAMETER = "ProviderParameter"


@dataclass(frozen=True, slots=True)
class Pin:
    """One pinned model, as the manifest states it.

    `api_repo` is the repository an API call addresses, which is usually the
    model id itself. The native Whisper weights are the exception: their id is
    `<owner>/<repo>/<file>`, because a repository commit alone would not say
    which of that repository's many weight files ran.
    """

    model_id: str
    api_repo: str
    kind: PinKind
    revision: str


class ManifestParseError(RuntimeError):
    """The manifest could not be read as a set of pins.

    Raised when the file's shape no longer matches what this parser understands.
    It is deliberately fatal: a parser that silently found fewer entries than the
    manifest holds would report a clean result for models it never checked,
    which is the exact failure this script exists to prevent.
    """


# The manifest is Rust source, so the pins are read by a narrow, deliberately
# strict parse rather than by a second copy of the table. A second list would
# recreate the mirrored-table defect the manifest was written to delete, and it
# would be the list that goes stale. The parse is kept honest by a cross-check
# below: if it does not recover exactly as many entries as the file contains
# entry literals, it refuses to report anything at all.
_CONST_RE = re.compile(
    r'const\s+([A-Z0-9_]+)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]*)"\s*;'
)

# An entry is `id: <literal or CONST>, revision: ManifestRevision::<Kind>(<literal or CONST>)`.
_ENTRY_RE = re.compile(
    r'id:\s*(?:"(?P<id_literal>[^"]*)"|(?P<id_const>[A-Z0-9_]+))\s*,\s*'
    r"revision:\s*ManifestRevision::(?P<kind>[A-Za-z]+)\("
    r'(?:"(?P<revision_literal>[^"]*)"|(?P<revision_const>[A-Z0-9_]+))\)',
    re.S,
)

# Every `ManifestEntry {` that constructs a value. The struct definition and the
# inherent `impl` block both mention the name without constructing anything, and
# `MalformedManifestEntry {` is a different name entirely (the lookbehind rejects
# it, because the character before `ManifestEntry` there is a letter).
_ENTRY_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_])ManifestEntry\s*\{")
_ENTRY_NON_VALUE_RE = re.compile(r"(?:struct|impl)\s+ManifestEntry\s*\{")

_NATIVE_ID_CONST = "NATIVE_WHISPER_ID"
_NATIVE_REVISION_CONST = "NATIVE_WHISPER_REVISION"


def read_pins(manifest_path: Path) -> list[Pin]:
    """Read every pin the manifest states, or raise `ManifestParseError`."""
    try:
        source = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestParseError(f"cannot read {manifest_path}: {error}") from error

    constants = dict(_CONST_RE.findall(source))

    pins: list[Pin] = []
    for match in _ENTRY_RE.finditer(source):
        model_id = _resolve(match, "id_literal", "id_const", constants, manifest_path)
        revision = _resolve(
            match, "revision_literal", "revision_const", constants, manifest_path
        )
        kind_text = match.group("kind")
        try:
            kind = PinKind(kind_text)
        except ValueError as error:
            # A new revision variant is a decision about how to check it, which
            # belongs here rather than in a silent skip.
            raise ManifestParseError(
                f"{manifest_path}: unknown revision variant "
                f"`ManifestRevision::{kind_text}` for {model_id}. Teach this "
                f"script how to check it before pinning a model with it."
            ) from error
        pins.append(
            Pin(model_id=model_id, api_repo=model_id, kind=kind, revision=revision)
        )

    _cross_check(source, len(pins), manifest_path)
    pins.append(_native_whisper_pin(constants, manifest_path))
    return pins


def _resolve(
    match: re.Match[str],
    literal_group: str,
    const_group: str,
    constants: dict[str, str],
    manifest_path: Path,
) -> str:
    """Take a manifest value written either as a literal or as a named const."""
    literal = match.group(literal_group)
    if literal is not None:
        return literal
    name = match.group(const_group)
    if name not in constants:
        raise ManifestParseError(
            f"{manifest_path}: entry refers to `{name}`, which this parser "
            f"could not resolve to a string constant."
        )
    return constants[name]


def _cross_check(source: str, parsed: int, manifest_path: Path) -> None:
    """Refuse to report unless the parse recovered every entry in the file.

    This is the guard that makes a narrow regex parse safe to rely on. The two
    counts are arrived at independently: one by parsing `id`/`revision` pairs,
    the other by counting entry literals. If the manifest grows an entry written
    in a shape this parser does not understand, the counts disagree and the
    script fails loudly instead of reporting a clean result for a model it never
    looked at.
    """
    literals = len(_ENTRY_LITERAL_RE.findall(source)) - len(
        _ENTRY_NON_VALUE_RE.findall(source)
    )
    if parsed != literals:
        raise ManifestParseError(
            f"{manifest_path}: parsed {parsed} pinned entries but the file "
            f"contains {literals} entry literals. The manifest's shape has "
            f"changed and this parser must be updated before its result can be "
            f"trusted."
        )
    if parsed == 0:
        raise ManifestParseError(
            f"{manifest_path}: no pinned entries found at all, which cannot be right."
        )


def _native_whisper_pin(constants: dict[str, str], manifest_path: Path) -> Pin:
    """The native speech model, pinned by its repository commit.

    It is stated as two consts rather than as an entry, because its id names a
    repository AND a file inside it: the repository commit says which revision,
    and the file name says which of that repository's weight files ran.
    """
    missing = [
        name
        for name in (_NATIVE_ID_CONST, _NATIVE_REVISION_CONST)
        if name not in constants
    ]
    if missing:
        raise ManifestParseError(
            f"{manifest_path}: expected {' and '.join(missing)} to be declared. "
            f"The native speech model's pin has moved or been renamed, and this "
            f"script must be updated to find it."
        )
    model_id = constants[_NATIVE_ID_CONST]
    owner, _, remainder = model_id.partition("/")
    repo_name, _, _file_name = remainder.partition("/")
    if not owner or not repo_name:
        raise ManifestParseError(
            f"{manifest_path}: {_NATIVE_ID_CONST} is {model_id!r}, which this "
            f"script cannot read as `<owner>/<repository>/<file>`."
        )
    return Pin(
        model_id=model_id,
        api_repo=f"{owner}/{repo_name}",
        kind=PinKind.COMMIT,
        revision=constants[_NATIVE_REVISION_CONST],
    )


def distinct(pins: Iterable[Pin]) -> list[Pin]:
    """Collapse pins that name the same repository at the same revision.

    Two languages can share one model deliberately; they are one upstream
    question and should be one lookup and one reported line.
    """
    seen: set[tuple[str, str, PinKind]] = set()
    unique: list[Pin] = []
    for pin in pins:
        key = (pin.api_repo, pin.revision, pin.kind)
        if key not in seen:
            seen.add(key)
            unique.append(pin)
    return unique


# ---------------------------------------------------------------------------
# The outcomes
# ---------------------------------------------------------------------------


class Severity(Enum):
    """What a verdict means for the exit code.

    The value IS the exit code contribution, and the run exits with the highest
    one seen. `UNCHECKED` outranks `DRIFTED` because an unknown is worse than a
    known difference.
    """

    NO_ACTION = 0
    DRIFTED = 1
    UNCHECKED = 2


@dataclass(frozen=True, slots=True)
class Matches:
    """Upstream is still at the pinned revision."""

    revision: str

    severity: ClassVar[Severity] = Severity.NO_ACTION
    label: ClassVar[str] = "MATCH"

    def detail(self) -> str:
        return f"upstream is still at {self.revision}"


@dataclass(frozen=True, slots=True)
class Moved:
    """Upstream has moved away from the pin. Both revisions are named."""

    pinned: str
    upstream: str

    severity: ClassVar[Severity] = Severity.DRIFTED
    label: ClassVar[str] = "MOVED"

    def detail(self) -> str:
        return f"pinned {self.pinned}; upstream now {self.upstream}"


@dataclass(frozen=True, slots=True)
class TagPresentCommitNotExposed:
    """The pinned tag exists, but the hub exposes no commit behind it.

    NOT a match. The tag existing proves only that the name is still published;
    the hub can re-point a tag at new bytes and this check cannot see that.
    """

    tag: str

    severity: ClassVar[Severity] = Severity.NO_ACTION
    label: ClassVar[str] = "TAG-ONLY"

    def detail(self) -> str:
        return (
            f"tag {self.tag} still exists, but this hub exposes no commit "
            f"behind a tag, so movement under the tag is not detectable"
        )


@dataclass(frozen=True, slots=True)
class NotObservable:
    """The source exposes no revision at all, so drift is invisible.

    NOT a match, and not a failure either: it is a permanent property of the
    provider. Reporting it as "no drift" would be a green line that means
    nothing.
    """

    why: str

    severity: ClassVar[Severity] = Severity.NO_ACTION
    label: ClassVar[str] = "NOT-OBSERVABLE"

    def detail(self) -> str:
        return self.why


@dataclass(frozen=True, slots=True)
class CouldNotCheck:
    """The check did not run to an answer, and says why."""

    why: str

    severity: ClassVar[Severity] = Severity.UNCHECKED
    label: ClassVar[str] = "UNCHECKED"

    def detail(self) -> str:
        return self.why


Verdict = Matches | Moved | TagPresentCommitNotExposed | NotObservable | CouldNotCheck


@dataclass(frozen=True, slots=True)
class Checked:
    """One pin and what the upstream said about it."""

    pin: Pin
    verdict: Verdict


# ---------------------------------------------------------------------------
# Talking to the upstreams
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FetchFailed:
    """A lookup that produced no usable body, with the reason a reader needs."""

    why: str


def fetch_json(url: str) -> dict[str, object] | FetchFailed:
    """GET one public JSON document, mapping every failure to a stated reason.

    Every branch here becomes a distinct "could not check" reason, because
    "the network was down" and "the repository is gone" call for different
    actions and must never read alike.
    """
    # The URL is always one of this module's two fixed https API endpoints with
    # a repository name substituted in, never a value from an untrusted caller.
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        return FetchFailed(_http_reason(error.code))
    except urllib.error.URLError as error:
        return FetchFailed(f"network failure: {error.reason}")
    except TimeoutError:
        return FetchFailed(
            f"network failure: no response within {REQUEST_TIMEOUT_SECONDS}s"
        )
    except OSError as error:
        return FetchFailed(f"network failure: {error}")

    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        return FetchFailed(f"upstream returned a body that is not JSON: {error}")
    if not isinstance(parsed, dict):
        return FetchFailed("upstream returned JSON that is not an object")
    return parsed


def _http_reason(code: int) -> str:
    """Say what an HTTP status means for a pin, without overclaiming.

    401 is the important one. An unauthenticated lookup gets the same refusal
    for a repository that was renamed, withdrawn, or made private, so this names
    all three rather than picking one and being wrong two thirds of the time.
    """
    if code in (401, 403):
        return (
            f"HTTP {code}: refused without credentials. The repository is "
            f"renamed, withdrawn, or private, and an unauthenticated lookup "
            f"cannot tell those apart"
        )
    if code == 404:
        return f"HTTP {code}: upstream has no such repository (renamed or withdrawn)"
    if code == 429:
        return f"HTTP {code}: rate limited by the upstream; retry later"
    if 500 <= code < 600:
        return f"HTTP {code}: upstream server error"
    return f"HTTP {code}: unexpected response from upstream"


def check_hugging_face(pin: Pin) -> Verdict:
    """Compare a commit pin against the repository's current commit."""
    document = fetch_json(HUGGING_FACE_API.format(repo=pin.api_repo))
    if isinstance(document, FetchFailed):
        return CouldNotCheck(document.why)
    sha = document.get("sha")
    if not isinstance(sha, str) or not sha:
        return CouldNotCheck(
            "the API response carried no `sha`, so the upstream commit is unknown"
        )
    if sha == pin.revision:
        return Matches(sha)
    return Moved(pinned=pin.revision, upstream=sha)


def check_modelscope(pin: Pin) -> Verdict:
    """Check that a tag pin still names a published tag.

    Tag existence is all this hub allows. It publishes no commit behind a tag,
    so a matching tag is reported as a limit rather than as a confirmed match.
    A tag that has DISAPPEARED is real, detectable drift: the pin names a
    revision upstream no longer publishes.
    """
    document = fetch_json(MODELSCOPE_API.format(repo=pin.api_repo))
    if isinstance(document, FetchFailed):
        return CouldNotCheck(document.why)

    data = document.get("Data")
    revision_map = data.get("RevisionMap") if isinstance(data, dict) else None
    raw_tags = revision_map.get("Tags") if isinstance(revision_map, dict) else None
    if raw_tags is None:
        return CouldNotCheck(
            "the API response carried no tag list, so tag existence is unknown"
        )
    if not isinstance(raw_tags, list):
        return CouldNotCheck("the API response carried a tag list of an unknown shape")

    tags = [
        entry["Revision"]
        for entry in raw_tags
        if isinstance(entry, dict) and isinstance(entry.get("Revision"), str)
    ]
    if pin.revision in tags:
        return TagPresentCommitNotExposed(pin.revision)
    published = ", ".join(tags) if tags else "no tags at all"
    return Moved(
        pinned=pin.revision,
        upstream=f"has no such tag; it publishes {published}",
    )


def check(pin: Pin) -> Verdict:
    """Route one pin to the source that can answer for it."""
    if pin.kind is PinKind.COMMIT:
        return check_hugging_face(pin)
    if pin.kind is PinKind.TAG:
        return check_modelscope(pin)
    return NotObservable(
        f"a cloud provider exposes no revision, so drift in {pin.model_id} "
        f"(selector {pin.revision!r}) cannot be observed from outside"
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(results: Sequence[Checked]) -> str:
    """Render one line per pin, then a summary that separates the outcomes."""
    id_width = max(len(result.pin.model_id) for result in results)
    label_width = max(len(result.verdict.label) for result in results)
    lines = [
        f"{result.pin.model_id:<{id_width}}  "
        f"{result.verdict.label:<{label_width}}  {result.verdict.detail()}"
        for result in results
    ]

    counts: dict[str, int] = {}
    for result in results:
        counts[result.verdict.label] = counts.get(result.verdict.label, 0) + 1
    summary = ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
    lines.append("")
    lines.append(f"{len(results)} pinned models: {summary}")

    # The legend is not decoration. A reader who takes TAG-ONLY or
    # NOT-OBSERVABLE for a confirmed match would read this report as proving
    # something it does not prove.
    if any(
        isinstance(result.verdict, TagPresentCommitNotExposed) for result in results
    ):
        lines.append(
            "  TAG-ONLY means the pinned tag still exists and nothing more; "
            "that hub publishes no commit behind a tag."
        )
    if any(isinstance(result.verdict, NotObservable) for result in results):
        lines.append(
            "  NOT-OBSERVABLE means the provider publishes no revision at all; "
            "it is not a statement that nothing changed."
        )
    return "\n".join(lines)


def exit_code(results: Iterable[Checked]) -> int:
    """The highest severity any pin reached.

    Derived from the verdicts themselves rather than from flags set along the
    way, so a new verdict cannot be added without deciding what it means for the
    exit code.
    """
    return max(result.verdict.severity.value for result in results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check the pinned models in the manifest against their upstreams. "
            "Queries public APIs only: no credentials, no writes, no downloads, "
            "and no local model cache is consulted."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "the manifest to read pins from "
            "(default: crates/batchalign/src/model_manifest.rs)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        pins = distinct(read_pins(args.manifest))
    except ManifestParseError as error:
        # Exit 2: nothing was checked, so the answer is unknown rather than clean.
        print(f"could not read the pins: {error}", file=sys.stderr)
        return Severity.UNCHECKED.value

    results = [Checked(pin=pin, verdict=check(pin)) for pin in pins]
    print(render(results))
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
