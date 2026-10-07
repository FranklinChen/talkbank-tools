"""Typed classification of Hugging Face Hub access/credential failures.

A pinned model artifact can fail to download for reasons that are a
configuration problem on the OPERATOR's machine, not a defect in batchalign:
the artifact's repository requires accepting license terms, the operator has
no Hub token, the token lacks access, or the Hub is unreachable and no cached
copy exists. Left unclassified, that failure surfaces at the worker boundary
as an undifferentiated runtime crash, which the server reports as a
generic/validation-class failure, which the dashboard used to label a
pipeline bug (it now labels validation "Input Refused"), even though neither
batchalign nor the input is at fault.

:func:`classify_huggingface_access_error` maps the specific
``huggingface_hub`` exception CLASSES raised for those cases (never their
free-form message text, which can be reworded between releases without
notice) into one typed :class:`ModelAccessDeniedError`. A caller that catches
this exception, rather than the underlying Hub exception, gets a stable
identity the Rust worker boundary can recognize via
``pyo3::import_exception!`` without inspecting anything huggingface_hub owns.
"""

from __future__ import annotations

import re


class ModelAccessDeniedError(RuntimeError):
    """A pinned Hugging Face Hub artifact refused this machine's request.

    ``resource_url`` is the exact Hub resource whose download failed.
    ``repo_id`` is derived from it (``owner/name``) when the URL has the
    expected shape, and names the repository an operator would need to visit
    to accept its terms; it is ``None`` when the URL could not be parsed that
    way, in which case the message falls back to the raw resource string
    rather than fabricating a repo id.
    """

    def __init__(self, resource_url: str, reason: str) -> None:
        self.resource_url = resource_url
        self.reason = reason
        self.repo_id = _repo_id_from_hub_url(resource_url)
        super().__init__(_render_message(self.repo_id, resource_url, reason))


def _repo_id_from_hub_url(url: str) -> str | None:
    """Extract an ``owner/name`` repo id from a Hub resource URL, if shaped that way."""

    match = re.search(r"huggingface\.co/([^/\s]+/[^/\s]+?)(?:/resolve/|/blob/|$)", url)
    return match.group(1) if match else None


def _render_message(repo_id: str | None, resource_url: str, reason: str) -> str:
    target = repo_id or resource_url
    accept_terms = (
        f"Visit https://huggingface.co/{repo_id} to accept its terms, then "
        if repo_id is not None
        else ""
    )
    remedy = (
        f"{accept_terms}set hf_token under [auth] in ~/.batchalign.ini (or run "
        "`hf auth login`), or choose a different --speaker-engine."
    )
    return f"could not download the Hugging Face model at {target}: {reason}. {remedy}"


def classify_huggingface_access_error(
    error: Exception,
) -> ModelAccessDeniedError | None:
    """Translate a huggingface_hub exception into a typed access failure.

    Returns ``None`` for any exception that is not a Hub access/credential
    failure, so the caller re-raises the original exception untouched.
    Classification is by EXCEPTION CLASS (and, for a bare ``HfHubHTTPError``,
    its HTTP status code) only, never by parsing the exception's message.
    """

    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    if isinstance(error, GatedRepoError):
        return ModelAccessDeniedError(
            _hub_error_resource(error),
            "its repository is gated and requires accepted terms",
        )
    if isinstance(error, RepositoryNotFoundError):
        return ModelAccessDeniedError(
            _hub_error_resource(error),
            "its repository was not found, or is private and this machine has no access",
        )
    if isinstance(error, LocalEntryNotFoundError):
        return ModelAccessDeniedError(
            str(error), "no cached copy exists and the Hugging Face Hub is unreachable"
        )
    if isinstance(error, HfHubHTTPError):
        status = error.response.status_code if error.response is not None else None
        if status in (401, 403):
            return ModelAccessDeniedError(
                _hub_error_resource(error), f"the Hub returned HTTP {status}"
            )
    return None


def _hub_error_resource(error: Exception) -> str:
    """The resource URL a Hub HTTP error names, if its response carries one."""

    response = getattr(error, "response", None)
    if response is not None and getattr(response, "url", None):
        return str(response.url)
    return str(error)


__all__ = [
    "ModelAccessDeniedError",
    "classify_huggingface_access_error",
]
