"""Batchalign exception hierarchy.

Defines the structured error types raised throughout the pipeline, CLI, and
server subsystems.  All exceptions carry enough context for callers to produce
user-facing diagnostics or structured JSON error responses.

Error Hierarchy
---------------
::

    Exception
    +-- SkipFileWarning          # Graceful skip (file passes through unchanged)
    +-- CHATValidationException  # Structural CHAT problems (parse or validation)
    +-- DocumentValidationException  # Non-CHAT document payload problems
    +-- ConfigNotFoundError      # Required config file missing on disk
    +-- ConfigError              # Config present but semantically invalid

The :func:`classify_error` helper maps arbitrary exceptions to one of five
user-facing categories (``validation``, ``input``, ``media``, ``system``,
``processing``) for CLI progress display and server job status reporting.

Key Exports
-----------
SkipFileWarning
    Raised when a file should be skipped with a warning, not failed.
CHATValidationException
    Raised when CHAT validation detects structural problems.
DocumentValidationException
    Raised when validating a non-CHAT document payload fails.
ConfigNotFoundError
    Raised when required Batchalign config files are missing.
ConfigError
    Raised for syntactically present but semantically invalid config.
classify_error
    Map an exception to a user-facing error category string.
"""

from typing import TypedDict


class ValidationErrorEntry(TypedDict, total=False):
    """One structured validation error from ``validate_structured()``."""

    code: str
    severity: str
    line: int
    column: int
    message: str
    suggestion: str


class SkipFileWarning(Exception):
    """Raised when a file should be skipped with a warning, not failed.

    Unlike ``CHATValidationException`` (which signals a processing failure),
    ``SkipFileWarning`` signals a graceful skip: the file passes through to
    output unchanged and a warning is logged.  No error is recorded.

    Attributes
    ----------
    chat_text : str | None
        The raw CHAT text to copy to output.  ``None`` when the content
        is not available at the raise site (e.g. the server path, where
        the caller already holds the content).
    """

    def __init__(self, message: str, chat_text: str | None = None) -> None:
        """Initialize with a warning message and optional raw CHAT text.

        Parameters
        ----------
        message : str
            Human-readable reason for skipping the file.
        chat_text : str or None
            The raw CHAT text to copy to output unchanged.  ``None`` when the
            content is not available at the raise site (e.g. the server path,
            where the caller already holds the content).
        """
        super().__init__(message)
        self.chat_text: str | None = chat_text


class CHATValidationException(Exception):
    """Raised when CHAT validation detects structural problems.

    Attributes
    ----------
    errors : list[ValidationErrorEntry]
        Structured error entries from ``validate_structured()``.
        Each dict has keys: code, severity, line, column, message, suggestion.
        Empty list for exceptions raised with a plain string message.
    """

    def __init__(self, message: str,
                 errors: list[ValidationErrorEntry] | None = None,
                 bug_report_id: str | None = None) -> None:
        """Initialize with a summary message and optional structured errors.

        Parameters
        ----------
        message : str
            Human-readable summary of the validation failure.
        errors : list[ValidationErrorEntry] or None
            Structured error entries from ``validate_structured()``.  Each dict
            has keys: ``code``, ``severity``, ``line``, ``column``,
            ``message``, ``suggestion``.  Empty list when raised with a plain
            string message.
        bug_report_id : str or None
            Unique identifier for an automatically filed bug report.  Present
            only when the validation failure is a pipeline bug (not user input
            error).  Used by :func:`classify_error` to distinguish
            ``"validation"`` from ``"input"`` category.
        """
        super().__init__(message)
        self.errors: list[ValidationErrorEntry] = errors or []
        self.bug_report_id: str | None = bug_report_id

class DocumentValidationException(Exception):
    """Raised when validating a non-CHAT document payload fails.

    Used by the server's content-mode submission path to reject payloads that
    are neither valid CHAT nor a recognized alternative format.  Carries only
    a plain string message (no structured error list).
    """
    pass

class ConfigNotFoundError(Exception):
    """Raised when required Batchalign config files are missing from disk.

    Typical trigger: a pipeline engine needs a configuration file (e.g.
    ``server.yaml`` or a Rev.AI API key file) that does not
    exist at the expected path.  The message includes the missing path.
    """
    pass

class ConfigError(Exception):
    """Raised for syntactically present but semantically invalid configuration.

    Covers cases where a config file exists and is parseable (valid YAML/JSON)
    but contains values that violate business rules -- for example, an unknown
    engine name, a negative port number, or missing required fields.
    """
    pass


def classify_error(exc: BaseException) -> str:
    """Classify an exception into a user-facing error category.

    Used by the CLI progress display and the server's job status reporting to
    assign a human-readable category to each per-file failure.

    Parameters
    ----------
    exc : BaseException
        The exception to classify.

    Returns
    -------
    str
        One of:

        - ``"validation"`` -- pipeline-produced validation bug (the exception
          is a ``CHATValidationException`` with a ``bug_report_id``).
        - ``"input"`` -- malformed CHAT input that the user should fix.
        - ``"media"`` -- missing audio/video file or filesystem path error.
        - ``"system"`` -- memory exhaustion or other infrastructure failure.
        - ``"processing"`` -- catch-all for all other processing failures.
    """
    if isinstance(exc, CHATValidationException):
        if getattr(exc, "bug_report_id", None) is not None:
            return "validation"
        return "input"

    if isinstance(exc, ValueError) and ("CHAT" in str(exc) or "Parse error" in str(exc)):
        return "input"

    if isinstance(exc, FileNotFoundError):
        return "media"

    if isinstance(exc, MemoryError):
        return "system"

    return "processing"
