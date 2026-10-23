"""Provider refusals whose identity must survive the worker boundary."""

from __future__ import annotations


class ProviderQuotaExhaustedError(RuntimeError):
    """The provider requires quota/resource-pack intervention before retrying.

    Adapters raise this only from a structured provider error code. The Rust
    bridge recognizes the exception class, never words in its message.
    """
