# Release Contract

**Status:** Current
**Last updated:** 2026-09-06

The canonical Batchalign contract is
[Release Contract in the public book](../book/src/batchalign/developer/release-contract.md).
It owns the preview classification, supported CLI/server/dashboard surfaces,
internal Python and Rust APIs, evidence guarantees, and distribution policy.
This entry point is retained for existing links; it does not maintain a second
surface table.

The CHAT core belongs to [TalkBank/chatter](https://github.com/TalkBank/chatter),
whose release authority is independent. This repository consumes released
Chatter crates and ships BA3 through GitHub Releases, never PyPI.

See [Versioning](VERSIONING.md) for canonical version sources and bump rules,
[Platform Support](PLATFORM-SUPPORT.md) for the platform-policy entry point,
and the [release checklist](../book/src/batchalign/developer/release-checklist.md)
for required gates and artifact verification.
