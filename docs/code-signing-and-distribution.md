# Code Signing and Distribution

**Status:** Current
**Last updated:** 2026-09-06

Batchalign distribution and signing claims are defined in the
[canonical release contract](../book/src/batchalign/developer/release-contract.md#distribution-and-signing).
The release workflow is `.github/workflows/batchalign-release.yml`.

BA3 ships five wheels, one source distribution, shell and PowerShell installers,
and a SHA-256 manifest through GitHub Releases. It is not published to PyPI.
Artifacts are not currently code-signed or notarized. Checksums establish
integrity against the manifest, not independent publisher authentication.

The experimental desktop is outside the supported release. Adding native GUI
installers requires an explicit surface-specific signing policy and automation
before public distribution. Do not claim signing, notarization or platform
trust unless the relevant workflow performs and verifies it.

Use the [release checklist](../book/src/batchalign/developer/release-checklist.md)
for artifact, installer and packaged server-health verification. Chatter's
artifacts and signing policy belong to
[its separate repository](https://github.com/TalkBank/chatter).
