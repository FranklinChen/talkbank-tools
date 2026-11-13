"""The installed Stanza must be one whose behaviour we have adjudicated.

This is the check that turns the "Stanza version confirmed" column in the
defect-mitigation map from a note into something that fires. Without it, a bump
inherits every recorded confirmation for free and the only evidence that
anything changed is whichever golden tests happen to notice.

Deliberately NOT marked `golden`: it must run in the default suite, since its
whole job is to be seen immediately after someone edits the version constraint,
rather than during a model-downloading run they may not think to do.

This is a POLICY test, not an invariant a type could carry: "we have looked at
this version's behaviour" is a claim about work performed by people, and no
signature can express it. The type does as much as a type can, by making the
comparison total and correctly ordered; the rest is the record.
"""

from __future__ import annotations

from batchalign.tests._stanza_confirmation import (
    BUMP_PROCEDURE,
    CONFIRMED_STANZA_VERSIONS,
    StanzaVersion,
)


def test_installed_stanza_version_has_been_adjudicated() -> None:
    """Guards: The installed dependency has an explicit human-reviewed qualification entry.
    Kind: policy
    Not a type: A parsed semantic version cannot prove that reviewers attributed real model outputs for that installed release and resource population.
    State: Installed package metadata supplies the actual runtime version under review.
    Concurrency: Read-only package metadata and an immutable confirmation registry are shared safely.
    """
    installed = StanzaVersion.installed()
    # Sorted as VERSIONS, not as strings: "1.10.1" sorts before "1.9.0".
    confirmed = ", ".join(str(v) for v in sorted(CONFIRMED_STANZA_VERSIONS))
    assert installed in CONFIRMED_STANZA_VERSIONS, (
        f"Stanza {installed} is installed, and no one has recorded adjudicating "
        f"it. Confirmed: {confirmed}.\n\n"
        "This is not a request to pin an old version. Bumping is expected and "
        f"wanted. It is a request to finish the bump:\n\n{BUMP_PROCEDURE}"
    )
