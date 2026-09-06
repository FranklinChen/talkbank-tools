#!/usr/bin/env bash
# Verify the receipt from make gate against the working tree and pushed objects.
# This hook does no compilation. Install with make install-hooks.
# The protocol is shared with Chatter and tb and exercised by test-gate-receipts.sh.
set -euo pipefail

# The stamp hashes working-tree CONTENT, not staging or commit state.
# Committing the same bytes preserves it; changing any included file after
# the gate requires a fresh gate. See tree-stamp.sh for the ownership of that
# content calculation.
#
# `tree-stamp.sh` exits non-zero rather than printing an empty stamp; under
# `set -e` that aborts the push here, which is the point. Its first version
# printed nothing on failure and this hook then compared "" to "" and passed.
# Keep Git's exact ref stream for verification and the optional local hook.
refs_dir="$(mktemp -d)"
trap 'rm -r "$refs_dir"' EXIT
cat > "$refs_dir/refs"

TREE_STATE="$(bash "$(git rev-parse --show-toplevel)/scripts/tree-stamp.sh")"

# `git rev-parse --git-dir`, never a literal `.git`: in a worktree `.git` is a
# file, and the gate writes its stamp to the same resolved directory.
stamp="$(git rev-parse --git-dir)/gate-passed"
if [ ! -f "$stamp" ]; then
    echo "[pre-push] REFUSED: no stamp from 'make gate'." >&2
    exit 1
fi
got="$(cat "$stamp")"
if [ "$got" != "$TREE_STATE" ]; then
    echo "[pre-push] REFUSED: the gate stamp is for a different tree." >&2
    echo "           stamped: $got" >&2
    echo "           pushing: $TREE_STATE" >&2
    exit 1
fi

# Working-tree equality alone cannot authorize a different committed tree.
# Annotated release tags are peeled by ^{tree}; deletion refs contain no tree.
while read -r local_ref local_oid remote_ref remote_oid extra || [[ -n "$local_ref" ]]; do
    if [[ -z "$local_ref" || -z "$local_oid" || -z "$remote_ref" || -z "$remote_oid" || -n "$extra" ]]; then
        echo "[pre-push] REFUSED: malformed Git ref update." >&2
        exit 1
    fi
    if [[ ! "$local_oid" =~ ^[0-9a-f]{40}([0-9a-f]{24})?$ ]]; then
        echo "[pre-push] REFUSED: invalid local object ID." >&2
        exit 1
    fi
    if [[ "$local_oid" == "${TREE_STATE//?/0}" ]]; then
        continue
    fi
    if ! pushed_tree="$(git rev-parse --verify "$local_oid^{tree}")"; then
        echo "[pre-push] REFUSED: $local_ref does not resolve to a commit tree." >&2
        exit 1
    fi
    if [[ "$pushed_tree" != "$TREE_STATE" ]]; then
        echo "[pre-push] REFUSED: $local_ref contains bytes that were not gated." >&2
        echo "           gated:   $TREE_STATE" >&2
        echo "           pushing: $pushed_tree" >&2
        exit 1
    fi
done < "$refs_dir/refs"

echo "[pre-push] the gate stamp matches the working tree and every pushed tree."

# Chain to an optional local pre-push hook, AFTER the gate stamp, so a stamp
# failure short-circuits first and a contributor's own check never masks it.
#
# Replay the unchanged ref bytes after verification. Calling rather than
# execing lets the EXIT trap remove the temporary ref stream.
#
# Keep this free of any specific local hook's identity. It is tracked and
# public; what an individual checks on their own machine is theirs.
LOCAL_HOOK="$(git rev-parse --git-dir)/hooks/pre-push.local"
if [ -x "$LOCAL_HOOK" ]; then
    "$LOCAL_HOOK" "$@" < "$refs_dir/refs"
fi
