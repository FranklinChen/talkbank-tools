#!/usr/bin/env bash
# Refuse to run the media tests against any ffmpeg but the pinned release.
#
# Runs first in `make batchalign-ci-rust`, which both CI and the local pre-push
# gate invoke, so the gate and CI test against the same decoder. A different
# release on PATH (a Homebrew upgrade, a distribution package) fails here with
# the reason, instead of later as a test whose expected length moved.
#
# Usage: scripts/check-ffmpeg-pin.sh

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/ffmpeg-pin.env
source "$here/ffmpeg-pin.env"

for tool in ffmpeg ffprobe; do
    if ! banner="$("$tool" -version 2>/dev/null)"; then
        echo "ERROR: $tool is not runnable; the media tests need ffmpeg $FFMPEG_VERSION." >&2
        exit 1
    fi
    first_line="${banner%%$'\n'*}"
    if [[ "$first_line" != "$tool version $FFMPEG_VERSION "* ]]; then
        echo "ERROR: $tool on PATH reports '$first_line'," >&2
        echo "  but the media tests are pinned to $FFMPEG_VERSION (scripts/ffmpeg-pin.env)." >&2
        echo "  Install that release (scripts/install-pinned-ffmpeg.sh builds it), or" >&2
        echo "  re-measure the media tests against the new release and bump the pin." >&2
        exit 1
    fi
    echo "$tool $FFMPEG_VERSION (pinned)"
done
