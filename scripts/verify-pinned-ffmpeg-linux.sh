#!/usr/bin/env bash
# Prove scripts/install-pinned-ffmpeg.sh builds the pinned release on the CI
# runner's distribution, before CI is the first place it runs.
#
# Builds the pin in an Ubuntu 24.04 container (the runner's release), runs the
# pin check against the result, and decodes the fixture that first exposed the
# version skew: a 20.3 s AAC tone in M4A, which ffmpeg 6.1 decodes 17.5 ms long
# and the pinned release must decode to exactly 20300 ms. Run it when bumping
# the pin. Needs a running Docker engine (for example `colima start`).
#
# Usage: scripts/verify-pinned-ffmpeg-linux.sh

set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"

# The single-quoted body runs inside the container, so nothing in it may
# expand here.
# shellcheck disable=SC2016
docker run --rm -v "$repo/scripts:/pin:ro" ubuntu:24.04 bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq build-essential nasm pkg-config libmp3lame-dev curl xz-utils perl > /dev/null
    bash /pin/install-pinned-ffmpeg.sh /opt/ffmpeg-pinned > /tmp/build.log 2>&1 \
        || { tail -40 /tmp/build.log; exit 1; }
    export PATH="/opt/ffmpeg-pinned/bin:$PATH"
    bash /pin/check-ffmpeg-pin.sh
    ffmpeg -nostdin -v error -y -f lavfi \
        -i "sine=frequency=440:sample_rate=44100:duration=20300ms" -acodec aac /tmp/aac.m4a
    stated="$(ffprobe -v error -select_streams a:0 -show_entries stream=duration -of csv=p=0 /tmp/aac.m4a)"
    bytes="$(ffmpeg -nostdin -v error -i /tmp/aac.m4a -ac 1 -f s16le -acodec pcm_s16le - | wc -c)"
    decoded_ms=$(( (bytes / 2 * 1000 + 44099) / 44100 ))
    echo "aac.m4a: stream states ${stated} s, decodes to ${decoded_ms} ms"
    if (( decoded_ms != 20300 )); then
        echo "ERROR: the pinned release decodes the AAC tone to ${decoded_ms} ms, not 20300" >&2
        exit 1
    fi
'
