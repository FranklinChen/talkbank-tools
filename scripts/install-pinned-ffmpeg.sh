#!/usr/bin/env bash
# Build the pinned ffmpeg release from ffmpeg.org's source tarball into PREFIX.
#
# Why from source: no Linux binary distribution offers an exact, durable
# release. Ubuntu packages one old release per distribution; the static-build
# sites publish release-branch heads and prune them. ffmpeg.org keeps every
# release tarball, and the pin names its hash, so this build is the pinned
# release and nothing else. CI caches PREFIX by the pin, so it builds once.
#
# The configuration is ffmpeg's defaults (every native codec, demuxer and
# lavfi source the tests use) plus libmp3lame, the one external encoder the
# MP3 tests need. Build prerequisites: a C toolchain, nasm, pkg-config and
# libmp3lame's headers.
#
# Usage: scripts/install-pinned-ffmpeg.sh <prefix>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <prefix>" >&2
    exit 2
fi
prefix="$1"

here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/ffmpeg-pin.env
source "$here/ffmpeg-pin.env"

work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT

tarball="ffmpeg-${FFMPEG_VERSION}.tar.xz"
curl -fsSL --retry 3 -o "$work/$tarball" "https://ffmpeg.org/releases/$tarball"
# A tarball that is not the pinned one stops the build before any of it runs.
echo "$FFMPEG_SOURCE_SHA256  $work/$tarball" | shasum -a 256 -c -

tar -xJf "$work/$tarball" -C "$work"
cd "$work/ffmpeg-$FFMPEG_VERSION" || exit 1
./configure --prefix="$prefix" --enable-gpl --enable-libmp3lame \
    --disable-doc --disable-ffplay --disable-debug
make -j"$(getconf _NPROCESSORS_ONLN)"
make install
