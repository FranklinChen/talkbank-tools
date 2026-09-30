# shellcheck shell=bash
# The one ffmpeg release every media test is measured against.
#
# The media tests generate audio with ffmpeg's encoders and compare what
# ffprobe states and what ffmpeg decodes, so their answers are properties of
# the RELEASE, not of the code under test. ffmpeg 6.1 (Ubuntu 24.04's package)
# decodes an AAC-in-MP4 tone 17.5 ms longer than ffmpeg 9.0 does, because it
# ignores the edit list's duration; tests that passed on the developer's 9.0
# failed on the CI runner's 6.1 (2026-09-30). One pinned release, checked
# before the tests run locally and in CI, makes those answers deterministic.
#
# Bumping: re-run the media tests against the new release, re-measure anything
# they report, then change both lines together. The hash is of ffmpeg.org's
# release tarball, taken from a download whose signature by the FFmpeg release
# key (FCF986EA15E6E293A5644F10B4322F04D67658D8) verified.
#
# Sourced (never run) by scripts/check-ffmpeg-pin.sh (the gate) and
# scripts/install-pinned-ffmpeg.sh (CI's build of exactly this release), which
# is why the assignments below are unused within this file.
# shellcheck disable=SC2034  # read by the scripts that source this file
FFMPEG_VERSION=9.0.2
# shellcheck disable=SC2034  # read by the scripts that source this file
FFMPEG_SOURCE_SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
