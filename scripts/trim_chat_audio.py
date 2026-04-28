#!/usr/bin/env python3
"""Trim a CHAT file and its audio to a range of main-tier utterance lines.

Creates a minimal but valid CHAT file containing only the specified utterance
range (with headers preserved), plus the corresponding audio segment. Useful
for isolating problem regions for algorithmic comparison.

Usage:
    python3 trim_chat_audio.py INPUT.cha --lines 50-80 --output trimmed/
    python3 trim_chat_audio.py INPUT.cha --lines 50-80 --context 5 --output trimmed/

The --lines range refers to main-tier lines (*SPK:), not raw file line numbers.
Use --context N to include N extra utterances before and after the range.

Audio trimming uses the timing from a reference CHAT file (the aligned output)
to determine the time range. If the input has no timing, pass --ref ALIGNED.cha
to extract timestamps from the aligned version.

Requirements: ffmpeg (for audio trimming)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

BULLET = "\x15"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Input CHAT file")
    p.add_argument("--lines", required=True, help="Utterance range, e.g. 20-40 (1-based, main-tier lines only)")
    p.add_argument("--context", type=int, default=0, help="Extra utterances before/after range")
    p.add_argument("--ref", help="Reference aligned CHAT for timing (if input is untimed)")
    p.add_argument("--audio", help="Audio file (default: inferred from @Media header)")
    p.add_argument("--output", default=".", help="Output directory")
    p.add_argument("--padding-ms", type=int, default=2000, help="Audio padding before/after (ms)")
    return p.parse_args()


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def find_main_tier_indices(lines):
    """Return list of (line_index, utterance_number) for main-tier lines."""
    result = []
    utt_num = 0
    for i, line in enumerate(lines):
        if line.startswith("*"):
            utt_num += 1
            result.append((i, utt_num))
    return result


def extract_timing(line):
    """Extract (start_ms, end_ms) from a line with bullet markers, or None."""
    m = re.search(BULLET + r"(\d+)_(\d+)" + BULLET, line)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def find_time_range(lines, main_indices, start_utt, end_utt):
    """Find the audio time range covering utterances start_utt..end_utt.

    Scans the selected utterances (and their continuation/dependent lines)
    for bullet timing. Returns (start_ms, end_ms) or None.
    """
    min_ms = float("inf")
    max_ms = 0

    for line_idx, utt_num in main_indices:
        if utt_num < start_utt or utt_num > end_utt:
            continue
        # Check main line and continuation lines
        j = line_idx
        while j < len(lines):
            t = extract_timing(lines[j])
            if t:
                min_ms = min(min_ms, t[0])
                max_ms = max(max_ms, t[1])
            j += 1
            if j >= len(lines) or (lines[j].startswith("*") or lines[j].startswith("@")):
                break

    if min_ms == float("inf"):
        return None
    return int(min_ms), int(max_ms)


def extract_headers(lines):
    """Extract header lines (@-lines before @G or first *)."""
    headers = []
    for line in lines:
        if line.startswith("@") and not line.startswith("@G:"):
            headers.append(line)
        elif line.startswith("*"):
            break
        elif line.startswith("@G:"):
            break
    return headers


def find_audio_file(lines, input_dir):
    """Find audio file from @Media header."""
    for line in lines:
        if line.startswith("@Media:"):
            parts = line.split("\t", 1)
            if len(parts) > 1:
                media = parts[1].strip().split(",")[0].strip()
                for ext in [".mp3", ".wav", ".mp4", ".m4a", ".flac"]:
                    candidate = input_dir / (media + ext)
                    if candidate.exists():
                        return candidate
    return None


def rebase_timing(lines, offset_ms):
    """Subtract offset_ms from all timing bullets so they're relative to trimmed audio."""
    rebased = []
    for line in lines:
        def rebase_match(m):
            start = max(0, int(m.group(1)) - offset_ms)
            end = max(0, int(m.group(2)) - offset_ms)
            return f"{BULLET}{start}_{end}{BULLET}"

        new_line = re.sub(BULLET + r"(\d+)_(\d+)" + BULLET, rebase_match, line)
        # Also handle bare timing (no bullet markers) at end of main tier lines
        if line.startswith("*") and not BULLET in line:
            new_line = re.sub(r"\b(\d{4,})_(\d{4,})\b",
                              lambda m: f"{max(0, int(m.group(1)) - offset_ms)}_{max(0, int(m.group(2)) - offset_ms)}",
                              new_line)
        rebased.append(new_line)
    return rebased


def build_trimmed_chat(lines, main_indices, start_utt, end_utt):
    """Build a trimmed CHAT file with headers + selected utterances."""
    headers = extract_headers(lines)

    # Find line range: from first selected utterance to last selected + its deps
    first_line = None
    last_line = None
    for line_idx, utt_num in main_indices:
        if start_utt <= utt_num <= end_utt:
            if first_line is None:
                first_line = line_idx
            last_line = line_idx

    if first_line is None:
        return None

    # Include @G line before first utterance if present
    g_lines = []
    for i in range(max(0, first_line - 5), first_line):
        if lines[i].startswith("@G:"):
            g_lines.append(lines[i])

    # Extend last_line to include continuation/dependent lines
    j = last_line + 1
    while j < len(lines) and not lines[j].startswith("*") and not lines[j].startswith("@"):
        j += 1
    last_line = j - 1

    body = lines[first_line:last_line + 1]

    result = headers + g_lines + body
    if not any(l.strip() == "@End" for l in result):
        result.append("@End\n")

    return result


def trim_audio(audio_path, start_ms, end_ms, padding_ms, output_path):
    """Trim audio using ffmpeg."""
    start_s = max(0, (start_ms - padding_ms)) / 1000.0
    end_s = (end_ms + padding_ms) / 1000.0
    duration = end_s - start_s

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration:.3f}",
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Retry with re-encoding if copy fails
        cmd[-2] = "-acodec"
        cmd[-1] = "libmp3lame"
        cmd.append(str(output_path))
        subprocess.run(cmd, capture_output=True, text=True, check=True)


def main():
    args = parse_args()

    input_path = Path(args.input)
    input_dir = input_path.parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse utterance range
    parts = args.lines.split("-")
    start_utt = int(parts[0])
    end_utt = int(parts[1]) if len(parts) > 1 else start_utt

    # Apply context
    start_utt = max(1, start_utt - args.context)
    end_utt = end_utt + args.context

    # Read input
    input_lines = read_lines(input_path)
    main_indices = find_main_tier_indices(input_lines)
    total_utts = len(main_indices)
    end_utt = min(end_utt, total_utts)

    print(f"Input: {input_path.name} ({total_utts} utterances)")
    print(f"Selecting utterances {start_utt}-{end_utt} ({end_utt - start_utt + 1} utterances)")

    # Determine time range from reference or input
    ref_path = Path(args.ref) if args.ref else input_path
    ref_lines = read_lines(ref_path) if args.ref else input_lines
    ref_indices = find_main_tier_indices(ref_lines)

    time_range = find_time_range(ref_lines, ref_indices, start_utt, end_utt)
    if time_range is None:
        print("WARNING: no timing found in reference — audio will not be trimmed")
    else:
        start_ms, end_ms = time_range
        print(f"Time range: {start_ms}ms - {end_ms}ms ({(end_ms - start_ms) / 1000:.1f}s)")

    # Build trimmed CHAT
    trimmed = build_trimmed_chat(input_lines, main_indices, start_utt, end_utt)
    if trimmed is None:
        print("ERROR: no utterances found in range")
        sys.exit(1)

    # Rebase timing bullets: subtract the audio trim start offset so bullets
    # are relative to the trimmed audio (which starts at 0).
    if time_range:
        audio_start_ms = max(0, start_ms - args.padding_ms)
        trimmed = rebase_timing(trimmed, audio_start_ms)

    stem = input_path.stem
    chat_out = output_dir / f"{stem}-trimmed.cha"
    with open(chat_out, "w", encoding="utf-8") as f:
        f.writelines(trimmed)
    utt_count = sum(1 for l in trimmed if l.startswith("*"))
    print(f"Wrote {chat_out.name} ({utt_count} utterances)")

    # Update @Media header to reference trimmed audio
    # (handled by user if needed)

    # Trim audio
    if time_range:
        audio_path = Path(args.audio) if args.audio else find_audio_file(input_lines, input_dir)
        if audio_path and audio_path.exists():
            audio_out = output_dir / f"{stem}-trimmed{audio_path.suffix}"
            trim_audio(audio_path, start_ms, end_ms, args.padding_ms, audio_out)
            print(f"Wrote {audio_out.name} ({(end_ms - start_ms + 2 * args.padding_ms) / 1000:.1f}s)")

            # Update @Media in trimmed CHAT
            new_lines = []
            for line in open(chat_out, encoding="utf-8"):
                if line.startswith("@Media:"):
                    media_type = line.split(",")[-1].strip() if "," in line else "audio"
                    new_lines.append(f"@Media:\t{stem}-trimmed, {media_type}\n")
                else:
                    new_lines.append(line)
            with open(chat_out, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        else:
            print("WARNING: audio file not found — skipping audio trim")

    print("\nDone.")


if __name__ == "__main__":
    main()
