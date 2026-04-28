#!/usr/bin/env python3
"""Copy a CHAT file plus matching media from a remote host, then trim to a line range.

Resolves media files via media_mappings in a remote server.yaml, copies them
locally via SCP, then delegates trimming to trim_chat_audio.py. Works with any
SSH-accessible media server -- not limited to TalkBank infrastructure.

Examples:
    # TalkBank workspace (defaults work):
    python3 scripts/prepare_corpus_media_fixture.py \\
        data/childes-eng-na-data/Eng-NA/Sprott/01.cha \\
        --lines 4-10 --output /tmp/fixture

    # External user with their own media server:
    python3 scripts/prepare_corpus_media_fixture.py \\
        my_corpus/sample.cha \\
        --ssh-host user@myserver \\
        --mapping-key my-corpus \\
        --lines 10-20 --output /tmp/fixture
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

MEDIA_EXTS = [".wav", ".mp3", ".mp4", ".m4a", ".flac"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat", help="CHAT file under data/*-data (absolute or workspace-relative)")
    parser.add_argument("--lines", required=True, help="Utterance range, e.g. 4-10")
    parser.add_argument("--output", required=True, help="Local output directory")
    parser.add_argument("--context", type=int, default=0, help="Extra utterances before/after range")
    parser.add_argument("--padding-ms", type=int, default=2000, help="Audio padding before/after trim")
    parser.add_argument(
        "--workspace-root",
        help="Workspace root (default: inferred from this script location)",
    )
    parser.add_argument(
        "--ssh-host",
        default="operator@server",
        help="SSH target that hosts media_mappings (default: operator@server)",
    )
    parser.add_argument(
        "--server-config",
        default="~/.batchalign3/server.yaml",
        help="Remote server.yaml path used for media_mappings",
    )
    parser.add_argument(
        "--mapping-key",
        help="Override the media_mappings key (default: infer from data/<repo>)",
    )
    parser.add_argument(
        "--ref",
        help="Reference CHAT for timing if the input CHAT is untimed (absolute or workspace-relative)",
    )
    return parser.parse_args()


def workspace_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def extract_chat_repo(root: Path, chat_path: Path) -> tuple[str, Path] | None:
    """Infer the data repo name and repo-relative path from a workspace layout.

    Returns None if the path doesn't follow the data/<repo>/ convention
    (e.g., external users with --mapping-key override).
    """
    try:
        rel = chat_path.relative_to(root)
    except ValueError:
        return None

    if len(rel.parts) < 3 or rel.parts[0] != "data":
        return None
    repo = rel.parts[1]
    repo_relative = Path(*rel.parts[2:])
    return repo, repo_relative


def parse_media_name(chat_path: Path) -> str:
    for line in chat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@Media:"):
            parts = line.split("\t", 1)
            if len(parts) == 2:
                name = parts[1].split(",", 1)[0].strip()
                if name:
                    return name
    raise SystemExit(f"No @Media header found in {chat_path}")


def run_checked(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def fetch_remote_server_config(host: str, config_path: str) -> str:
    remote_path = config_path if config_path.startswith("~") else shlex.quote(config_path)
    remote_cmd = f"sed -n '1,240p' {remote_path}"
    result = run_checked(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote_cmd],
        capture_output=True,
    )
    return result.stdout


def parse_media_mappings(yaml_text: str) -> dict[str, str]:
    mappings: dict[str, str] = {}
    in_section = False
    for raw_line in yaml_text.splitlines():
        if raw_line.strip().startswith("#"):
            continue
        if raw_line.startswith("media_mappings:"):
            in_section = True
            continue
        if in_section:
            if not raw_line.strip():
                continue
            if not raw_line.startswith("  "):
                break
            key, sep, value = raw_line.strip().partition(":")
            if not sep:
                continue
            mappings[key.strip()] = value.strip()
    return mappings


def find_remote_media(host: str, remote_dir: PurePosixPath, media_name: str) -> PurePosixPath:
    find_cmd = (
        f"find {shlex.quote(str(remote_dir))} -maxdepth 1 -type f "
        f"\\( -name {shlex.quote(media_name + '.*')} -o -name {shlex.quote(media_name)} \\)"
    )
    result = run_checked(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, find_cmd],
        capture_output=True,
    )
    candidates = [PurePosixPath(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not candidates:
        raise SystemExit(
            f"No remote media found for {media_name!r} under {remote_dir}"
        )

    def sort_key(path: PurePosixPath) -> tuple[int, str]:
        suffix = path.suffix.lower()
        try:
            rank = MEDIA_EXTS.index(suffix)
        except ValueError:
            rank = len(MEDIA_EXTS)
        return (rank, str(path))

    return sorted(candidates, key=sort_key)[0]


def copy_remote_media(host: str, remote_path: PurePosixPath, output_dir: Path) -> Path:
    run_checked(["scp", "-q", f"{host}:{remote_path}", str(output_dir)])
    return output_dir / remote_path.name


def main() -> int:
    args = parse_args()
    root = workspace_root(args.workspace_root)
    chat_path = resolve_path(root, args.chat)
    repo_info = extract_chat_repo(root, chat_path)
    if repo_info:
        repo_name, repo_relative = repo_info
    else:
        repo_name = None
        repo_relative = Path(chat_path.name)
    mapping_key = args.mapping_key or repo_name
    if not mapping_key:
        raise SystemExit(
            "Cannot infer media mapping key from path. Use --mapping-key explicitly."
        )
    media_name = parse_media_name(chat_path)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_copy = output_dir / chat_path.name
    shutil.copy2(chat_path, chat_copy)

    config_text = fetch_remote_server_config(args.ssh_host, args.server_config)
    media_mappings = parse_media_mappings(config_text)
    if mapping_key not in media_mappings:
        known = ", ".join(sorted(media_mappings))
        raise SystemExit(
            f"media_mappings has no key {mapping_key!r}; known keys: {known}"
        )

    remote_root = PurePosixPath(media_mappings[mapping_key])
    remote_dir = remote_root / PurePosixPath(repo_relative.parent.as_posix())
    remote_media = find_remote_media(args.ssh_host, remote_dir, media_name)
    media_copy = copy_remote_media(args.ssh_host, remote_media, output_dir)

    trim_script = Path(__file__).resolve().parent / "trim_chat_audio.py"
    trim_output = output_dir / "trimmed"
    trim_output.mkdir(parents=True, exist_ok=True)

    trim_cmd = [
        sys.executable,
        str(trim_script),
        str(chat_copy),
        "--lines",
        args.lines,
        "--context",
        str(args.context),
        "--padding-ms",
        str(args.padding_ms),
        "--audio",
        str(media_copy),
        "--output",
        str(trim_output),
    ]
    if args.ref:
        trim_cmd.extend(["--ref", str(resolve_path(root, args.ref))])
    run_checked(trim_cmd)

    print(f"Workspace root: {root}")
    print(f"CHAT:           {chat_path}")
    print(f"Repo key:       {mapping_key}")
    print(f"Remote media:   {remote_media}")
    print(f"Output dir:     {output_dir}")
    print(f"Trimmed dir:    {trim_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
