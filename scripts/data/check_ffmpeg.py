"""Verify this machine's FFmpeg matches the version the conda env pins.

Checks the FFmpeg that torchcodec actually loaded, not the ``ffmpeg`` binary on
PATH — they are frequently different. torchcodec dlopens whichever of its bundled
``libtorchcodec_core4..core9`` matches the installed ``libavutil`` soname, so the
shared library is what decides how audio decodes; a stray ``ffmpeg`` executable is
irrelevant to it.

The expected version is read from conda_environment.yaml so the pin has a single
source of truth (uv_to_conda.sh writes it there from FFMPEG_VERSION).

    python -m scripts.data.check_ffmpeg                  # exact match required
    python -m scripts.data.check_ffmpeg --major-only     # only the major must match
    python -m scripts.data.check_ffmpeg --expected 7.1.1

Installing a matching FFmpeg is a per-machine setup step; see the README.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.constants import REPO_ROOT

CONDA_YAML = REPO_ROOT / "conda_environment.yaml"
# torchcodec ships loaders for these FFmpeg majors; outside this range it cannot
# load at all, regardless of what the pin says.
SUPPORTED_MAJORS = range(4, 10)

# libavutil's soname major runs 52 ahead of the FFmpeg major (FFmpeg 7 -> so.59).
AVUTIL_OFFSET = 52


def pinned_version(conda_yaml: Path) -> str | None:
    """Return the FFmpeg version pinned in the conda YAML, or None if absent."""
    if not conda_yaml.is_file():
        return None
    match = re.search(r"^\s*-\s*ffmpeg=([0-9][^\s\"']*)", conda_yaml.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def loaded_ffmpeg() -> tuple[str, int]:
    """Return (version string, libavutil major) for the FFmpeg torchcodec loaded."""
    from torchcodec._core.ops import _get_json_ffmpeg_library_versions

    info = json.loads(_get_json_ffmpeg_library_versions())
    return info["ffmpeg_version"], info["libavutil"][0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--expected", help="version to require (default: read from conda YAML)")
    p.add_argument(
        "--major-only",
        action="store_true",
        help="accept any version sharing the expected major",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compare the loaded FFmpeg against the pin, returning a process exit code."""
    args = parse_args(argv)

    expected = args.expected or pinned_version(CONDA_YAML)
    if expected is None:
        print(
            f"error: no ffmpeg pin found in {CONDA_YAML}; pass --expected or run "
            "uv_to_conda.sh to generate it",
            file=sys.stderr,
        )
        return 2

    try:
        actual, avutil_major = loaded_ffmpeg()
    except Exception as exc:  # any import/load failure is itself the answer
        print(f"FAIL: torchcodec could not load FFmpeg: {exc}", file=sys.stderr)
        return 1

    # "4.4.2-0ubuntu0.22.04.1" -> ("4", "4", "2"); distro suffixes are not part of
    # the upstream version and must not defeat the comparison.
    actual_parts = re.split(r"[.\-+]", actual)
    expected_parts = expected.split(".")
    depth = 1 if args.major_only else len(expected_parts)
    matches = actual_parts[:depth] == expected_parts[:depth]

    print(f"expected  {expected}" + ("  (major only)" if args.major_only else ""))
    print(f"actual    {actual}")
    print(f"libavutil so.{avutil_major}  -> libtorchcodec_core{avutil_major - AVUTIL_OFFSET}.so")

    if matches:
        if avutil_major - AVUTIL_OFFSET not in SUPPORTED_MAJORS:
            print(
                f"FAIL: FFmpeg major {avutil_major - AVUTIL_OFFSET} is outside torchcodec's "
                f"supported range {SUPPORTED_MAJORS.start}-{SUPPORTED_MAJORS.stop - 1}",
                file=sys.stderr,
            )
            return 1
        print("OK: FFmpeg matches the pinned version")
        return 0

    print(
        f"FAIL: FFmpeg mismatch (expected {expected}, got {actual}). "
        "See the README for installing a matching FFmpeg.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
