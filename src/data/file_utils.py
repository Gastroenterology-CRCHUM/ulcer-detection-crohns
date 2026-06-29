"""File-system utilities for scanning and deduplicating image datasets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def scan(root: Path, extensions: tuple[str, ...] = _IMAGE_EXTS) -> dict[str, list[Path]]:
    """Walk *root* recursively and index each image file by its full stem.

    Returns:
        Mapping ``{stem: [list of Path]}`` — each key is the filename without
        extension; values are all files sharing that stem.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            index[p.stem].append(p)
    return index


def find_duplicates(index: dict[str, list[Path]]) -> list[dict]:
    """Return entries from *index* that appear in more than one location.

    Returns:
        Sorted list of ``{"stem": str, "n_copies": int, "paths": list[Path]}``.
    """
    dupes = [
        {"stem": stem, "n_copies": len(paths), "paths": paths}
        for stem, paths in index.items()
        if len(paths) > 1
    ]
    return sorted(dupes, key=lambda d: d["stem"])


def duplicate_report(dupes: list[dict], root: Path) -> pd.DataFrame:
    """Build a DataFrame of duplicates — one row per file copy.

    Columns: stem, folder, filename, path, size_kb, mtime.
    """
    rows = []
    for d in dupes:
        for p in d["paths"]:
            rows.append(
                {
                    "stem": d["stem"],
                    "folder": str(p.parent.relative_to(root)),
                    "filename": p.name,
                    "path": str(p),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "mtime": p.stat().st_mtime,
                }
            )
    return pd.DataFrame(rows)


def check_and_abort(root: Path | str, verbose: bool = True) -> bool:
    """Return True if *root* contains no duplicate image files, False otherwise.

    Intended as a pre-flight guard in preprocessing pipelines.
    """
    root = Path(root)
    index = scan(root)
    dupes = find_duplicates(index)

    if dupes:
        if verbose:
            print(f"\n[file_utils] ⚠  {len(dupes)} duplicate frame(s) in {root}")
            print("[file_utils] Fix with:")
            print(
                f"[file_utils]   python scripts/data/check_duplicates.py --raw-inf {root} --fix\n"
            )
        return False

    if verbose:
        print(f"[file_utils] ✓ No duplicates in {root}")
    return True
