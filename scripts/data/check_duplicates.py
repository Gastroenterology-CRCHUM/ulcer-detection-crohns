"""
scripts/check_duplicates.py
============================
Verify that no frame appears in more than one subfolder of raw_inf/.

A frame is identified by the full stem of its filename.

Modes
-----
    --report   : display duplicates only (default)
    --fix      : remove duplicates, keeping the most recently modified copy
    --fix-keep : remove duplicates, keeping the copy in --keep-dir

Usage
-----
    # Check only
    python scripts/check_duplicates.py --raw-inf data/raw_inf

    # Check and fix (keep the most recent file)
    python scripts/check_duplicates.py --raw-inf data/raw_inf --fix

    # Check and fix, always keeping the copy in Informative/
    python scripts/check_duplicates.py --raw-inf data/raw_inf \\
        --fix-keep Informative

    # Export the report to CSV
    python scripts/check_duplicates.py --raw-inf data/raw_inf \\
        --output duplicates.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.file_utils import duplicate_report, find_duplicates, scan

# ---------------------------------------------------------------------------
# CLI-only helpers (printing / file deletion)
# ---------------------------------------------------------------------------


def _relative_class(path: Path, raw_inf_root: Path) -> str:
    return str(path.parent.relative_to(raw_inf_root))


def print_report(dupes: list[dict], raw_inf_root: Path) -> None:
    if not dupes:
        print("✓ No duplicates detected.")
        return
    print(f"⚠  {len(dupes)} duplicate frames:\n")
    for d in dupes:
        print(f"  {d['stem']}  ({d['n_copies']} copies)")
        for p in d["paths"]:
            print(f"    [{_relative_class(p, raw_inf_root)}]")
    print()


def fix_duplicates(
    dupes: list[dict],
    raw_inf_root: Path,
    strategy: str,
    keep_dir: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Delete excess copies according to the chosen strategy.

    Strategies:
        "newest"   : keep the most recently modified file
        "keep_dir" : keep the copy whose relative folder starts with keep_dir
    """
    n_deleted = 0
    n_errors = 0
    for d in dupes:
        paths = d["paths"]
        if strategy == "newest":
            keeper = max(paths, key=lambda p: p.stat().st_mtime)
        elif strategy == "keep_dir":
            preferred = [
                p for p in paths if _relative_class(p, raw_inf_root).startswith(keep_dir or "")
            ]
            keeper = preferred[0] if preferred else max(paths, key=lambda p: p.stat().st_mtime)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for p in [p for p in paths if p != keeper]:
            folder = _relative_class(p, raw_inf_root)
            if dry_run:
                print(f"  DRY-RUN  would delete [{folder}]  {p.name}")
            else:
                try:
                    p.unlink()
                    print(f"  DEL  [{folder}]  {p.name}")
                    n_deleted += 1
                except Exception as e:
                    print(f"  ERR  {p}  ->  {e}")
                    n_errors += 1
    return n_deleted, n_errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    from src.config.paths import get_default_paths

    default_paths = get_default_paths()

    parser = argparse.ArgumentParser(description="Check (and fix) duplicate frames in raw_inf/")
    parser.add_argument(
        "--raw-inf",
        default=str(default_paths.informative_raw_dir),
        help=f"Root of the raw_inf/ directory (default: {default_paths.informative_raw_dir})",
    )
    parser.add_argument("--output", default=None, help="Export duplicate report to CSV.")
    parser.add_argument(
        "--fix", action="store_true", help="Delete duplicates, keeping the newest copy."
    )
    parser.add_argument(
        "--fix-keep",
        default=None,
        metavar="DIR",
        help="Delete duplicates, keeping the copy in DIR (e.g. Informative).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be deleted without deleting."
    )
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png"])
    args = parser.parse_args()

    raw_inf_root = Path(args.raw_inf)
    if not raw_inf_root.exists():
        print(f"Error: {raw_inf_root} does not exist.")
        return

    extensions = tuple(e if e.startswith(".") else f".{e}" for e in args.extensions)

    print(f"Scanning {raw_inf_root} ...")
    index = scan(raw_inf_root, extensions)
    n_total = sum(len(v) for v in index.values())
    print(f"  {len(index):,} unique filenames ({n_total:,} total files)\n")

    dupes = find_duplicates(index)
    print_report(dupes, raw_inf_root)

    if not dupes:
        return

    if args.output:
        df = duplicate_report(dupes, raw_inf_root)
        df.to_csv(args.output, index=False)
        print(f"Report exported -> {args.output}")

    if args.fix or args.fix_keep:
        strategy = "keep_dir" if args.fix_keep else "newest"
        print(
            f"Strategy: {strategy}"
            + (f" (preferred folder: {args.fix_keep})" if args.fix_keep else "")
        )
        if args.dry_run:
            print("Dry-run mode — no files will be deleted\n")
        n_del, n_err = fix_duplicates(
            dupes=dupes,
            raw_inf_root=raw_inf_root,
            strategy=strategy,
            keep_dir=args.fix_keep,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"\n{n_del} files deleted, {n_err} errors")
            if n_del > 0:
                print("-> Re-run preprocess_inf.py --skip-preprocess to rebuild the manifest")
    else:
        print("To fix automatically:")
        print(f"  python scripts/data/check_duplicates.py --raw-inf {raw_inf_root} --fix")
        print(
            f"  python scripts/data/check_duplicates.py --raw-inf {raw_inf_root}"
            f" --fix-keep Informative"
        )


if __name__ == "__main__":
    main()
