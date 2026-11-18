"""
src/evaluation/reporting.py
----------------------------
Table writers shared by the result scripts. No MLflow, no matplotlib; callers
own the DataFrame and the destination.

Public API
----------
to_markdown(df) -> str
    Pipe-table markdown, no index.
write_table(df, path_stem)
    <path_stem>.csv and <path_stem>.md, no index.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a pipe-table markdown string, no index column.

    Deliberately NOT DataFrame.to_markdown() (needs the optional `tabulate`
    dependency), raw str(v) per cell, matching every existing .md file
    already on disk under results/ulcer/.
    """
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)
    )
    return f"{header}\n{sep}\n{body}\n"


def write_table(df: pd.DataFrame, path_stem: Path) -> None:
    """Write `df` to <path_stem>.csv and <path_stem>.md (UTF-8, no index)."""
    path_stem = Path(path_stem)
    df.to_csv(path_stem.with_name(path_stem.name + ".csv"), index=False)
    path_stem.with_name(path_stem.name + ".md").write_text(to_markdown(df), encoding="utf-8")
