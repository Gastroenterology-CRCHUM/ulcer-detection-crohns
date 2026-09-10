"""
src/evaluation/reporting.py
----------------------------
Tiny shared report-writing helpers. No MLflow, no plotting, no matplotlib --
callers own the DataFrame and the destination path.

Public API
----------
to_markdown(df) -> str
    Pipe-table markdown, no index. Was duplicated byte-for-byte in
    scripts/ulcer/cv_results.py and scripts/ulcer/effect_decomposition.py.
"""

from __future__ import annotations

import pandas as pd


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a pipe-table markdown string, no index column.

    Deliberately NOT DataFrame.to_markdown() (needs the optional `tabulate`
    dependency) -- raw str(v) per cell, matching every existing .md file
    already on disk under results/ulcer/.
    """
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)
    )
    return f"{header}\n{sep}\n{body}\n"
