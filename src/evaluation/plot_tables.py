"""
src/evaluation/plot_tables.py
-----------------------------
Tables rendered as figures (one row per model configuration), shared by the
CV and held-out reports.

Public API
----------
text_lengths(rows) -> longest cell text per column
column_widths(lengths) -> relative column widths
draw_config_table(ax, header, cell_text, row_keys, col_widths, *, fontsize, row_scale)
    Shaded bold header; first cell of each configuration row in its config color.
"""

from __future__ import annotations

from src.evaluation.style import config_color

HEADER_COLOR = "#e1e0d9"


def text_lengths(rows: list[list]) -> list[int]:
    """Longest cell text per column, over every row given (header included)."""
    return [max(len(str(row[j])) for row in rows) for j in range(len(rows[0]))]


def column_widths(lengths: list[int]) -> list[float]:
    return [n / sum(lengths) for n in lengths]


def draw_config_table(
    ax, header, cell_text, row_keys, col_widths, *, fontsize: float, row_scale: float
):
    """Draw the table on `ax` (axis turned off) and return it for extra styling.
    `row_keys` are the MODEL_REGISTRY keys of the first len(row_keys) body rows."""
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=header, colWidths=col_widths, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, row_scale)
    for j in range(len(header)):
        table[0, j].set_facecolor(HEADER_COLOR)
        table[0, j].set_text_props(fontweight="bold")
    for i, key in enumerate(row_keys, start=1):
        table[i, 0].set_facecolor(config_color(key))
    return table
