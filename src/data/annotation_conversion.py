"""Annotation format converters (labelme JSON → binary masks, etc.)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def json_to_mask(
    json_path: Path,
    output_dir: Path,
    target_label: str | None = None,
    suffix: str = "_mask",
) -> Path:
    """Convert a single labelme JSON annotation to a binary PNG mask.

    Args:
        json_path:    Path to the .json annotation file.
        output_dir:   Directory where the mask PNG will be saved.
        target_label: If set, only draw shapes whose label matches.
                      If None, all shapes are drawn.
        suffix:       String appended to the stem before '.png'.

    Returns:
        Path to the saved PNG mask.

    Raises:
        ValueError: If *json_path* is missing imageHeight / imageWidth fields.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if h is None or w is None:
        raise ValueError(f"{json_path.name}: missing imageHeight / imageWidth fields.")

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    shapes_drawn = 0
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if target_label is not None and label != target_label:
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            print(f"  [warn] {json_path.name}: shape '{label}' has < 3 points — skipped.")
            continue
        draw.polygon([tuple(pt) for pt in points], fill=255)
        shapes_drawn += 1

    if shapes_drawn == 0:
        label_info = f" for label '{target_label}'" if target_label else ""
        print(f"  [warn] {json_path.name}: no polygons drawn{label_info}.")

    out_path = output_dir / (json_path.stem + suffix + ".png")
    mask.save(out_path)
    return out_path
