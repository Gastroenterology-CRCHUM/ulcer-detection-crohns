"""
PyTorch Dataset for colonoscopy ulcer detection.

Accepts a DataFrame directly so folds and splits can be sliced in memory
without writing intermediate CSVs to disk.

Exports
-------
    UlcerDataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class UlcerDataset(Dataset):
    """
    Binary ulcer classification dataset.

    Args:
        df:        DataFrame with at least the columns:
                       relative_path, label, video_id, segment_id, patient_id
        data_dir:  Root directory from which `relative_path` is resolved.
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self, df: pd.DataFrame, data_dir: Path, transform=None, label_col: str = "label"
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_col = label_col

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, csv_path: Path, data_dir: Path, transform=None) -> UlcerDataset:
        """Build from a pre-written split CSV (e.g. train.csv / test.csv)."""
        return cls(pd.read_csv(csv_path), data_dir, transform)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        data_dir: Path,
        split: str,
        transform=None,
    ) -> UlcerDataset:
        """
        Build from the full manifest, filtering by the `split` column.

        Args:
            split: One of 'train', 'test'.
        """
        df = pd.read_csv(manifest_path)
        return cls(df[df["split"] == split].copy(), data_dir, transform)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            image_path = self.data_dir / row["relative_path"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            with Image.open(image_path) as im:
                image = im.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {row['relative_path']}: {e}") from e

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        clip_id = f"{row['video_id']}_{row['segment_id']}"
        id_frame = str(row["relative_path"])  # full relative path — globally unique

        if self.transform:
            image = self.transform(image)

        return image, label, clip_id, id_frame

    # ------------------------------------------------------------------
    # Helpers exposed for stratification / sampling
    # ------------------------------------------------------------------

    @property
    def patient_ids(self) -> pd.Series:
        """Patient IDs aligned with dataset indices."""
        return self.df["patient_id"]

    @property
    def labels(self) -> pd.Series:
        """Integer labels aligned with dataset indices."""
        return self.df["label"]
