"""
tests/test_data_dataset.py
===========================
Unit tests for UlcerDataset.
"""

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import UlcerDataset


class TestUlcerDataset:
    """Test UlcerDataset class."""

    def test_dataset_init(self, sample_dataset):
        """Test dataset initialization."""
        data_dir, df = sample_dataset
        dataset = UlcerDataset(df, data_dir)
        assert len(dataset) == 3
        assert dataset.label_col == "label"

    def test_getitem_all_indices(self, sample_dataset):
        """Test getting all items from dataset."""
        data_dir, df = sample_dataset
        dataset = UlcerDataset(df, data_dir)

        for i in range(len(dataset)):
            image, label, clip_id = dataset[i]
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"
            assert isinstance(label, torch.Tensor)
            assert label.dtype == torch.long
            assert int(label) in (0, 1)
            assert isinstance(clip_id, str)

    def test_getitem_missing_image(self, sample_dataset):
        """Test handling of missing image files."""
        data_dir, df = sample_dataset
        # Add a row with non-existent image
        df_bad = df.copy()
        df_bad = pd.concat(
            [
                df_bad,
                pd.DataFrame(
                    {
                        "relative_path": ["missing.jpg"],
                        "label": [1],
                        "video_id": ["v999"],
                        "segment_id": [1],
                        "patient_id": ["p999"],
                        "split": ["test"],
                    }
                ),
            ],
            ignore_index=True,
        )

        dataset = UlcerDataset(df_bad, data_dir)
        # Should raise error when accessing missing image
        with pytest.raises((FileNotFoundError, RuntimeError)):
            dataset[len(df)]

    def test_from_csv_classmethod(self, sample_dataset):
        """Test from_csv class method."""
        data_dir, df = sample_dataset

        # Save df as CSV
        csv_path = data_dir.parent / "test_split.csv"
        df.to_csv(csv_path, index=False)

        dataset = UlcerDataset.from_csv(csv_path, data_dir)
        assert len(dataset) == 3

    def test_from_manifest_train_split(self, sample_dataset):
        """Test from_manifest filtering by train split."""
        data_dir, df = sample_dataset

        manifest_path = data_dir.parent / "manifest.csv"
        df.to_csv(manifest_path, index=False)

        dataset = UlcerDataset.from_manifest(manifest_path, data_dir, split="train")
        assert len(dataset) == 2  # Only train split rows

    def test_from_manifest_val_split(self, sample_dataset):
        """Test from_manifest filtering by val split."""
        data_dir, df = sample_dataset

        manifest_path = data_dir.parent / "manifest.csv"
        df.to_csv(manifest_path, index=False)

        dataset = UlcerDataset.from_manifest(manifest_path, data_dir, split="val")
        assert len(dataset) == 1  # Only val split rows

    def test_patient_ids_property(self, sample_dataset):
        """Test patient_ids property."""
        data_dir, df = sample_dataset
        dataset = UlcerDataset(df, data_dir)

        patient_ids = dataset.patient_ids
        assert isinstance(patient_ids, pd.Series)
        assert len(patient_ids) == 3
        assert patient_ids[0] == "p001"

    def test_labels_property(self, sample_dataset):
        """Test labels property."""
        data_dir, df = sample_dataset
        dataset = UlcerDataset(df, data_dir)

        labels = dataset.labels
        assert isinstance(labels, pd.Series)
        assert len(labels) == 3
        assert list(labels) == [0, 1, 0]

    def test_custom_label_column(self, sample_dataset):
        """Test dataset with custom label column."""
        data_dir, df = sample_dataset

        # Add a custom label column
        df["ulcer_size"] = [0, 1, 2]

        dataset = UlcerDataset(df, data_dir, label_col="ulcer_size")
        image, label, clip_id = dataset[0]
        assert isinstance(label, torch.Tensor)
        assert int(label) == 0

    def test_transform_application(self, sample_dataset):
        """Test that transforms are applied to images."""
        data_dir, df = sample_dataset

        # Simple mock transform that checks input
        def mock_transform(img):
            assert isinstance(img, Image.Image)
            return img

        dataset = UlcerDataset(df, data_dir, transform=mock_transform)
        image, label, clip_id = dataset[0]
        assert isinstance(image, Image.Image)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
