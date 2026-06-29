"""Tests for src/data/dataloader.py."""

import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataloader import (
    get_all_folds,
    get_cv_loaders,
    get_loaders,
    get_split_loaders,
    get_test_loader,
    get_val_loader,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_fixture(tmp_path):
    """Minimal manifest: 4 train patients, 1 val patient, 1 test patient."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rows = []
    specs = [
        # (patient_id, split, label)
        ("p001", "train", 0),
        ("p001", "train", 0),
        ("p002", "train", 1),
        ("p002", "train", 1),
        ("p003", "train", 0),
        ("p003", "train", 0),
        ("p004", "train", 1),
        ("p004", "train", 1),
        ("p005", "val", 0),
        ("p005", "val", 0),
        ("p006", "test", 1),
        ("p006", "test", 1),
    ]
    for i, (patient_id, split, label) in enumerate(specs):
        fname = f"img_{i:04d}.jpg"
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(data_dir / fname)
        rows.append(
            {
                "relative_path": fname,
                "label": label,
                "video_id": f"v{i:03d}",
                "segment_id": 1,
                "patient_id": patient_id,
                "split": split,
                "clip_key": f"{patient_id}_v{i:03d}",
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, data_dir


# ---------------------------------------------------------------------------
# get_loaders (dispatch)
# ---------------------------------------------------------------------------


class TestGetLoaders:
    def test_dispatch_split_mode(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_loaders(
            mode="split",
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_dispatch_cv_mode(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_loaders(
            mode="cv",
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            fold=0,
            n_splits=2,
            num_workers=0,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_invalid_mode_raises(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        with pytest.raises(ValueError, match="mode must be"):
            get_loaders(
                mode="invalid",
                manifest_path=manifest_path,
                data_dir=data_dir,
                batch_size=4,
                img_size=32,
                num_workers=0,
            )


# ---------------------------------------------------------------------------
# get_split_loaders
# ---------------------------------------------------------------------------


class TestGetSplitLoaders:
    def test_with_val_in_manifest(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_split_loaders(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
        )
        assert len(train_loader.dataset) == 8  # rows with split=="train"
        assert len(val_loader.dataset) == 2  # rows with split=="val"

    def test_without_val_in_manifest(self, tmp_path):
        """When no 'val' rows exist, assigns_val_split carves val from train."""
        data_dir = tmp_path / "data_nov"
        data_dir.mkdir()

        rows = []
        for i, (pid, label) in enumerate(
            [("a", 0), ("a", 0), ("b", 1), ("b", 1), ("c", 0), ("c", 0), ("d", 1), ("d", 1)]
        ):
            fname = f"img_{i:04d}.jpg"
            Image.new("RGB", (64, 64)).save(data_dir / fname)
            rows.append(
                {
                    "relative_path": fname,
                    "label": label,
                    "video_id": f"v{i:03d}",
                    "segment_id": 1,
                    "patient_id": pid,
                    "split": "train",
                    "clip_key": f"{pid}_v{i:03d}",
                }
            )

        manifest_path = tmp_path / "manifest_nov.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        train_loader, val_loader = get_split_loaders(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
            val_ratio=0.5,
        )
        total = len(train_loader.dataset) + len(val_loader.dataset)
        assert total == 8
        assert len(val_loader.dataset) > 0

    def test_loaders_are_dataloaders(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_split_loaders(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=2,
            img_size=32,
            num_workers=0,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)


# ---------------------------------------------------------------------------
# get_cv_loaders
# ---------------------------------------------------------------------------


class TestGetCvLoaders:
    def test_basic_cv_fold(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_cv_loaders(
            fold=0,
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            n_splits=2,
            num_workers=0,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        # All train rows are partitioned between train and val loaders
        total = len(train_loader.dataset) + len(val_loader.dataset)
        assert total == 8  # 4 patients × 2 frames in "train" split

    def test_both_folds_disjoint(self, manifest_fixture):
        """Val sets for fold 0 and fold 1 should be disjoint (no shared frames)."""
        manifest_path, data_dir = manifest_fixture
        kwargs = dict(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            n_splits=2,
            num_workers=0,
        )
        _, val0 = get_cv_loaders(fold=0, **kwargs)
        _, val1 = get_cv_loaders(fold=1, **kwargs)

        paths0 = set(val0.dataset.df["relative_path"].tolist())
        paths1 = set(val1.dataset.df["relative_path"].tolist())
        assert paths0.isdisjoint(paths1)

    def test_invalid_fold_raises(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        with pytest.raises(ValueError, match="fold must be in"):
            get_cv_loaders(
                fold=5,
                manifest_path=manifest_path,
                data_dir=data_dir,
                batch_size=4,
                img_size=32,
                n_splits=2,
                num_workers=0,
            )

    def test_use_full_trainset(self, manifest_fixture):
        """use_full_trainset=True merges 'val' rows into the train pool."""
        manifest_path, data_dir = manifest_fixture
        _, val_normal = get_cv_loaders(
            fold=0,
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            n_splits=2,
            num_workers=0,
            use_full_trainset=False,
        )
        _, val_full = get_cv_loaders(
            fold=0,
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            n_splits=2,
            num_workers=0,
            use_full_trainset=True,
        )
        # With full trainset the pool is larger so the val fold is larger too
        assert len(val_full.dataset) >= len(val_normal.dataset)

    def test_use_all_splits(self, manifest_fixture):
        """use_all_splits=True uses all rows regardless of the split column."""
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_cv_loaders(
            fold=0,
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            n_splits=2,
            num_workers=0,
            use_all_splits=True,
        )
        total = len(train_loader.dataset) + len(val_loader.dataset)
        assert total == 12  # all rows in the manifest


# ---------------------------------------------------------------------------
# get_test_loader
# ---------------------------------------------------------------------------


class TestGetTestLoader:
    def test_returns_test_rows(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        loader = get_test_loader(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)
        assert len(loader.dataset) == 2  # rows with split=="test"

    def test_shuffle_is_false(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        loader = get_test_loader(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
        )
        assert loader.sampler.__class__.__name__ != "RandomSampler"


# ---------------------------------------------------------------------------
# get_val_loader
# ---------------------------------------------------------------------------


class TestGetValLoader:
    def test_with_val_in_manifest(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        loader = get_val_loader(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)
        assert len(loader.dataset) == 2  # rows with split=="val"

    def test_without_val_carves_from_train(self, tmp_path):
        data_dir = tmp_path / "data_vl"
        data_dir.mkdir()
        rows = []
        for i, (pid, label) in enumerate(
            [("a", 0), ("a", 0), ("b", 1), ("b", 1), ("c", 0), ("c", 0), ("d", 1), ("d", 1)]
        ):
            fname = f"img_{i:04d}.jpg"
            Image.new("RGB", (64, 64)).save(data_dir / fname)
            rows.append(
                {
                    "relative_path": fname,
                    "label": label,
                    "video_id": f"v{i:03d}",
                    "segment_id": 1,
                    "patient_id": pid,
                    "split": "train",
                    "clip_key": f"{pid}_v{i:03d}",
                }
            )
        manifest_path = tmp_path / "manifest_vl.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

        loader = get_val_loader(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
            val_ratio=0.5,
        )
        assert isinstance(loader, DataLoader)
        assert len(loader.dataset) > 0


# ---------------------------------------------------------------------------
# get_all_folds
# ---------------------------------------------------------------------------


class TestSubsetRatio:
    """Tests that exercise _sampling_train (subset_ratio < 1.0)."""

    def test_split_mode_subset(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_split_loaders(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            num_workers=0,
            subset_ratio=0.5,
        )
        # Subsampled train must be smaller than the full 8 train rows
        assert len(train_loader.dataset) <= 8
        assert len(train_loader.dataset) >= 1

    def test_cv_mode_subset(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        train_loader, val_loader = get_cv_loaders(
            fold=0,
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            n_splits=2,
            num_workers=0,
            subset_ratio=0.5,
        )
        assert isinstance(train_loader, DataLoader)
        assert len(train_loader.dataset) >= 1


class TestGetAllFolds:
    def test_returns_n_splits_pairs(self, manifest_fixture):
        manifest_path, data_dir = manifest_fixture
        folds = get_all_folds(
            manifest_path=manifest_path,
            data_dir=data_dir,
            batch_size=4,
            img_size=32,
            n_splits=2,
            num_workers=0,
        )
        assert len(folds) == 2
        for train_loader, val_loader in folds:
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
