"""Extended tests for src/data/splits.py (covers gaps at ~72% coverage)."""

import pandas as pd
import pytest

from src.data.splits import (
    assign_train_val_test_split,
    build_strat_bin,
    modal_patient_label,
    split_with_rare_strata,
    ulcer_presence_bin,
)

# ---------------------------------------------------------------------------
# modal_patient_label — "unknown" paths
# ---------------------------------------------------------------------------


class TestModalPatientLabel:
    def test_unknown_when_patient_not_found(self):
        df = pd.DataFrame({"patient_id": ["p001"], "label": [1]})
        result = modal_patient_label("p999", df)
        assert result == "unknown"

    def test_unknown_when_all_labels_nan(self):
        df = pd.DataFrame({"patient_id": ["p001", "p001"], "label": [float("nan"), float("nan")]})
        result = modal_patient_label("p001", df)
        assert result == "unknown"

    def test_returns_mode_for_known_patient(self):
        df = pd.DataFrame({"patient_id": ["p001", "p001", "p001"], "label": [0, 1, 1]})
        result = modal_patient_label("p001", df)
        assert result == "1"


# ---------------------------------------------------------------------------
# ulcer_presence_bin
# ---------------------------------------------------------------------------


class TestUlcerPresenceBin:
    def test_no_ulcer(self):
        df = pd.DataFrame({"patient_id": ["p001", "p001"], "label": [0, 0]})
        assert ulcer_presence_bin("p001", df) == "no_ulcer"

    def test_has_ulcer(self):
        df = pd.DataFrame({"patient_id": ["p001", "p001"], "label": [0, 1]})
        assert ulcer_presence_bin("p001", df) == "ulcer"


# ---------------------------------------------------------------------------
# build_strat_bin — all modes
# ---------------------------------------------------------------------------


class TestBuildStratBin:
    def _df(self):
        return pd.DataFrame(
            {
                "patient_id": ["p001", "p001", "p002", "p002"],
                "label": [1, 1, 0, 0],
                "ulcer_size": [1, 1, None, None],
            }
        )

    def test_presence_mode(self):
        df = self._df()
        result = build_strat_bin("p001", df, mode="presence")
        assert result == "ulcer"
        result_neg = build_strat_bin("p002", df, mode="presence")
        assert result_neg == "no_ulcer"

    def test_size_mode_with_size_column(self):
        df = self._df()
        result = build_strat_bin("p001", df, mode="size")
        assert result == "1"

    def test_size_mode_no_ulcer_patient_returns_none(self):
        # p002 has no ulcer frames; dominant_ulcer_size returns "none"
        df = self._df()
        result = build_strat_bin("p002", df, mode="size")
        assert result == "none"

    def test_size_and_presence_mode(self):
        df = self._df()
        result = build_strat_bin("p001", df, mode="size_and_presence")
        assert "ulcer" in result and "1" in result

    def test_ulcer_ratio_mode(self):
        df = self._df()
        result = build_strat_bin("p001", df, mode="ulcer_ratio")
        assert result in ("no_ulcer", "low_ulcer", "high_ulcer")

    def test_invalid_mode_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="Unknown strat_mode"):
            build_strat_bin("p001", df, mode="invalid_mode")


# ---------------------------------------------------------------------------
# split_with_rare_strata — rare strata assignment
# ---------------------------------------------------------------------------


class TestSplitWithRareStrata:
    def test_all_common_stratified(self):
        ids = [f"p{i:03d}" for i in range(20)]
        labels = ["a"] * 10 + ["b"] * 10
        train, val, test, strategy, rare = split_with_rare_strata(
            ids, labels, 0.7, 0.15, 0.15, random_seed=42
        )
        assert len(train) + len(val) + len(test) == 20
        assert strategy in ("stratified", "partial")

    def test_rare_strata_get_assigned(self):
        ids = [f"p{i:03d}" for i in range(10)]
        # "rare_label" only appears once — must be manually assigned
        labels = ["common"] * 9 + ["rare_label"]
        train, val, test, strategy, rare = split_with_rare_strata(
            ids, labels, 0.7, 0.15, 0.15, random_seed=42
        )
        assert len(train) + len(val) + len(test) == 10
        assert strategy == "partial"
        # The rare ID appears exactly once across all splits
        rare_id = ids[9]
        count = sum(1 for split in (train, val, test) if rare_id in split)
        assert count == 1

    def test_all_ids_covered(self):
        ids = list(range(30))
        labels = ["x"] * 15 + ["y"] * 15
        train, val, test, _, _ = split_with_rare_strata(ids, labels, 0.6, 0.2, 0.2, random_seed=7)
        assert set(train) | set(val) | set(test) == set(ids)

    def test_single_item_stratum(self):
        ids = ["lonely"]
        labels = ["singleton"]
        train, val, test, strategy, rare = split_with_rare_strata(
            ids, labels, 0.7, 0.15, 0.15, random_seed=0
        )
        assert len(train) + len(val) + len(test) == 1


# ---------------------------------------------------------------------------
# assign_train_val_test_split — empty dataset + missing columns
# ---------------------------------------------------------------------------


class TestAssignTrainValTestSplit:
    def test_empty_dataset_returns_empty_split_info(self):
        df = pd.DataFrame({"patient_id": [], "label": []})
        out, info = assign_train_val_test_split(df, 0.7, 0.15, 0.15, random_seed=42)
        assert info["strategy"] == "empty"
        assert info["splits"]["train"]["n_patients"] == 0

    def test_missing_patient_col_raises(self):
        df = pd.DataFrame({"label": [0, 1]})
        with pytest.raises(KeyError, match="patient_id"):
            assign_train_val_test_split(df, 0.7, 0.15, 0.15, random_seed=42)

    def test_all_patients_assigned(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i:02d}" for i in range(15)],
                "label": [i % 2 for i in range(15)],
            }
        )
        out, info = assign_train_val_test_split(df, 0.7, 0.15, 0.15, random_seed=42)
        assert "split" in out.columns
        total = sum(info["splits"][s]["n_patients"] for s in ("train", "val", "test"))
        assert total == 15

    def test_split_info_has_expected_keys(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i:02d}" for i in range(10)],
                "label": [i % 2 for i in range(10)],
            }
        )
        _, info = assign_train_val_test_split(df, 0.6, 0.2, 0.2, random_seed=0)
        assert "strategy" in info
        assert "splits" in info
        for split in ("train", "val", "test"):
            assert split in info["splits"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
