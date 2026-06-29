"""Extended tests for src/utils/common.py (covers gaps at ~76% coverage)."""

from unittest.mock import patch

import pytest

from src.utils.common import (
    ConfigurationError,
    DataError,
    format_metrics,
    get_device,
    set_seed,
    validate_value_range,
)


class TestGetDeviceErrors:
    def test_device_id_exceeds_count_raises(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            pytest.raises(ValueError, match="not available"),
        ):
            get_device(device_id=5)

    def test_negative_device_id_returns_cpu(self):
        device = get_device(device_id=-1)
        assert str(device) == "cpu"


class TestSetSeed:
    def test_set_seed_runs_without_error(self):
        set_seed(123)

    def test_set_seed_with_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            set_seed(0)

    def test_set_seed_with_cuda_available(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed_all") as mock_seed,
        ):
            set_seed(99)
            mock_seed.assert_called_with(99)


class TestFindLatestCheckpointExtended:
    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        from src.utils.common import find_latest_checkpoint

        result = find_latest_checkpoint(tmp_path / "does_not_exist")
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        from src.utils.common import find_latest_checkpoint

        empty = tmp_path / "empty"
        empty.mkdir()
        result = find_latest_checkpoint(empty)
        assert result is None


class TestFormatMetricsExtended:
    def test_non_float_value_formatted_without_decimals(self):
        metrics = {"model": "resnet50", "epochs": 100}
        result = format_metrics(metrics)
        assert "model: resnet50" in result
        assert "epochs: 100" in result

    def test_mixed_float_and_non_float(self):
        metrics = {"name": "test", "score": 0.75}
        result = format_metrics(metrics)
        assert "name: test" in result
        assert "score: 0.7500" in result


class TestValidateValueRange:
    def test_valid_value_returns_value(self):
        result = validate_value_range(0.5, 0.0, 1.0, "rate")
        assert result == 0.5

    def test_value_below_min_raises(self):
        with pytest.raises(ConfigurationError, match="rate"):
            validate_value_range(-0.1, 0.0, 1.0, "rate")

    def test_value_above_max_raises(self):
        with pytest.raises(ConfigurationError, match="rate"):
            validate_value_range(1.1, 0.0, 1.0, "rate")

    def test_boundary_values_pass(self):
        assert validate_value_range(0.0, 0.0, 1.0) == 0.0
        assert validate_value_range(1.0, 0.0, 1.0) == 1.0


class TestValidateFileExists:
    def test_existing_file_returns_path(self, tmp_path):
        from src.utils.common import validate_file_exists

        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = validate_file_exists(f)
        assert result == f

    def test_missing_file_raises_data_error(self, tmp_path):
        from src.utils.common import validate_file_exists

        with pytest.raises(DataError):
            validate_file_exists(tmp_path / "missing.txt")

    def test_directory_raises_data_error(self, tmp_path):
        from src.utils.common import validate_file_exists

        with pytest.raises(DataError):
            validate_file_exists(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
