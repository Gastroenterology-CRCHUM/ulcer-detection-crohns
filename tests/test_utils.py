"""
tests/test_utils.py
==================
Unit tests for utility functions.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.utils import (
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
    compute_confidence_interval,
    ensure_dir,
    find_latest_checkpoint,
    format_metrics,
    get_device,
    get_device_info,
    get_logger,
    setup_logging,
    validate_path_exists,
)


class TestExceptions:
    """Test custom exceptions."""

    @pytest.mark.parametrize(
        "exc_class, msg",
        [
            (ConfigurationError, "Invalid config"),
            (DataError, "Data loading failed"),
            (ModelError, "Model creation failed"),
            (TrainingError, "Training failed"),
        ],
    )
    def test_custom_exception(self, exc_class, msg):
        error = exc_class(msg)
        assert str(error) == msg
        assert isinstance(error, Exception)


class TestLogging:
    """Test logging utilities."""

    def test_setup_logging(self):
        """Test setting up logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_logging(
                name="test_logger",
                level=20,  # INFO
                log_dir=Path(temp_dir),
                use_color=False,
            )
            assert logger.name == "test_logger"
            assert logger.level == 20
            # Clean up handlers
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test_module")
        assert logger.name == "test_module"

        # Test singleton behavior
        logger2 = get_logger("test_module")
        assert logger is logger2


class TestDevice:
    """Test device utilities."""

    def test_get_device_cpu(self):
        """Test getting CPU device when no GPU available."""
        with patch("torch.cuda.is_available", return_value=False):
            device = get_device(device_id=0)
            assert str(device) == "cpu"

    def test_get_device_gpu(self):
        """Test getting GPU device when available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            device = get_device(device_id=1)
            assert str(device) == "cuda:1"

    def test_get_device_info(self):
        """Test getting device information."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_name", return_value="Mock GPU"),
        ):
            info = get_device_info()
            assert info["cuda_available"] is True
            assert info["cuda_device_count"] == 1


class TestPathUtilities:
    """Test path utility functions."""

    def test_ensure_dir(self):
        """Test ensuring directory exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_subdir"
            result = ensure_dir(test_dir)
            assert result.exists()
            assert result.is_dir()

    def test_validate_path_exists(self):
        """Test validating path exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = Path(temp_dir) / "existing_file.txt"
            existing_path.write_text("test")

            # Should not raise
            result = validate_path_exists(existing_path, "test context")
            assert result == existing_path

    def test_validate_path_not_exists(self):
        """Test validating non-existent path raises error."""
        with pytest.raises(DataError):
            validate_path_exists(Path("non_existent_path"), "test context")

    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "checkpoints"
            base_dir.mkdir(parents=True, exist_ok=True)

            # find_latest_checkpoint expects model/timestamp/best.pt
            ckpt_old = base_dir / "resnet50" / "20250101_120000" / "best.pt"
            ckpt_new = base_dir / "resnet50" / "20250101_130000" / "best.pt"
            ckpt_old.parent.mkdir(parents=True, exist_ok=True)
            ckpt_new.parent.mkdir(parents=True, exist_ok=True)
            ckpt_old.write_text("checkpoint_old")
            ckpt_new.write_text("checkpoint_new")

            latest = find_latest_checkpoint(base_dir)
            assert latest is not None
            assert latest.name == "best.pt"


class TestMetricsUtilities:
    """Test metrics utility functions."""

    def test_compute_confidence_interval(self):
        """Test computing confidence interval."""
        data = np.array([0.8, 0.85, 0.9, 0.87, 0.83])
        mean, lower, upper = compute_confidence_interval(data, confidence=0.95)

        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= mean <= upper
        # CI bounds are reasonable
        assert lower < mean
        assert upper > mean

    def test_format_metrics(self):
        """Test formatting metrics."""
        metrics = {"accuracy": 0.95, "f1_score": 0.87, "precision": 0.92}

        formatted = format_metrics(metrics, prefix="test_")
        lines = formatted.strip().split("\n")

        assert len(lines) == 3
        assert "test_accuracy: 0.9500" in lines[0]
        assert "test_f1_score: 0.8700" in lines[1]
        assert "test_precision: 0.9200" in lines[2]

    def test_format_metrics_no_prefix(self):
        """Test formatting metrics without prefix."""
        metrics = {"accuracy": 0.95}
        formatted = format_metrics(metrics)
        assert "accuracy: 0.9500" in formatted


if __name__ == "__main__":
    pytest.main([__file__])
