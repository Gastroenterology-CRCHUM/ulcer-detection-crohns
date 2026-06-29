"""Tests for src/config/validation.py."""

from types import SimpleNamespace

import pytest

from src.config.validation import validate_config


def _make_config(**overrides):
    """Build a SimpleNamespace config that passes all validation checks."""
    model = overrides.pop(
        "model",
        SimpleNamespace(
            model="resnet50_gastronet",
            img_size=None,
            dropout_rate=0.5,
            freeze_layers=0,
        ),
    )
    training = overrides.pop(
        "training",
        SimpleNamespace(learning_rate=5e-5, batch_size=64),
    )
    cv = overrides.pop("cv", SimpleNamespace(n_splits=5))
    return SimpleNamespace(model=model, training=training, cv=cv, **overrides)


class TestNoneConfig:
    def test_none_returns_early(self):
        validate_config(None)  # must not raise


class TestModelValidation:
    def test_unknown_model_raises(self):
        cfg = _make_config(
            model=SimpleNamespace(model="not_a_real_model", dropout_rate=0.5, freeze_layers=0)
        )
        with pytest.raises(ValueError, match="not_a_real_model"):
            validate_config(cfg)

    def test_valid_model_passes(self):
        cfg = _make_config()
        validate_config(cfg)  # must not raise


class TestLearningRateValidation:
    def test_lr_too_low_raises(self):
        cfg = _make_config(training=SimpleNamespace(learning_rate=1e-10, batch_size=64))
        with pytest.raises(ValueError, match="Learning rate"):
            validate_config(cfg)

    def test_lr_too_high_raises(self):
        cfg = _make_config(training=SimpleNamespace(learning_rate=0.5, batch_size=64))
        with pytest.raises(ValueError, match="Learning rate"):
            validate_config(cfg)

    def test_lr_at_upper_bound_passes(self):
        cfg = _make_config(training=SimpleNamespace(learning_rate=1e-2, batch_size=64))
        validate_config(cfg)

    def test_lr_at_lower_bound_passes(self):
        cfg = _make_config(training=SimpleNamespace(learning_rate=1e-8, batch_size=64))
        validate_config(cfg)


class TestBatchSizeValidation:
    def test_batch_size_zero_raises(self):
        cfg = _make_config(training=SimpleNamespace(learning_rate=5e-5, batch_size=0))
        with pytest.raises(ValueError, match="batch_size"):
            validate_config(cfg)

    def test_batch_size_too_large_raises(self):
        cfg = _make_config(training=SimpleNamespace(learning_rate=5e-5, batch_size=513))
        with pytest.raises(ValueError, match="batch_size"):
            validate_config(cfg)

    def test_batch_size_boundary_passes(self):
        for size in (1, 512):
            cfg = _make_config(training=SimpleNamespace(learning_rate=5e-5, batch_size=size))
            validate_config(cfg)


class TestDropoutValidation:
    def test_dropout_negative_raises(self):
        cfg = _make_config(
            model=SimpleNamespace(
                model="resnet50_gastronet",
                dropout_rate=-0.1,
                freeze_layers=0,
            )
        )
        with pytest.raises(ValueError, match="dropout_rate"):
            validate_config(cfg)

    def test_dropout_one_raises(self):
        cfg = _make_config(
            model=SimpleNamespace(
                model="resnet50_gastronet",
                dropout_rate=1.0,
                freeze_layers=0,
            )
        )
        with pytest.raises(ValueError, match="dropout_rate"):
            validate_config(cfg)

    def test_dropout_zero_passes(self):
        cfg = _make_config(
            model=SimpleNamespace(
                model="resnet50_gastronet",
                dropout_rate=0.0,
                freeze_layers=0,
            )
        )
        validate_config(cfg)


class TestFreezeLayersValidation:
    def test_freeze_layers_minus_two_raises(self):
        cfg = _make_config(
            model=SimpleNamespace(
                model="resnet50_gastronet",
                dropout_rate=0.5,
                freeze_layers=-2,
            )
        )
        with pytest.raises(ValueError, match="freeze_layers"):
            validate_config(cfg)

    def test_freeze_layers_minus_one_passes(self):
        cfg = _make_config(
            model=SimpleNamespace(
                model="resnet50_gastronet",
                dropout_rate=0.5,
                freeze_layers=-1,
            )
        )
        validate_config(cfg)


class TestCVValidation:
    def test_n_splits_too_low_raises(self):
        cfg = _make_config(cv=SimpleNamespace(n_splits=1))
        with pytest.raises(ValueError, match="n_splits"):
            validate_config(cfg)

    def test_n_splits_too_high_raises(self):
        cfg = _make_config(cv=SimpleNamespace(n_splits=11))
        with pytest.raises(ValueError, match="n_splits"):
            validate_config(cfg)

    def test_n_splits_boundary_passes(self):
        for n in (2, 10):
            cfg = _make_config(cv=SimpleNamespace(n_splits=n))
            validate_config(cfg)


class TestMultipleErrors:
    def test_multiple_errors_collected_in_single_raise(self):
        cfg = _make_config(
            model=SimpleNamespace(
                model="bad_model",
                dropout_rate=-1.0,
                freeze_layers=-5,
            ),
            training=SimpleNamespace(learning_rate=99.0, batch_size=9999),
        )
        with pytest.raises(ValueError) as exc_info:
            validate_config(cfg)
        msg = str(exc_info.value)
        # At least two independent errors are reported in one raise
        assert "bad_model" in msg
        assert "Learning rate" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
