# Tests

This directory contains unit tests for the Ulcer Detection project.

Current status: 381 tests, all passing.

## Running Tests

### Run all tests

```bash
pytest
```

Default output groups by file, one filename line, then one dot per test in
it (not a per-test listing). Use `-v`/`--verbose` for the full per-test
listing, or `-q` for a plain progress bar with no filenames.

### Run specific test file

```bash
pytest tests/test_config.py
```

### Run specific test class

```bash
pytest tests/test_config.py::TestModelConfig
```

### Run specific test method

```bash
pytest tests/test_config.py::TestModelConfig::test_valid_model_config
```

### Run with coverage

```bash
pytest --cov=src --cov-report=html
```

## Test Structure

**Config** (`src/config/`)
- `test_config.py`: Configuration dataclasses, `PathConfig`, legacy dict conversion, `load_config`
- `test_config_validation.py`: Cross-field config validation

**Data** (`src/data/`)
- `test_annotation_loaders.py`: Excel annotation parsing (`load_ulcer_annotations`, timestamp helpers)
- `test_data_dataset.py`: `UlcerDataset` loading, CSV/manifest helpers, labels
- `test_data_dataloader.py`: DataLoader construction, stratified subsampling for CV
- `test_data_splits.py`, `test_data_splits_extended.py`: Patient-level stratification, train/val splitting, CV folds
- `test_data_subsampling.py`, `test_subsampling.py`: Visual-diversity subsampling (backbone-free paths)
- `test_data_transforms.py`: `ResizeWithPad`, `CLAHE_Y`, and transform pipelines
- `test_video_extraction.py`: Frame extraction from video

**Evaluation** (`src/evaluation/`)
- `test_evaluation_metrics.py`, `test_evaluation_metrics_extended.py`: Metric computation and bootstrap confidence intervals
- `test_evaluation_threshold.py`: Threshold sweeps and best-threshold selection
- `test_evaluation_aggregation.py`: Frame-to-clip aggregation methods and ranking
- `test_evaluation_delong.py`: DeLong AUROC comparison test
- `test_evaluation_runner.py`: `run_delong` orchestration
- `test_evaluation_model_loader.py`: Checkpoint loading (`load_model`, `load_best_models`)
- `test_evaluation_mlflow_utils.py`: Run tags, metric/artifact logging, model registry, run comparison

**Models / training**
- `test_training_trainer.py`: Checkpoint discovery and loading
- `test_training_run_modes.py`: `PipelineDef`, pure-logic branch of `_compute_fold_metrics`, explainability fallback, see note below on what's *not* covered here
- `test_noninformative_model.py`: Non-informative-frame RF classifier

**Scripts**
- `test_ulcer_scripts.py`: `scripts/ulcer/create_manifest.py` and `scripts/ulcer/eda.py`
- `test_scripts_data.py`: `src/data/eda_utils.py` pure-logic helpers

**Utilities**
- `test_utils.py`, `test_utils_extended.py`: Logging, device management, path utilities, formatting helpers

Shared fixtures live in `conftest.py`. Prefer using those over recreating local setup code when possible.
Notably `mlflow_tmp_tracking`: points MLflow at a throwaway per-test sqlite DB
(`tmp_path/mlflow.db`) and restores the global tracking URI afterward. Use it
instead of mocking `mlflow.*` calls, the local sqlite backend is fully
functional (tracking, artifacts, model registry), so tests exercise the real
client and catch API-shape drift a mock would hide. MLflow's plain file-store
backend is deprecated as of MLflow 3.x and rejects writes by default, hence
sqlite rather than `file:///...`.

## Test Categories

- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test interactions between components (marked with `@pytest.mark.integration`)
- **Slow tests**: Tests that take longer to run (marked with `@pytest.mark.slow`)

## Writing Tests

### Basic test structure

```python
import pytest
from src.module import function_to_test

def test_function_name():
    """Test description."""
    # Arrange
    input_data = ...

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_result
```

### Testing exceptions

```python
def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

### Parametrized tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_function_with_multiple_inputs(input, expected):
    """Test function with multiple input/output pairs."""
    assert function_to_test(input) == expected
```

## Continuous Integration

No CI is configured in this repository yet, `.github/workflows/` doesn't
exist. Tests are run locally (`pytest`) before pushing. If CI gets set up
later, update this section to point at the workflow file.

## Test Coverage

Current coverage report available at `htmlcov/index.html` after running:

```bash
pytest --cov=src --cov-report=html
```

## Maintenance Guide

1. Start from the module you changed and add or update the nearest `test_*.py` file.
2. Reuse shared fixtures from `conftest.py` before adding new local setup code.
3. Keep assertions close to the real API: tensor types, metric key names, return tuples, and error types should match production code exactly.
4. Run the targeted test file first, then the full suite if the change touches shared code.
5. Update this README when you add a new test file or a new test category.

## Adding New Tests

1. Create the test file in `tests/` using the `test_*.py` naming pattern.
2. Prefer small, deterministic fixtures over large integration setups.
3. Match the real API exactly; if a helper returns tensors or capitalized metric keys, assert that directly.
4. Add tests near the module they cover so regressions are easy to locate.
5. Run the targeted test file first, then the wider suite.
