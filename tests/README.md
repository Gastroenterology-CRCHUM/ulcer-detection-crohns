# Tests

This directory contains unit tests for the Ulcer Detection project.

Current status: the full test suite passes on this branch (see CI for up-to-date results).

## Running Tests

### Run all tests

```bash
pytest
```

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

- `test_config.py`: Configuration dataclasses, validation, loading
- `test_utils.py`: Logging, device management, path utilities, formatting helpers
- `test_data_dataset.py`: `UlcerDataset` loading, CSV/manifest helpers, labels
- `test_data_transforms.py`: `ResizeWithPad`, `CLAHE_Y`, and transform pipelines
- `test_data_splits.py`: Patient-level stratification, train/val splitting, CV folds
- `test_evaluation_threshold.py`: Threshold sweeps and best-threshold selection
- `test_evaluation_metrics.py`: Metric computation and bootstrap confidence intervals
- `test_evaluation_aggregation.py`: Frame-to-clip aggregation methods and ranking
- `test_training_trainer.py`: Checkpoint discovery and loading

Shared fixtures live in `conftest.py`. Prefer using those over recreating local setup code when possible.

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

Tests are automatically run on:

- Pull requests
- Pushes to main branch
- Manual triggers

### CI Configuration

See `.github/workflows/test.yml` for CI configuration.

## Test Coverage

Aim for >90% code coverage. Focus first on the critical data and evaluation layers; model-training coverage can be extended after that.

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
