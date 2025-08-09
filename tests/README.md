# Test Suite

This directory contains unit tests for the data-analysis-agent package.

## Test Structure

- `test_planner.py` - Tests for the planner module
- `test_automotive_quality.py` - Tests for automotive quality assessment including silent mode functionality
- `test_enhanced_automotive_quality.py` - Tests for enhanced automotive quality with signal dictionary
- `test_cli.py` - Tests for CLI functionality
- `test_paths.py` - Tests for path utilities
- `conftest.py` - Shared test configuration and fixtures
- `assets/` - Test data files (CSV files, JSONL files)

## Running Tests

### All Tests (excluding integration tests)
```bash
pytest tests/ -v
```

### Unit Tests Only (no API calls)
```bash
pytest tests/ -k "not real_llm" -v
```

### Integration Tests (requires OpenAI API key)
```bash
pytest tests/ -m "requires_api_key" -v
```

### Specific Test Suites
```bash
pytest tests/test_planner.py::TestTask -v
pytest tests/test_planner.py::TestPlanGenerator -v
pytest tests/test_automotive_quality.py -v
pytest tests/test_enhanced_automotive_quality.py -v
```

### Using the Test Runner
```bash
# Unit tests only
python run_tests.py unit

# All automotive quality tests
python run_tests.py automotive

# Planner tests only
python run_tests.py planner

# All tests
python run_tests.py all
```

## Test Data

The `assets/` directory contains:
- Sample CSV files from the InfiAgent-DABench dataset
- Short question/answer pairs for testing
- Additional test datasets (titanic.csv, abalone.csv, etc.)

## Environment Variables

For integration tests that use real LLM calls, set:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Without this key, integration tests will be automatically skipped.

## Test Categories

- **Unit Tests**: Test individual components without external dependencies
- **Integration Tests**: Test complete workflows including LLM calls
- **Mocked Tests**: Test behavior with mocked LLM responses (future enhancement)

## Adding New Tests

When adding new tests:
1. Follow the existing naming convention (`test_*.py`)
2. Use descriptive class and method names
3. Add appropriate markers for integration tests
4. Include docstrings explaining what each test validates
5. Use fixtures for common setup (see `conftest.py`)

## Common Test Patterns

### Testing with Sample Data
```python
def test_with_sample_data(self, assets_dir):
    csv_path = assets_dir / "titanic.csv"
    df = pd.read_csv(csv_path)
    # Your test code here
```

### Testing Plan Generation
```python
def test_plan_creation(self):
    plan = create_plan(question, df_json, csv_path)
    assert isinstance(plan, Plan)
    assert len(plan.task_list) > 0
```

### Testing with Temporary Directories
```python
def test_with_temp_dir(self, tmp_path):
    # tmp_path is a pytest fixture for temporary directories
    test_file = tmp_path / "test.csv"
    # Your test code here
```

```

### Testing Automotive Quality Silent Mode

```python
# Test automatic silent mode detection
def test_silent_mode_auto_detection():
    result = generate_automotive_quality_report(df, json_output_file='report.json')
    assert result is None  # Silent mode returns None

# Test explicit silent mode control
def test_silent_mode_explicit():
    result = generate_automotive_quality_report(df, silent=True)
    assert result[0] is None  # Text report is None in silent mode
    assert isinstance(result[1], dict)  # JSON data still available

# Test silent mode override
def test_silent_mode_override():
    result = generate_automotive_quality_report(df, json_output_file='report.json', silent=False)
    assert isinstance(result[0], str)  # Forces text output even with file
```
