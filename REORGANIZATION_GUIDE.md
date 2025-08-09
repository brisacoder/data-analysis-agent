# Project Reorganization Guide

## What Changed

The Python source files have been moved from the project root to a `src/` directory to improve project organization and maintainability. The path detection in `paths.py` has been updated to correctly locate the project root.

### Files Moved

The following files were moved from the root directory to `src/`:
- `main.py` → `src/main.py`
- `async_main.py` → `src/async_main.py`
- `planner.py` → `src/planner.py`
- `coder.py` → `src/coder.py`
- `prompts.py` → `src/prompts.py`
- `paths.py` → `src/paths.py`
- `dataframe_to_dict.py` → `src/dataframe_to_dict.py`
- `dataframe_time.py` → `src/dataframe_time.py`
- `data_quality_assessment.py` → `src/data_quality_assessment.py`
- `point_geom.py` → `src/point_geom.py`

### New Entry Points

New entry point scripts have been created in the root directory:
- `main.py` - Entry point for the main data analysis workflow
- `async_main.py` - Entry point for the async data analysis workflow
- `setup_notebook_path.py` - Helper script for notebooks to setup import paths

## How to Use

### Running the Application

The application can still be run from the root directory using the same commands:

```bash
# Standard workflow
python main.py

# Async workflow  
python async_main.py

# With options
python main.py --log-level DEBUG
python main.py --skip-cleanup
```

### Working with Notebooks

For Jupyter notebooks that need to import modules from the `src/` directory, add this import at the beginning of your notebook:

```python
import setup_notebook_path
```

This will automatically add the `src/` directory to the Python path, allowing you to import modules normally:

```python
from planner import create_plan
from coder import create_code
from dataframe_to_dict import parse_dataframe_info_with_columns
```

### Development

When developing or importing these modules in other Python scripts, you have two options:

1. **Use the entry points** (recommended for end users):
   ```python
   # Import from root directory entry points
   import main
   import async_main
   ```

2. **Direct import from src** (recommended for development):
   ```python
   import sys
   from pathlib import Path
   
   # Add src to path
   src_path = Path(__file__).parent / "src"
   sys.path.insert(0, str(src_path))
   
   # Now import normally
   from main import main
   from planner import create_plan
   ```

## Benefits

- ✅ Cleaner root directory structure
- ✅ Better separation of source code from configuration and documentation
- ✅ Follows Python packaging best practices
- ✅ Makes the project more maintainable and scalable
- ✅ Easier to set up proper testing and CI/CD
- ✅ Backward compatibility maintained through entry points

## Troubleshooting

If you encounter import errors:

1. Make sure you're using the entry points (`main.py`, `async_main.py`) from the root directory
2. For notebooks, ensure you import `setup_notebook_path` first
3. For direct imports, make sure the `src/` directory is in your Python path

The project structure now follows modern Python best practices while maintaining all existing functionality.
