"""
Notebook setup script to add src directory to Python path.
Import this at the beginning of any notebook that needs to use the data analysis agent modules.

Usage in notebook:
    import setup_notebook_path
"""

import sys
from pathlib import Path

# Add the src directory to Python path if not already there
src_path = Path(__file__).parent / "src"
src_path_str = str(src_path.resolve())

if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)
    print(f"Added {src_path_str} to Python path")
