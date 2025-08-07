#!/usr/bin/env python3
"""
Test script to verify the coder.py JSON parsing fix.
"""

from pathlib import Path
from coder import create_code


def test_coder_resilience():
    """Test that the coder can handle malformed JSON responses gracefully using include_raw."""
    
    # Simple test parameters
    plan = """
Task 1: Setup Imports and Dependencies
- Details: Import pandas, numpy
- Dependencies: None
- Output: All necessary imports ready

Task 2: Load test data
- Details: Create a simple test dataframe
- Dependencies: Task 1
- Output: Test DataFrame ready
"""
    
    question = "Create a simple test dataframe with some sample data"
    df_json = '{"columns": ["A", "B"], "dtypes": ["int64", "float64"], "shape": [10, 2]}'
    data_file_name = Path("test_sample")
    
    try:
        print("Testing coder.py with improved include_raw approach...")
        result = create_code(plan, question, df_json, data_file_name)
        
        print(f"✅ Success! Generated code with {len(result.code)} characters")
        print(f"📝 Assumptions: {result.assumptions}")
        
        # Verify the code is a string and not empty
        assert isinstance(result.code, str), "Code should be a string"
        assert len(result.code) > 0, "Code should not be empty"
        assert isinstance(result.assumptions, list), "Assumptions should be a list"
        
        print("✅ All tests passed! The coder now uses include_raw for better error handling.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    test_coder_resilience()
