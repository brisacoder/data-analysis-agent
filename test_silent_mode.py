#!/usr/bin/env python3
"""
Quick test script for the new silent mode functionality
"""

import pandas as pd
import tempfile
import os
from data_analysis_agent.automotive_data_quality import generate_automotive_quality_report

# Create a simple test dataset
test_data = {
    'vehicle_speed_kph': [0, 10, 20, 30, 40, 50],
    'engine_rpm': [800, 1000, 1200, 1400, 1600, 1800],
    'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:00:01', 
                  '2024-01-01 10:00:02', '2024-01-01 10:00:03',
                  '2024-01-01 10:00:04', '2024-01-01 10:00:05']
}
df = pd.DataFrame(test_data)

print("Testing silent mode functionality...")

# Test 1: Normal mode (should return tuple)
print("\n1. Normal mode:")
result1 = generate_automotive_quality_report(df)
print(f"   Return type: {type(result1)}")
print(f"   Text report exists: {result1[0] is not None}")
print(f"   JSON data exists: {result1[1] is not None}")

# Test 2: Silent mode with JSON output
print("\n2. Silent mode with JSON output:")
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json_file = f.name
try:
    result2 = generate_automotive_quality_report(df, json_output_file=json_file)
    print(f"   Return type: {type(result2)}")
    print(f"   JSON file created: {os.path.exists(json_file)}")
finally:
    if os.path.exists(json_file):
        os.unlink(json_file)

# Test 3: Silent mode with text output
print("\n3. Silent mode with text output:")
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    text_file = f.name
try:
    result3 = generate_automotive_quality_report(df, output_file=text_file)
    print(f"   Return type: {type(result3)}")
    print(f"   Text file created: {os.path.exists(text_file)}")
finally:
    if os.path.exists(text_file):
        os.unlink(text_file)

# Test 4: Explicit silent=False with JSON output (should still show output)
print("\n4. Explicit silent=False with JSON output:")
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json_file = f.name
try:
    result4 = generate_automotive_quality_report(df, json_output_file=json_file, silent=False)
    print(f"   Return type: {type(result4)}")
    print(f"   Text report exists: {result4[0] is not None}")
    print(f"   JSON file created: {os.path.exists(json_file)}")
finally:
    if os.path.exists(json_file):
        os.unlink(json_file)

print("\nAll tests completed successfully!")
