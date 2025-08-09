#!/usr/bin/env python3
"""
Test runner script for the data-analysis-agent package.

This script provides convenient ways to run different test suites.
"""

import sys
import subprocess


def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, capture_output=False, check=False)
    return result.returncode == 0


def main():
    """Main test runner."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python run_tests.py [option]")
        print("Options:")
        print("  unit        - Run unit tests only (no API calls)")
        print("  integration - Run integration tests (requires API key)")
        print("  all         - Run all tests")
        print("  planner     - Run planner tests only")
        print("  automotive  - Run automotive quality tests only")
        print("  -h, --help  - Show this help message")
        return
    
    # Get the test option
    test_type = sys.argv[1] if len(sys.argv) > 1 else "unit"
    
    # Base pytest command
    base_cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    if test_type == "unit":
        cmd = base_cmd + ["-k", "not real_llm"]
        success = run_command(cmd, "Unit Tests (no API calls)")
        
    elif test_type == "integration":
        cmd = base_cmd + ["-m", "requires_api_key"]
        success = run_command(cmd, "Integration Tests (requires API key)")
        
    elif test_type == "all":
        cmd = base_cmd
        success = run_command(cmd, "All Tests")
        
    elif test_type == "planner":
        cmd = base_cmd + ["tests/test_planner.py"]
        success = run_command(cmd, "Planner Tests")
        
    elif test_type == "automotive":
        cmd = base_cmd + ["tests/test_automotive_quality.py", "tests/test_enhanced_automotive_quality.py"]
        success = run_command(cmd, "Automotive Quality Tests")
        
    else:
        print(f"Unknown test type: {test_type}")
        print("Use -h or --help for usage information.")
        return
    
    if success:
        print(f"\n✅ {test_type.title()} tests completed successfully!")
    else:
        print(f"\n❌ {test_type.title()} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
