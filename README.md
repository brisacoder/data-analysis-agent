# data-analysis-agent

A robust data analysis agent with comprehensive error handling, logging, and validation features.

## Features

- **Comprehensive Logging**: Detailed logging to both file and console with configurable levels
- **Error Handling**: Robust exception handling for file operations, data processing, and API calls
- **Input Validation**: Validates environment variables, file existence, and data integrity
- **Progress Tracking**: Real-time progress updates during processing
- **Graceful Failure**: Continues processing even when individual records fail
- **Resource Validation**: Checks for required files and directories before processing
- **Path Resolution**: Automatically resolves relative paths based on script location, allowing execution from any directory
- **Automatic Cleanup**: Removes old output files before processing to ensure clean runs

## Setup

1. Copy `.env.example` to `.env` and configure your file paths:
   ```bash
   cp .env.example .env
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Run the agent:
   ```bash
   python main.py
   ```

   Or with options:
   ```bash
   # Skip cleanup of old files
   python main.py --skip-cleanup
   
   # Set debug logging level
   python main.py --log-level DEBUG
   ```

## Command Line Options

- `--skip-cleanup`: Skip cleanup of old output files before processing
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Configuration

The application requires the following environment variables:

- `QUESTIONS_FILE`: Path to the questions JSON file
- `ANSWERS_FILE`: Path to the answers JSON file
- `LOG_LEVEL` (optional): Logging level (DEBUG, INFO, WARNING, ERROR)

## Logging

The application creates a log file `data_analysis_agent.log` in the project directory with detailed information about:

- Processing progress
- Error conditions
- File validation results
- Performance metrics

## Error Handling

The application handles various error conditions gracefully:

- Missing or invalid environment variables
- File not found errors
- Invalid CSV/JSON data
- API call failures
- Network connectivity issues
- Path resolution issues when running from different directories

Processing continues even when individual records fail, with detailed logging of failures.

## Troubleshooting

### Path Issues
If you see unexpected directory structures like `data/code/data/plan`, this was caused by running the script from different directories in earlier versions. The current version:

1. Automatically resolves all paths relative to the script location
2. Changes the working directory during execution to ensure consistency  
3. Includes comprehensive cleanup that removes incorrectly nested directories

### Cleanup
The `--skip-cleanup` option can be used if you need to preserve existing output files, but normally the automatic cleanup ensures a fresh start for each run.