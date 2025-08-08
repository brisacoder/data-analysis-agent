# Async Orchestrator (async_main.py)

This document explains the new asynchronous entry point that processes planning and coding in parallel across rows to reduce time lost to I/O and model calls.

## What it is

- A drop-in, parallel alternative to `main.py`, implemented in `async_main.py`.
- Runs the per-row pipeline concurrently:
  1) Read CSV
  2) Build DataFrame schema JSON from `df.info()`
  3) Create a plan (`planner.create_plan`)
  4) Create code (`coder.create_code`)
- Preserves everything else: logging, directories, outputs, and inputs.
- Does not modify existing files; it only adds a new entry point.

## Why it’s faster

Most time is spent waiting on I/O (reading CSVs, LLM calls). `async_main.py` uses `asyncio` to schedule many rows at once. Blocking work runs in background threads using `asyncio.to_thread()` while a semaphore bounds concurrency to avoid overload.

## Key features

- Per-row concurrency with a configurable limit (`--max-concurrency`).
- Same environment variables and data loading as `main.py`.
- Same output files as `main.py` (merged plans and merged code CSVs).
- Robust logging to both console and the existing log file.

## Inputs and environment

- QUESTIONS_FILE: path to JSONL with questions (must contain `id`, `file_name`, `question`).
- ANSWERS_FILE: path to JSONL with pre-labeled answers (must contain `id`).
- LLM credentials: managed by LangChain’s environment variables; for OpenAI, set `OPENAI_API_KEY` (and any other provider-specific variables if used in your environment).
- `.env` is loaded with `dotenv`. Variables in `.env` override the process environment when the app starts.

Example `.env` (adjust paths to your repo):

```env
QUESTIONS_FILE=curated/15_DEVRT-DACIA-SPRING-questions.jsonl
ANSWERS_FILE=curated/15_DEVRT-DACIA-SPRING-labels.jsonl
# OpenAI (example)
OPENAI_API_KEY=sk-...redacted...
```

## How it works (architecture)

- Validate and resolve QUESTIONS_FILE and ANSWERS_FILE to absolute paths.
- Load both JSONL files and merge on `id` (inner join).
- Ensure required columns exist (at minimum: `file_name`, `question`).
- For each row, build a `RowInput` and schedule `process_row_pipeline()`:
  - Read CSV from `app_paths.tables_dir / file_name` (threaded).
  - Convert `df.info()` to a JSON-like schema string via `parse_dataframe_info_with_columns` (threaded).
  - Call `create_plan(question, df_json, file_path)` (threaded).
  - Call `create_code(plan_json, question, df_json, file_path)` (threaded).
- As tasks complete, write `plan` and `code` JSON back into the DataFrame row.
- Save merged results to the same output CSVs the sync version uses.

Concurrency is limited by an `asyncio.Semaphore` to keep memory and API usage in check.

## Outputs and directories

Paths come from `paths.get_paths()` (no changes needed):

- Plans and code files are saved in their usual directories by the existing `planner.py` and `coder.py` logic.
- The merged CSVs are written to `app_paths.merged_plans_file` and `app_paths.merged_code_file`.
- Logs go to `app_paths.log_file` and stdout.

## Running the async orchestrator (PowerShell)

```powershell
# Activate your venv (example)
& .\.venv\Scripts\Activate.ps1

# Basic run with default concurrency and INFO logging
python async_main.py

# Tuning concurrency (try 4, 8, 12, etc.)
python async_main.py --max-concurrency 8

# More verbose logs
python async_main.py --log-level DEBUG

# Keep existing outputs (skip cleanup)
python async_main.py --skip-cleanup
```

Tip: If you see API rate limit errors, lower `--max-concurrency`.

## Differences vs main.py

- The synchronous version processes rows one-by-one; the async version processes many rows in parallel.
- Both use the same planner and coder modules, the same file/directory layout, and the same merged output CSVs.
- Logging format/targets are the same.

## Error handling and edge cases

- Missing env vars or input files: startup validation will raise a clear error.
- Missing columns: requires at least `file_name` and `question` in the merged DataFrame.
- CSV issues: empty or unreadable CSVs cause that row to be skipped with a warning.
- Planner/coder errors: recorded per-row; the run continues for other rows.
- Working directory is set to the project root during execution and restored on exit.

## Performance guidance

- Start with `--max-concurrency 4`–`8`.
- If your CSVs are large or the model is rate-limited, reduce the concurrency.
- If your machine and API capacity allow, you can increase concurrency; watch memory, CPU, and provider limits.

## Extending or customizing

- Add timeouts/retries: wrap the threaded calls (`create_plan`, `create_code`, CSV reads) with your preferred retry policy.
- API batch control: if your provider supports batching, adapt planner/coder to use async HTTP clients and remove `to_thread`.
- Selective pipeline: if you want “plan-only” or “code-only”, split `process_row_pipeline` or add CLI flags.

## Troubleshooting

- “File not found” for CSV: confirm `file_name` entries point to files under `app_paths.tables_dir`.
- No rows processed: verify the `id` join between QUESTIONS and ANSWERS is correct, and required columns exist.
- Rate limit / model errors: reduce `--max-concurrency`, check credentials, and provider quotas.
- Windows path quirks: paths are handled via `pathlib.Path`; ensure there are no stray quotes or trailing spaces in your inputs.

## Contract and assumptions

- Inputs: QUESTIONS_FILE, ANSWERS_FILE (JSONL), both having an `id` column; merged rows must have `file_name` and `question`.
- Outputs: merged CSVs at the standard locations defined by `paths.py`.
- Failure modes: per-row errors recorded in logs; the run continues.
- Assumptions: the planner/coder LLMs are available through LangChain configuration and your environment.

## Quick reference

- Entry point: `async_main.py`
- CLI: `--max-concurrency`, `--log-level`, `--skip-cleanup`
- Saves: same merged CSVs as `main.py`
- Logging: same targets and format as `main.py`
