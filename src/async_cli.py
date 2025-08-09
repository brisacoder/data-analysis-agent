"""
Async orchestrator for the Data Analysis Agent.

This module mirrors `main.py` but executes per-row work (read CSV -> df info ->
plan -> code) concurrently using asyncio, so I/O-heavy steps run in parallel.

It does NOT modify existing files; it only adds an alternative entry point.
"""

from __future__ import annotations

import os
import io
import sys
import argparse
import logging
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from dataframe_to_dict import parse_dataframe_info_with_columns
from planner import create_plan
from coder import create_code
from paths import get_paths


# Get paths instance for the application
app_paths = get_paths()


# Configure logging with Unicode support and better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(app_paths.log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
)
logger = logging.getLogger(__name__)

# Configure console handler to use UTF-8 encoding
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        # Force UTF-8 encoding for console output
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        break

# Suppress noisy HTTP debug logging from OpenAI and httpcore to avoid Unicode issues
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ------------------------- Utility and validation ------------------------- #


def validate_environment_variables() -> tuple[Path, Path]:
    """Validate and retrieve required environment variables.

    Returns absolute resolved paths for QUESTIONS_FILE and ANSWERS_FILE.
    """
    load_dotenv(override=True)

    questions_file = os.getenv("QUESTIONS_FILE")
    answers_file = os.getenv("ANSWERS_FILE")

    if not questions_file:
        raise ValueError("QUESTIONS_FILE environment variable is not set")
    if not answers_file:
        raise ValueError("ANSWERS_FILE environment variable is not set")

    script_dir = app_paths.project_root

    questions_path = Path(questions_file)
    if not questions_path.is_absolute():
        questions_path = script_dir / questions_path

    answers_path = Path(answers_file)
    if not answers_path.is_absolute():
        answers_path = script_dir / answers_path

    questions_path = questions_path.resolve()
    answers_path = answers_path.resolve()

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not answers_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")

    logger.info(f"Using questions file: {questions_path}")
    logger.info(f"Using answers file: {answers_path}")

    return questions_path, answers_path


def load_and_merge_data(questions_path: Path, answers_path: Path) -> pd.DataFrame:
    """Load questions and answers data and merge them."""
    try:
        logger.info("Loading questions and answers data...")
        df_questions = pd.read_json(questions_path, lines=True)
        df_answers = pd.read_json(answers_path, lines=True)

        logger.info(
            f"Loaded {len(df_questions)} questions and {len(df_answers)} answers"
        )

        if "id" not in df_questions.columns:
            raise ValueError("Questions dataframe missing 'id' column")
        if "id" not in df_answers.columns:
            raise ValueError("Answers dataframe missing 'id' column")

        df_merged = df_answers.merge(
            df_questions, left_on="id", right_on="id", how="inner"
        )
        logger.info(f"Merged data contains {len(df_merged)} records")

        if len(df_merged) == 0:
            logger.warning("No matching records found after merge")

        return df_merged

    except Exception as e:
        logger.error(f"Error loading and merging data: {e}")
        raise


def cleanup_output_directories() -> None:
    """Clean up old plan and code files before starting new processing."""
    try:
        files_removed = app_paths.clean_output_directories(skip_log_files=True)
        if files_removed > 0:
            logger.info(f"Cleanup completed: {files_removed} files removed")
        else:
            logger.info("Cleanup completed: No files to remove")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def df_info_to_json(df: pd.DataFrame) -> str:
    """Convert DataFrame info to JSON format with error handling."""
    try:
        buffer = io.StringIO()
        df.info(buf=buffer, show_counts=True)
        return parse_dataframe_info_with_columns(buffer.getvalue(), list(df.columns))
    except Exception as e:
        logger.error(f"Error converting DataFrame info to JSON: {e}")
        raise


# ------------------------- Async processing pipeline ---------------------- #


@dataclass
class RowInput:
    index: int
    file_name: Path
    question: str


@dataclass
class RowResult:
    index: int
    plan_json: Optional[str] = None
    code_json: Optional[str] = None
    error: Optional[str] = None


async def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV in a worker thread (blocking I/O)."""
    return await asyncio.to_thread(pd.read_csv, path)


async def _save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    await asyncio.to_thread(df.to_csv, output_path, index=False)


async def process_row_pipeline(
    row: RowInput,
    *,
    semaphore: asyncio.Semaphore,
) -> RowResult:
    """Process a single row: read CSV -> df_info -> plan -> code.

    Uses a semaphore to bound concurrency. All blocking operations run in a
    thread executor via asyncio.to_thread.
    """
    async with semaphore:
        try:
            logger.info(f"Starting Row {row.index}: Processing '{row.file_name.name}' with question: '{row.question[:100]}{'...' if len(row.question) > 100 else ''}'")
            
            if not row.file_name.exists():
                msg = f"File not found: {row.file_name}"
                logger.warning(f"Row {row.index}: {msg}")
                return RowResult(index=row.index, error=msg)

            # Load CSV
            try:
                logger.debug(f"Row {row.index}: Loading CSV file...")
                df = await _read_csv(row.file_name)
                if df.empty:
                    msg = f"Empty CSV file: {row.file_name}"
                    logger.warning(f"Row {row.index}: {msg}")
                    return RowResult(index=row.index, error=msg)
                logger.debug(f"Row {row.index}: CSV loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns)")
            except Exception as e:
                msg = f"Error reading CSV {row.file_name}: {e}"
                logger.error(f"Row {row.index}: {msg}")
                return RowResult(index=row.index, error=msg)

            # DataFrame info -> JSON
            try:
                logger.debug(f"Row {row.index}: Converting DataFrame info to JSON...")
                df_json = await asyncio.to_thread(df_info_to_json, df)
                logger.debug(f"Row {row.index}: DataFrame info conversion completed")
            except Exception as e:
                msg = f"Error converting DataFrame info to JSON: {e}"
                logger.error(f"Row {row.index}: {msg}")
                return RowResult(index=row.index, error=msg)

            # Create plan (potentially network-bound)
            try:
                start_time = time.time()
                logger.info(f"Row {row.index}: 📋 PLANNING - Creating analysis plan...")
                plan = await asyncio.to_thread(create_plan, row.question, df_json, row.file_name)
                plan_json = plan.model_dump_json()
                duration = time.time() - start_time
                logger.info(f"Row {row.index}: ✅ PLANNING COMPLETE - Plan created successfully in {duration:.1f}s")
            except Exception as e:
                msg = f"Error creating plan: {e}"
                logger.error(f"Row {row.index}: ❌ PLANNING FAILED - {msg}")
                return RowResult(index=row.index, error=msg)

            # Create code (dependent on plan)
            try:
                start_time = time.time()
                logger.info(f"Row {row.index}: 💻 CODING - Generating code from plan...")
                code = await asyncio.to_thread(create_code, plan_json, row.question, df_json, row.file_name)
                code_json = code.model_dump_json()
                duration = time.time() - start_time
                logger.info(f"Row {row.index}: ✅ CODING COMPLETE - Code generated successfully in {duration:.1f}s")
            except Exception as e:
                msg = f"Error creating code: {e}"
                logger.error(f"Row {row.index}: ❌ CODING FAILED - {msg}")
                return RowResult(index=row.index, plan_json=plan_json, error=msg)

            logger.info(f"Row {row.index}: 🎉 PIPELINE COMPLETE - Both planning and coding successful for '{row.file_name.name}'")
            return RowResult(index=row.index, plan_json=plan_json, code_json=code_json)

        except Exception as e:  # Catch-all to avoid task cancellation
            msg = f"Unexpected error: {e}"
            logger.error(f"Row {row.index}: ❌ PIPELINE FAILED - {msg}")
            return RowResult(index=row.index, error=msg)


def _validate_dataframe_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


async def run_pipeline(
    df_merged: pd.DataFrame,
    *,
    max_concurrency: int,
) -> pd.DataFrame:
    """Run the per-row pipeline concurrently and return an updated DataFrame.

    The returned DataFrame will include 'plan' and 'code' columns when available.
    """
    required_cols = ["file_name", "question"]
    missing = _validate_dataframe_columns(df_merged, required_cols)
    if missing:
        raise ValueError(f"Missing required columns in merged data: {missing}")

    path_prefix = app_paths.tables_dir
    if not path_prefix.exists():
        raise FileNotFoundError(f"Data directory not found: {path_prefix}")

    sem = asyncio.Semaphore(max_concurrency)

    tasks: list[asyncio.Task[RowResult]] = []
    for index, row in df_merged.iterrows():
        # Validate minimal inputs quickly
        file_name = row.get("file_name")
        question = row.get("question")

        if pd.isna(file_name) or pd.isna(question):
            logger.warning(f"Row {index}: Missing file_name or question, skipping")
            continue

        row_input = RowInput(
            index=index,
            file_name=path_prefix / str(file_name),
            question=str(question),
        )
        tasks.append(asyncio.create_task(process_row_pipeline(row_input, semaphore=sem)))

    logger.info(f"Starting concurrent processing with concurrency={max_concurrency} for {len(tasks)} rows")

    processed, failed = 0, 0
    total_tasks = len(tasks)
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result.error:
            failed += 1
            logger.warning(f"❌ Row {result.index} failed: {result.error}")
        else:
            processed += 1
            if result.plan_json is not None:
                df_merged.at[result.index, "plan"] = result.plan_json
            if result.code_json is not None:
                df_merged.at[result.index, "code"] = result.code_json

        # Progress reporting with percentage
        completed = processed + failed
        percentage = (completed / total_tasks) * 100 if total_tasks > 0 else 0
        
        if completed % 5 == 0 or completed == total_tasks:  # Report every 5 completions or at end
            logger.info(f"📊 Progress: {completed}/{total_tasks} ({percentage:.1f}%) | ✅ {processed} successful, ❌ {failed} failed")

    logger.info(f"🏁 Concurrent processing completed: {processed} successful, {failed} failed out of {total_tasks} total")
    return df_merged


async def _save_outputs(df: pd.DataFrame) -> None:
    # Ensure required directories exist
    app_paths.ensure_directories()

    # Save merged with plans
    if "plan" in df.columns:
        await asyncio.to_thread(df.to_csv, app_paths.merged_plans_file, index=False)
        logger.info(f"Saved merged plans to {app_paths.merged_plans_file}")

    # Save merged with code
    if "code" in df.columns:
        await asyncio.to_thread(df.to_csv, app_paths.merged_code_file, index=False)
        logger.info(f"Saved merged code to {app_paths.merged_code_file}")


# ------------------------------ CLI / main -------------------------------- #


async def async_main(
    skip_cleanup: bool, 
    max_concurrency: int,
    max_rows: Optional[int] = None,
    start_row: int = 0,
    file_pattern: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    start_time = time.time()
    try:
        logger.info("🚀 Starting async data analysis agent...")

        # Validate env and load data
        logger.info("⚙️  Validating environment variables and loading data...")
        questions_path, answers_path = validate_environment_variables()
        df_merged = load_and_merge_data(questions_path, answers_path)
        if len(df_merged) == 0:
            logger.error("❌ No data to process after merging")
            return 2

        # Change working directory to project root so relative paths work
        original_cwd = Path.cwd()
        os.chdir(app_paths.project_root)
        logger.info(f"📁 Changed working directory to: {app_paths.project_root}")

        try:
            if not skip_cleanup:
                logger.info("🧹 Cleaning up old output files...")
                cleanup_output_directories()
            else:
                logger.info("⏭️  Skipping cleanup of old output files")

            # Run concurrent pipeline
            pipeline_start = time.time()
            logger.info(f"🔄 Starting pipeline with {len(df_merged)} rows and max concurrency of {max_concurrency}...")
            df_merged = await run_pipeline(df_merged, max_concurrency=max_concurrency)
            pipeline_duration = time.time() - pipeline_start

            # Save outputs
            logger.info("💾 Saving outputs...")
            await _save_outputs(df_merged)

            total_duration = time.time() - start_time
            logger.info("🎉 Async data analysis agent completed successfully!")
            logger.info(f"⏱️  Total execution time: {total_duration:.1f}s (Pipeline: {pipeline_duration:.1f}s)")
            return 0

        finally:
            try:
                os.chdir(original_cwd)
                logger.debug(f"Restored working directory to: {original_cwd}")
            except Exception as e:
                logger.warning(f"Could not restore working directory: {e}")

    except KeyboardInterrupt:
        total_duration = time.time() - start_time
        logger.info(f"⚠️  Process interrupted by user after {total_duration:.1f}s")
        return 1
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"💥 Fatal error in async main after {total_duration:.1f}s: {e}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async Data Analysis Agent (parallel per-row plan+code)"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup of old output files before processing",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="Maximum number of concurrent rows to process (default: 32, optimized for LLM I/O-bound operations)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (useful for testing)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Start processing from this row index (0-based)",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=None,
        help="Only process files matching this pattern (e.g., 'titanic')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually doing it",
    )
    args = parser.parse_args()

    if args.log_level != "INFO":
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.info(f"Logging level set to {args.log_level}")

    # Run asyncio program
    exit_code = asyncio.run(async_main(
        args.skip_cleanup, 
        args.max_concurrency,
        args.max_rows,
        args.start_row,
        args.file_pattern,
        args.dry_run,
    ))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
