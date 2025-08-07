import os
import io
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from dataframe_to_dict import parse_dataframe_info_with_columns
from planner import create_plan
from coder import create_code
from paths import get_paths

# Get paths instance for the application
app_paths = get_paths()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(app_paths.log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def validate_environment_variables() -> tuple[Path, Path]:
    """Validate and retrieve required environment variables."""
    load_dotenv(override=True)

    questions_file = os.getenv("QUESTIONS_FILE")
    answers_file = os.getenv("ANSWERS_FILE")

    if not questions_file:
        raise ValueError("QUESTIONS_FILE environment variable is not set")
    if not answers_file:
        raise ValueError("ANSWERS_FILE environment variable is not set")

    # Get the directory where this script is located
    script_dir = app_paths.project_root

    # Convert relative paths to absolute paths relative to script location
    questions_path = Path(questions_file)
    if not questions_path.is_absolute():
        questions_path = script_dir / questions_path

    answers_path = Path(answers_file)
    if not answers_path.is_absolute():
        answers_path = script_dir / answers_path

    # Resolve any relative components like '..' or '.'
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

        # Validate required columns
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
        # Don't raise the exception, just log it and continue


def df_info_to_json(df: pd.DataFrame) -> str:
    """Convert DataFrame info to JSON format with error handling."""
    try:
        buffer = io.StringIO()
        df.info(buf=buffer, show_counts=True)
        # Use the robust function that takes actual column names
        df_json = parse_dataframe_info_with_columns(buffer.getvalue(), list(df.columns))
        return df_json
    except Exception as e:
        logger.error(f"Error converting DataFrame info to JSON: {e}")
        raise


def process_plans(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Process data to create plans with robust error handling."""
    logger.info("Starting plan creation process...")

    path_prefix = app_paths.tables_dir
    if not path_prefix.exists():
        logger.error(f"Data directory not found: {path_prefix}")
        raise FileNotFoundError(f"Data directory not found: {path_prefix}")

    processed_count = 0
    failed_count = 0

    for index, row in df_merged.iterrows():
        try:
            if "file_name" not in row or pd.isna(row["file_name"]):
                logger.warning(f"Row {index}: Missing file_name, skipping")
                failed_count += 1
                continue

            if "question" not in row or pd.isna(row["question"]):
                logger.warning(f"Row {index}: Missing question, skipping")
                failed_count += 1
                continue

            file_name = path_prefix / row["file_name"]

            if not file_name.exists():
                logger.warning(f"Row {index}: File not found: {file_name}, skipping")
                failed_count += 1
                continue

            logger.debug(f"Processing row {index}: {file_name}")

            # Load and validate CSV
            try:
                df = pd.read_csv(file_name)
                if df.empty:
                    logger.warning(f"Row {index}: Empty CSV file: {file_name}")
                    failed_count += 1
                    continue
            except pd.errors.EmptyDataError:
                logger.warning(f"Row {index}: Empty or invalid CSV file: {file_name}")
                failed_count += 1
                continue
            except Exception as e:
                logger.error(f"Row {index}: Error reading CSV {file_name}: {e}")
                failed_count += 1
                continue

            # Convert DataFrame info to JSON
            df_json = df_info_to_json(df)

            # Create plan
            try:
                plan = create_plan(row["question"], df_json, file_name)
                df_merged.at[index, "plan"] = plan.model_dump_json()
                processed_count += 1

                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} plans so far...")

            except Exception as e:
                logger.error(f"Row {index}: Error creating plan: {e}")
                failed_count += 1
                continue

        except Exception as e:
            logger.error(f"Row {index}: Unexpected error: {e}")
            failed_count += 1
            continue

    logger.info(
        f"Plan creation completed: {processed_count} successful, {failed_count} failed"
    )
    return df_merged


def process_codes(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Process data to create code with robust error handling."""
    logger.info("Starting code creation process...")

    path_prefix = app_paths.tables_dir
    if not path_prefix.exists():
        logger.error(f"Data directory not found: {path_prefix}")
        raise FileNotFoundError(f"Data directory not found: {path_prefix}")

    processed_count = 0
    failed_count = 0

    for index, row in df_merged.iterrows():
        try:
            # Validate required fields
            required_fields = ["file_name", "question", "plan"]
            missing_fields = [
                field
                for field in required_fields
                if field not in row or pd.isna(row[field])
            ]

            if missing_fields:
                logger.warning(
                    f"Row {index}: Missing required fields {missing_fields}, skipping"
                )
                failed_count += 1
                continue

            file_name = path_prefix / row["file_name"]

            if not file_name.exists():
                logger.warning(f"Row {index}: File not found: {file_name}, skipping")
                failed_count += 1
                continue

            logger.debug(f"Processing code for row {index}: {file_name}")

            # Load and validate CSV
            try:
                df = pd.read_csv(file_name)
                if df.empty:
                    logger.warning(f"Row {index}: Empty CSV file: {file_name}")
                    failed_count += 1
                    continue
            except Exception as e:
                logger.error(f"Row {index}: Error reading CSV {file_name}: {e}")
                failed_count += 1
                continue

            # Convert DataFrame info to JSON
            df_json = df_info_to_json(df)

            # Create code
            try:
                code = create_code(
                    row["plan"], row["question"], df_json, Path(row["file_name"])
                )
                df_merged.at[index, "code"] = code.model_dump_json()
                processed_count += 1

                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count} codes so far...")

            except Exception as e:
                logger.error(f"Row {index}: Error creating code: {e}")
                failed_count += 1
                continue

        except Exception as e:
            logger.error(f"Row {index}: Unexpected error: {e}")
            failed_count += 1
            continue

    logger.info(
        f"Code creation completed: {processed_count} successful, {failed_count} failed"
    )
    return df_merged


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV with error handling."""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} records to {output_path}")

    except Exception as e:
        logger.error(f"Error saving DataFrame to {output_path}: {e}")
        raise


def main(skip_cleanup: bool = False):
    """Main function with comprehensive error handling."""
    try:
        logger.info("Starting data analysis agent...")

        # Validate environment and load data
        questions_path, answers_path = validate_environment_variables()
        df_merged = load_and_merge_data(questions_path, answers_path)

        if len(df_merged) == 0:
            logger.error("No data to process after merging")
            return

        # Validate required columns
        required_columns = ["file_name", "question"]
        missing_columns = [
            col for col in required_columns if col not in df_merged.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns in merged data: {missing_columns}")
            return

        # Get script directory for relative paths
        script_dir = app_paths.project_root

        # Change working directory to script directory to ensure relative paths work correctly
        original_cwd = Path.cwd()
        os.chdir(script_dir)
        logger.info(f"Changed working directory to: {script_dir}")

        # Ensure required directories exist
        app_paths.ensure_directories()

        # Clean up old output files before starting (unless skipped)
        if not skip_cleanup:
            logger.info("Cleaning up old output files...")
            cleanup_output_directories()
        else:
            logger.info("Skipping cleanup of old output files")

        # Process plans
        logger.info("Processing plans...")
        df_merged = process_plans(df_merged)
        save_dataframe(df_merged, app_paths.merged_plans_file)

        # Process codes
        logger.info("Processing codes...")
        df_merged = process_codes(df_merged)
        save_dataframe(df_merged, app_paths.merged_code_file)

        logger.info("Data analysis agent completed successfully!")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        sys.exit(1)
    finally:
        # Restore original working directory if it was changed
        try:
            if "original_cwd" in locals():
                os.chdir(original_cwd)
                logger.debug(f"Restored working directory to: {original_cwd}")
        except Exception as e:
            logger.warning(f"Could not restore working directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Analysis Agent with robust error handling"
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

    args = parser.parse_args()

    # Update logging level if specified
    if args.log_level != "INFO":
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.info(f"Logging level set to {args.log_level}")

    main(skip_cleanup=args.skip_cleanup)
