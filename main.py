import os
import io
import sys
import logging
import shutil
import argparse
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from dataframe_to_dict import parse_dataframe_info
from planner import create_plan
from coder import create_code

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_analysis_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
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
    script_dir = Path(__file__).parent.absolute()
    
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
        
        logger.info(f"Loaded {len(df_questions)} questions and {len(df_answers)} answers")
        
        # Validate required columns
        if 'id' not in df_questions.columns:
            raise ValueError("Questions dataframe missing 'id' column")
        if 'id' not in df_answers.columns:
            raise ValueError("Answers dataframe missing 'id' column")
        
        df_merged = df_answers.merge(df_questions, left_on="id", right_on="id", how='inner')
        logger.info(f"Merged data contains {len(df_merged)} records")
        
        if len(df_merged) == 0:
            logger.warning("No matching records found after merge")
        
        return df_merged
    
    except Exception as e:
        logger.error(f"Error loading and merging data: {e}")
        raise


def cleanup_output_directories(script_dir: Path) -> None:
    """Clean up old plan and code files before starting new processing."""
    try:
        # The planner and coder modules use relative paths, so we need to clean both:
        # 1. Paths relative to script directory (what we want)
        # 2. Paths relative to current working directory (what might be created)
        
        current_dir = Path.cwd()
        
        # Define possible directories to clean
        possible_dirs = [
            # Relative to script directory (intended)
            script_dir / "data" / "plan",
            script_dir / "data" / "code",
            # Relative to current working directory (actual behavior)
            current_dir / "data" / "plan",
            current_dir / "data" / "code",
            # In case there are nested paths
            current_dir / "data" / "code" / "data" / "plan",
            script_dir / "data" / "code" / "data" / "plan",
        ]
        
        # Define old output files to clean
        old_csv_files = [
            script_dir / "data" / "merged_with_plans.csv",
            script_dir / "data" / "merged_with_code.csv",
            current_dir / "data" / "merged_with_plans.csv",
            current_dir / "data" / "merged_with_code.csv",
        ]
        
        directories_cleaned = 0
        files_removed = 0
        
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Current working directory: {current_dir}")
        
        # Clean directories
        for directory in possible_dirs:
            if directory.exists():
                logger.info(f"Cleaning directory: {directory}")
                
                # Remove all files in the directory
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            files_removed += 1
                            logger.debug(f"Removed file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove file {file_path}: {e}")
                    elif file_path.is_dir():
                        try:
                            shutil.rmtree(file_path)
                            files_removed += 1
                            logger.debug(f"Removed directory: {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove directory {file_path}: {e}")
                
                # Remove the directory itself if it's empty
                try:
                    if not any(directory.iterdir()):
                        directory.rmdir()
                        logger.debug(f"Removed empty directory: {directory}")
                except Exception as e:
                    logger.debug(f"Could not remove directory {directory}: {e}")
                
                directories_cleaned += 1
            else:
                logger.debug(f"Directory does not exist: {directory}")
        
        # Clean old CSV files
        for csv_file in old_csv_files:
            if csv_file.exists():
                try:
                    csv_file.unlink()
                    files_removed += 1
                    logger.info(f"Removed old output file: {csv_file}")
                except Exception as e:
                    logger.warning(f"Could not remove old output file {csv_file}: {e}")
        
        if files_removed > 0:
            logger.info(f"Cleanup completed: {files_removed} files removed, checked {len(possible_dirs)} directories")
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
        df_json = parse_dataframe_info(buffer.getvalue())
        return df_json
    except Exception as e:
        logger.error(f"Error converting DataFrame info to JSON: {e}")
        raise


def process_plans(df_merged: pd.DataFrame, path_prefix: Path) -> pd.DataFrame:
    """Process data to create plans with robust error handling."""
    logger.info("Starting plan creation process...")
    
    if not path_prefix.exists():
        logger.error(f"Data directory not found: {path_prefix}")
        raise FileNotFoundError(f"Data directory not found: {path_prefix}")
    
    processed_count = 0
    failed_count = 0
    
    for index, row in df_merged.iterrows():
        try:
            if 'file_name' not in row or pd.isna(row['file_name']):
                logger.warning(f"Row {index}: Missing file_name, skipping")
                failed_count += 1
                continue
                
            if 'question' not in row or pd.isna(row['question']):
                logger.warning(f"Row {index}: Missing question, skipping")
                failed_count += 1
                continue
            
            file_name = path_prefix / row['file_name']
            
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
                plan = create_plan(row['question'], df_json, file_name)
                df_merged.at[index, 'plan'] = plan.model_dump_json()
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
    
    logger.info(f"Plan creation completed: {processed_count} successful, {failed_count} failed")
    return df_merged


def process_codes(df_merged: pd.DataFrame, path_prefix: Path) -> pd.DataFrame:
    """Process data to create code with robust error handling."""
    logger.info("Starting code creation process...")
    
    if not path_prefix.exists():
        logger.error(f"Data directory not found: {path_prefix}")
        raise FileNotFoundError(f"Data directory not found: {path_prefix}")
    
    processed_count = 0
    failed_count = 0
    
    for index, row in df_merged.iterrows():
        try:
            # Validate required fields
            required_fields = ['file_name', 'question', 'plan']
            missing_fields = [field for field in required_fields if field not in row or pd.isna(row[field])]
            
            if missing_fields:
                logger.warning(f"Row {index}: Missing required fields {missing_fields}, skipping")
                failed_count += 1
                continue
            
            file_name = path_prefix / row['file_name']
            
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
                code = create_code(row['plan'], row['question'], df_json, Path(row['file_name']))
                df_merged.at[index, 'code'] = code.model_dump_json()
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
    
    logger.info(f"Code creation completed: {processed_count} successful, {failed_count} failed")
    return df_merged


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
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
        required_columns = ['file_name', 'question']
        missing_columns = [col for col in required_columns if col not in df_merged.columns]
        if missing_columns:
            logger.error(f"Missing required columns in merged data: {missing_columns}")
            return
        
        # Get script directory for relative paths
        script_dir = Path(__file__).parent.absolute()
        
        # Change working directory to script directory to ensure relative paths work correctly
        original_cwd = Path.cwd()
        os.chdir(script_dir)
        logger.info(f"Changed working directory to: {script_dir}")
        
        # Clean up old output files before starting (unless skipped)
        if not skip_cleanup:
            logger.info("Cleaning up old output files...")
            cleanup_output_directories(script_dir)
        else:
            logger.info("Skipping cleanup of old output files")
        
        path_prefix = script_dir / "data/InfiAgent-DABench/da-dev-tables/"
        
        # Process plans
        logger.info("Processing plans...")
        df_merged = process_plans(df_merged, path_prefix)
        save_dataframe(df_merged, str(script_dir / "data/merged_with_plans.csv"))
        
        # Process codes
        logger.info("Processing codes...")
        df_merged = process_codes(df_merged, path_prefix)
        save_dataframe(df_merged, str(script_dir / "data/merged_with_code.csv"))
        
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
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
                logger.debug(f"Restored working directory to: {original_cwd}")
        except Exception as e:
            logger.warning(f"Could not restore working directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Analysis Agent with robust error handling")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Skip cleanup of old output files before processing")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Update logging level if specified
    if args.log_level != "INFO":
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.info(f"Logging level set to {args.log_level}")
    
    main(skip_cleanup=args.skip_cleanup)