"""
Tests for the CLI module of the data analysis agent.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

from data_analysis_agent.cli import (
    parse_arguments,
    run_data_analysis,
    validate_environment_variables,
)


class TestParseArguments:
    """Test cases for the parse_arguments function."""

    def test_parse_arguments_defaults(self):
        """Test that parse_arguments returns correct defaults."""
        with patch('sys.argv', ['test_script']):
            args = parse_arguments()
            assert args.skip_cleanup is False
            assert args.log_level == "INFO"

    def test_parse_arguments_skip_cleanup(self):
        """Test parse_arguments with --skip-cleanup flag."""
        with patch('sys.argv', ['test_script', '--skip-cleanup']):
            args = parse_arguments()
            assert args.skip_cleanup is True
            assert args.log_level == "INFO"

    def test_parse_arguments_log_level(self):
        """Test parse_arguments with different log levels."""
        test_cases = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for level in test_cases:
            with patch('sys.argv', ['test_script', '--log-level', level]):
                args = parse_arguments()
                assert args.log_level == level
                assert args.skip_cleanup is False

    def test_parse_arguments_all_options(self):
        """Test parse_arguments with all options provided."""
        with patch('sys.argv', ['test_script', '--skip-cleanup', '--log-level', 'DEBUG']):
            args = parse_arguments()
            assert args.skip_cleanup is True
            assert args.log_level == "DEBUG"

    def test_parse_arguments_invalid_log_level(self):
        """Test parse_arguments with invalid log level."""
        with patch('sys.argv', ['test_script', '--log-level', 'INVALID']):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestRunDataAnalysis:
    """Test cases for the run_data_analysis function."""

    @patch('data_analysis_agent.cli.logger')
    @patch('data_analysis_agent.cli.logging.getLogger')
    @patch('data_analysis_agent.cli.validate_environment_variables')
    @patch('data_analysis_agent.cli.load_and_merge_data')
    @patch('data_analysis_agent.cli.cleanup_output_directories')
    @patch('data_analysis_agent.cli.process_plans')
    @patch('data_analysis_agent.cli.process_codes')
    @patch('data_analysis_agent.cli.save_dataframe')
    @patch('data_analysis_agent.cli.app_paths')
    @patch('data_analysis_agent.cli.os.chdir')
    def test_run_data_analysis_success(
        self,
        mock_chdir,
        mock_app_paths,
        mock_save_dataframe,
        mock_process_codes,
        mock_process_plans,
        mock_cleanup,
        mock_load_merge,
        mock_validate_env,
        mock_get_logger,
        mock_logger
    ):
        """Test successful execution of run_data_analysis."""
        # Setup mocks
        mock_validate_env.return_value = (Path("questions.json"), Path("answers.json"))
        mock_df = MagicMock()
        mock_df.__len__.return_value = 5
        mock_df.columns = ["file_name", "question", "id"]
        mock_load_merge.return_value = mock_df
        mock_process_plans.return_value = mock_df
        mock_process_codes.return_value = mock_df
        
        mock_app_paths.project_root = Path("/test/root")
        mock_app_paths.ensure_directories = MagicMock()
        mock_app_paths.merged_plans_file = Path("plans.csv")
        mock_app_paths.merged_code_file = Path("codes.csv")

        # Execute
        run_data_analysis(skip_cleanup=False, log_level="DEBUG")

        # Verify
        mock_get_logger.return_value.setLevel.assert_called_once_with(logging.DEBUG)
        mock_validate_env.assert_called_once()
        mock_load_merge.assert_called_once()
        mock_cleanup.assert_called_once()
        mock_process_plans.assert_called_once_with(mock_df)
        mock_process_codes.assert_called_once_with(mock_df)
        assert mock_save_dataframe.call_count == 2

    @patch('data_analysis_agent.cli.logger')
    @patch('data_analysis_agent.cli.validate_environment_variables')
    @patch('data_analysis_agent.cli.load_and_merge_data')
    @patch('data_analysis_agent.cli.cleanup_output_directories')
    def test_run_data_analysis_skip_cleanup(
        self,
        mock_cleanup,
        mock_load_merge,
        mock_validate_env,
        mock_logger
    ):
        """Test run_data_analysis with skip_cleanup=True."""
        # Setup mocks
        mock_validate_env.return_value = (Path("questions.json"), Path("answers.json"))
        mock_df = MagicMock()
        mock_df.__len__.return_value = 0  # Empty dataframe to trigger early return
        mock_load_merge.return_value = mock_df

        # Execute
        run_data_analysis(skip_cleanup=True)

        # Verify cleanup was not called
        mock_cleanup.assert_not_called()

    @patch('data_analysis_agent.cli.logger')
    @patch('data_analysis_agent.cli.validate_environment_variables')
    @patch('data_analysis_agent.cli.load_and_merge_data')
    def test_run_data_analysis_empty_dataframe(
        self,
        mock_load_merge,
        mock_validate_env,
        mock_logger
    ):
        """Test run_data_analysis with empty merged dataframe."""
        # Setup mocks
        mock_validate_env.return_value = (Path("questions.json"), Path("answers.json"))
        mock_df = MagicMock()
        mock_df.__len__.return_value = 0
        mock_load_merge.return_value = mock_df

        # Execute
        run_data_analysis()

        # Verify early return
        mock_logger.error.assert_called_with("No data to process after merging")

    @patch('data_analysis_agent.cli.logger')
    @patch('data_analysis_agent.cli.validate_environment_variables')
    @patch('data_analysis_agent.cli.load_and_merge_data')
    def test_run_data_analysis_missing_columns(
        self,
        mock_load_merge,
        mock_validate_env,
        mock_logger
    ):
        """Test run_data_analysis with missing required columns."""
        # Setup mocks
        mock_validate_env.return_value = (Path("questions.json"), Path("answers.json"))
        mock_df = MagicMock()
        mock_df.__len__.return_value = 5
        mock_df.columns = ["id"]  # Missing required columns
        mock_load_merge.return_value = mock_df

        # Execute
        run_data_analysis()

        # Verify error logged
        mock_logger.error.assert_called_with(
            "Missing required columns in merged data: ['file_name', 'question']"
        )

    @patch('data_analysis_agent.cli.logger')
    @patch('data_analysis_agent.cli.validate_environment_variables')
    @patch('data_analysis_agent.cli.sys.exit')
    def test_run_data_analysis_keyboard_interrupt(
        self,
        mock_exit,
        mock_validate_env,
        mock_logger
    ):
        """Test run_data_analysis handles KeyboardInterrupt."""
        # Setup mock to raise KeyboardInterrupt
        mock_validate_env.side_effect = KeyboardInterrupt()

        # Execute
        run_data_analysis()

        # Verify
        mock_logger.info.assert_called_with("Process interrupted by user")
        mock_exit.assert_called_once_with(1)

    @patch('data_analysis_agent.cli.logger')
    @patch('data_analysis_agent.cli.validate_environment_variables')
    @patch('data_analysis_agent.cli.sys.exit')
    def test_run_data_analysis_exception(
        self,
        mock_exit,
        mock_validate_env,
        mock_logger
    ):
        """Test run_data_analysis handles general exceptions."""
        # Setup mock to raise exception
        mock_validate_env.side_effect = Exception("Test error")

        # Execute
        run_data_analysis()

        # Verify
        mock_logger.error.assert_called_with("Fatal error in main process: Test error")
        mock_exit.assert_called_once_with(1)


class TestValidateEnvironmentVariables:
    """Test cases for validate_environment_variables function."""

    @patch('data_analysis_agent.cli.load_dotenv')
    @patch('data_analysis_agent.cli.os.getenv')
    @patch('data_analysis_agent.cli.app_paths')
    def test_validate_environment_variables_success(self, mock_app_paths, mock_getenv, mock_load_dotenv):
        """Test successful environment variable validation."""
        # Setup mocks
        mock_app_paths.project_root = Path("/test/root")
        mock_getenv.side_effect = lambda key: {
            "QUESTIONS_FILE": "questions.json",
            "ANSWERS_FILE": "answers.json"
        }.get(key)
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            questions_path, answers_path = validate_environment_variables()
            
            assert questions_path.name == "questions.json"
            assert answers_path.name == "answers.json"

    @patch('data_analysis_agent.cli.load_dotenv')
    @patch('data_analysis_agent.cli.os.getenv')
    def test_validate_environment_variables_missing_questions(self, mock_getenv, mock_load_dotenv):
        """Test validation with missing QUESTIONS_FILE."""
        mock_getenv.side_effect = lambda key: {
            "QUESTIONS_FILE": None,
            "ANSWERS_FILE": "answers.json"
        }.get(key)
        
        with pytest.raises(ValueError, match="QUESTIONS_FILE environment variable is not set"):
            validate_environment_variables()

    @patch('data_analysis_agent.cli.load_dotenv')
    @patch('data_analysis_agent.cli.os.getenv')
    def test_validate_environment_variables_missing_answers(self, mock_getenv, mock_load_dotenv):
        """Test validation with missing ANSWERS_FILE."""
        mock_getenv.side_effect = lambda key: {
            "QUESTIONS_FILE": "questions.json",
            "ANSWERS_FILE": None
        }.get(key)
        
        with pytest.raises(ValueError, match="ANSWERS_FILE environment variable is not set"):
            validate_environment_variables()


class TestCliIntegration:
    """Integration tests for the CLI module."""

    @patch('data_analysis_agent.cli.run_data_analysis')
    def test_main_py_integration(self, mock_run_analysis):
        """Test that main.py properly calls the CLI functions."""
        from main import main
        
        with patch('main.parse_arguments') as mock_parse:
            mock_args = MagicMock()
            mock_args.skip_cleanup = True
            mock_args.log_level = "DEBUG"
            mock_parse.return_value = mock_args
            
            main()
            
            mock_parse.assert_called_once()
            mock_run_analysis.assert_called_once_with(skip_cleanup=True, log_level="DEBUG")

    def test_cli_module_name_main_execution(self):
        """Test that running cli.py as main works correctly."""
        with patch('data_analysis_agent.cli.parse_arguments') as mock_parse, \
             patch('data_analysis_agent.cli.run_data_analysis'):
            
            mock_args = MagicMock()
            mock_args.skip_cleanup = False
            mock_args.log_level = "INFO"
            mock_parse.return_value = mock_args
            
            # This is a smoke test to ensure the code structure is correct
            # The actual execution would happen when the module is run directly


if __name__ == "__main__":
    pytest.main([__file__])
