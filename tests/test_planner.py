"""
Unit tests for the planner module.

These tests validate the planner's ability to create analysis plans
based on user questions and DataFrame structures. The tests use real
LLM calls to ensure end-to-end functionality.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
import io

import pandas as pd
from dotenv import load_dotenv

from data_analysis_agent.planner import Plan, Task, PlanGenerator, create_plan
from data_analysis_agent.dataframe_to_dict import parse_dataframe_info_with_columns

# Load environment variables
load_dotenv()


def create_dataframe_json(df: pd.DataFrame) -> str:
    """Create DataFrame info JSON using the same method as the CLI."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return parse_dataframe_info_with_columns(buffer.getvalue(), list(df.columns))


class TestTask:
    """Test the Task model."""
    
    def test_task_creation(self):
        """Test that a Task can be created with all required fields."""
        task = Task(
            task_name="Load and examine data",
            task_number=1,
            details="Load the CSV file and examine its structure",
            dependencies="Input CSV file",
            output="DataFrame with loaded data",
            assumptions="CSV file is well-formatted"
        )
        
        assert task.task_name == "Load and examine data"
        assert task.task_number == 1
        assert task.details == "Load the CSV file and examine its structure"
        assert task.dependencies == "Input CSV file"
        assert task.output == "DataFrame with loaded data"
        assert task.assumptions == "CSV file is well-formatted"


class TestPlan:
    """Test the Plan model."""
    
    def test_plan_creation(self):
        """Test that a Plan can be created with a list of tasks."""
        task1 = Task(
            task_name="Load data",
            task_number=1,
            details="Load CSV file",
            dependencies="CSV file",
            output="DataFrame",
            assumptions="File exists"
        )
        task2 = Task(
            task_name="Analyze data",
            task_number=2,
            details="Perform analysis",
            dependencies="DataFrame from task 1",
            output="Analysis results",
            assumptions="Data is clean"
        )
        
        plan = Plan(task_list=[task1, task2])
        
        assert len(plan.task_list) == 2
        assert plan.task_list[0].task_name == "Load data"
        assert plan.task_list[1].task_name == "Analyze data"


class TestPlanGenerator:
    """Test the PlanGenerator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40],
            'salary': [50000, 60000, 70000, 80000],
            'department': ['IT', 'HR', 'Finance', 'IT']
        })
    
    @pytest.fixture
    def sample_csv_file(self, temp_dir, sample_dataframe):
        """Create a sample CSV file for testing."""
        csv_path = temp_dir / "sample_data.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        return csv_path
    
    def test_plan_generator_singleton(self, temp_dir):
        """Test that PlanGenerator follows singleton pattern."""
        # Reset the singleton for testing
        PlanGenerator._instance = None
        PlanGenerator._initialized = False
        
        generator1 = PlanGenerator(plan_dir=temp_dir)
        generator2 = PlanGenerator(plan_dir=temp_dir)
        
        assert generator1 is generator2
    
    def test_plan_generator_initialization(self, temp_dir):
        """Test PlanGenerator initialization."""
        # Reset the singleton for testing
        PlanGenerator._instance = None
        PlanGenerator._initialized = False
        
        generator = PlanGenerator(plan_dir=temp_dir, clean_on_first_use=False)
        
        assert generator.plan_dir == temp_dir
        assert generator.clean_on_first_use is False
    
    def test_ensure_directory_ready(self, temp_dir):
        """Test that _ensure_directory_ready creates the directory."""
        # Reset the singleton for testing
        PlanGenerator._instance = None
        PlanGenerator._initialized = False
        
        plan_subdir = temp_dir / "plans"
        generator = PlanGenerator(plan_dir=plan_subdir, clean_on_first_use=False)
        
        assert not plan_subdir.exists()
        generator._ensure_directory_ready()
        assert plan_subdir.exists()
    
    def test_ensure_directory_ready_cleanup(self, temp_dir):
        """Test that _ensure_directory_ready cleans existing files when requested."""
        # Reset the singleton for testing
        PlanGenerator._instance = None
        PlanGenerator._initialized = False
        
        # Create some existing files
        existing_file = temp_dir / "existing_plan.json"
        existing_file.write_text('{"old": "plan"}')
        
        generator = PlanGenerator(plan_dir=temp_dir, clean_on_first_use=True)
        generator._ensure_directory_ready()
        
        assert not existing_file.exists()
    
    @pytest.mark.integration
    def test_create_plan_with_real_llm(self, temp_dir, sample_csv_file, sample_dataframe):
        """Test create_plan with a real LLM call."""
        # Skip if no API key is available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OpenAI API key available for integration testing")
        
        # Reset the singleton for testing
        PlanGenerator._instance = None
        PlanGenerator._initialized = False
        
        generator = PlanGenerator(plan_dir=temp_dir, clean_on_first_use=False)
        
        # Create DataFrame info JSON using the helper function
        df_json = create_dataframe_json(sample_dataframe)
        
        question = "What is the average salary by department?"
        
        plan = generator.create_plan(question, df_json, sample_csv_file)
        
        # Verify the plan structure
        assert isinstance(plan, Plan)
        assert len(plan.task_list) > 0
        
        # Verify that each task has the required fields
        for task in plan.task_list:
            assert isinstance(task, Task)
            assert len(task.task_name) > 0
            assert isinstance(task.task_number, int)
            assert task.task_number > 0
            assert len(task.details) > 0
            assert len(task.dependencies) > 0
            assert len(task.output) > 0
            assert len(task.assumptions) > 0
        
        # Verify that a plan file was created
        plan_files = list(temp_dir.glob(f"{sample_csv_file.stem}_*.json"))
        assert len(plan_files) == 1
        
        # Verify the content of the written file
        with open(plan_files[0], 'r') as f:
            saved_plan = json.load(f)
        
        assert 'task_list' in saved_plan
        assert len(saved_plan['task_list']) == len(plan.task_list)


class TestTitanicData:
    """Test with real Titanic dataset."""
    
    @pytest.fixture
    def titanic_csv_path(self):
        """Path to the Titanic CSV file."""
        return Path(__file__).parent / "assets" / "titanic.csv"
    
    @pytest.fixture
    def titanic_dataframe(self, titanic_csv_path):
        """Load the Titanic dataset."""
        if not titanic_csv_path.exists():
            pytest.skip(f"Titanic dataset not found at {titanic_csv_path}")
        return pd.read_csv(titanic_csv_path)
    
    @pytest.mark.integration
    def test_create_plan_titanic_fare_analysis(self, titanic_csv_path, titanic_dataframe):
        """Test creating a plan for Titanic fare analysis."""
        # Skip if no API key is available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OpenAI API key available for integration testing")
            
        if not titanic_csv_path.exists():
            pytest.skip(f"Titanic dataset not found at {titanic_csv_path}")
        
        # Reset the singleton for testing
        PlanGenerator._instance = None
        PlanGenerator._initialized = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            generator = PlanGenerator(plan_dir=temp_path, clean_on_first_use=False)
            
            # Create DataFrame info JSON using the helper function
            df_json = create_dataframe_json(titanic_dataframe)
            
            question = ("Perform a distribution analysis on the 'Fare' column for each passenger class "
                        "('Pclass') separately. Calculate the mean, median, and standard deviation of the "
                        "fare for each class.")
            
            plan = generator.create_plan(question, df_json, titanic_csv_path)
            
            # Verify the plan structure
            assert isinstance(plan, Plan)
            assert len(plan.task_list) > 0
            
            # Check that the plan contains relevant tasks for this analysis
            task_names = [task.task_name.lower() for task in plan.task_list]
            task_details = " ".join([task.details.lower() for task in plan.task_list])
            
            # Should include data loading
            assert any("load" in name or "read" in name for name in task_names)
            
            # Should mention fare and pclass analysis
            assert "fare" in task_details
            assert "pclass" in task_details or "class" in task_details
            
            # Should mention statistical measures
            assert any(stat in task_details for stat in ["mean", "median", "standard deviation"])


class TestCreatePlanFunction:
    """Test the standalone create_plan function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'product': ['A', 'B', 'C', 'D'],
            'sales': [100, 200, 150, 300],
            'region': ['North', 'South', 'North', 'West']
        })
    
    @pytest.fixture
    def sample_csv_file(self, temp_dir, sample_dataframe):
        """Create a sample CSV file for testing."""
        csv_path = temp_dir / "sales_data.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.mark.integration
    def test_create_plan_function(self, sample_csv_file, sample_dataframe):
        """Test the standalone create_plan function."""
        # Skip if no API key is available
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("No OpenAI API key available for integration testing")
        
        # Reset the singleton for testing
        PlanGenerator._instance = None
        PlanGenerator._initialized = False
        
        # Create DataFrame info JSON using the helper function
        df_json = create_dataframe_json(sample_dataframe)
        
        question = "What are the total sales by region?"
        
        plan = create_plan(question, df_json, sample_csv_file)
        
        # Verify the plan structure
        assert isinstance(plan, Plan)
        assert len(plan.task_list) > 0
        
        # Verify that each task has the required fields
        for task in plan.task_list:
            assert isinstance(task, Task)
            assert len(task.task_name) > 0
            assert isinstance(task.task_number, int)
            assert task.task_number > 0


class TestErrorHandling:
    """Test error handling in the planner."""
    
    def test_invalid_plan_response(self):
        """Test handling of invalid LLM response."""
        # This would require mocking the LLM to return invalid data
        # For now, we'll just ensure the structure is correct
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Reset the singleton for testing
            PlanGenerator._instance = None
            PlanGenerator._initialized = False
            
            generator = PlanGenerator(plan_dir=temp_path, clean_on_first_use=False)
            
            # The actual error handling would be tested with mocked LLM responses
            # This is a placeholder for that test
            assert generator.plan_dir == temp_path


if __name__ == "__main__":
    pytest.main([__file__])
