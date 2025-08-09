import json
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from data_analysis_agent.prompts import SystemPrompts
from data_analysis_agent.paths import get_paths

load_dotenv(override=True)


class Task(BaseModel):
    """
    Represents a single task in a data analysis workflow.
    This class defines the structure for breaking down complex data analysis requests
    into manageable, executable tasks. Each task contains detailed information about
    what needs to be accomplished, its dependencies, and expected outputs.
    Attributes:
        task_name (str): Short description of the coding task.
        details (str): Step-by-step description of what must be done, including
            any transformations or conditions.
        dependencies (str): Data or previous tasks this step depends on.
        output (str): What this task should produce.
        assumptions (str): Any assumptions made to proceed with the task,
            especially if the user request was unclear.
    """

    task_name: str = Field(description="Short description of the coding task.")
    task_number: int = Field(description="Task number in the plan, starting from 1.")
    details: str = Field(
        description="Step-by-step description of what must be done, including any transformations or conditions."
    )
    dependencies: str = Field(
        description="Data or previous tasks this step depends on."
    )
    output: str = Field(description="What this task should produce.")
    assumptions: str = Field(
        description="Any assumptions made to proceed with the task, especially if the user request was unclear."
    )


class Plan(BaseModel):
    """
    A data structure representing a collection of tasks organized as a plan.

    This class serves as a container for a list of Task objects, providing
    a structured way to organize and manage multiple tasks within a data
    analysis workflow.

    Attributes:
        task_list (List[Task]): A list of Task objects that make up this plan.
            Each task represents a specific action or operation to be performed
            as part of the overall plan execution.
    """

    task_list: List[Task]


class PlanGenerator:
    """
    A singleton-like class for managing plan generation and file cleanup.

    This class ensures that the plan directory is cleaned only once per session,
    preventing the accumulation of hundreds of test files while avoiding
    repeated cleanup on every plan generation call.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, plan_dir: Optional[Path] = None, clean_on_first_use: bool = True
    ):
        # Only initialize once
        if not hasattr(self, "plan_dir"):
            self.plan_dir = plan_dir if plan_dir is not None else get_paths().plan_dir
            self.clean_on_first_use = clean_on_first_use

    def _ensure_directory_ready(self):
        """Ensure directory exists and clean it only on first use."""
        if not self._initialized:
            if self.clean_on_first_use and self.plan_dir.exists():
                # Clean existing files only on first use
                for file in self.plan_dir.iterdir():
                    if file.is_file():
                        file.unlink()

            self.plan_dir.mkdir(parents=True, exist_ok=True)
            PlanGenerator._initialized = True

    def create_plan(self, question: str, df_json: str, data_file_name: Path) -> Plan:
        """
        Create an analysis plan based on a user question and DataFrame structure.
        This method uses a language model to generate a structured plan for data analysis
        by combining a user's question with DataFrame metadata. The plan is generated using
        OpenAI's GPT-4.1 model and follows a predefined schema.
        Args:
            question (str): The user's question or analysis request
            df_json (str): JSON string containing the DataFrame structure and metadata
            file_name (Path): Name for the output plan file (without extension)
        Returns:
            Plan: A structured plan object containing the analysis steps
        Raises:
            ValueError: If the language model doesn't return a Plan object as expected
        Side Effects:
            - Creates and cleans the plan directory only on first use
            - Writes the generated plan to '{data_file_name}_{timestamp}.json' in the plan directory
        """

        system_message = SystemMessage(content=SystemPrompts.planner)
        # data_file_name is already an absolute path from main.py
        data = {
            "question": question,
            "file_name": data_file_name.as_posix(),
            "data_frame_structure": df_json,
        }
        human_message = HumanMessage(content=json.dumps(data, indent=2))
        messages = [system_message, human_message]

        llm = init_chat_model(
            "openai:gpt-5",
            # temperature=0.7,
            max_retries=3,
            output_version="responses/v1",
        )
        structured_llm = llm.with_structured_output(schema=Plan)
        result = structured_llm.invoke(messages)

        if isinstance(result, Plan):
            resp = result
        else:
            raise ValueError(f"Expected Plan, got {type(result)}: {result}")

        # Ensure directory is ready (clean only on first use)
        try:
            self._ensure_directory_ready()
        except Exception as e:
            raise ValueError(f"Error preparing directory: {e}") from e

        # Generate timestamped filename
        datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = self.plan_dir / f"{data_file_name.stem}_{datetime_str}.json"

        # Write the generated plan
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(resp.model_dump(), f, indent=2)

        return resp


# Global instance for backward compatibility
_plan_generator = PlanGenerator()


def create_plan(question: str, df_json: str, data_file_name: Path) -> Plan:
    """
    Create an analysis plan based on a user question and DataFrame structure.
    This function maintains backward compatibility by delegating to the PlanGenerator singleton.

    Args:
        question (str): The user's question or analysis request
        df_json (str): JSON string containing the DataFrame structure and metadata
        data_file_name (Path): Name for the output plan file (without extension)
    Returns:
        Plan: A structured plan object containing the analysis steps
    """
    return _plan_generator.create_plan(question, df_json, data_file_name)
