import json
import os
from datetime import datetime, timezone
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from prompts import SystemPrompts

load_dotenv(override=True)

PLAN_DIR = Path("data") / "plan"


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


def create_plan(question: str, df_json: str, file_name: Path) -> Plan:
    """
    Create an analysis plan based on a user question and DataFrame structure.
    This function uses a language model to generate a structured plan for data analysis
    by combining a user's question with DataFrame metadata. The plan is generated using
    OpenAI's GPT-4.1 model and follows a predefined schema.
    Args:
        question (str): The user's question or analysis request
        df_json (str): JSON string containing the DataFrame structure and metadata
    Returns:
        Plan: A structured plan object containing the analysis steps
    Raises:
        ValueError: If the language model doesn't return a Plan object as expected
    Side Effects:
        - Writes the generated plan to "planner_output.json" file
        - Makes API calls to OpenAI's language model
    Example:
        >>> plan = create_plan("What are the top 5 products by sales?", df_structure_json)
        >>> print(plan.steps)
    """

    system_message = SystemMessage(
        content=SystemPrompts.planner,
    )

    data = {
        "question": question,
        "file_name": file_name.as_posix(),
        "data_frame_structure": df_json
    }

    human_message = HumanMessage(content=json.dumps(data, indent=2))

    messages = [system_message, human_message]

    llm = init_chat_model(
        "openai:gpt-4.1", temperature=0.7, max_retries=3, output_version="responses/v1"
    )
    structured_llm = llm.with_structured_output(schema=Plan)
    result = structured_llm.invoke(messages)
    if isinstance(result, Plan):
        resp = result
    else:
        raise ValueError(f"Expected Plan, got {type(result)}: {result}")

    try:
        # If directory exists, remove all files in it
        if PLAN_DIR.exists():
            for file in PLAN_DIR.iterdir():
                if file.is_file():
                    file.unlink()
        
        # Ensure the directory exists
        os.makedirs(PLAN_DIR, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Error managing directory: {e}") from e

    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = PLAN_DIR / f"{file_name.stem}_{datetime_str}.py"

    with open(PLAN_DIR / f"{file_name.stem}.json", "w", encoding="utf-8") as f:
        json.dump(resp.model_dump(), f, indent=2)
    return resp
