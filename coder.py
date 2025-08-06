import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from prompts import SystemPrompts

load_dotenv(override=True)

CODE_DIR = Path("data") / "code"


class CodeResponse(BaseModel):
    """
    A Pydantic model representing a response containing executable Python code and related assumptions.
    This model is used to structure responses that include both the generated Python code
    and any assumptions that were made during the code generation process.
    Attributes:
        code (str): The Python code to execute the task.
        assumptions (List[str]): Assumptions made while generating the code, 
                                especially if the user request was unclear.
    """

    code: str = Field(description="The Python code to execute the task.")
    assumptions: List[str] = Field(
        description="Assumptions made while generating the code, especially if the user request was unclear."
    )


def create_code(plan: str, question: str, df_json: str, file_name: Path) -> CodeResponse:
    """
    Generate Python code based on a data analysis plan and question.
    This function uses a language model to create code that answers a specific question
    about a dataset, following a provided analysis plan. The generated code is saved
    to a file and returned as a structured response.
    Args:
        plan (str): The analysis plan or strategy to follow when generating code
        question (str): The specific question to be answered by the generated code
        df_json (str): JSON representation of the DataFrame structure/schema
        file_name (str): Name for the output Python file (without .py extension)
    Returns:
        CodeResponse: Structured response containing the generated code and metadata
    Raises:
        ValueError: If the language model doesn't return a CodeResponse object
    Side Effects:
        - Creates a directory 'data/code' if it doesn't exist
        - Writes the generated code to '{file_name}.py' in the current directory
    """

    system_message = SystemMessage(
        content=SystemPrompts.coder,
    )

    data = {"question": question, "plan": plan, "data_frame_structure": df_json}

    human_message = HumanMessage(content=json.dumps(data, indent=2))

    messages = [system_message, human_message]

    llm = init_chat_model(
        "openai:gpt-4.1", temperature=0.7, max_retries=3, output_version="responses/v1"
    )
    structured_llm = llm.with_structured_output(schema=CodeResponse)
    result = structured_llm.invoke(messages)
    if isinstance(result, CodeResponse):
        resp = result
    else:
        raise ValueError(f"Expected CodeResponse, got {type(result)}: {result}")

    # Save the generated code to a file

    try:

        # If directory exists, remove all files in it
        if CODE_DIR.exists():
            for file in CODE_DIR.iterdir():
                if file.is_file():
                    file.unlink()

        os.makedirs(CODE_DIR, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Error creating directory: {e}") from e
        # Ensure the directory exists before writing the file

    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = CODE_DIR / f"{file_name.stem}_{datetime_str}.py"

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(resp.code)
    return resp
