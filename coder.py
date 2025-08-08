"""
Code generation module for data analysis tasks.
This module provides functionality to generate Python code for data analysis based on user
questions and analysis plans. It uses language models to create executable code that can
answer specific questions about datasets.
The module implements a singleton pattern for the CodeGenerator class to ensure efficient
resource management and prevent unnecessary file cleanup operations during a session.
Key Features:
- Automatic code generation from natural language questions and analysis plans
- Structured output using Pydantic models for code and assumptions
- Singleton-based directory management with cleanup on first use
- Support for DataFrame schema integration
- Timestamped file generation for code tracking
- Robust error handling and fallback parsing mechanisms
Classes:
    CodeResponse: Pydantic model for structured code generation responses
    CodeGenerator: Singleton class managing code generation and file operations
Functions:
    create_code: Backward-compatible function for code generation
Dependencies:
    - langchain: For language model integration
    - pydantic: For data validation and structured outputs
    - dotenv: For environment variable management
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from prompts import SystemPrompts
from paths import get_paths

load_dotenv(override=True)


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


class CodeGenerator:
    """
    A singleton-like class for managing code generation and file cleanup.

    This class ensures that the code directory is cleaned only once per session,
    preventing the accumulation of hundreds of test files while avoiding
    repeated cleanup on every code generation call.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, code_dir: Optional[Path] = None, clean_on_first_use: bool = True
    ):
        # Only initialize once
        if not hasattr(self, "code_dir"):
            self.code_dir = code_dir if code_dir is not None else get_paths().code_dir
            self.clean_on_first_use = clean_on_first_use

    def _ensure_directory_ready(self):
        """Ensure directory exists and clean it only on first use."""
        if not self._initialized:
            if self.clean_on_first_use and self.code_dir.exists():
                # Clean existing files only on first use (skip log files)
                for file in self.code_dir.iterdir():
                    if file.is_file() and not file.name.endswith(".log"):
                        file.unlink()

            os.makedirs(self.code_dir, exist_ok=True)
            CodeGenerator._initialized = True

    def create_code(
        self, plan: str, question: str, df_json: str, data_file_name: Path
    ) -> CodeResponse:
        """
        Generate Python code based on a data analysis plan and question.
        This method uses a language model to create code that answers a specific question
        about a dataset, following a provided analysis plan. The generated code is saved
        to a file and returned as a structured response.
        Args:
            plan (str): The analysis plan or strategy to follow when generating code
            question (str): The specific question to be answered by the generated code
            df_json (str): JSON representation of the DataFrame structure/schema
            data_file_name (Path): Name for the output Python file (without .py extension)
        Returns:
            CodeResponse: Structured response containing the generated code and metadata
        Raises:
            ValueError: If the language model doesn't return a CodeResponse object
        Side Effects:
            - Creates and cleans the code directory only on first use
            - Writes the generated code to '{file_name}_{timestamp}.py' in the code directory
        """

        system_message = SystemMessage(content=SystemPrompts.coder)
        data = {"question": question, "plan": plan, "data_frame_structure": df_json}
        human_message = HumanMessage(content=json.dumps(data, indent=2))
        messages = [system_message, human_message]

        llm = init_chat_model(
            "openai:gpt-5",
            # temperature=0.7,
            max_retries=3,
            output_version="responses/v1",
        )

        # Use structured output with raw response as fallback
        structured_llm = llm.with_structured_output(
            schema=CodeResponse, include_raw=True
        )

        try:
            result: Dict[str, Any] = structured_llm.invoke(messages)

            # If structured parsing succeeded, use the parsed result
            if (
                "parsed" in result
                and result["parsed"] is not None
                and isinstance(result["parsed"], CodeResponse)
            ):
                resp = result["parsed"]
            else:
                # Fallback: try to parse JSON from the raw response
                raw_content: str = str(result["raw"].content)

                # Try to extract JSON from the response
                json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Clean up common JSON issues
                    json_str = json_str.replace("\n", "\\n").replace("\r", "\\r")
                    parsed_data = json.loads(json_str)
                    resp = CodeResponse(**parsed_data)
                else:
                    # If no JSON found, treat entire content as code
                    resp = CodeResponse(
                        code=raw_content,
                        assumptions=[
                            "Generated from raw LLM output due to JSON parsing issues"
                        ],
                    )

        except Exception as e:
            raise ValueError(f"Error invoking model: {e}") from e

        # Ensure directory is ready (clean only on first use)
        try:
            self._ensure_directory_ready()
        except Exception as e:
            raise ValueError(f"Error preparing directory: {e}") from e

        # Generate timestamped filename
        datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = self.code_dir / f"{data_file_name.stem}_{datetime_str}.py"

        # Write the generated code
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(resp.code)

        return resp


# Global instance for backward compatibility
_code_generator = CodeGenerator()


def create_code(
    plan: str, question: str, df_json: str, data_file_name: Path
) -> CodeResponse:
    """
    Generate Python code based on a data analysis plan and question.
    This function maintains backward compatibility by delegating to the CodeGenerator singleton.

    Args:
        plan (str): The analysis plan or strategy to follow when generating code
        question (str): The specific question to be answered by the generated code
        df_json (str): JSON representation of the DataFrame structure/schema
        data_file_name (Path): Name for the output Python file (without .py extension)
    Returns:
        CodeResponse: Structured response containing the generated code and metadata
    """
    return _code_generator.create_code(plan, question, df_json, data_file_name)
