import json
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


class CodeResponse(BaseModel):
    code: str = Field(description="The Python code to execute the task.")
    assumptions: List[str] = Field(
        description="Assumptions made while generating the code, especially if the user request was unclear."
    )


def create_code(plan: str, question: str, df_json: str) -> CodeResponse:
    system_message = SystemMessage(
        content=SystemPrompts.coder,
    )

    df_structure = "DataFrame Structure:\n" + df_json

    human_message = HumanMessage(
        content=f"Human Request:\n{question}\n\n"
        + f"Plan: {plan}\n\n"
        + df_structure
    )

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

    with open("coder_output.json", "w", encoding="utf-8") as f:
        json.dump(resp.model_dump(), f, indent=2)
    return resp
