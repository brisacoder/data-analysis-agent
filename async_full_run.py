import asyncio
import io
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
# Import LangChain modules
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from coder import CodeResponse
# Import your existing modules
from dataframe_to_dict import parse_dataframe_info
from planner import Plan
from prompts import SystemPrompts


def df_info_to_json(df):
    """Convert DataFrame info to JSON format."""
    buffer = io.StringIO()
    df.info(buf=buffer, show_counts=True)
    df_json = parse_dataframe_info(buffer.getvalue())
    return df_json


async def create_plan_async(question: str, df_json: str) -> Plan:
    """Async version of create_plan function."""
    system_message = SystemMessage(
        content=SystemPrompts.planner,
    )

    df_structure = f"DataFrame Structure:\n{df_json}"
    human_message = HumanMessage(content=f"{question}\n\n{df_structure}")
    messages = [system_message, human_message]

    llm = init_chat_model(
        "openai:gpt-4.1", temperature=0.7, max_retries=3, output_version="responses/v1"
    )
    structured_llm = llm.with_structured_output(schema=Plan)

    # Use asyncio to run the synchronous invoke in a thread pool
    result = await asyncio.get_event_loop().run_in_executor(
        None, structured_llm.invoke, messages
    )

    if isinstance(result, Plan):
        return result
    else:
        raise ValueError(f"Expected Plan, got {type(result)}: {result}")


async def create_code_async(plan: str, question: str, df_json: str) -> CodeResponse:
    """Async version of create_code function."""
    system_message = SystemMessage(
        content=SystemPrompts.coder,
    )

    df_structure = "DataFrame Structure:\n" + df_json

    human_message = HumanMessage(
        content=f"Plan: {plan}\n\n" + f"Human Request:\n{question}\n\n" + df_structure
    )

    messages = [system_message, human_message]

    llm = init_chat_model(
        "openai:gpt-4.1", temperature=0.7, max_retries=3, output_version="responses/v1"
    )
    structured_llm = llm.with_structured_output(schema=CodeResponse)

    # Use asyncio to run the synchronous invoke in a thread pool
    result = await asyncio.get_event_loop().run_in_executor(
        None, structured_llm.invoke, messages
    )

    if isinstance(result, CodeResponse):
        return result
    else:
        raise ValueError(f"Expected CodeResponse, got {type(result)}: {result}")


async def process_row_plan(index: int, row: pd.Series, path_prefix: Path) -> tuple:
    """Process a single row to create a plan."""
    df = pd.read_csv(path_prefix / row["file_name"])
    df_json = df_info_to_json(df)
    plan = await create_plan_async(row["question"], df_json)
    return index, plan.model_dump_json()


async def process_row_code(index: int, row: pd.Series, path_prefix: Path) -> tuple:
    """Process a single row to create code."""
    df = pd.read_csv(path_prefix / row["file_name"])
    df_json = df_info_to_json(df)
    code = await create_code_async(row["plan"], row["question"], df_json)
    return index, code.model_dump_json()


async def create_plans_parallel(
    df_merged: pd.DataFrame, path_prefix: Path, max_concurrent: int = 5
) -> pd.DataFrame:
    """Create plans for all rows in parallel with concurrency control."""
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(index, row):
        async with semaphore:
            return await process_row_plan(index, row, path_prefix)

    # Create tasks for all rows
    tasks = [process_with_semaphore(index, row) for index, row in df_merged.iterrows()]

    print(
        f"🚀 Starting async planning for {len(tasks)} rows with max {max_concurrent} concurrent requests..."
    )

    # Execute all tasks and collect results
    results = await asyncio.gather(*tasks)

    # Update the dataframe with results
    df_result = df_merged.copy()
    for index, plan_json in results:
        df_result.at[index, "plan"] = plan_json

    return df_result


async def create_code_parallel(
    df_merged: pd.DataFrame, path_prefix: Path, max_concurrent: int = 5
) -> pd.DataFrame:
    """Create code for all rows in parallel with concurrency control."""
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(index, row):
        async with semaphore:
            return await process_row_code(index, row, path_prefix)

    # Create tasks for all rows
    tasks = [process_with_semaphore(index, row) for index, row in df_merged.iterrows()]

    print(
        f"🚀 Starting async coding for {len(tasks)} rows with max {max_concurrent} concurrent requests..."
    )

    # Execute all tasks and collect results
    results = await asyncio.gather(*tasks)

    # Update the dataframe with results
    df_result = df_merged.copy()
    for index, code_json in results:
        df_result.at[index, "code"] = code_json

    return df_result


async def process_complete_workflow(
    df_merged: pd.DataFrame, path_prefix: Path, max_concurrent: int = 3
) -> pd.DataFrame:
    """Complete async workflow that processes planning and coding for all rows."""
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total_rows = len(df_merged)

    async def process_single_row(index: int, row: pd.Series) -> tuple:
        nonlocal completed
        async with semaphore:
            # Load the dataframe once
            df = pd.read_csv(path_prefix / row["file_name"])
            df_json = df_info_to_json(df)

            # Create plan
            plan = await create_plan_async(row["question"], df_json)
            plan_json = plan.model_dump_json()

            # Create code using the plan
            code = await create_code_async(plan_json, row["question"], df_json)
            code_json = code.model_dump_json()

            completed += 1
            if completed % max(1, total_rows // 10) == 0:  # Progress every 10%
                progress = (completed / total_rows) * 100
                print(f"✅ Progress: {completed}/{total_rows} ({progress:.1f}%)")

            return index, plan_json, code_json

    # Create tasks for all rows
    tasks = [process_single_row(index, row) for index, row in df_merged.iterrows()]

    print(
        f"🚀 Starting complete async workflow for {len(tasks)} rows with max {max_concurrent} concurrent requests..."
    )

    # Execute all tasks and collect results
    results = await asyncio.gather(*tasks)

    # Update the dataframe with results
    df_result = df_merged.copy()
    for index, plan_json, code_json in results:
        df_result.at[index, "plan"] = plan_json
        df_result.at[index, "code"] = code_json

    return df_result


async def process_with_timing(
    df_merged: pd.DataFrame, path_prefix: Path, max_concurrent: int = 3
):
    """Run the async workflow with timing and progress monitoring."""
    start_time = time.time()
    total_rows = len(df_merged)

    print(f"🚀 Starting async processing at {time.strftime('%H:%M:%S')}")
    print(
        f"📊 Processing {total_rows} rows with max {max_concurrent} concurrent requests"
    )
    print(
        f"⚡ Expected speedup: ~{min(max_concurrent, total_rows)}x compared to sequential processing"
    )
    print("-" * 60)

    # Process with the complete workflow
    df_result = await process_complete_workflow(df_merged, path_prefix, max_concurrent)

    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 60)
    print(f"🎉 Completed processing {total_rows} rows in {total_time:.2f} seconds")
    print(f"⏱️ Average time per row: {total_time / total_rows:.2f} seconds")
    print(
        f"🚄 Estimated sequential time would have been: ~{total_time * max_concurrent:.1f} seconds"
    )

    return df_result


def load_data():
    """Load the data using environment variables."""
    load_dotenv(override=True)

    questions = Path(os.getenv("QUESTIONS_FILE"))
    answers = Path(os.getenv("ANSWERS_FILE"))

    df_questions = pd.read_json(questions, lines=True)
    df_answers = pd.read_json(answers, lines=True)
    df_merged = df_answers.merge(df_questions, left_on="id", right_on="id", how="inner")

    return df_merged


async def main():
    """Main async function demonstrating different usage patterns."""
    # Load data
    print("📂 Loading data...")
    df_merged = load_data()
    path_prefix = Path("data/InfiAgent-DABench/da-dev-tables/")

    print(f"📊 Loaded {len(df_merged)} rows for processing")
    print("\n" + "=" * 60)

    # Option 1: Planning only
    print("Option 1: Planning phase only")
    df_with_plans = await create_plans_parallel(
        df_merged, path_prefix, max_concurrent=5
    )
    df_with_plans.to_csv("data/async_merged_with_plans.csv", index=False)
    print(f"✅ Plans saved to 'data/async_merged_with_plans.csv'")

    print("\n" + "=" * 60)

    # Option 2: Coding phase (using plans from above)
    print("Option 2: Coding phase")
    df_with_code = await create_code_parallel(
        df_with_plans, path_prefix, max_concurrent=5
    )
    df_with_code.to_csv("data/async_merged_with_code.csv", index=False)
    print(f"✅ Complete results saved to 'data/async_merged_with_code.csv'")

    print("\n" + "=" * 60)

    # Option 3: Complete workflow with timing (using subset for demo)
    print("Option 3: Complete workflow with timing (demo with 5 rows)")
    df_complete = await process_with_timing(
        df_merged.head(5), path_prefix, max_concurrent=3
    )
    print(f"✅ Demo complete workflow finished")

    print("\n🎉 All async processing completed!")


if __name__ == "__main__":
    # Usage examples:
    print("🔧 Async Data Analysis Agent")
    print("\n📝 Available functions:")
    print("1. create_plans_parallel() - Parallel planning only")
    print("2. create_code_parallel() - Parallel coding only")
    print("3. process_complete_workflow() - Combined planning + coding")
    print("4. process_with_timing() - Complete workflow with timing")
    print("\n⚠️ Recommended max_concurrent values:")
    print("   - Conservative: 3-5 (good for API rate limits)")
    print("   - Aggressive: 8-10 (if you have high API limits)")

    # Run the main demo
    asyncio.run(main())
