# Copilot Instructions for Data Analysis Agent

## Big Picture Architecture

This is a **dual-agent system** implementing the **plan-and-execute pattern** for automated data analysis. The architecture consists of:

### Core Agents
- **Planner Agent** (`data_analysis_agent/planner.py`): Decomposes natural language questions into structured task lists using LLMs
- **Coder Agent** (`data_analysis_agent/coder.py`): Converts plans into executable Python scripts with comprehensive error handling

### Data Flow
```
Questions (JSONL) + CSV Data → Planner Agent → Plans (JSON) → Coder Agent → Python Scripts → Results
```

### Key Components
- **CLI Controllers** (`cli.py`, `async_cli.py`): Main entry points with batch processing
- **Data Quality Framework** (`data_quality_assessment.py`): Comprehensive quality assessment with automotive specialization
- **Path Management** (`paths.py`): Centralized file system organization with auto-cleanup
- **Prompt Engineering** (`prompts.py`): LLM system prompts for agent behavior

## Critical Developer Workflows

### Environment Setup (Windows PowerShell 7 + uv)
```powershell
# Required: Python 3.13+, OpenAI API key in .env
# First time setup
uv venv                                # Create virtual environment
.venv\Scripts\Activate.ps1            # Activate venv (PowerShell)
uv pip install -e ".[dev]"           # Development mode with test dependencies

# Daily workflow activation
.venv\Scripts\Activate.ps1            # Always activate venv first
```

### Running the System
```powershell
# Activate venv first, then run commands
.venv\Scripts\Activate.ps1
data-analysis-agent                   # CLI entry point (preferred)
data-analysis-agent-async            # Async version for performance
python -m data_analysis_agent.cli    # Module invocation

# Or with full path (if venv not activated)
.venv\Scripts\python.exe -m data_analysis_agent.cli
```

### Testing Strategy
```powershell
# Always activate venv first
.venv\Scripts\Activate.ps1

# Then run tests
pytest -m "not requires_api_key"     # Skip API integration tests
pytest tests/test_planner.py -v      # Test specific components
pytest --cov=data_analysis_agent     # Coverage reporting

# Or direct execution without activation
.venv\Scripts\python.exe -m pytest tests/test_planner.py -v
```

## Project-Specific Conventions

### Singleton Pattern for Resource Management
Both `PlanGenerator` and `CodeGenerator` use singletons with **cleanup-on-first-use**:
```python
# Directory cleanup happens only once per session to prevent file accumulation
class CodeGenerator:
    _instance = None
    _initialized = False
    
    def _ensure_directory_ready(self):
        if not self._initialized:
            # Clean existing files only on first use
```

### Pydantic Models for Agent Communication
All agent inputs/outputs use structured Pydantic models:
```python
class Task(BaseModel):
    task_name: str
    task_number: int
    details: str        # Step-by-step instructions
    dependencies: str   # What this task depends on
    output: str        # Expected deliverables
    assumptions: str   # Clarifying assumptions
```

### Mandatory Task Sequence
Every plan begins with standardized setup tasks:
1. Setup Imports and Dependencies
2. Define Configuration Constants  
3. Setup Logging and Reproducibility
4. Load and Validate Input Data
5. **Data Quality Assessment and Cleaning** ← Critical step
6. Create Output Directory

### Data Quality Integration Pattern
The data quality framework integrates at multiple levels:
- **General Assessment**: `data_quality_assessment.py` for any dataset
- **Domain-Specific**: `automotive_data_quality.py` for automotive signals
- **API Wrapper**: `automotive_data_quality_api.py` for programmatic access

## Integration Points

### LLM Integration
- Uses `langchain.chat_models.init_chat_model()` with structured output
- Default model: `"openai:gpt-5"` with retry logic
- Structured output via `llm.with_structured_output(schema=Plan)`

### File System Organization
```
data/
├── plan/           # Generated JSON plans (timestamped)
├── code/           # Generated Python scripts  
├── InfiAgent-DABench/  # Input datasets
└── data_analysis_agent.log  # Centralized logging
```

### Error Handling Pattern
Use **graceful degradation** with detailed logging:
```python
try:
    plan = create_plan(question, df_json, file_name)
    processed_count += 1
except Exception as e:
    logger.error(f"Row {index}: Error creating plan: {e}")
    failed_count += 1
    continue  # Continue processing other items
```

## Key Implementation Details

### DataFrame Schema Generation
Use `dataframe_to_dict.py` for LLM-friendly data descriptions:
```python
df_json = df_info_to_json(df)  # Converts DataFrame to JSON schema
```

### Data Quality Context Awareness
The quality assessment is **context-aware** - what constitutes a "problem" depends on expected data type:
```python
check_problematic_values(series, expected_type="categorical")  # Won't flag text as problematic
check_problematic_values(series, expected_type="numeric")     # Will flag text as problematic
```

### Automotive Domain Specialization
For automotive data, use the specialized API:
```python
from data_analysis_agent.automotive_data_quality_api import AutomotiveDataQualityAPI
api = AutomotiveDataQualityAPI("your_data.csv")
```

### Testing with API Dependencies
Tests are marked for selective execution:
```python
@pytest.mark.requires_api_key
def test_integration():
    if not os.getenv('OPENAI_API_KEY'):
        pytest.skip("No API key available")
```

### Windows PowerShell 7 Command Patterns
```powershell
# Virtual environment management
.venv\Scripts\Activate.ps1                    # Activate venv
.venv\Scripts\python.exe script.py           # Direct execution
.venv\Scripts\python.exe -m module.name      # Module execution

# File operations
Remove-Item file1.py, file2.py               # Delete files
Get-ChildItem -Path .\data\ -Recurse         # List directory contents
Test-Path .\data\file.csv                    # Check file existence
```

## Development Patterns

- **Windows PowerShell 7 Environment**: Always activate venv with `.venv\Scripts\Activate.ps1` before running commands
- **Virtual Environment Workflow**: Use `uv venv` for creation, `.venv\Scripts\python.exe` for direct execution
- **Timestamped Outputs**: All generated files include UTC timestamps for tracking
- **Robust File Handling**: Always check file existence before processing
- **Batch Processing**: Process multiple items before I/O operations for efficiency
- **Silent Mode**: Support silent operation for automated workflows
- **JSON Serialization**: Use `convert_numpy_types()` for JSON compatibility
- **PowerShell Path Handling**: Use backslashes `\` for Windows paths in commands
