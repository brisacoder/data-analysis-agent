import json
import re
from typing import Dict, Any


def parse_dataframe_info(info_output: str) -> str:
    """
    Parse the string output of DataFrame.info() into a JSON object.
    
    This function extracts column information from pandas DataFrame.info() output,
    ignoring the header lines and footer statistics. It focuses only on parsing
    the column details including index, name, non-null count, and data type.
    
    NOTE: This function attempts to extract the actual column names as they appear
    in the DataFrame, including any leading/trailing whitespace that may exist
    due to CSV formatting issues.
    
    Args:
        info_output (str): The string output from DataFrame.info()
        
    Returns:
        str: JSON string containing parsed column information
        
    Raises:
        ValueError: If the input format is not recognized or cannot be parsed
    """
    lines = info_output.strip().split('\n')
    columns_data = []
    
    # Find the start of column data (skip first 3 lines)
    column_start_index = 3
    
    # Find where column data ends (before dtypes: or memory usage:)
    column_end_index = len(lines)
    for i, line in enumerate(lines[column_start_index:], column_start_index):
        if line.strip().startswith('dtypes:') or line.strip().startswith('memory usage:'):
            column_end_index = i
            break
    
    # Parse each column line
    for line in lines[column_start_index:column_end_index]:
        line = line.strip()
        
        # Skip separator lines (lines with only dashes and spaces)
        if re.match(r'^[-\s]+$', line) or not line:
            continue
            
        # Parse column information using regex
        # Pattern matches: index, column_name, non_null_count, dtype
        # Updated pattern to handle column names with spaces and special characters
        pattern = r'^\s*(\d+)\s+(.+?)\s+(\d+)\s+non-null\s+(\S+)\s*$'
        match = re.match(pattern, line)
        
        if match:
            column_index = int(match.group(1))
            column_name = match.group(2)
            non_null_count = int(match.group(3))
            dtype = match.group(4)
            
            column_info = {
                'index': column_index,
                'column_name': column_name,
                'non_null_count': non_null_count,
                'dtype': dtype
            }
            
            columns_data.append(column_info)
    
    # Create the final JSON structure
    result = {
        'columns': columns_data,
        'total_columns': len(columns_data)
    }
    
    return json.dumps(result, indent=2)


def parse_dataframe_info_with_columns(info_output: str, actual_columns: list) -> str:
    """
    Parse the string output of DataFrame.info() into a JSON object using actual column names.
    
    This function is more robust than parse_dataframe_info() as it uses the actual column names
    from the DataFrame rather than trying to parse them from the formatted info() output.
    This prevents issues with whitespace, special characters, or formatting inconsistencies.
    
    Args:
        info_output (str): The string output from DataFrame.info()
        actual_columns (list): The actual column names from DataFrame.columns
        
    Returns:
        str: JSON string containing parsed column information
        
    Raises:
        ValueError: If the input format is not recognized or cannot be parsed
    """
    lines = info_output.strip().split('\n')
    columns_data = []
    
    # Find the start of column data (skip first 3 lines)
    column_start_index = 3
    
    # Find where column data ends (before dtypes: or memory usage:)
    column_end_index = len(lines)
    for i, line in enumerate(lines[column_start_index:], column_start_index):
        if line.strip().startswith('dtypes:') or line.strip().startswith('memory usage:'):
            column_end_index = i
            break
    
    # Parse each column line and match with actual column names
    column_lines = []
    for line in lines[column_start_index:column_end_index]:
        line = line.strip()
        # Skip separator lines (lines with only dashes and spaces)
        if re.match(r'^[-\s]+$', line) or not line:
            continue
        # Skip the header line (contains "Column" and "Non-Null Count")
        if 'Column' in line and 'Non-Null Count' in line:
            continue
        column_lines.append(line)
    
    # Ensure we have the same number of columns in both sources
    if len(column_lines) != len(actual_columns):
        raise ValueError(f"Mismatch: {len(column_lines)} lines in info() but {len(actual_columns)} actual columns")
    
    # Parse each column line and use the actual column name
    for i, line in enumerate(column_lines):
        # Parse non-null count and dtype from the line
        pattern = r'^\s*(\d+)\s+.+?\s+(\d+)\s+non-null\s+(\S+)\s*$'
        match = re.match(pattern, line)
        
        if match:
            column_index = int(match.group(1))
            non_null_count = int(match.group(2))
            dtype = match.group(3)
            
            # Use the actual column name from the DataFrame
            actual_column_name = actual_columns[i]
            
            column_info = {
                'index': column_index,
                'column_name': actual_column_name,
                'non_null_count': non_null_count,
                'dtype': dtype
            }
            
            columns_data.append(column_info)
        else:
            raise ValueError(f"Could not parse column info line: {line}")
    
    # Create the final JSON structure
    result = {
        'columns': columns_data,
        'total_columns': len(columns_data)
    }
    
    return json.dumps(result, indent=2)


def parse_dataframe_info_to_dict(info_output: str) -> Dict[str, Any]:
    """
    Parse the string output of DataFrame.info() into a Python dictionary.
    
    This function is similar to parse_dataframe_info() but returns a dictionary
    instead of a JSON string, which can be more convenient for further processing.
    
    Args:
        info_output (str): The string output from DataFrame.info()
        
    Returns:
        Dict[str, Any]: Dictionary containing parsed column information
        
    Raises:
        ValueError: If the input format is not recognized or cannot be parsed
    """
    json_string = parse_dataframe_info(info_output)
    return json.loads(json_string)


def parse_dataframe_info_with_columns_to_dict(info_output: str, actual_columns: list) -> Dict[str, Any]:
    """
    Parse the string output of DataFrame.info() into a Python dictionary using actual column names.
    
    This function is similar to parse_dataframe_info_with_columns() but returns a dictionary
    instead of a JSON string, which can be more convenient for further processing.
    
    Args:
        info_output (str): The string output from DataFrame.info()
        actual_columns (list): The actual column names from DataFrame.columns
        
    Returns:
        Dict[str, Any]: Dictionary containing parsed column information
        
    Raises:
        ValueError: If the input format is not recognized or cannot be parsed
    """
    json_string = parse_dataframe_info_with_columns(info_output, actual_columns)
    return json.loads(json_string)
