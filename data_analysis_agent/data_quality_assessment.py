"""
Comprehensive Data Quality Assessment Framework

This module provides a complete toolkit for assessing data quality before cleaning.
It analyzes individual columns, detects patterns, checks distributions, and generates
comprehensive quality reports.

Key Features:
- Detects problematic values (NaN, infinite, extreme values)
- Analyzes statistical distributions and outliers
- Identifies data patterns and anomalies
- Performs temporal and categorical data analysis
- Generates detailed quality reports
- Provides actionable recommendations

Example Usage:
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'numeric': [1, 2, 3, np.nan, 5],
    ...     'categorical': ['A', 'B', 'A', 'C', 'B'],
    ...     'dates': pd.date_range('2023-01-01', periods=5)
    ... })
    >>> 
    >>> # Generate quality report
    >>> report = generate_quality_report(df)
    >>> print(report)
    >>> 
    >>> # Analyze single column
    >>> analysis = analyze_column_comprehensive(df['numeric'])
    >>> print(analysis['basic_quality']['summary'])

Author: Reinaldo Penno
License: MIT
Version: 2.0.0
"""

import json
import logging
import warnings
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from scipy import stats


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Parameters
    ----------
    obj : Any
        Object that may contain numpy types
        
    Returns
    -------
    Any
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityAssessmentConfig:
    """
    Configuration class for data quality assessment parameters.
    
    This class centralizes all configurable thresholds and parameters
    used throughout the quality assessment process, making the framework
    more flexible and maintainable.
    """
    
    def __init__(
        self,
        # Problematic values detection
        extreme_threshold: float = 1e10,
        check_zeros_default: bool = False,
        check_negatives_default: bool = False,
        
        # Missing value thresholds
        high_missing_threshold: float = 20.0,    # %
        critical_missing_threshold: float = 50.0,  # %
        
        # Outlier detection
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        high_outlier_threshold: float = 10.0,    # %
        
        # Correlation analysis
        high_correlation_threshold: float = 0.95,
        
        # Quality scoring weights
        missing_weight: float = 0.3,
        outlier_weight: float = 0.2,
        quality_issues_weight: float = 0.3,
        distribution_weight: float = 0.2,
        
        # Performance settings
        chunk_size: Optional[int] = None,  # None = no chunking
        max_memory_mb: float = 500.0,
        enable_progress_tracking: bool = False,
        
        # Datetime analysis
        future_date_threshold_years: int = 5,
        old_date_threshold_year: int = 1900,
        
        # Categorical analysis
        high_cardinality_threshold: float = 0.95,  # uniqueness ratio
        mixed_case_tolerance: float = 0.1,  # % of mixed case values to tolerate
    ):
        """
        Initialize quality assessment configuration.
        
        Parameters
        ----------
        extreme_threshold : float, default 1e10
            Threshold for detecting extremely large values
        check_zeros_default : bool, default False
            Whether to flag zeros as problematic by default
        check_negatives_default : bool, default False
            Whether to flag negative values as problematic by default
        high_missing_threshold : float, default 20.0
            Percentage threshold for flagging high missing values
        critical_missing_threshold : float, default 50.0
            Percentage threshold for flagging critical missing values
        outlier_method : str, default "iqr"
            Method for outlier detection ("iqr", "zscore", "isolation")
        outlier_threshold : float, default 1.5
            Threshold for outlier detection (IQR multiplier or z-score)
        high_outlier_threshold : float, default 10.0
            Percentage threshold for flagging high outlier rates
        high_correlation_threshold : float, default 0.95
            Correlation threshold for flagging multicollinearity
        missing_weight : float, default 0.3
            Weight for missing values in quality score calculation
        outlier_weight : float, default 0.2
            Weight for outliers in quality score calculation
        quality_issues_weight : float, default 0.3
            Weight for general quality issues in quality score calculation
        distribution_weight : float, default 0.2
            Weight for distribution analysis in quality score calculation
        chunk_size : int, optional
            Number of rows to process at once for large datasets
        max_memory_mb : float, default 500.0
            Maximum memory usage before triggering chunked processing
        enable_progress_tracking : bool, default False
            Whether to show progress bars for long operations
        future_date_threshold_years : int, default 5
            Years into future to flag as problematic dates
        old_date_threshold_year : int, default 1900
            Year threshold for flagging very old dates
        high_cardinality_threshold : float, default 0.95
            Uniqueness ratio threshold for flagging high cardinality
        mixed_case_tolerance : float, default 0.1
            Tolerance for mixed case values in categorical data
        """
        self.extreme_threshold = extreme_threshold
        self.check_zeros_default = check_zeros_default
        self.check_negatives_default = check_negatives_default
        
        self.high_missing_threshold = high_missing_threshold
        self.critical_missing_threshold = critical_missing_threshold
        
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.high_outlier_threshold = high_outlier_threshold
        
        self.high_correlation_threshold = high_correlation_threshold
        
        self.missing_weight = missing_weight
        self.outlier_weight = outlier_weight
        self.quality_issues_weight = quality_issues_weight
        self.distribution_weight = distribution_weight
        
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.enable_progress_tracking = enable_progress_tracking
        
        self.future_date_threshold_years = future_date_threshold_years
        self.old_date_threshold_year = old_date_threshold_year
        
        self.high_cardinality_threshold = high_cardinality_threshold
        self.mixed_case_tolerance = mixed_case_tolerance
    
    def should_use_chunking(self, df: pd.DataFrame) -> bool:
        """
        Determine if chunked processing should be used based on DataFrame size.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to assess
            
        Returns
        -------
        bool
            True if chunking should be used
        """
        if self.chunk_size is not None:
            return len(df) > self.chunk_size
        
        # Estimate memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        return memory_mb > self.max_memory_mb
    
    def get_effective_chunk_size(self, df: pd.DataFrame) -> int:
        """
        Calculate effective chunk size based on DataFrame characteristics.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to assess
            
        Returns
        -------
        int
            Recommended chunk size
        """
        if self.chunk_size is not None:
            return self.chunk_size
        
        # Calculate based on memory constraints
        row_memory = df.memory_usage(deep=True).sum() / len(df)  # bytes per row
        target_memory_bytes = self.max_memory_mb * 1024 * 1024
        estimated_chunk_size = int(target_memory_bytes / row_memory)
        
        # Ensure reasonable bounds
        return max(1000, min(100000, estimated_chunk_size))


# Global default configuration instance
DEFAULT_CONFIG = QualityAssessmentConfig()


class ProgressTracker:
    """
    Simple progress tracking utility for data quality assessment operations.
    """
    
    def __init__(self, enabled: bool = False, total: Optional[int] = None):
        """
        Initialize progress tracker.
        
        Parameters
        ----------
        enabled : bool, default False
            Whether progress tracking is enabled
        total : int, optional
            Total number of items to process
        """
        self.enabled = enabled
        self.total = total
        self.current = 0
        self.last_reported = 0
    
    def update(self, increment: int = 1, description: str = "Processing"):
        """
        Update progress counter.
        
        Parameters
        ----------
        increment : int, default 1
            Number of items processed
        description : str, default "Processing"
            Description of current operation
        """
        if not self.enabled:
            return
            
        self.current += increment
        
        # Report progress every 10% or every 10 items, whichever is larger
        if self.total:
            report_interval = max(10, self.total // 10)
        else:
            report_interval = 10
            
        if self.current - self.last_reported >= report_interval:
            if self.total:
                percentage = (self.current / self.total) * 100
                logger.info(f"{description}: {self.current}/{self.total} ({percentage:.1f}%)")
            else:
                logger.info(f"{description}: {self.current} items processed")
            self.last_reported = self.current
    
    def finish(self, description: str = "Processing"):
        """Mark processing as complete."""
        if self.enabled:
            logger.info(f"{description}: Complete ({self.current} items processed)")


def estimate_memory_usage(df: pd.DataFrame) -> float:
    """
    Estimate memory usage of a DataFrame in MB.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns
    -------
    float
        Estimated memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def should_use_chunked_processing(df: pd.DataFrame, config: QualityAssessmentConfig) -> bool:
    """
    Determine if chunked processing should be used for the given DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to assess
    config : QualityAssessmentConfig
        Configuration settings
        
    Returns
    -------
    bool
        True if chunked processing is recommended
    """
    return config.should_use_chunking(df)


class DataLeakageResults(TypedDict):
    """
    Type definition for data leakage detection results.
    
    Attributes:
        potential_id_columns: Columns that might be identifiers
        perfect_predictors: Columns perfectly correlated with target
        constant_columns: Columns with only one unique value
        duplicate_columns: Pairs of identical columns
        temporal_leakage_risk: Columns suggesting future information
        recommendations: List of actionable recommendations
        summary: Brief summary of findings
    """
    potential_id_columns: List[Dict[str, Any]]
    perfect_predictors: List[Dict[str, Any]]
    constant_columns: List[str]
    duplicate_columns: List[tuple]
    temporal_leakage_risk: List[Dict[str, Any]]
    recommendations: List[str]
    summary: str


class OutlierResults(TypedDict):
    """
    Type definition for outlier analysis results.
    
    Attributes:
        method: Method used for outlier detection
        threshold: Threshold value used
        outlier_count: Number of outliers found
        outlier_percentage: Percentage of data that are outliers
        lower_bound: Lower boundary for normal values
        upper_bound: Upper boundary for normal values
        outlier_indices: Row indices of outlier values
        summary: Brief summary of findings
    """
    method: str
    threshold: float
    outlier_count: int
    outlier_percentage: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    outlier_indices: List[int]
    summary: str


class PatternResults(TypedDict):
    """
    Type definition for pattern analysis results.
    
    Attributes:
        is_constant: Whether all values are the same
        is_monotonic_increasing: Whether values always increase
        is_monotonic_decreasing: Whether values always decrease
        has_duplicates: Whether duplicate values exist
        duplicate_count: Number of duplicate values
        unique_count: Number of unique values
        uniqueness_ratio: Ratio of unique to total values
        most_frequent_values: Most common values and their counts
        suspected_rounding: Whether values appear to be rounded
        decimal_places_distribution: Distribution of decimal places
        summary: Brief summary of findings
    """
    is_constant: bool
    is_monotonic_increasing: bool
    is_monotonic_decreasing: bool
    has_duplicates: bool
    duplicate_count: int
    unique_count: int
    uniqueness_ratio: float
    most_frequent_values: Dict[Any, int]
    suspected_rounding: bool
    decimal_places_distribution: Dict[int, int]
    summary: str


class TemporalResults(TypedDict):
    """
    Type definition for temporal analysis results.
    
    Attributes:
        is_datetime: Whether data can be converted to datetime
        conversion_errors: Number of values that couldn't be converted
        future_dates: Number of dates in the future
        very_old_dates: Number of dates before 1900
        date_range: Min, max, and span of dates
        gaps_analysis: Analysis of time gaps between consecutive dates
        timezone_info: Timezone information if available
        most_common_times: Most frequent time components
        summary: Brief summary of findings
    """
    is_datetime: bool
    conversion_errors: int
    future_dates: int
    very_old_dates: int
    date_range: Dict[str, Any]
    gaps_analysis: Dict[str, Any]
    timezone_info: Optional[str]
    most_common_times: Dict[str, int]
    summary: str


class CategoricalResults(TypedDict):
    """
    Type definition for categorical analysis results.
    
    Attributes:
        unique_count: Number of unique categories
        most_frequent: Most common categories and their counts
        least_frequent: Least common categories and their counts
        has_leading_trailing_spaces: Whether values have extra spaces
        has_special_characters: Whether values contain special characters
        has_mixed_case: Whether values have inconsistent capitalization
        encoding_issues: Whether encoding problems were detected
        length_statistics: Statistics about string lengths
        summary: Brief summary of findings
    """
    unique_count: int
    most_frequent: Dict[str, int]
    least_frequent: Dict[str, int]
    has_leading_trailing_spaces: bool
    has_special_characters: bool
    has_mixed_case: bool
    encoding_issues: bool
    length_statistics: Dict[str, float]
    summary: str


def log_analysis_warning(function_name: str, message: str, exception: Exception) -> None:
    """
    Log analysis warnings with proper context.

    This function provides consistent warning logging across the module,
    helping with debugging and analysis transparency.

    Parameters
    ----------
    function_name : str
        Name of the function where the warning occurred
    message : str
        Descriptive message about the warning
    exception : Exception
        The exception that was caught

    Examples
    --------
    >>> try:
    ...     result = risky_operation()
    ... except ValueError as e:
    ...     log_analysis_warning('analyze_data', 'Failed to convert values', e)
    """
    logger.warning(
        f"{function_name}: {message} - {type(exception).__name__}: {str(exception)}"
    )


def handle_conversion_error(series_name: str, operation: str, exception: Exception) -> None:
    """
    Handle and log data conversion errors.

    This function provides consistent error handling for data type conversions
    throughout the analysis process.

    Parameters
    ----------
    series_name : str
        Name of the series being processed
    operation : str
        Description of the operation being attempted
    exception : Exception
        The exception that was caught

    Examples
    --------
    >>> try:
    ...     numeric_data = pd.to_numeric(series, errors='raise')
    ... except ValueError as e:
    ...     handle_conversion_error('my_column', 'numeric conversion', e)
    """
    logger.debug(
        f"Column '{series_name}': {operation} failed - {type(exception).__name__}: {str(exception)}"
    )


def _check_nan_values(series: pd.Series) -> Dict[str, Any]:
    """
    Check for NaN values in a series.
    
    Parameters
    ----------
    series : pd.Series
        The series to check for NaN values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with nan_count and nan_indices
    """
    nan_mask = pd.isna(series)
    return {
        'nan_count': nan_mask.sum(),
        'nan_indices': series[nan_mask].index.tolist()
    }


def _check_infinite_values(numeric_series: pd.Series) -> Dict[str, Any]:
    """
    Check for infinite values in a numeric series.
    
    Parameters
    ----------
    numeric_series : pd.Series
        The numeric series to check
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with inf_count, ninf_count, and inf_indices
    """
    inf_mask = np.isposinf(numeric_series)
    ninf_mask = np.isneginf(numeric_series)
    
    return {
        'inf_count': inf_mask.sum(),
        'ninf_count': ninf_mask.sum(),
        'inf_indices': numeric_series[inf_mask | ninf_mask].index.tolist()
    }


def _check_extreme_values(numeric_series: pd.Series, threshold: float = 1e10) -> Dict[str, Any]:
    """
    Check for extremely large or small values.
    
    Parameters
    ----------
    numeric_series : pd.Series
        The numeric series to check
    threshold : float, default 1e10
        Threshold for extreme values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with extreme value counts and indices
    """
    abs_values = np.abs(numeric_series)
    extreme_large_mask = abs_values > threshold
    extreme_small_mask = (abs_values < 1 / threshold) & (abs_values != 0)
    
    return {
        'extreme_large_count': extreme_large_mask.sum(),
        'extreme_small_count': extreme_small_mask.sum(),
        'extreme_indices': numeric_series[extreme_large_mask | extreme_small_mask].index.tolist()
    }


def _check_special_numeric_values(
    numeric_series: pd.Series, 
    check_zeros: bool = True, 
    check_negatives: bool = False
) -> Dict[str, Any]:
    """
    Check for special numeric values like zeros and negatives.
    
    Parameters
    ----------
    numeric_series : pd.Series
        The numeric series to check
    check_zeros : bool, default True
        Whether to flag zero values
    check_negatives : bool, default False
        Whether to flag negative values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with counts and indices of special values
    """
    result = {
        'zero_count': 0, 
        'negative_count': 0, 
        'special_indices': [],
        'zero_indices': [],
        'negative_indices': []
    }
    
    if check_zeros:
        zero_mask = numeric_series == 0
        result['zero_count'] = zero_mask.sum()
        if result['zero_count'] > 0:
            zero_indices = numeric_series[zero_mask].index.tolist()
            result['zero_indices'] = zero_indices
            result['special_indices'].extend(zero_indices)
    
    if check_negatives:
        neg_mask = numeric_series < 0
        result['negative_count'] = neg_mask.sum()
        if result['negative_count'] > 0:
            neg_indices = numeric_series[neg_mask].index.tolist()
            result['negative_indices'] = neg_indices
            result['special_indices'].extend(neg_indices)
    
    return result


def check_problematic_values(
    series: pd.Series,
    check_zeros: bool = True,
    check_negatives: bool = False,
    extreme_threshold: float = 1e10,
    return_details: bool = True,
    expected_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive check for problematic values in a pandas Series.

    This function identifies various types of problematic values that can cause
    issues in data analysis, including NaN, infinite, non-numeric, extreme values,
    and other edge cases.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to check for problematic values.
    check_zeros : bool, default True
        Whether to flag zero values as potentially problematic.
        Useful for ratio calculations or log transformations.
    check_negatives : bool, default False
        Whether to flag negative values as potentially problematic.
        Useful for data that should only contain positive values.
    extreme_threshold : float, default 1e10
        Threshold for flagging extremely large or small values.
        Values with absolute value > threshold or < 1/threshold are flagged.
    return_details : bool, default True
        Whether to return detailed information about problematic rows.
        If False, only returns counts to save memory.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comprehensive analysis of problematic values:
        - total_rows: Total number of rows in the series
        - nan_count: Number of NaN/null values
        - inf_count: Number of positive infinite values
        - ninf_count: Number of negative infinite values
        - non_numeric_count: Number of values that couldn't be converted to numeric
        - zero_count: Number of zero values (if check_zeros=True)
        - negative_count: Number of negative values (if check_negatives=True)
        - extreme_large_count: Number of extremely large values
        - extreme_small_count: Number of extremely small values
        - empty_string_count: Number of empty/whitespace strings
        - mixed_types: Whether the series contains mixed data types
        - variance: Statistical variance of numeric values
        - is_clean: Whether the series is free of problematic values
        - problematic_indices: List of row indices with issues (if return_details=True)
        - summary: Text summary of findings

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create problematic data
    >>> data = pd.Series([1, 2, np.nan, np.inf, -np.inf, 0, 1e15, "text"])
    >>> result = check_problematic_values(data)
    >>> print(f"Clean data: {result['is_clean']}")
    >>> print(f"Issues: {result['summary']}")
    
    >>> # Check only specific issues
    >>> result = check_problematic_values(data, check_zeros=False, check_negatives=True)
    >>> print(f"Negative values: {result['negative_count']}")
    """
    # Initialize results dictionary
    results: Dict[str, Any] = {
        "total_rows": len(series),
        "nan_count": 0,
        "inf_count": 0,
        "ninf_count": 0,
        "non_numeric_count": 0,
        "zero_count": 0,
        "negative_count": 0,
        "extreme_large_count": 0,
        "extreme_small_count": 0,
        "empty_string_count": 0,
        "mixed_types": False,
        "variance": None,
        "is_clean": True,
        "problematic_indices": [],
        "summary": "",
    }

    if len(series) == 0:
        results["summary"] = "Empty series"
        results["is_clean"] = False
        return results

    # Check for NaN values
    nan_results = _check_nan_values(series)
    results["nan_count"] = nan_results["nan_count"]
    if results["nan_count"] > 0:
        results["is_clean"] = False
        if return_details:
            results["problematic_indices"].extend(nan_results["nan_indices"])

    # Work with non-NaN values for remaining checks
    non_nan_series = series.dropna()
    if len(non_nan_series) == 0:
        results["summary"] = "All values are NaN"
        results["is_clean"] = False
        return results

    # Check for mixed data types
    unique_types = set(type(val).__name__ for val in non_nan_series)
    if len(unique_types) > 1:
        results["mixed_types"] = True
        results["is_clean"] = False

    # Attempt numeric conversion and analyze
    try:
        numeric_series = pd.to_numeric(non_nan_series, errors="coerce")
        newly_nan = pd.isna(numeric_series) & pd.notna(non_nan_series)
        results["non_numeric_count"] = newly_nan.sum()

        # Only flag non-numeric values as problematic if we expect numeric data
        if results["non_numeric_count"] > 0 and expected_type == "numeric":
            results["is_clean"] = False
            if return_details:
                non_numeric_indices = non_nan_series[newly_nan].index.tolist()
                results["problematic_indices"].extend(non_numeric_indices)

        # Continue with successfully converted numeric values
        clean_numeric = numeric_series.dropna()

        if len(clean_numeric) > 0:
            # Check for infinite values (always problematic for any data type)
            inf_results = _check_infinite_values(clean_numeric)
            results["inf_count"] = inf_results["inf_count"]
            results["ninf_count"] = inf_results["ninf_count"]
            
            if results["inf_count"] > 0 or results["ninf_count"] > 0:
                results["is_clean"] = False
                if return_details:
                    results["problematic_indices"].extend(inf_results["inf_indices"])

            # Remove infinite values for remaining analysis
            finite_numeric = clean_numeric[np.isfinite(clean_numeric)]

            if len(finite_numeric) > 0 and expected_type == "numeric":
                # Only do numeric-specific checks if we expect numeric data
                
                # Check for special numeric values (zeros, negatives)
                special_results = _check_special_numeric_values(
                    finite_numeric, check_zeros, check_negatives
                )
                results["zero_count"] = special_results["zero_count"]
                results["negative_count"] = special_results["negative_count"]
                
                if check_zeros and special_results["zero_count"] > 0:
                    results["is_clean"] = False
                    if return_details:
                        results["problematic_indices"].extend(special_results.get("zero_indices", []))
                        
                if check_negatives and special_results["negative_count"] > 0:
                    results["is_clean"] = False
                    if return_details:
                        results["problematic_indices"].extend(special_results.get("negative_indices", []))

                # Check for extreme values
                extreme_results = _check_extreme_values(finite_numeric, extreme_threshold)
                results["extreme_large_count"] = extreme_results["extreme_large_count"]
                results["extreme_small_count"] = extreme_results["extreme_small_count"]
                
                if results["extreme_large_count"] > 0 or results["extreme_small_count"] > 0:
                    results["is_clean"] = False
                    if return_details:
                        results["problematic_indices"].extend(extreme_results["extreme_indices"])

                # Calculate variance safely
                try:
                    results["variance"] = finite_numeric.var()
                    if pd.isna(results["variance"]):
                        results["is_clean"] = False
                except (MemoryError, OverflowError, ArithmeticError) as e:
                    log_analysis_warning(
                        "check_problematic_values",
                        "Could not calculate variance due to computational error",
                        e,
                    )
                    results["variance"] = None
                    results["is_clean"] = False

    except Exception as e:
        results["is_clean"] = False
        results["summary"] = f"Error during numeric conversion: {str(e)}"
        return results

    # Check for empty strings or whitespace (for object dtype)
    if series.dtype == "object":
        try:
            str_series = non_nan_series.astype(str)
            empty_mask = (str_series.str.strip() == "") | (str_series == "")
            results["empty_string_count"] = empty_mask.sum()
            if results["empty_string_count"] > 0:
                results["is_clean"] = False
                if return_details:
                    empty_indices = non_nan_series[empty_mask].index.tolist()
                    results["problematic_indices"].extend(empty_indices)
        except (AttributeError, ValueError, TypeError) as e:
            log_analysis_warning(
                "check_problematic_values",
                "Could not check for empty strings due to incompatible object types",
                e,
            )

    # Remove duplicates from problematic_indices
    if return_details:
        results["problematic_indices"] = list(set(results["problematic_indices"]))

    # Generate comprehensive summary
    results["summary"] = _generate_problematic_values_summary(results, check_zeros, check_negatives, expected_type)

    return results


def _generate_problematic_values_summary(
    results: Dict[str, Any], 
    check_zeros: bool, 
    check_negatives: bool,
    expected_type: Optional[str] = None
) -> str:
    """
    Generate a summary of problematic values found.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from problematic values check
    check_zeros : bool
        Whether zeros were checked
    check_negatives : bool
        Whether negatives were checked
    expected_type : str, optional
        Expected data type (numeric, categorical, datetime)
        
    Returns
    -------
    str
        Summary string
    """
    issues = []
    
    if results["nan_count"] > 0:
        issues.append(f"{results['nan_count']} NaN values")
    if results["inf_count"] > 0:
        issues.append(f"{results['inf_count']} positive infinite values")
    if results["ninf_count"] > 0:
        issues.append(f"{results['ninf_count']} negative infinite values")
    
    # Only include non-numeric as issue if we expect numeric data
    if results["non_numeric_count"] > 0 and expected_type == "numeric":
        issues.append(f"{results['non_numeric_count']} non-numeric values")
    
    if check_zeros and results["zero_count"] > 0:
        issues.append(f"{results['zero_count']} zero values")
    if check_negatives and results["negative_count"] > 0:
        issues.append(f"{results['negative_count']} negative values")
    if results["extreme_large_count"] > 0:
        issues.append(f"{results['extreme_large_count']} extremely large values")
    if results["extreme_small_count"] > 0:
        issues.append(f"{results['extreme_small_count']} extremely small values")
    if results["empty_string_count"] > 0:
        issues.append(f"{results['empty_string_count']} empty/whitespace strings")
    if results["mixed_types"]:
        issues.append("mixed data types")
    
    # Only include undefined variance as issue if we expect numeric data
    if pd.isna(results["variance"]) and expected_type == "numeric":
        issues.append("undefined variance")

    return f"Found: {', '.join(issues)}" if issues else "No problematic values detected"


def detect_data_leakage_patterns(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None,
) -> DataLeakageResults:
    """
    Detect potential data leakage patterns in a DataFrame.

    This function identifies columns that might cause data leakage in machine learning
    models, such as columns with perfect correlation to target, ID-like columns, or
    future-looking features. Data leakage occurs when information from outside the
    training dataset is used to create the model, leading to overly optimistic
    performance that doesn't generalize to new data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze for potential data leakage patterns.
    target_column : str, optional
        The target column name for machine learning. If provided, the function
        will check for perfect predictors that correlate too strongly with the target.
    id_columns : List[str], optional
        Known identifier columns to exclude from analysis. These are typically
        primary keys, foreign keys, or other unique identifiers.
    datetime_columns : List[str], optional
        Known datetime columns for temporal leakage detection. Used to identify
        features that might contain future information.

    Returns
    -------
    DataLeakageResults
        TypedDict containing detailed leakage analysis:
        - potential_id_columns: Columns with very high uniqueness (>95%)
        - perfect_predictors: Columns perfectly correlated with target
        - constant_columns: Columns with only one unique value
        - duplicate_columns: Pairs of identical columns
        - temporal_leakage_risk: Features suggesting future information
        - recommendations: Actionable recommendations to address issues
        - summary: Brief summary of total issues found

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data with leakage issues
    >>> df = pd.DataFrame({
    ...     'id': range(100),  # ID column
    ...     'target': np.random.choice([0, 1], 100),
    ...     'perfect_pred': lambda x: x['target'],  # Perfect predictor
    ...     'constant': [1] * 100,  # Constant column
    ...     'duplicate': np.random.normal(0, 1, 100),
    ...     'duplicate_copy': lambda x: x['duplicate'],  # Duplicate column
    ...     'future_sales': np.random.normal(100, 10, 100)  # Suspicious name
    ... })
    >>> 
    >>> leakage = detect_data_leakage_patterns(df, target_column='target')
    >>> print(f"Issues found: {leakage['summary']}")
    >>> for rec in leakage['recommendations']:
    ...     print(f"- {rec}")

    Notes
    -----
    This function performs several checks:
    1. **ID Detection**: Looks for columns with >95% unique values
    2. **Perfect Predictors**: Finds features perfectly correlated with target
    3. **Constant Features**: Identifies columns with no variation
    4. **Duplicate Features**: Detects identical columns
    5. **Temporal Leakage**: Searches for suspicious column names
    """
    results: DataLeakageResults = {
        "potential_id_columns": [],
        "perfect_predictors": [],
        "constant_columns": [],
        "duplicate_columns": [],
        "temporal_leakage_risk": [],
        "recommendations": [],
        "summary": "",
    }

    id_columns = id_columns or []
    datetime_columns = datetime_columns or []

    # Analyze each column for potential leakage patterns
    for col in df.columns:
        if col == target_column or col in id_columns:
            continue

        series = df[col]

        # Check for potential ID columns (very high uniqueness)
        if series.dtype in ["object", "int64", "int32"]:
            uniqueness_ratio = series.nunique() / len(series)
            if uniqueness_ratio > 0.95:
                results["potential_id_columns"].append({
                    "column": col,
                    "uniqueness_ratio": uniqueness_ratio,
                    "unique_values": series.nunique(),
                })

        # Check for constant columns (no variation)
        if series.nunique() == 1:
            results["constant_columns"].append(col)

        # Check for perfect predictors if target is provided
        if target_column and target_column in df.columns:
            perfect_predictor_info = _check_perfect_predictor(series, df[target_column], col)
            if perfect_predictor_info:
                results["perfect_predictors"].append(perfect_predictor_info)

    # Check for duplicate columns
    results["duplicate_columns"] = _find_duplicate_columns(df)

    # Check for temporal leakage patterns
    if datetime_columns:
        results["temporal_leakage_risk"] = _check_temporal_leakage(df, datetime_columns)

    # Generate recommendations and summary
    results["recommendations"] = _generate_leakage_recommendations(results)
    results["summary"] = _generate_leakage_summary(results)

    return results


def _check_perfect_predictor(
    series: pd.Series, 
    target: pd.Series, 
    column_name: str
) -> Optional[Dict[str, Any]]:
    """
    Check if a column is a perfect predictor of the target.
    
    Parameters
    ----------
    series : pd.Series
        The feature column to check
    target : pd.Series
        The target column
    column_name : str
        Name of the feature column
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with correlation info if perfect predictor found, None otherwise
    """
    try:
        # For numeric columns, check correlation
        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
            corr = series.corr(target)
            if abs(corr) > 0.99:
                return {"column": column_name, "correlation": corr}
        
        # For categorical, check perfect separation (only for low cardinality)
        elif series.nunique() < 100:
            crosstab = pd.crosstab(series, target)
            # Check if each category maps to only one target value
            perfect_separation = all((crosstab > 0).sum(axis=1) == 1)
            if perfect_separation:
                return {"column": column_name, "type": "perfect_separation"}
                
    except (ValueError, TypeError, KeyError, pd.errors.DataError) as e:
        logger.debug(f"Could not check correlation for {column_name}: {str(e)}")
    
    return None


def _find_duplicate_columns(df: pd.DataFrame) -> List[tuple]:
    """
    Find pairs of duplicate columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check for duplicate columns
        
    Returns
    -------
    List[tuple]
        List of tuples containing pairs of duplicate column names
    """
    duplicate_pairs = []
    
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i + 1:]:
            try:
                if df[col1].equals(df[col2]):
                    duplicate_pairs.append((col1, col2))
            except (TypeError, ValueError, AttributeError):
                # Some columns might not be comparable (different dtypes, etc.)
                pass
    
    return duplicate_pairs


def _check_temporal_leakage(df: pd.DataFrame, datetime_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Check for potential temporal leakage patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check
    datetime_columns : List[str]
        List of datetime column names
        
    Returns
    -------
    List[Dict[str, Any]]
        List of temporal leakage risks found
    """
    temporal_risks = []
    
    for dt_col in datetime_columns:
        if dt_col in df.columns:
            try:
                # Validate that the column can be converted to datetime
                pd.to_datetime(df[dt_col], errors="coerce")
                
                # Check if any columns contain future information
                for col in df.columns:
                    if col != dt_col and "future" in col.lower():
                        temporal_risks.append({
                            "column": col,
                            "reason": "Column name suggests future information",
                        })
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                logger.debug(f"Could not analyze datetime column {dt_col}: {str(e)}")
    
    return temporal_risks


def _generate_leakage_recommendations(results: DataLeakageResults) -> List[str]:
    """
    Generate recommendations based on leakage detection results.
    
    Parameters
    ----------
    results : DataLeakageResults
        The leakage detection results
        
    Returns
    -------
    List[str]
        List of actionable recommendations
    """
    recommendations = []
    
    if results["potential_id_columns"]:
        id_cols = [item['column'] for item in results['potential_id_columns']]
        recommendations.append(f"Remove potential ID columns: {id_cols}")
    
    if results["constant_columns"]:
        recommendations.append(f"Remove constant columns: {results['constant_columns']}")
    
    if results["perfect_predictors"]:
        perfect_cols = [item['column'] for item in results['perfect_predictors']]
        recommendations.append(f"Investigate perfect predictors for leakage: {perfect_cols}")
    
    if results["duplicate_columns"]:
        recommendations.append(f"Remove duplicate columns: {results['duplicate_columns']}")
    
    return recommendations


def _generate_leakage_summary(results: DataLeakageResults) -> str:
    """
    Generate summary of leakage detection results.
    
    Parameters
    ----------
    results : DataLeakageResults
        The leakage detection results
        
    Returns
    -------
    str
        Summary string
    """
    issues_found = sum([
        len(results["potential_id_columns"]),
        len(results["constant_columns"]),
        len(results["perfect_predictors"]),
        len(results["duplicate_columns"]),
        len(results["temporal_leakage_risk"]),
    ])
    
    return f"Found {issues_found} potential data leakage issues"


def analyze_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5,
    return_indices: bool = True,
) -> OutlierResults:
    """
    Detect outliers in a numeric series using various statistical methods.

    This function implements multiple outlier detection methods commonly used
    in data analysis and machine learning preprocessing. Outliers can significantly
    impact model performance and statistical analysis, so identifying them is
    crucial for data quality assessment.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to analyze for outliers. Must contain numeric data
        or data that can be converted to numeric.
    method : str, default 'iqr'
        Method for outlier detection. Available options:
        - 'iqr': Interquartile Range method (robust, good for skewed data)
        - 'zscore': Z-score method (assumes normal distribution)
        - 'mad': Median Absolute Deviation (robust alternative to z-score)
    threshold : float, default 1.5
        Threshold multiplier for outlier detection:
        - For IQR: typically 1.5 (mild outliers) or 3.0 (extreme outliers)
        - For Z-score: typically 2.0 (5% outliers) or 3.0 (0.3% outliers)
        - For MAD: typically 2.5 or 3.0 (similar to z-score)
    return_indices : bool, default True
        Whether to return the specific row indices of detected outliers.
        Set to False for large datasets to save memory.

    Returns
    -------
    OutlierResults
        TypedDict containing comprehensive outlier analysis:
        - method: The detection method used
        - threshold: The threshold value applied
        - outlier_count: Number of outliers detected
        - outlier_percentage: Percentage of data that are outliers
        - lower_bound: Lower boundary for normal values
        - upper_bound: Upper boundary for normal values
        - outlier_indices: Row indices of outlier values (if return_indices=True)
        - summary: Brief text summary of findings

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create data with outliers
    >>> data = pd.Series([1, 2, 3, 4, 5, 100, -50, 6, 7, 8])
    >>> 
    >>> # Detect outliers using IQR method
    >>> result = analyze_outliers(data, method='iqr', threshold=1.5)
    >>> print(f"Outliers found: {result['outlier_count']} ({result['outlier_percentage']:.1f}%)")
    >>> print(f"Outlier indices: {result['outlier_indices']}")
    >>> 
    >>> # Use Z-score for normally distributed data
    >>> normal_data = pd.Series(np.random.normal(0, 1, 1000))
    >>> normal_data.iloc[0] = 10  # Add an outlier
    >>> result = analyze_outliers(normal_data, method='zscore', threshold=3.0)
    >>> print(f"Z-score outliers: {result['summary']}")

    Notes
    -----
    **Method Details:**
    
    1. **IQR (Interquartile Range)**:
       - Outliers: Q1 - threshold×IQR > value > Q3 + threshold×IQR
       - Robust to skewness and doesn't assume normal distribution
       - threshold=1.5 is standard, threshold=3.0 for extreme outliers only
    
    2. **Z-Score**:
       - Outliers: |value - mean| / std > threshold
       - Assumes normal distribution, sensitive to extreme values
       - threshold=2.0 captures ~5% of data, threshold=3.0 captures ~0.3%
    
    3. **MAD (Median Absolute Deviation)**:
       - Modified z-score using median instead of mean
       - More robust than z-score, good for skewed distributions
       - Uses factor 0.6745 to make it comparable to standard deviation
    """
    results: OutlierResults = {
        "method": method,
        "threshold": threshold,
        "outlier_count": 0,
        "outlier_percentage": 0.0,
        "lower_bound": None,
        "upper_bound": None,
        "outlier_indices": [],
        "summary": "",
    }

    # Convert to numeric and remove NaN values
    try:
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) == 0:
            results["summary"] = "No numeric values found"
            return results
    except Exception as e:
        results["summary"] = f"Error converting to numeric: {str(e)}"
        return results

    # Apply the selected outlier detection method
    try:
        if method == "iqr":
            outlier_mask, lower_bound, upper_bound = _detect_outliers_iqr(
                numeric_series, threshold
            )
        elif method == "zscore":
            outlier_mask, lower_bound, upper_bound = _detect_outliers_zscore(
                numeric_series, threshold
            )
        elif method == "mad":
            outlier_mask, lower_bound, upper_bound = _detect_outliers_mad(
                numeric_series, threshold
            )
        else:
            results["summary"] = f"Unknown method: {method}"
            return results

        # Populate results
        results["outlier_count"] = outlier_mask.sum()
        results["outlier_percentage"] = (results["outlier_count"] / len(numeric_series)) * 100
        results["lower_bound"] = lower_bound
        results["upper_bound"] = upper_bound

        if return_indices and results["outlier_count"] > 0:
            results["outlier_indices"] = numeric_series[outlier_mask].index.tolist()

        results["summary"] = (
            f"Found {results['outlier_count']} outliers ({results['outlier_percentage']:.2f}%)"
        )

    except Exception as e:
        results["summary"] = f"Error in {method} detection: {str(e)}"

    return results


def _detect_outliers_iqr(series: pd.Series, threshold: float) -> tuple:
    """
    Detect outliers using the Interquartile Range method.
    
    Parameters
    ----------
    series : pd.Series
        Numeric series to analyze
    threshold : float
        IQR multiplier threshold
        
    Returns
    -------
    tuple
        (outlier_mask, lower_bound, upper_bound)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    
    return outlier_mask, lower_bound, upper_bound


def _detect_outliers_zscore(series: pd.Series, threshold: float) -> tuple:
    """
    Detect outliers using the Z-score method.
    
    Parameters
    ----------
    series : pd.Series
        Numeric series to analyze
    threshold : float
        Z-score threshold
        
    Returns
    -------
    tuple
        (outlier_mask, lower_bound, upper_bound)
    """
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        raise ValueError("Standard deviation is zero, cannot detect outliers")
    
    z_scores = np.abs((series - mean) / std)
    outlier_mask = z_scores > threshold
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    
    return outlier_mask, lower_bound, upper_bound


def _detect_outliers_mad(series: pd.Series, threshold: float) -> tuple:
    """
    Detect outliers using the Median Absolute Deviation method.
    
    Parameters
    ----------
    series : pd.Series
        Numeric series to analyze
    threshold : float
        MAD threshold
        
    Returns
    -------
    tuple
        (outlier_mask, lower_bound, upper_bound)
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    
    if mad == 0:
        raise ValueError("MAD is zero, cannot detect outliers")
    
    modified_z_scores = 0.6745 * (series - median) / mad
    outlier_mask = np.abs(modified_z_scores) > threshold
    lower_bound = median - threshold * mad / 0.6745
    upper_bound = median + threshold * mad / 0.6745
    
    return outlier_mask, lower_bound, upper_bound


def analyze_distribution(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze the statistical distribution of a numeric series.

    This function provides comprehensive statistical analysis of numeric data,
    including descriptive statistics, distribution shape analysis, and normality
    testing. Understanding data distribution is crucial for choosing appropriate
    statistical methods and machine learning algorithms.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to analyze. Should contain numeric data or data
        that can be converted to numeric. Non-numeric values will be excluded.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comprehensive distribution analysis:
        - count: Number of valid (non-NaN) numeric values
        - mean: Arithmetic mean of the data
        - median: 50th percentile (middle value)
        - mode: Most frequently occurring value
        - std: Standard deviation (measure of spread)
        - skewness: Measure of asymmetry (0=symmetric, >0=right-skewed, <0=left-skewed)
        - kurtosis: Measure of tail heaviness (0=normal, >0=heavy tails, <0=light tails)
        - normality_test: Statistical test results for normal distribution
        - distribution_type: Classified distribution type based on characteristics
        - quantiles: Key percentiles (min, Q1, median, Q3, max)
        - summary: Brief text description of the distribution

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Analyze normal distribution
    >>> normal_data = pd.Series(np.random.normal(100, 15, 1000))
    >>> result = analyze_distribution(normal_data)
    >>> print(f"Distribution type: {result['distribution_type']}")
    >>> print(f"Skewness: {result['skewness']:.3f}")
    >>> print(f"Is normal: {result['normality_test'].get('is_normal', 'Unknown')}")
    >>> 
    >>> # Analyze skewed distribution
    >>> skewed_data = pd.Series(np.random.exponential(2, 1000))
    >>> result = analyze_distribution(skewed_data)
    >>> print(f"Summary: {result['summary']}")

    Notes
    -----
    **Distribution Classification:**
    - **normal**: Passes normality test (Shapiro-Wilk p-value > 0.05)
    - **highly_skewed**: |skewness| > 2
    - **high_variance**: Coefficient of variation > 1
    - **other**: Doesn't fit the above categories
    
    **Skewness Interpretation:**
    - |skewness| < 0.5: Approximately symmetric
    - 0.5 ≤ |skewness| < 1: Moderately skewed
    - |skewness| ≥ 1: Highly skewed
    
    **Normality Testing:**
    - Uses Shapiro-Wilk test for samples ≤ 5000 (most powerful for small samples)
    - For larger samples, consider using Kolmogorov-Smirnov or Anderson-Darling tests
    """
    results: Dict[str, Any] = {
        "count": 0,
        "mean": None,
        "median": None,
        "mode": None,
        "std": None,
        "skewness": None,
        "kurtosis": None,
        "normality_test": {},
        "distribution_type": "unknown",
        "quantiles": {},
        "summary": "",
    }

    # Convert to numeric and validate data
    try:
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) < 3:
            results["summary"] = "Insufficient data for distribution analysis (need ≥3 values)"
            return results
    except Exception as e:
        results["summary"] = f"Error converting to numeric: {str(e)}"
        return results

    # Calculate basic descriptive statistics
    results["count"] = len(numeric_series)
    results["mean"] = float(numeric_series.mean())
    results["median"] = float(numeric_series.median())
    results["std"] = float(numeric_series.std())

    # Calculate mode (handle potential multiple modes)
    try:
        mode_result = stats.mode(numeric_series, keepdims=True)
        results["mode"] = float(mode_result.mode[0]) if len(mode_result.mode) > 0 else None
    except Exception as e:
        logger.debug(f"Could not calculate mode: {str(e)}")
        results["mode"] = None

    # Calculate distribution shape measures
    if results["count"] >= 3:
        try:
            results["skewness"] = float(stats.skew(numeric_series))
            results["kurtosis"] = float(stats.kurtosis(numeric_series))
        except Exception as e:
            logger.debug(f"Could not calculate skewness/kurtosis: {str(e)}")

    # Calculate quantiles for comprehensive distribution summary
    try:
        results["quantiles"] = {
            "min": float(numeric_series.min()),
            "q1": float(numeric_series.quantile(0.25)),
            "q2": float(numeric_series.quantile(0.50)),  # Same as median
            "q3": float(numeric_series.quantile(0.75)),
            "max": float(numeric_series.max()),
        }
    except Exception as e:
        logger.debug(f"Could not calculate quantiles: {str(e)}")
        results["quantiles"] = {}

    # Perform normality test (Shapiro-Wilk for reasonable sample sizes)
    if 3 <= results["count"] <= 5000:
        try:
            stat, p_value = stats.shapiro(numeric_series)
            results["normality_test"] = {
                "test": "Shapiro-Wilk",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05,
            }
        except Exception as e:
            results["normality_test"] = {"error": str(e)}

    # Classify distribution type based on statistical characteristics
    results["distribution_type"] = _classify_distribution_type(results)

    # Generate human-readable summary
    results["summary"] = _generate_distribution_summary(results)

    return results


def _classify_distribution_type(results: Dict[str, Any]) -> str:
    """
    Classify distribution type based on statistical characteristics.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Distribution analysis results
        
    Returns
    -------
    str
        Classification of distribution type
    """
    # Check if normally distributed
    if results["normality_test"].get("is_normal", False):
        return "normal"
    
    # Check for high skewness
    skewness = results.get("skewness", 0)
    if skewness is not None and abs(skewness) > 2:
        return "highly_skewed"
    
    # Check for high variance relative to mean
    mean = results.get("mean", 0)
    std = results.get("std", 0)
    if mean != 0 and std != 0 and (std / abs(mean)) > 1:
        return "high_variance"
    
    return "other"


def _generate_distribution_summary(results: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of distribution analysis.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Distribution analysis results
        
    Returns
    -------
    str
        Summary string
    """
    skewness = results.get("skewness", 0)
    
    if skewness is None:
        skew_desc = "unknown skewness"
    elif abs(skewness) < 0.5:
        skew_desc = "symmetric"
    elif abs(skewness) < 1:
        skew_desc = "moderately skewed"
    else:
        skew_desc = "highly skewed"
    
    distribution_type = results.get("distribution_type", "unknown")
    return f"Distribution: {distribution_type}, {skew_desc}"


def analyze_patterns(series: pd.Series) -> PatternResults:
    """
    Detect patterns and anomalies in data values.

    This function analyzes data for various patterns that might indicate data quality
    issues, systematic biases, or interesting characteristics that need attention
    during analysis. Pattern detection helps identify potential preprocessing needs
    and data collection issues.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to analyze for patterns. Can contain any data type,
        though some pattern checks are specific to numeric data.

    Returns
    -------
    PatternResults
        TypedDict containing comprehensive pattern analysis:
        - is_constant: Whether all non-null values are identical
        - is_monotonic_increasing: Whether values always increase (numeric only)
        - is_monotonic_decreasing: Whether values always decrease (numeric only)
        - has_duplicates: Whether any values appear more than once
        - duplicate_count: Number of duplicate occurrences
        - unique_count: Number of distinct values
        - uniqueness_ratio: Proportion of values that are unique (0-1)
        - most_frequent_values: Top 5 most common values and their counts
        - suspected_rounding: Whether numeric values appear artificially rounded
        - decimal_places_distribution: Distribution of decimal places in numeric data
        - summary: Brief text description of patterns found

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Analyze constant data
    >>> constant_data = pd.Series([5, 5, 5, 5, 5])
    >>> result = analyze_patterns(constant_data)
    >>> print(f"Is constant: {result['is_constant']}")
    >>> 
    >>> # Analyze rounded data
    >>> rounded_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> result = analyze_patterns(rounded_data)
    >>> print(f"Suspected rounding: {result['suspected_rounding']}")
    >>> 
    >>> # Analyze monotonic sequence
    >>> sequence = pd.Series([1, 2, 3, 4, 5, 6])
    >>> result = analyze_patterns(sequence)
    >>> print(f"Monotonic increasing: {result['is_monotonic_increasing']}")

    Notes
    -----
    **Pattern Categories:**
    
    1. **Uniqueness Patterns:**
       - Constant values (all identical)
       - High duplication rates
       - Very low or high uniqueness ratios
    
    2. **Sequence Patterns (Numeric):**
       - Monotonic increasing/decreasing sequences
       - Regular intervals or steps
    
    3. **Precision Patterns (Numeric):**
       - Suspected rounding (most values have same decimal places)
       - Artificial precision (too many decimal places)
    
    4. **Frequency Patterns:**
       - Highly skewed value distributions
       - Unexpected common values
    
    **Rounding Detection:**
    Values are considered "suspected rounding" if >90% have the same number
    of decimal places, which might indicate artificial precision or data
    collection constraints.
    """
    results: PatternResults = {
        "is_constant": False,
        "is_monotonic_increasing": False,
        "is_monotonic_decreasing": False,
        "has_duplicates": False,
        "duplicate_count": 0,
        "unique_count": 0,
        "uniqueness_ratio": 0.0,
        "most_frequent_values": {},
        "suspected_rounding": False,
        "decimal_places_distribution": {},
        "summary": "",
    }

    if len(series) == 0:
        results["summary"] = "Empty series"
        return results

    # Work with non-null values
    clean_series = series.dropna()
    if len(clean_series) == 0:
        results["summary"] = "All values are NaN"
        return results

    # Analyze uniqueness and duplication patterns
    results["unique_count"] = clean_series.nunique()
    results["uniqueness_ratio"] = results["unique_count"] / len(clean_series)
    results["has_duplicates"] = results["unique_count"] < len(clean_series)
    results["duplicate_count"] = len(clean_series) - results["unique_count"]

    # Check for constant values
    results["is_constant"] = results["unique_count"] == 1

    # Analyze value frequency patterns
    value_counts = clean_series.value_counts()
    results["most_frequent_values"] = dict(value_counts.head(5))

    # Perform numeric-specific pattern analysis
    numeric_patterns = _analyze_numeric_patterns(clean_series)
    results.update(numeric_patterns)

    # Generate summary of detected patterns
    results["summary"] = _generate_pattern_summary(results)

    return results


def _analyze_numeric_patterns(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze patterns specific to numeric data.
    
    Parameters
    ----------
    series : pd.Series
        Series to analyze for numeric patterns
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with numeric pattern results
    """
    numeric_results = {
        "is_monotonic_increasing": False,
        "is_monotonic_decreasing": False,
        "suspected_rounding": False,
        "decimal_places_distribution": {},
    }
    
    try:
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_series) <= 1:
            return numeric_results

        # Check for monotonic patterns
        numeric_results["is_monotonic_increasing"] = numeric_series.is_monotonic_increasing
        numeric_results["is_monotonic_decreasing"] = numeric_series.is_monotonic_decreasing

        # Analyze decimal places for rounding detection
        decimal_places = _extract_decimal_places(numeric_series)
        if decimal_places:
            decimal_counter = Counter(decimal_places)
            numeric_results["decimal_places_distribution"] = dict(decimal_counter)

            # Check for suspected rounding (>90% of values have same decimal places)
            most_common_decimal = decimal_counter.most_common(1)[0]
            if most_common_decimal[1] / len(decimal_places) > 0.9:
                numeric_results["suspected_rounding"] = True

    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"Could not analyze numeric patterns: {str(e)}")

    return numeric_results


def _extract_decimal_places(numeric_series: pd.Series) -> List[int]:
    """
    Extract the number of decimal places for each value in a numeric series.
    
    Parameters
    ----------
    numeric_series : pd.Series
        Numeric series to analyze
        
    Returns
    -------
    List[int]
        List of decimal place counts for each value
    """
    decimal_places = []
    
    for val in numeric_series:
        if pd.notna(val) and np.isfinite(val):
            decimal_str = str(val).split(".")
            if len(decimal_str) > 1:
                # Count trailing zeros as no decimal places
                decimal_places.append(len(decimal_str[1].rstrip("0")))
            else:
                decimal_places.append(0)
    
    return decimal_places


def _generate_pattern_summary(results: PatternResults) -> str:
    """
    Generate summary of detected patterns.
    
    Parameters
    ----------
    results : PatternResults
        Pattern analysis results
        
    Returns
    -------
    str
        Summary string
    """
    patterns = []
    
    if results["is_constant"]:
        patterns.append("constant values")
    if results["is_monotonic_increasing"]:
        patterns.append("monotonic increasing")
    if results["is_monotonic_decreasing"]:
        patterns.append("monotonic decreasing")
    if results["suspected_rounding"]:
        patterns.append("suspected rounding")
    if results["uniqueness_ratio"] < 0.01:
        patterns.append("very low uniqueness")

    return f"Patterns detected: {', '.join(patterns)}" if patterns else "No special patterns detected"


def analyze_temporal_data(series: pd.Series) -> TemporalResults:
    """
    Analyze temporal/datetime data for quality issues and patterns.

    This function performs comprehensive analysis of datetime data, checking for
    conversion issues, logical inconsistencies, and temporal patterns that might
    indicate data quality problems or interesting characteristics for time series
    analysis.

    Parameters
    ----------
    series : pd.Series
        The pandas Series containing datetime data or data that should be
        converted to datetime format. Can contain strings, timestamps, or
        other datetime-like formats.

    Returns
    -------
    TemporalResults
        TypedDict containing comprehensive temporal analysis:
        - is_datetime: Whether data can be successfully converted to datetime
        - conversion_errors: Number of values that couldn't be parsed as dates
        - future_dates: Number of dates that occur in the future
        - very_old_dates: Number of dates before 1900 (potentially erroneous)
        - date_range: Min, max date and total span in days
        - gaps_analysis: Analysis of time gaps between consecutive dates
        - timezone_info: Timezone information if detected
        - most_common_times: Most frequent time-of-day values
        - summary: Brief text summary of temporal data quality

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime, timedelta
    >>> 
    >>> # Analyze clean datetime data
    >>> dates = pd.date_range('2023-01-01', periods=100, freq='D')
    >>> result = analyze_temporal_data(dates)
    >>> print(f"Is datetime: {result['is_datetime']}")
    >>> print(f"Date range: {result['date_range']['span_days']} days")
    >>> 
    >>> # Analyze problematic datetime data
    >>> problematic_dates = pd.Series([
    ...     '2023-01-01', '2023-01-02', 'invalid_date', 
    ...     '2025-12-31', '1800-01-01'
    ... ])
    >>> result = analyze_temporal_data(problematic_dates)
    >>> print(f"Conversion errors: {result['conversion_errors']}")
    >>> print(f"Future dates: {result['future_dates']}")
    >>> print(f"Very old dates: {result['very_old_dates']}")

    Notes
    -----
    **Quality Checks Performed:**
    
    1. **Conversion Validation:**
       - Attempts pandas datetime conversion with error tracking
       - Identifies unparseable date strings or formats
    
    2. **Logical Validation:**
       - Future dates (beyond current time)
       - Historically implausible dates (before 1900)
       - Extreme date ranges that might indicate errors
    
    3. **Pattern Analysis:**
       - Time series gaps and frequency detection
       - Common time-of-day patterns
       - Timezone information extraction
    
    4. **Temporal Consistency:**
       - Regular vs. irregular time intervals
       - Missing time periods or unusual gaps
    
    **Common Issues Detected:**
    - Invalid date formats or unparseable strings
    - Data entry errors (wrong century, impossible dates)
    - Future dates in historical datasets
    - Inconsistent datetime formats within the same column
    - Missing timezone information in time-sensitive data
    """
    results: TemporalResults = {
        "is_datetime": False,
        "conversion_errors": 0,
        "future_dates": 0,
        "very_old_dates": 0,
        "date_range": {},
        "gaps_analysis": {},
        "timezone_info": None,
        "most_common_times": {},
        "summary": "",
    }

    # Attempt datetime conversion and track errors
    try:
        datetime_series = pd.to_datetime(series, errors="coerce")
        valid_dates = datetime_series.dropna()
        
        # Count conversion errors (new NaNs that weren't in original)
        results["conversion_errors"] = datetime_series.isna().sum() - series.isna().sum()

        if len(valid_dates) == 0:
            results["summary"] = "No valid datetime values found"
            return results

        results["is_datetime"] = True

        # Analyze date range and extreme values
        date_range_analysis = _analyze_date_range(valid_dates)
        results.update(date_range_analysis)

        # Analyze time series gaps and patterns
        if len(valid_dates) > 1:
            gaps_analysis = _analyze_temporal_gaps(valid_dates)
            results["gaps_analysis"] = gaps_analysis

        # Extract timezone information
        results["timezone_info"] = _extract_timezone_info(valid_dates)

        # Analyze time-of-day patterns (if timestamps include time)
        time_patterns = _analyze_time_patterns(valid_dates)
        results["most_common_times"] = time_patterns

        # Generate summary
        results["summary"] = _generate_temporal_summary(results)

    except Exception as e:
        results["summary"] = f"Not datetime data or conversion failed: {str(e)}"

    return results


def _analyze_date_range(valid_dates: pd.Series) -> Dict[str, Any]:
    """
    Analyze the date range and identify extreme values.
    
    Parameters
    ----------
    valid_dates : pd.Series
        Series of valid datetime values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with date range analysis
    """
    current_date = pd.Timestamp.now()
    old_date_threshold = pd.Timestamp("1900-01-01")
    
    min_date = valid_dates.min()
    max_date = valid_dates.max()
    span_days = (max_date - min_date).days
    
    future_dates = (valid_dates > current_date).sum()
    very_old_dates = (valid_dates < old_date_threshold).sum()
    
    return {
        "date_range": {
            "min": str(min_date),
            "max": str(max_date),
            "span_days": span_days,
        },
        "future_dates": future_dates,
        "very_old_dates": very_old_dates,
    }


def _analyze_temporal_gaps(valid_dates: pd.Series) -> Dict[str, Any]:
    """
    Analyze gaps between consecutive dates in a time series.
    
    Parameters
    ----------
    valid_dates : pd.Series
        Series of valid datetime values
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with gap analysis results
    """
    sorted_dates = valid_dates.sort_values()
    gaps = sorted_dates.diff().dropna()

    if len(gaps) == 0:
        return {}

    return {
        "min_gap": str(gaps.min()),
        "max_gap": str(gaps.max()),
        "median_gap": str(gaps.median()),
        "has_regular_frequency": gaps.nunique() == 1,
    }


def _extract_timezone_info(valid_dates: pd.Series) -> Optional[str]:
    """
    Extract timezone information from datetime series.
    
    Parameters
    ----------
    valid_dates : pd.Series
        Series of valid datetime values
        
    Returns
    -------
    Optional[str]
        Timezone string if found, None otherwise
    """
    try:
        if len(valid_dates) > 0 and hasattr(valid_dates.iloc[0], "tz"):
            return str(valid_dates.iloc[0].tz)
    except (AttributeError, IndexError):
        pass
    
    return None


def _analyze_time_patterns(valid_dates: pd.Series) -> Dict[str, int]:
    """
    Analyze time-of-day patterns in datetime data.
    
    Parameters
    ----------
    valid_dates : pd.Series
        Series of valid datetime values
        
    Returns
    -------
    Dict[str, int]
        Dictionary of most common times and their frequencies
    """
    try:
        # Check if any dates have non-zero time components
        has_time = any(d.time() != pd.Timestamp("00:00:00").time() for d in valid_dates)
        
        if has_time:
            time_counts = valid_dates.dt.time.value_counts().head(5)
            return {str(k): v for k, v in time_counts.items()}
    except (AttributeError, TypeError):
        pass
    
    return {}


def _generate_temporal_summary(results: TemporalResults) -> str:
    """
    Generate summary of temporal analysis results.
    
    Parameters
    ----------
    results : TemporalResults
        Temporal analysis results
        
    Returns
    -------
    str
        Summary string
    """
    issues = []
    
    if results["conversion_errors"] > 0:
        issues.append(f"{results['conversion_errors']} conversion errors")
    if results["future_dates"] > 0:
        issues.append(f"{results['future_dates']} future dates")
    if results["very_old_dates"] > 0:
        issues.append(f"{results['very_old_dates']} very old dates")

    return f"Issues found: {', '.join(issues)}" if issues else "Temporal data appears valid"


def analyze_categorical_data(series: pd.Series) -> CategoricalResults:
    """
    Analyze categorical/string data for quality issues and characteristics.

    This function performs comprehensive analysis of categorical data, checking for
    common data quality issues like inconsistent formatting, encoding problems,
    and structural issues that can impact analysis and machine learning models.

    Parameters
    ----------
    series : pd.Series
        The pandas Series containing categorical data. Can contain strings,
        mixed types, or any data that should be treated as categories.

    Returns
    -------
    CategoricalResults
        TypedDict containing comprehensive categorical analysis:
        - unique_count: Number of distinct categories
        - most_frequent: Top 5 most common categories and their counts
        - least_frequent: Bottom 5 least common categories and their counts
        - has_leading_trailing_spaces: Whether any values have extra whitespace
        - has_special_characters: Whether values contain non-alphanumeric characters
        - has_mixed_case: Whether values show inconsistent capitalization
        - encoding_issues: Whether non-ASCII characters suggest encoding problems
        - length_statistics: Min, max, mean, std of string lengths
        - summary: Brief text summary of categorical data quality

    Examples
    --------
    >>> import pandas as pd
    >>> 
    >>> # Analyze clean categorical data
    >>> clean_categories = pd.Series(['A', 'B', 'C', 'A', 'B'])
    >>> result = analyze_categorical_data(clean_categories)
    >>> print(f"Unique categories: {result['unique_count']}")
    >>> 
    >>> # Analyze problematic categorical data
    >>> messy_categories = pd.Series([
    ...     'Category A', '  Category B  ', 'category_c', 
    ...     'CATEGORY A', 'Cat@gory!', 'Catégorie'
    ... ])
    >>> result = analyze_categorical_data(messy_categories)
    >>> print(f"Has spaces: {result['has_leading_trailing_spaces']}")
    >>> print(f"Mixed case: {result['has_mixed_case']}")
    >>> print(f"Special chars: {result['has_special_characters']}")
    >>> print(f"Encoding issues: {result['encoding_issues']}")

    Notes
    -----
    **Quality Issues Detected:**
    
    1. **Formatting Issues:**
       - Leading/trailing whitespace that can cause duplicate categories
       - Inconsistent capitalization (e.g., 'Apple' vs 'apple' vs 'APPLE')
       - Special characters that might indicate data corruption
    
    2. **Encoding Issues:**
       - Non-ASCII characters that might cause problems in some systems
       - Potentially corrupted text due to encoding conversion errors
    
    3. **Structural Issues:**
       - Very long category names that might be descriptions rather than categories
       - Very short names that might be codes rather than readable categories
       - Extreme frequency imbalances in category distribution
    
    **Length Statistics:**
    Provides insights into whether categories are:
    - Codes (typically short, uniform length)
    - Names (moderate length, variable)
    - Descriptions (long, highly variable)
    
    **Common Recommendations:**
    - Strip leading/trailing spaces: `series.str.strip()`
    - Standardize case: `series.str.lower()` or `series.str.title()`
    - Handle encoding: Convert to consistent encoding before analysis
    - Validate categories: Check against expected category lists
    """
    results: CategoricalResults = {
        "unique_count": 0,
        "most_frequent": {},
        "least_frequent": {},
        "has_leading_trailing_spaces": False,
        "has_special_characters": False,
        "has_mixed_case": False,
        "encoding_issues": False,
        "length_statistics": {},
        "summary": "",
    }

    clean_series = series.dropna()
    if len(clean_series) == 0:
        results["summary"] = "All values are NaN"
        return results

    # Convert to string for consistent analysis
    try:
        str_series = clean_series.astype(str)
    except Exception as e:
        results["summary"] = f"Error converting to string: {str(e)}"
        return results

    # Basic category analysis
    results["unique_count"] = str_series.nunique()

    # Frequency analysis
    value_counts = str_series.value_counts()
    results["most_frequent"] = dict(value_counts.head(5))
    results["least_frequent"] = dict(value_counts.tail(5))

    # Check for formatting issues
    formatting_issues = _check_categorical_formatting(str_series)
    results.update(formatting_issues)

    # Check for encoding issues
    results["encoding_issues"] = _check_encoding_issues(str_series)

    # Calculate string length statistics
    results["length_statistics"] = _calculate_length_statistics(str_series)

    # Generate summary
    results["summary"] = _generate_categorical_summary(results)

    return results


def _check_categorical_formatting(str_series: pd.Series) -> Dict[str, bool]:
    """
    Check for common formatting issues in categorical data.
    
    Parameters
    ----------
    str_series : pd.Series
        String series to check for formatting issues
        
    Returns
    -------
    Dict[str, bool]
        Dictionary indicating presence of formatting issues
    """
    # Check for leading/trailing spaces
    trimmed_series = str_series.str.strip()
    has_leading_trailing_spaces = not str_series.equals(trimmed_series)

    # Check for special characters (non-alphanumeric, space, dash, underscore, period)
    special_char_pattern = r"[^a-zA-Z0-9\s\-_.]"
    has_special_characters = str_series.str.contains(special_char_pattern, regex=True).any()

    # Check for mixed case inconsistencies
    lower_series = str_series.str.lower()
    upper_series = str_series.str.upper()
    has_mixed_case = (str_series != lower_series).any() and (str_series != upper_series).any()

    return {
        "has_leading_trailing_spaces": has_leading_trailing_spaces,
        "has_special_characters": has_special_characters,
        "has_mixed_case": has_mixed_case,
    }


def _check_encoding_issues(str_series: pd.Series) -> bool:
    """
    Check for potential encoding issues in string data.
    
    Parameters
    ----------
    str_series : pd.Series
        String series to check for encoding issues
        
    Returns
    -------
    bool
        True if encoding issues detected
    """
    try:
        # Sample up to 100 values to check for ASCII encoding issues
        sample_size = min(100, len(str_series))
        sample_values = str_series.sample(sample_size) if sample_size > 0 else str_series
        
        for val in sample_values:
            val.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True
    except Exception:
        return False


def _calculate_length_statistics(str_series: pd.Series) -> Dict[str, float]:
    """
    Calculate statistics about string lengths in categorical data.
    
    Parameters
    ----------
    str_series : pd.Series
        String series to analyze
        
    Returns
    -------
    Dict[str, float]
        Dictionary with length statistics
    """
    try:
        lengths = str_series.str.len()
        return {
            "min": float(lengths.min()),
            "max": float(lengths.max()),
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
        }
    except Exception:
        return {"min": 0, "max": 0, "mean": 0, "std": 0}


def _generate_categorical_summary(results: CategoricalResults) -> str:
    """
    Generate summary of categorical analysis results.
    
    Parameters
    ----------
    results : CategoricalResults
        Categorical analysis results
        
    Returns
    -------
    str
        Summary string
    """
    issues = []
    
    if results["has_leading_trailing_spaces"]:
        issues.append("leading/trailing spaces")
    if results["has_special_characters"]:
        issues.append("special characters")
    if results["encoding_issues"]:
        issues.append("encoding issues")
    if results["has_mixed_case"]:
        issues.append("mixed case formatting")

    return f"Issues found: {', '.join(issues)}" if issues else "Categorical data appears clean"


def _detect_column_type(series: pd.Series) -> str:
    """
    Automatically detect the most appropriate data type for a series.
    
    Parameters
    ----------
    series : pd.Series
        Series to analyze for type detection
        
    Returns
    -------
    str
        Detected type: 'numeric', 'datetime', or 'categorical'
    """
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return "categorical"  # Default for empty series

    # Try datetime detection first (but only if data looks date-like)
    try:
        # Quick heuristic: check if values contain common date patterns
        str_sample = non_null_series.astype(str).iloc[:min(10, len(non_null_series))]
        
        # Look for common date patterns (YYYY, MM, DD, /, -, :)
        date_patterns = any(
            any(pattern in str(val) for pattern in ["-", "/", ":", "T"])
            and any(char.isdigit() for char in str(val))
            for val in str_sample
        )

        if date_patterns:
            # Suppress warnings during datetime conversion attempts
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                
                # Check conversion success rate on sample
                datetime_converted = pd.to_datetime(non_null_series, errors="coerce")
                conversion_rate = datetime_converted.notna().sum() / len(non_null_series)
                
                if conversion_rate > 0.5:
                    return "datetime"
    except (ValueError, TypeError, OverflowError, pd.errors.OutOfBoundsDatetime):
        pass

    # Try numeric detection
    try:
        numeric_converted = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_converted.notna().sum() / len(series)
        if numeric_ratio > 0.5:
            return "numeric"
    except (ValueError, TypeError, OverflowError):
        pass

    # Default to categorical
    return "categorical"


def analyze_column_comprehensive(
    series: pd.Series,
    expected_type: Optional[str] = None,
    config: Optional[QualityAssessmentConfig] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a single column.

    This function serves as the main entry point for analyzing individual columns,
    automatically detecting data type and performing appropriate analyses. It
    combines basic quality checks, distribution analysis, pattern detection, and
    type-specific analysis to provide a complete picture of column quality.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to analyze comprehensively.
    column_type : str, optional
        Force specific type analysis. Options: 'numeric', 'datetime', 'categorical'.
        If None, the function will automatically detect the most appropriate type
        based on the data content and conversion success rates.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comprehensive analysis results:
        - column_name: Name of the analyzed column
        - detected_type: Auto-detected or specified data type
        - basic_quality: Results from problematic values check
        - distribution: Statistical distribution analysis (numeric only)
        - outliers: Outlier detection results (numeric only)
        - patterns: Pattern detection results (all types)
        - type_specific: Specialized analysis based on detected type
        - recommendations: List of actionable improvement suggestions

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Comprehensive analysis of numeric column
    >>> numeric_data = pd.Series([1, 2, 3, np.nan, 100, 4, 5], name='sales')
    >>> result = analyze_column_comprehensive(numeric_data)
    >>> print(f"Type: {result['detected_type']}")
    >>> print(f"Quality: {'CLEAN' if result['basic_quality']['is_clean'] else 'ISSUES'}")
    >>> print(f"Recommendations: {result['recommendations']}")
    >>> 
    >>> # Force datetime analysis
    >>> date_strings = pd.Series(['2023-01-01', '2023-01-02', 'invalid'], name='dates')
    >>> result = analyze_column_comprehensive(date_strings, column_type='datetime')
    >>> print(f"Conversion errors: {result['type_specific']['conversion_errors']}")
    >>> 
    >>> # Analyze categorical data
    >>> categories = pd.Series(['A', '  B  ', 'a', 'C'], name='category')
    >>> result = analyze_column_comprehensive(categories)
    >>> print(f"Formatting issues: {result['type_specific']['has_leading_trailing_spaces']}")

    Notes
    -----
    **Analysis Flow:**
    1. **Type Detection**: Automatically determines the most appropriate data type
    2. **Basic Quality**: Checks for NaN, infinite, extreme, and problematic values
    3. **Pattern Analysis**: Detects constants, duplicates, and structural patterns
    4. **Type-Specific Analysis**: Performs specialized analysis based on data type
    5. **Recommendations**: Generates actionable suggestions for data improvement

    **Type-Specific Analyses:**
    - **Numeric**: Distribution analysis, outlier detection, statistical summaries
    - **Datetime**: Temporal validation, date range checks, gap analysis
    - **Categorical**: Frequency analysis, formatting checks, encoding validation

    **Recommendation Categories:**
    - Data cleaning suggestions (handle missing values, outliers)
    - Formatting improvements (strip spaces, standardize case)
    - Type conversion recommendations
    - Feature engineering suggestions
    """
    results: Dict[str, Any] = {
        "column_name": series.name or "unnamed",
        "detected_type": None,
        "basic_quality": {},
        "distribution": {},
        "outliers": {},
        "patterns": {},
        "type_specific": {},
        "recommendations": [],
    }

    # Detect or use specified column type
    if expected_type is None:
        expected_type = _detect_column_type(series)
    
    results["detected_type"] = expected_type

    # Perform basic quality assessment (for all types)
    results["basic_quality"] = check_problematic_values(series, expected_type=expected_type, check_zeros=False)

    # Perform pattern analysis (for all types)
    results["patterns"] = analyze_patterns(series)

    # Perform type-specific analysis
    if expected_type == "numeric":
        results["distribution"] = analyze_distribution(series)
        results["outliers"] = analyze_outliers(series)
        
    elif expected_type == "datetime":
        results["type_specific"] = analyze_temporal_data(series)
        # Flag datetime columns with conversion errors as problematic
        if results["type_specific"].get("conversion_errors", 0) > 0:
            results["basic_quality"]["is_clean"] = False
        
    elif expected_type == "categorical":
        results["type_specific"] = analyze_categorical_data(series)

    # Generate recommendations based on findings
    results["recommendations"] = _generate_column_recommendations(results, expected_type, config)

    return results


def _calculate_column_quality_score(
    column_analysis: Dict[str, Any],
    config: QualityAssessmentConfig
) -> float:
    """
    Calculate quality score for a single column based on analysis results.
    
    Parameters
    ----------
    column_analysis : Dict[str, Any]
        Complete analysis results for a column
    config : QualityAssessmentConfig
        Configuration with scoring weights
        
    Returns
    -------
    float
        Quality score between 0.0 and 1.0
    """
    score = 1.0
    basic_quality = column_analysis.get("basic_quality", {})
    
    # Missing values penalty
    missing_pct = (basic_quality.get("nan_count", 0) / basic_quality.get("total_rows", 1)) * 100
    if missing_pct > config.critical_missing_threshold:
        score -= config.missing_weight * 0.8  # Severe penalty
    elif missing_pct > config.high_missing_threshold:
        score -= config.missing_weight * 0.4  # Moderate penalty
    elif missing_pct > 0:
        score -= config.missing_weight * 0.1  # Minor penalty
    
    # Quality issues penalty
    if not basic_quality.get("is_clean", True):
        score -= config.quality_issues_weight * 0.5
    
    # Outliers penalty (for numeric columns)
    outliers = column_analysis.get("outliers", {})
    outlier_pct = outliers.get("outlier_percentage", 0)
    if outlier_pct > config.high_outlier_threshold:
        score -= config.outlier_weight * 0.6
    elif outlier_pct > 5:
        score -= config.outlier_weight * 0.3
    
    # Distribution issues penalty (for numeric columns)
    distribution = column_analysis.get("distribution", {})
    if distribution.get("distribution_type") == "unknown":
        score -= config.distribution_weight * 0.2
    
    return max(0.0, score)


def analyze_categorical_data_enhanced(series: pd.Series, config: QualityAssessmentConfig) -> Dict[str, Any]:
    """
    Enhanced categorical data analysis with configuration support.
    
    Parameters
    ----------
    series : pd.Series
        Categorical data to analyze
    config : QualityAssessmentConfig
        Configuration settings
        
    Returns
    -------
    Dict[str, Any]
        Analysis results for categorical data
    """
    # Use existing implementation but with enhanced configuration
    # This is a placeholder - the full implementation would use config parameters
    # for now, fall back to existing implementation
    try:
        return analyze_categorical_data_original(series)
    except NameError:
        # Simplified implementation if original doesn't exist
        return {
            "unique_count": series.nunique(),
            "most_frequent": dict(series.value_counts().head().items()) if len(series) > 0 else {},
            "has_leading_trailing_spaces": False,
            "has_special_characters": False,
            "has_mixed_case": False,
            "encoding_issues": False,
            "length_statistics": {},
            "summary": "Basic categorical analysis"
        }


def _generate_column_recommendations(
    results: Dict[str, Any],
    expected_type: str,
    config: QualityAssessmentConfig
) -> List[str]:
    """
    Generate actionable recommendations based on column analysis results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Complete column analysis results
    column_type : str
        The detected or specified column type
        
    Returns
    -------
    List[str]
        List of actionable recommendations
    """
    recommendations = []
    
    # Basic quality recommendations
    basic_quality = results.get("basic_quality", {})
    if basic_quality.get("nan_count", 0) > 0:
        nan_percentage = (basic_quality["nan_count"] / basic_quality["total_rows"]) * 100
        if nan_percentage > 50:
            recommendations.append("High missing rate - consider dropping column")
        else:
            recommendations.append("Handle missing values (imputation or removal)")
    
    if basic_quality.get("inf_count", 0) > 0 or basic_quality.get("ninf_count", 0) > 0:
        recommendations.append("Replace infinite values with appropriate substitutes")
    
    if basic_quality.get("mixed_types", False):
        recommendations.append("Standardize data types within column")

    # Type-specific recommendations
    if expected_type == "numeric":
        outliers = results.get("outliers", {})
        outlier_pct = outliers.get("outlier_percentage", 0)
        if outlier_pct > config.high_outlier_threshold:
            recommendations.append(f"High outlier percentage ({outlier_pct:.1f}%) - consider outlier treatment")
        elif outlier_pct > 5:
            recommendations.append("Consider outlier treatment (removal, capping, or transformation)")
        
        distribution = results.get("distribution", {})
        if abs(distribution.get("skewness", 0)) > 1:
            recommendations.append("Consider log transformation for highly skewed data")
    
    elif expected_type == "datetime":
        type_specific = results.get("type_specific", {})
        if type_specific.get("conversion_errors", 0) > 0:
            recommendations.append("Fix datetime parsing errors or standardize date formats")
        if type_specific.get("future_dates", 0) > 0:
            recommendations.append("Review future dates for validity")
        if type_specific.get("very_old_dates", 0) > 0:
            recommendations.append("Validate very old dates for accuracy")
    
    elif expected_type == "categorical":
        type_specific = results.get("type_specific", {})
        if type_specific.get("has_leading_trailing_spaces", False):
            recommendations.append("Strip leading/trailing spaces from categories")
        if type_specific.get("has_mixed_case", False):
            recommendations.append("Standardize case formatting (e.g., title case or lowercase)")
        if type_specific.get("encoding_issues", False):
            recommendations.append("Address encoding issues for non-ASCII characters")

    return recommendations


def analyze_dataframe_comprehensive(
    df: pd.DataFrame,
    column_types: Optional[Dict[str, str]] = None,
    correlation_threshold: float = 0.95,
    config: Optional[QualityAssessmentConfig] = None,
    progress_tracker: Optional[ProgressTracker] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive quality analysis on an entire DataFrame.

    This function provides a complete data quality assessment for a DataFrame,
    analyzing each column individually and examining relationships between columns.
    It's designed to give data scientists and analysts a thorough understanding
    of their data quality before beginning analysis or model building.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze comprehensively.
    column_types : Dict[str, str], optional
        Dictionary mapping column names to types ('numeric', 'datetime', 'categorical').
        If None, types will be automatically detected for each column.
        Example: {'sales': 'numeric', 'date': 'datetime', 'category': 'categorical'}
    correlation_threshold : float, default 0.95
        Threshold for flagging highly correlated numeric columns. Correlations
        above this threshold may indicate redundant features or data leakage.
    config : QualityAssessmentConfig, optional
        Configuration object with analysis parameters. If None, uses DEFAULT_CONFIG.
    progress_tracker : ProgressTracker, optional
        Progress tracker for reporting analysis progress.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comprehensive DataFrame analysis:
        - shape: Dimensions of the DataFrame (rows, columns)
        - memory_usage_mb: Total memory usage in megabytes
        - column_analyses: Individual analysis results for each column
        - correlations: Analysis of highly correlated numeric column pairs
        - duplicate_rows: Information about duplicate rows in the dataset
        - overall_quality_score: Aggregate quality score (0.0 to 1.0)
        - priority_issues: Columns requiring immediate attention
        - summary: Executive summary of data quality assessment

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample DataFrame with quality issues
    >>> df = pd.DataFrame({
    ...     'id': range(1000),
    ...     'sales': np.random.normal(100, 15, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000),
    ...     'date': pd.date_range('2023-01-01', periods=1000, freq='D'),
    ...     'outlier_col': np.concatenate([np.random.normal(50, 5, 990), [1000] * 10])
    ... })
    >>> 
    >>> # Add some quality issues
    >>> df.loc[10:20, 'sales'] = np.nan
    >>> df.loc[100, 'outlier_col'] = np.inf
    >>> 
    >>> # Perform comprehensive analysis with custom config
    >>> config = QualityAssessmentConfig(enable_progress_tracking=True)
    >>> analysis = analyze_dataframe_comprehensive(df, config=config)
    >>> print(f"Overall Quality Score: {analysis['overall_quality_score']:.2f}")
    >>> print(f"Priority Issues: {len(analysis['priority_issues'])} columns")
    >>> 
    >>> # Review specific column issues
    >>> for issue in analysis['priority_issues']:
    ...     print(f"Column '{issue['column']}': {', '.join(issue['issues'])}")

    Notes
    -----
    **Analysis Components:**
    
    1. **Individual Column Analysis**: Each column receives comprehensive analysis
       including type detection, quality assessment, and type-specific checks
    
    2. **Cross-Column Analysis**: Examines relationships between columns including
       correlations and potential duplicate columns
    
    3. **Dataset-Level Metrics**: Overall statistics like memory usage, duplicate
       rows, and aggregate quality scores
    
    4. **Prioritized Issues**: Identifies the most critical data quality problems
       that should be addressed first
    
    **Quality Score Calculation:**
    The overall quality score is computed as the average of individual column
    quality scores, where each column score is based on the absence of:
    - Missing values, infinite values, mixed types
    - Extreme outliers, encoding issues
    - Parsing errors, logical inconsistencies
    
    **Priority Issues Criteria:**
    Columns are flagged as priority issues if they have:
    - >20% missing values (configurable via config.high_missing_threshold)
    - >10% outliers (configurable via config.high_outlier_threshold)  
    - Significant data quality problems
    - Mixed data types or parsing errors
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG
    
    # Use default progress tracker if none provided
    if progress_tracker is None:
        progress_tracker = ProgressTracker(enabled=False)
    
    results = {
        "shape": df.shape,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "column_analyses": {},
        "correlations": {},
        "duplicate_rows": {"count": 0, "percentage": 0.0, "indices": []},
        "overall_quality_score": 0.0,
        "priority_issues": [],
        "summary": "",
    }

    # Analyze each column individually
    column_types = column_types or {}
    quality_scores = []

    for col in df.columns:
        col_type = column_types.get(col)
        results["column_analyses"][col] = analyze_column_comprehensive(
            df[col], expected_type=col_type, config=config
        )
        
        # Update progress
        progress_tracker.update(1, "Analyzing columns")
        
        # Calculate quality score for column
        col_quality_score = _calculate_column_quality_score(results["column_analyses"][col], config)
        quality_scores.append(col_quality_score)

    # Analyze duplicate rows
    results["duplicate_rows"] = _analyze_duplicate_rows(df)

    # Perform correlation analysis on numeric columns
    results["correlations"] = _analyze_correlations(df, results["column_analyses"], correlation_threshold)

    # Calculate overall quality score
    results["overall_quality_score"] = np.mean(quality_scores) if quality_scores else 0.0

    # Identify priority issues
    results["priority_issues"] = _identify_priority_issues(results["column_analyses"])

    # Generate executive summary
    results["summary"] = _generate_dataframe_summary(results)

    return results


def _analyze_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze duplicate rows in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for duplicates
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with duplicate row analysis
    """
    duplicate_mask = df.duplicated()
    duplicate_count = duplicate_mask.sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0.0
    
    # Get sample of duplicate indices (limit to first 100 for performance)
    duplicate_indices = df[duplicate_mask].index.tolist()[:100]
    
    return {
        "count": duplicate_count,
        "percentage": duplicate_percentage,
        "indices": duplicate_indices,
    }


def _analyze_correlations(
    df: pd.DataFrame, 
    column_analyses: Dict[str, Any], 
    correlation_threshold: float
) -> Dict[str, Any]:
    """
    Analyze correlations between numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    column_analyses : Dict[str, Any]
        Individual column analysis results
    correlation_threshold : float
        Threshold for flagging high correlations
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with correlation analysis results
    """
    # Identify numeric columns
    numeric_cols = [
        col for col, analysis in column_analyses.items()
        if analysis["detected_type"] == "numeric"
    ]

    if len(numeric_cols) <= 1:
        return {"high_correlation_pairs": [], "note": "Insufficient numeric columns for correlation analysis"}

    try:
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= correlation_threshold:
                    high_corr_pairs.append({
                        "col1": numeric_cols[i],
                        "col2": numeric_cols[j],
                        "correlation": float(corr_value),
                    })

        return {"high_correlation_pairs": high_corr_pairs}
    
    except Exception as e:
        return {"error": str(e), "high_correlation_pairs": []}


def _identify_priority_issues(column_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify columns that require priority attention based on analysis results.
    
    Parameters
    ----------
    column_analyses : Dict[str, Any]
        Analysis results for all columns
        
    Returns
    -------
    List[Dict[str, Any]]
        List of priority issues found
    """
    priority_issues = []

    for col, analysis in column_analyses.items():
        col_issues = []
        basic_quality = analysis.get("basic_quality", {})

        # Check for high missing rate
        if basic_quality.get("nan_count", 0) > 0:
            total_rows = basic_quality.get("total_rows", 1)
            nan_rate = (basic_quality["nan_count"] / total_rows) * 100
            if nan_rate > 20:
                col_issues.append(f"High missing rate ({nan_rate:.1f}%)")

        # Check for many outliers
        outliers = analysis.get("outliers", {})
        if outliers.get("outlier_percentage", 0) > 10:
            col_issues.append(f"Many outliers ({outliers['outlier_percentage']:.1f}%)")

        # Check for general data quality issues
        if not basic_quality.get("is_clean", True):
            col_issues.append("Data quality issues detected")

        # Check for type-specific issues
        type_specific = analysis.get("type_specific", {})
        if analysis.get("detected_type") == "datetime":
            if type_specific.get("conversion_errors", 0) > 0:
                col_issues.append("Datetime parsing errors")
        elif analysis.get("detected_type") == "categorical":
            if type_specific.get("encoding_issues", False):
                col_issues.append("Text encoding issues")

        # Add to priority list if issues found
        if col_issues:
            priority_issues.append({
                "column": col,
                "issues": col_issues,
                "recommendations": analysis.get("recommendations", []),
            })

    return priority_issues


def _generate_dataframe_summary(results: Dict[str, Any]) -> str:
    """
    Generate executive summary of DataFrame quality analysis.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Complete DataFrame analysis results
        
    Returns
    -------
    str
        Executive summary string
    """
    shape = results["shape"]
    quality_score = results["overall_quality_score"]
    memory_mb = results["memory_usage_mb"]
    duplicate_count = results["duplicate_rows"]["count"]
    priority_count = len(results["priority_issues"])
    
    summary_lines = [
        f"DataFrame Quality Score: {quality_score:.2f}/1.0",
        f"Shape: {shape[0]:,} rows × {shape[1]} columns",
        f"Memory Usage: {memory_mb:.2f} MB",
        f"Duplicate Rows: {duplicate_count:,}",
        f"Priority Issues: {priority_count} columns need attention"
    ]
    
    # Add correlation info if available
    high_corr_pairs = results.get("correlations", {}).get("high_correlation_pairs", [])
    if high_corr_pairs:
        summary_lines.append(f"High Correlations: {len(high_corr_pairs)} pairs detected")
    
    return "\n".join(summary_lines)


def generate_quality_report(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    json_output_file: Optional[str] = None,
    column_types: Optional[Dict[str, str]] = None,
    correlation_threshold: float = 0.95,
    silent: Optional[bool] = None,
    config: Optional[QualityAssessmentConfig] = None,
    enable_progress: Optional[bool] = None
) -> Optional[Tuple[Optional[str], Dict[str, Any]]]:
    """
    Generate a comprehensive, human-readable data quality report for a DataFrame.

    This function creates a detailed report that can be used by data scientists,
    analysts, and stakeholders to understand data quality issues and plan
    appropriate data cleaning and preprocessing steps.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze and report on.
    output_file : str, optional
        If provided, save the text report to this file path.
    json_output_file : str, optional
        If provided, save the JSON report to this file path.
    column_types : Dict[str, str], optional
        Dictionary mapping column names to types ('numeric', 'datetime', 'categorical').
        If None, types will be automatically detected.
    correlation_threshold : float, default 0.95
        Threshold for reporting correlations.
    silent : bool, optional
        If True, suppress console output. If None (default), automatically
        set to True when output_file or json_output_file is provided, False otherwise.

    Returns
    -------
    Optional[Tuple[Optional[str], Dict[str, Any]]]
        Text report (None if silent=True and files written) and JSON data,
        or None if silent=True and output files specified

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data with various quality issues
    >>> df = pd.DataFrame({
    ...     'id': range(100),
    ...     'sales': np.random.normal(1000, 200, 100),
    ...     'category': np.random.choice(['A', 'B', 'C'], 100),
    ...     'date': pd.date_range('2023-01-01', periods=100)
    ... })
    >>> 
    >>> # Add some quality issues
    >>> df.loc[10:15, 'sales'] = np.nan
    >>> df.loc[50, 'sales'] = np.inf
    >>> 
    >>> # Generate and print report
    >>> report = generate_quality_report(df)
    >>> if report and report[0]:
    ...     print(report[0])
    >>> 
    >>> # Save report to files
    >>> generate_quality_report(df, output_file='quality_report.txt', 
    ...                        json_output_file='quality_report.json')

    Notes
    -----
    **Report Sections:**
    
    1. **Executive Summary**: Overview of data quality score, dimensions, and key metrics
    2. **Column-by-Column Analysis**: Detailed analysis of each column including:
       - Data type detection and validation
       - Missing values and problematic data
       - Distribution characteristics (for numeric data)
       - Outlier detection and quantification
       - Type-specific quality checks
       - Actionable recommendations
    3. **Cross-Column Analysis**: Relationships between columns including correlations
    4. **Priority Issues**: Most critical problems requiring immediate attention
    5. **Recommendations Summary**: Consolidated list of suggested actions

    **Use Cases:**
    - **Data Exploration**: Initial assessment of new datasets
    - **Quality Monitoring**: Regular checks of data pipelines
    - **Documentation**: Formal quality assessments for data governance
    - **Preprocessing Planning**: Identify necessary cleaning steps
    - **Stakeholder Communication**: Share data quality status with non-technical teams
    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG
    
    # Auto-determine silent mode if not explicitly set
    if silent is None:
        silent = bool(output_file or json_output_file)
    
    # Auto-determine progress tracking if not explicitly set
    if enable_progress is None:
        enable_progress = config.enable_progress_tracking and not silent
    
    # Initialize progress tracker
    progress = ProgressTracker(enabled=enable_progress, total=len(df.columns))
    
    # Check if chunked processing should be used
    memory_mb = estimate_memory_usage(df)
    use_chunking = should_use_chunked_processing(df, config)
    
    if enable_progress:
        logger.info(f"Starting quality assessment for DataFrame: {df.shape} "
                    f"({memory_mb:.1f} MB)")
        if use_chunking:
            chunk_size = config.get_effective_chunk_size(df)
            logger.info(f"Using chunked processing with chunk size: {chunk_size}")
    
    # Perform comprehensive analysis
    analysis = analyze_dataframe_comprehensive(
        df,
        column_types=column_types,
        correlation_threshold=correlation_threshold,
        config=config,
        progress_tracker=progress
    )

    # Build report sections
    report_sections = []
    
    # Header
    report_sections.extend([
        "=" * 80,
        "DATA QUALITY ASSESSMENT REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ])

    # Executive Summary
    report_sections.extend([
        "EXECUTIVE SUMMARY",
        "-" * 40,
        analysis["summary"],
        "",
    ])

    # Column-by-column analysis
    report_sections.extend([
        "COLUMN-BY-COLUMN ANALYSIS",
        "-" * 40,
    ])

    for col, col_analysis in analysis["column_analyses"].items():
        report_sections.extend(_format_column_section(col, col_analysis))

    # Cross-column analysis
    if analysis["correlations"].get("high_correlation_pairs"):
        report_sections.extend(_format_correlation_section(analysis["correlations"]))

    # Priority issues
    if analysis["priority_issues"]:
        report_sections.extend(_format_priority_issues_section(analysis["priority_issues"]))

    # Recommendations summary
    all_recommendations = _collect_all_recommendations(analysis["column_analyses"])
    if all_recommendations:
        report_sections.extend(_format_recommendations_section(all_recommendations))

    # Footer
    report_sections.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])

    # Combine all sections
    text_report = "\n".join(report_sections)
    
    # Prepare JSON data
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'basic_stats': {
            'shape': analysis['shape'],
            'memory_usage_mb': analysis['memory_usage_mb'],
            'duplicate_rows': analysis['duplicate_rows']
        },
        'column_analyses': analysis['column_analyses'],
        'correlations': analysis['correlations'],
        'priority_issues': analysis['priority_issues'],
        'overall_quality_score': analysis['overall_quality_score'],
        'summary': analysis['summary']
    }

    # Save files if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        logger.info(f"Text report saved to {output_file}")
    
    if json_output_file:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            # Convert numpy types before JSON serialization
            json_serializable_data = convert_numpy_types(json_data)
            json.dump(json_serializable_data, f, indent=2, default=str)
        logger.info(f"JSON report saved to {json_output_file}")
    
    # Return results based on silent mode
    if silent and (output_file or json_output_file):
        return None
    elif silent:
        return None, convert_numpy_types(json_data)
    else:
        return text_report, convert_numpy_types(json_data)


def _format_column_section(col: str, col_analysis: Dict[str, Any]) -> List[str]:
    """
    Format the analysis section for a single column.
    
    Parameters
    ----------
    col : str
        Column name
    col_analysis : Dict[str, Any]
        Analysis results for the column
        
    Returns
    -------
    List[str]
        List of formatted report lines
    """
    lines = [
        "",
        f"Column: {col}",
        f"Type: {col_analysis['detected_type']}",
        f"Quality: {'CLEAN' if col_analysis['basic_quality']['is_clean'] else 'ISSUES FOUND'}",
    ]
    
    # Basic quality summary
    basic_summary = col_analysis['basic_quality']['summary']
    if basic_summary != "No problematic values detected":
        lines.append(f"Issues: {basic_summary}")
    
    # Type-specific insights
    detected_type = col_analysis["detected_type"]
    
    if detected_type == "numeric":
        distribution = col_analysis.get("distribution", {})
        if distribution:
            lines.append(f"  Distribution: {distribution.get('distribution_type', 'unknown')}")
            if distribution.get("skewness") is not None:
                lines.append(f"  Skewness: {distribution['skewness']:.3f}")
        
        outliers = col_analysis.get("outliers", {})
        if outliers.get("outlier_count", 0) > 0:
            lines.append(f"  Outliers: {outliers['outlier_count']} ({outliers['outlier_percentage']:.1f}%)")
    
    elif detected_type == "datetime":
        type_specific = col_analysis.get("type_specific", {})
        if type_specific.get("conversion_errors", 0) > 0:
            lines.append(f"  Conversion Errors: {type_specific['conversion_errors']}")
        if type_specific.get("future_dates", 0) > 0:
            lines.append(f"  Future Dates: {type_specific['future_dates']}")
    
    elif detected_type == "categorical":
        type_specific = col_analysis.get("type_specific", {})
        if type_specific.get("unique_count", 0) > 0:
            lines.append(f"  Unique Categories: {type_specific['unique_count']}")
    
    # Recommendations
    recommendations = col_analysis.get("recommendations", [])
    if recommendations:
        lines.append(f"Recommendations: {', '.join(recommendations)}")
    
    return lines


def _format_correlation_section(correlations: Dict[str, Any]) -> List[str]:
    """
    Format the correlation analysis section.
    
    Parameters
    ----------
    correlations : Dict[str, Any]
        Correlation analysis results
        
    Returns
    -------
    List[str]
        List of formatted report lines
    """
    lines = [
        "",
        "HIGH CORRELATIONS",
        "-" * 40,
    ]
    
    for pair in correlations["high_correlation_pairs"]:
        lines.append(f"{pair['col1']} <-> {pair['col2']}: {pair['correlation']:.3f}")
    
    return lines


def _format_priority_issues_section(priority_issues: List[Dict[str, Any]]) -> List[str]:
    """
    Format the priority issues section.
    
    Parameters
    ----------
    priority_issues : List[Dict[str, Any]]
        List of priority issues
        
    Returns
    -------
    List[str]
        List of formatted report lines
    """
    lines = [
        "",
        "PRIORITY ISSUES",
        "-" * 40,
    ]
    
    for issue in priority_issues:
        lines.extend([
            f"\n{issue['column']}:",
            f"  Issues: {', '.join(issue['issues'])}",
            f"  Actions: {', '.join(issue['recommendations'])}",
        ])
    
    return lines


def _collect_all_recommendations(column_analyses: Dict[str, Any]) -> List[str]:
    """
    Collect all unique recommendations from column analyses.
    
    Parameters
    ----------
    column_analyses : Dict[str, Any]
        All column analysis results
        
    Returns
    -------
    List[str]
        List of unique recommendations
    """
    all_recommendations = set()
    
    for col_analysis in column_analyses.values():
        recommendations = col_analysis.get("recommendations", [])
        all_recommendations.update(recommendations)
    
    return sorted(list(all_recommendations))


def _format_recommendations_section(recommendations: List[str]) -> List[str]:
    """
    Format the recommendations summary section.
    
    Parameters
    ----------
    recommendations : List[str]
        List of unique recommendations
        
    Returns
    -------
    List[str]
        List of formatted report lines
    """
    lines = [
        "",
        "RECOMMENDATIONS SUMMARY",
        "-" * 40,
    ]
    
    for i, recommendation in enumerate(recommendations, 1):
        lines.append(f"{i}. {recommendation}")
    
    return lines


# Example usage and demonstration
if __name__ == "__main__":
    """
    Demonstration of the data quality assessment framework.
    
    This example shows how to use the framework with realistic data that contains
    various quality issues commonly found in real-world datasets.
    """
    print("Data Quality Assessment Framework - Demonstration")
    print("=" * 60)
    
    # Create sample data with realistic quality issues
    np.random.seed(42)
    n_rows = 1000

    sample_data = pd.DataFrame({
        # Clean numeric column
        "revenue": np.random.normal(50000, 15000, n_rows),
        
        # Numeric column with outliers and missing values
        "profit_margin": np.concatenate([
            np.random.normal(0.15, 0.05, n_rows - 50),
            np.random.normal(0.80, 0.10, 30),  # outliers
            [np.nan] * 20  # missing values
        ]),
        
        # Categorical column with formatting issues
        "department": np.random.choice([
            "Sales", "  Marketing  ", "sales", "ENGINEERING", 
            "Support", "marketing", "Engineering"
        ], n_rows),
        
        # Date column with some issues
        "hire_date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
        
        # High-cardinality identifier
        "employee_id": [f"EMP_{i:04d}" for i in range(n_rows)],
        
        # Constant column (data quality issue)
        "company_code": ["ACME"] * n_rows,
        
        # Mixed data types
        "performance_rating": (["Excellent"] * 300 + ["Good"] * 400 + 
                             ["Average"] * 200 + [85, 90, 95] * 33 + [np.nan]),
    })

    # Introduce additional quality issues
    sample_data.loc[10:15, "revenue"] = np.inf  # infinite values
    sample_data.loc[100:110, "hire_date"] = pd.Timestamp("2030-12-31")  # future dates
    sample_data.loc[500, "profit_margin"] = -2.5  # impossible profit margin

    print(f"\nGenerated sample dataset with {len(sample_data)} rows and {len(sample_data.columns)} columns")
    print("Dataset intentionally contains various quality issues for demonstration.\n")

    # Demonstrate individual column analysis
    print("Example: Analyzing individual column (profit_margin)")
    print("-" * 50)
    
    profit_analysis = analyze_column_comprehensive(sample_data["profit_margin"])
    print(f"Detected Type: {profit_analysis['detected_type']}")
    print(f"Quality Status: {'CLEAN' if profit_analysis['basic_quality']['is_clean'] else 'ISSUES FOUND'}")
    print(f"Issues: {profit_analysis['basic_quality']['summary']}")
    print(f"Outliers: {profit_analysis['outliers']['outlier_count']} ({profit_analysis['outliers']['outlier_percentage']:.1f}%)")
    print(f"Recommendations: {', '.join(profit_analysis['recommendations'])}")

    # Demonstrate full DataFrame analysis and report generation
    print("\n" + "=" * 60)
    print("Generating comprehensive data quality report...")
    print("=" * 60)

    report = generate_quality_report(sample_data)
    print(report)

    # Demonstrate data leakage detection
    print("\n" + "=" * 60)
    print("Demonstrating data leakage detection...")
    print("=" * 60)

    # Create a target variable for demonstration
    sample_data['target'] = np.random.choice([0, 1], n_rows)
    
    leakage_results = detect_data_leakage_patterns(
        sample_data, 
        target_column='target',
        id_columns=['employee_id']
    )
    
    print(f"Leakage Analysis Summary: {leakage_results['summary']}")
    if leakage_results['recommendations']:
        print("Recommendations:")
        for rec in leakage_results['recommendations']:
            print(f"  - {rec}")

    print("\nDemonstration completed!")
    print("The framework is ready for use with your own datasets.")


# CLI interface functions for integration
def create_data_quality_cli():
    """Create CLI interface for data quality assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Data Quality Assessment"
    )
    parser.add_argument(
        "input_file",
        help="Path to the CSV file containing data to analyze"
    )
    parser.add_argument(
        "--output",
        help="Path to save the text report"
    )
    parser.add_argument(
        "--json-output",
        help="Path to save the JSON report"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for reporting (default: 0.95)"
    )
    parser.add_argument(
        "--column-types",
        help="JSON string mapping column names to types (numeric, datetime, categorical)"
    )
    
    return parser


if __name__ == "__main__":
    # Example usage
    parser = create_data_quality_cli()
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.input_file)
        
        # Parse column types if provided
        column_types = None
        if args.column_types:
            column_types = json.loads(args.column_types)
        
        result = generate_quality_report(
            df,
            output_file=args.output,
            json_output_file=args.json_output,
            column_types=column_types,
            correlation_threshold=args.correlation_threshold
        )
        
        if result and result[0]:  # Only print if result exists and text_report is not None
            print(result[0])
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise