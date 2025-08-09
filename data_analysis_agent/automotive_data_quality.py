"""
Automotive Telemetry Data Quality Assessment Framework

This module provides specialized data quality assessment for automotive telemetry data.
It understands the unique characteristics of car telemetry such as conditional signals,
expected correlations, and domain-specific validation rules.

Key Features:
- Automotive-specific signal validation (RPM, speed, temperature ranges)
- Conditional signal analysis (signals only active under certain conditions)
- Temporal consistency checks for physically impossible changes
- CAN bus signal quality assessment
- JSON and text output formats
- Configurable correlation reporting thresholds

Author: GitHub Copilot
License: MIT
Version: 1.0.0
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Automotive signal ranges and validation rules
AUTOMOTIVE_SIGNAL_RANGES = {
    # Engine signals
    'RPM': {'min': 0, 'max': 8000, 'unit': 'rpm'},
    'ENGINE_SPEED': {'min': 0, 'max': 8000, 'unit': 'rpm'},
    'THROTTLE': {'min': 0, 'max': 100, 'unit': '%'},
    'THROTTLE_POSITION': {'min': 0, 'max': 100, 'unit': '%'},
    
    # Vehicle dynamics
    'SPEED': {'min': 0, 'max': 300, 'unit': 'km/h'},
    'VEHICLE_SPEED': {'min': 0, 'max': 300, 'unit': 'km/h'},
    'ACCELERATION': {'min': -20, 'max': 20, 'unit': 'm/s²'},
    
    # Temperatures (typical automotive ranges)
    'ENGINE_TEMP': {'min': -40, 'max': 150, 'unit': '°C'},
    'COOLANT_TEMP': {'min': -40, 'max': 150, 'unit': '°C'},
    'OIL_TEMP': {'min': -40, 'max': 200, 'unit': '°C'},
    'INTAKE_TEMP': {'min': -40, 'max': 100, 'unit': '°C'},
    'AMBIENT_TEMP': {'min': -50, 'max': 60, 'unit': '°C'},
    
    # Pressures
    'OIL_PRESSURE': {'min': 0, 'max': 10, 'unit': 'bar'},
    'FUEL_PRESSURE': {'min': 0, 'max': 10, 'unit': 'bar'},
    'BOOST_PRESSURE': {'min': -1, 'max': 3, 'unit': 'bar'},
    
    # Electrical
    'BATTERY_VOLTAGE': {'min': 9, 'max': 16, 'unit': 'V'},
    'ALTERNATOR_VOLTAGE': {'min': 12, 'max': 15, 'unit': 'V'},
    
    # Fuel
    'FUEL_LEVEL': {'min': 0, 'max': 100, 'unit': '%'},
    'FUEL_FLOW': {'min': 0, 'max': 50, 'unit': 'L/h'},
    
    # GPS/Location
    'LATITUDE': {'min': -90, 'max': 90, 'unit': 'degrees'},
    'LONGITUDE': {'min': -180, 'max': 180, 'unit': 'degrees'},
    'ALTITUDE': {'min': -500, 'max': 9000, 'unit': 'm'},
}

# Signals that are expected to be mostly zero (only active under specific conditions)
CONDITIONAL_SIGNALS = {
    'BRAKE_PRESSURE', 'BRAKE_PEDAL', 'CLUTCH_PEDAL', 'HANDBRAKE',
    'TURN_SIGNAL_LEFT', 'TURN_SIGNAL_RIGHT', 'HAZARD_LIGHTS',
    'WIPER', 'HORN', 'REVERSE_GEAR', 'PARKING_BRAKE',
    'ABS_ACTIVE', 'ESP_ACTIVE', 'TRACTION_CONTROL',
    'CRUISE_CONTROL', 'ADAPTIVE_CRUISE', 'LANE_ASSIST',
}

# Expected high correlations (these are normal in automotive data)
EXPECTED_CORRELATIONS = {
    ('RPM', 'ENGINE_SPEED'): 'Same signal, different names',
    ('SPEED', 'VEHICLE_SPEED'): 'Same signal, different names',
    ('RPM', 'VEHICLE_SPEED'): 'Engine speed typically correlates with vehicle speed',
    ('THROTTLE', 'ENGINE_LOAD'): 'Throttle position affects engine load',
    ('FUEL_FLOW', 'ENGINE_LOAD'): 'Higher load requires more fuel',
    ('ENGINE_TEMP', 'COOLANT_TEMP'): 'Engine and coolant temperatures are related',
}


class AutomotiveDataQualityResults:
    """Container for automotive data quality assessment results."""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.basic_stats = {}
        self.signal_quality = {}
        self.range_violations = {}
        self.temporal_consistency = {}
        self.conditional_signals = {}
        self.correlations = {}
        self.priority_issues = []
        self.recommendations = []
        self.overall_score = 0.0


def detect_automotive_signal_type(column_name: str, data: pd.Series) -> Optional[str]:
    """
    Detect if a column represents a known automotive signal type.
    
    Parameters
    ----------
    column_name : str
        Name of the column to analyze
    data : pd.Series
        The data series
        
    Returns
    -------
    Optional[str]
        Detected signal type or None if not recognized
    """
    column_upper = column_name.upper()
    
    # Check for exact matches first
    for signal_type in AUTOMOTIVE_SIGNAL_RANGES:
        if signal_type in column_upper:
            return signal_type
    
    # Check for partial matches and common variations
    if any(term in column_upper for term in ['RPM', 'ENGINE_SPEED', 'ENG_SPEED']):
        return 'RPM'
    elif any(term in column_upper for term in ['SPEED', 'VELOCITY', 'VEL']):
        return 'SPEED'
    elif any(term in column_upper for term in ['THROTTLE', 'TPS', 'ACCEL_PEDAL']):
        return 'THROTTLE'
    elif any(term in column_upper for term in ['TEMP', 'TEMPERATURE']):
        if any(term in column_upper for term in ['ENGINE', 'COOLANT', 'WATER']):
            return 'ENGINE_TEMP'
        elif any(term in column_upper for term in ['OIL']):
            return 'OIL_TEMP'
        elif any(term in column_upper for term in ['INTAKE', 'AIR']):
            return 'INTAKE_TEMP'
        elif any(term in column_upper for term in ['AMBIENT', 'OUTSIDE', 'EXTERNAL']):
            return 'AMBIENT_TEMP'
    elif any(term in column_upper for term in ['PRESSURE', 'PRESS']):
        if any(term in column_upper for term in ['OIL']):
            return 'OIL_PRESSURE'
        elif any(term in column_upper for term in ['FUEL']):
            return 'FUEL_PRESSURE'
        elif any(term in column_upper for term in ['BOOST', 'TURBO', 'MANIFOLD']):
            return 'BOOST_PRESSURE'
    elif any(term in column_upper for term in ['VOLTAGE', 'VOLT']):
        if any(term in column_upper for term in ['BATTERY', 'BATT']):
            return 'BATTERY_VOLTAGE'
        elif any(term in column_upper for term in ['ALTERNATOR', 'ALT']):
            return 'ALTERNATOR_VOLTAGE'
    elif any(term in column_upper for term in ['FUEL_LEVEL', 'FUEL_TANK', 'TANK_LEVEL']):
        return 'FUEL_LEVEL'
    elif any(term in column_upper for term in ['LATITUDE', 'LAT']):
        return 'LATITUDE'
    elif any(term in column_upper for term in ['LONGITUDE', 'LON', 'LNG']):
        return 'LONGITUDE'
    
    return None


def check_signal_range_violations(column_name: str, data: pd.Series) -> Dict[str, Any]:
    """
    Check for values outside expected automotive signal ranges.
    
    Parameters
    ----------
    column_name : str
        Name of the signal column
    data : pd.Series
        The signal data
        
    Returns
    -------
    Dict[str, Any]
        Range violation analysis results
    """
    signal_type = detect_automotive_signal_type(column_name, data)
    
    if signal_type is None or signal_type not in AUTOMOTIVE_SIGNAL_RANGES:
        return {
            'signal_type': None,
            'has_violations': False,
            'violation_count': 0,
            'violation_percentage': 0.0,
            'violations': [],
            'expected_range': None,
            'actual_range': {'min': data.min(), 'max': data.max()},
            'summary': 'Signal type not recognized or no validation rules available'
        }
    
    expected_range = AUTOMOTIVE_SIGNAL_RANGES[signal_type]
    min_val, max_val = expected_range['min'], expected_range['max']
    
    # Find violations
    violations = data[(data < min_val) | (data > max_val)].dropna()
    violation_count = len(violations)
    violation_percentage = (violation_count / len(data.dropna())) * 100 if len(data.dropna()) > 0 else 0
    
    return {
        'signal_type': signal_type,
        'has_violations': violation_count > 0,
        'violation_count': violation_count,
        'violation_percentage': round(violation_percentage, 2),
        'violations': violations.tolist()[:10],  # Limit to first 10 for brevity
        'expected_range': expected_range,
        'actual_range': {'min': float(data.min()), 'max': float(data.max())},
        'summary': (
            f"Signal identified as {signal_type}. {violation_count} values "
            f"({violation_percentage:.1f}%) outside expected range "
            f"[{min_val}, {max_val}] {expected_range['unit']}"
        )
    }


def analyze_conditional_signal(column_name: str, data: pd.Series) -> Dict[str, Any]:
    """
    Analyze signals that are expected to be mostly zero (conditional signals).
    
    Parameters
    ----------
    column_name : str
        Name of the signal column
    data : pd.Series
        The signal data
        
    Returns
    -------
    Dict[str, Any]
        Conditional signal analysis results
    """
    column_upper = column_name.upper()
    is_conditional = any(cond_signal in column_upper for cond_signal in CONDITIONAL_SIGNALS)
    
    if not is_conditional:
        return {
            'is_conditional_signal': False,
            'zero_percentage': 0.0,
            'analysis': 'Not identified as a conditional signal'
        }
    
    # Calculate zero percentage
    non_null_data = data.dropna()
    if len(non_null_data) == 0:
        return {
            'is_conditional_signal': True,
            'zero_percentage': 100.0,
            'analysis': 'All values are null - this may be normal for conditional signals'
        }
    
    zero_count = (non_null_data == 0).sum()
    zero_percentage = (zero_count / len(non_null_data)) * 100
    
    analysis = f"Conditional signal with {zero_percentage:.1f}% zero values. "
    if zero_percentage > 80:
        analysis += "High zero percentage is normal for conditional signals."
    elif zero_percentage < 50:
        analysis += "Lower than expected zero percentage - verify signal behavior."
    else:
        analysis += "Moderate activity level for conditional signal."
    
    return {
        'is_conditional_signal': True,
        'zero_percentage': round(zero_percentage, 2),
        'active_percentage': round(100 - zero_percentage, 2),
        'active_count': len(non_null_data) - zero_count,
        'analysis': analysis
    }


def check_temporal_consistency(
    data: pd.Series,
    signal_type: str,
    max_change_per_second: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Check for physically impossible temporal changes in signals.
    
    Parameters
    ----------
    data : pd.Series
        The signal data (should have datetime index)
    signal_type : str
        Type of signal for validation rules
    max_change_per_second : Dict[str, float], optional
        Maximum allowed change per second for different signal types
        
    Returns
    -------
    Dict[str, Any]
        Temporal consistency analysis results
    """
    if max_change_per_second is None:
        max_change_per_second = {
            'SPEED': 10,  # km/h per second (aggressive acceleration/braking)
            'RPM': 1000,  # RPM per second
            'ENGINE_TEMP': 5,  # °C per second
            'BATTERY_VOLTAGE': 2,  # V per second
        }
    
    if signal_type not in max_change_per_second:
        return {
            'has_consistency_check': False,
            'analysis': f'No temporal consistency rules defined for {signal_type}'
        }
    
    if not isinstance(data.index, pd.DatetimeIndex):
        return {
            'has_consistency_check': False,
            'analysis': 'Data does not have datetime index for temporal analysis'
        }
    
    # Calculate time differences and value changes
    time_diffs = data.index.to_series().diff().dt.total_seconds()
    value_diffs = data.diff().abs()
    
    # Calculate rate of change per second
    rates = value_diffs / time_diffs
    max_allowed = max_change_per_second[signal_type]
    
    # Find violations
    violations = rates[rates > max_allowed].dropna()
    violation_count = len(violations)
    
    return {
        'has_consistency_check': True,
        'signal_type': signal_type,
        'max_allowed_change_per_sec': max_allowed,
        'violation_count': violation_count,
        'max_observed_rate': float(rates.max()) if not rates.empty else 0,
        'violations_summary': f"{violation_count} instances of rate changes exceeding {max_allowed} units/second",
        'analysis': f"Temporal consistency check for {signal_type}: {'PASS' if violation_count == 0 else 'FAIL'}"
    }


def analyze_correlations_automotive(df: pd.DataFrame, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Analyze correlations with automotive context.
    
    Parameters
    ----------
    df : pd.DataFrame
        The telemetry data
    threshold : float
        Correlation threshold for reporting
        
    Returns
    -------
    Dict[str, Any]
        Correlation analysis results with automotive context
    """
    # Calculate correlations for numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {
            'high_correlations_count': 0,
            'expected_correlations': [],
            'unexpected_correlations': [],
            'analysis': 'Insufficient numeric columns for correlation analysis'
        }
    
    corr_matrix = numeric_df.corr()
    
    # Find high correlations (excluding self-correlations)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                high_corr_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': round(corr_val, 3)
                })
    
    # Classify correlations as expected or unexpected
    expected_correlations = []
    unexpected_correlations = []
    
    for pair in high_corr_pairs:
        col1, col2 = pair['column1'].upper(), pair['column2'].upper()
        is_expected = False
        explanation = ""
        
        # Check if this is an expected correlation
        for (exp_col1, exp_col2), exp_explanation in EXPECTED_CORRELATIONS.items():
            if (exp_col1 in col1 or exp_col1 in col2) and (exp_col2 in col1 or exp_col2 in col2):
                is_expected = True
                explanation = exp_explanation
                break
        
        pair_with_context = {**pair, 'explanation': explanation}
        
        if is_expected:
            expected_correlations.append(pair_with_context)
        else:
            unexpected_correlations.append(pair_with_context)
    
    return {
        'high_correlations_count': len(high_corr_pairs),
        'expected_correlations': expected_correlations,
        'unexpected_correlations': unexpected_correlations,
        'analysis': (
            f"Found {len(high_corr_pairs)} high correlations (>{threshold}). "
            f"{len(expected_correlations)} expected, {len(unexpected_correlations)} unexpected."
        )
    }


def generate_automotive_quality_report(
    df: pd.DataFrame,
    output_file: Optional[str] = None,
    json_output_file: Optional[str] = None,
    correlation_threshold: float = 0.95,
    include_all_correlations: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a comprehensive automotive telemetry data quality report.
    
    Parameters
    ----------
    df : pd.DataFrame
        The automotive telemetry data to analyze
    output_file : str, optional
        Path to save the text report
    json_output_file : str, optional
        Path to save the JSON report
    correlation_threshold : float, default 0.95
        Threshold for reporting correlations
    include_all_correlations : bool, default False
        Whether to include all correlations or just unexpected ones
        
    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Text report and JSON data
    """
    results = AutomotiveDataQualityResults()
    
    # Basic statistics
    results.basic_stats = {
        'shape': df.shape,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'non_numeric_columns': len(df.select_dtypes(exclude=[np.number]).columns),
        'total_missing_values': int(df.isnull().sum().sum()),
        'missing_percentage': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    }
    
    # Analyze each column
    for column in df.columns:
        if column.upper() in ['SIGNAL EVENT TIME', 'TRIP_ID', 'TRIP_COLOR']:
            continue  # Skip metadata columns
            
        data = df[column]
        
        # Basic column stats
        col_analysis = {
            'data_type': str(data.dtype),
            'missing_count': int(data.isnull().sum()),
            'missing_percentage': round((data.isnull().sum() / len(data)) * 100, 2),
            'unique_count': int(data.nunique()),
            'zero_count': int((data == 0).sum()) if pd.api.types.is_numeric_dtype(data) else 0
        }
        
        if pd.api.types.is_numeric_dtype(data):
            # Range validation
            range_analysis = check_signal_range_violations(column, data)
            col_analysis['range_validation'] = range_analysis
            
            # Conditional signal analysis
            conditional_analysis = analyze_conditional_signal(column, data)
            col_analysis['conditional_signal'] = conditional_analysis
            
            # Basic numeric stats
            col_analysis.update({
                'min': float(data.min()) if not data.empty else None,
                'max': float(data.max()) if not data.empty else None,
                'mean': float(data.mean()) if not data.empty else None,
                'std': float(data.std()) if not data.empty else None
            })
        
        results.signal_quality[column] = col_analysis
    
    # Correlation analysis
    correlation_analysis = analyze_correlations_automotive(df, correlation_threshold)
    results.correlations = correlation_analysis
    
    # Generate priority issues
    priority_issues = []
    for column, analysis in results.signal_quality.items():
        if analysis.get('missing_percentage', 0) > 50:
            priority_issues.append(f"Column '{column}': {analysis['missing_percentage']:.1f}% missing values")
        
        range_validation = analysis.get('range_validation', {})
        if (range_validation.get('has_violations', False) and
                range_validation.get('violation_percentage', 0) > 5):
            violation_pct = range_validation['violation_percentage']
            priority_issues.append(
                f"Column '{column}': {violation_pct:.1f}% values outside expected range"
            )
    
    if len(correlation_analysis.get('unexpected_correlations', [])) > 10:
        priority_issues.append(f"High number of unexpected correlations: {len(correlation_analysis['unexpected_correlations'])}")
    
    results.priority_issues = priority_issues
    
    # Generate recommendations
    recommendations = []
    if results.basic_stats['missing_percentage'] > 20:
        recommendations.append("Consider investigating high missing value rates across the dataset")
    
    for column, analysis in results.signal_quality.items():
        conditional = analysis.get('conditional_signal', {})
        if conditional.get('is_conditional_signal', False) and conditional.get('zero_percentage', 0) < 50:
            recommendations.append(f"Verify signal '{column}' behavior - lower than expected zero percentage for conditional signal")
    
    if len(correlation_analysis.get('unexpected_correlations', [])) > 5:
        recommendations.append("Review unexpected high correlations for potential data quality issues or feature redundancy")
    
    results.recommendations = recommendations
    
    # Calculate overall quality score
    score_factors = []
    if results.basic_stats['missing_percentage'] < 10:
        score_factors.append(0.3)
    elif results.basic_stats['missing_percentage'] < 25:
        score_factors.append(0.2)
    else:
        score_factors.append(0.1)
    
    range_violation_score = 0.0
    for analysis in results.signal_quality.values():
        range_val = analysis.get('range_validation', {})
        if range_val.get('signal_type') and not range_val.get('has_violations', False):
            range_violation_score += 1
    
    if results.signal_quality:
        range_violation_score = (range_violation_score / len(results.signal_quality)) * 0.4
    
    correlation_score = 0.3 if len(correlation_analysis.get('unexpected_correlations', [])) < 5 else 0.2
    
    results.overall_score = sum(score_factors) + range_violation_score + correlation_score
    
    # Generate text report
    report_lines = [
        "=" * 80,
        "AUTOMOTIVE TELEMETRY DATA QUALITY ASSESSMENT REPORT",
        "=" * 80,
        f"Generated: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        f"Dataset shape: {results.basic_stats['shape'][0]:,} rows × {results.basic_stats['shape'][1]} columns",
        f"Memory usage: {results.basic_stats['memory_usage_mb']} MB",
        f"Missing values: {results.basic_stats['total_missing_values']:,} ({results.basic_stats['missing_percentage']:.1f}%)",
        f"Overall quality score: {results.overall_score:.2f}/1.0",
        "",
    ]
    
    if results.priority_issues:
        report_lines.extend([
            "PRIORITY ISSUES",
            "-" * 40,
        ])
        for issue in results.priority_issues:
            report_lines.append(f"⚠️  {issue}")
        report_lines.append("")
    
    # Signal quality summary
    report_lines.extend([
        "SIGNAL QUALITY SUMMARY",
        "-" * 40,
    ])
    
    automotive_signals = 0
    range_violations = 0
    conditional_signals = 0
    
    for column, analysis in results.signal_quality.items():
        range_val = analysis.get('range_validation', {})
        if range_val.get('signal_type'):
            automotive_signals += 1
            if range_val.get('has_violations', False):
                range_violations += 1
        
        conditional = analysis.get('conditional_signal', {})
        if conditional.get('is_conditional_signal', False):
            conditional_signals += 1
    
    report_lines.extend([
        f"Automotive signals identified: {automotive_signals}",
        f"Signals with range violations: {range_violations}",
        f"Conditional signals identified: {conditional_signals}",
        "",
    ])
    
    # Correlation summary
    corr_summary = results.correlations
    if include_all_correlations and corr_summary.get('high_correlations_count', 0) > 0:
        report_lines.extend([
            "HIGH CORRELATIONS ANALYSIS",
            "-" * 40,
            f"Total high correlations (>{correlation_threshold}): {corr_summary.get('high_correlations_count', 0)}",
            f"Expected correlations: {len(corr_summary.get('expected_correlations', []))}",
            f"Unexpected correlations: {len(corr_summary.get('unexpected_correlations', []))}",
            "",
        ])
        
        if corr_summary.get('unexpected_correlations'):
            report_lines.append("Unexpected high correlations:")
            for corr in corr_summary['unexpected_correlations'][:10]:  # Limit to first 10
                report_lines.append(f"  {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f}")
            if len(corr_summary['unexpected_correlations']) > 10:
                report_lines.append(f"  ... and {len(corr_summary['unexpected_correlations']) - 10} more")
            report_lines.append("")
    else:
        report_lines.extend([
            "CORRELATION ANALYSIS",
            "-" * 40,
            f"Unexpected high correlations: {len(corr_summary.get('unexpected_correlations', []))}",
            f"(Expected correlations hidden - use include_all_correlations=True to show)",
            "",
        ])
    
    # Recommendations
    if results.recommendations:
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 40,
        ])
        for rec in results.recommendations:
            report_lines.append(f"💡 {rec}")
        report_lines.append("")
    
    report_lines.extend([
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])
    
    text_report = "\n".join(report_lines)
    
    # Prepare JSON data
    json_data = {
        'timestamp': results.timestamp.isoformat(),
        'basic_stats': results.basic_stats,
        'signal_quality': results.signal_quality,
        'correlations': results.correlations,
        'priority_issues': results.priority_issues,
        'recommendations': results.recommendations,
        'overall_score': results.overall_score
    }
    
    # Save files if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        logger.info(f"Text report saved to {output_file}")
    
    if json_output_file:
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        logger.info(f"JSON report saved to {json_output_file}")
    
    return text_report, json_data


# CLI interface functions for integration
def create_automotive_quality_cli():
    """Create CLI interface for automotive quality assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automotive Telemetry Data Quality Assessment"
    )
    parser.add_argument(
        "input_file",
        help="Path to the CSV file containing telemetry data"
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
        "--include-all-correlations",
        action="store_true",
        help="Include all high correlations in the report"
    )
    
    return parser


if __name__ == "__main__":
    # Example usage
    parser = create_automotive_quality_cli()
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.input_file)
        text_report, json_data = generate_automotive_quality_report(
            df,
            output_file=args.output,
            json_output_file=args.json_output,
            correlation_threshold=args.correlation_threshold,
            include_all_correlations=args.include_all_correlations
        )
        print(text_report)
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise
