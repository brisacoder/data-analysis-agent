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
- Signal dictionary integration for enhanced reporting
- JSON and text output formats
- Configurable correlation reporting thresholds

Author: GitHub Copilot
License: MIT
Version: 1.2.0
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import signal dictionary functions
from .build_signal_dictionary import build_signal_dictionary, EXACT_MAP, apply_rules, expand_heuristic, classify_domain

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
    'ACCELERATOR_PEDAL': {'min': 0, 'max': 100, 'unit': '%'},
    'ENGINE_LOAD': {'min': 0, 'max': 100, 'unit': '%'},
    'MAF': {'min': 0, 'max': 400, 'unit': 'g/s'},
    'MAP': {'min': 10, 'max': 300, 'unit': 'kPa'},
    'AFR_LAMBDA': {'min': 0.7, 'max': 1.5, 'unit': 'λ'},
    'O2_VOLTAGE': {'min': 0.0, 'max': 1.2, 'unit': 'V'},
    
    # Vehicle dynamics
    'SPEED': {'min': 0, 'max': 300, 'unit': 'km/h'},
    'VEHICLE_SPEED': {'min': 0, 'max': 300, 'unit': 'km/h'},
    'ACCELERATION': {'min': -20, 'max': 20, 'unit': 'm/s²'},
    'WHEEL_SPEED': {'min': 0, 'max': 300, 'unit': 'km/h'},
    'STEERING_ANGLE': {'min': -900, 'max': 900, 'unit': 'deg'},
    'STEERING_TORQUE': {'min': -20, 'max': 20, 'unit': 'Nm'},
    'STEERING_SPEED': {'min': 0, 'max': 1200, 'unit': 'deg/s'},
    'GEAR': {'min': -1, 'max': 10, 'unit': 'gear'},
    
    # Temperatures (typical automotive ranges)
    'ENGINE_TEMP': {'min': -40, 'max': 150, 'unit': '°C'},
    'COOLANT_TEMP': {'min': -40, 'max': 150, 'unit': '°C'},
    'OIL_TEMP': {'min': -40, 'max': 200, 'unit': '°C'},
    'INTAKE_TEMP': {'min': -40, 'max': 100, 'unit': '°C'},
    'AMBIENT_TEMP': {'min': -50, 'max': 60, 'unit': '°C'},
    'ATF_TEMP': {'min': -40, 'max': 180, 'unit': '°C'},
    'CAT_TEMP': {'min': -40, 'max': 1000, 'unit': '°C'},
    
    # Pressures
    'OIL_PRESSURE': {'min': 0, 'max': 10, 'unit': 'bar'},
    'FUEL_PRESSURE': {'min': 0, 'max': 10, 'unit': 'bar'},
    'FUEL_RAIL_PRESSURE': {'min': 0, 'max': 2500, 'unit': 'kPa'},
    'BOOST_PRESSURE': {'min': -1, 'max': 3, 'unit': 'bar'},
    'TIRE_PRESSURE_KPA': {'min': 100, 'max': 400, 'unit': 'kPa'},
    
    # Electrical
    'BATTERY_VOLTAGE': {'min': 9, 'max': 16, 'unit': 'V'},
    'ALTERNATOR_VOLTAGE': {'min': 12, 'max': 15, 'unit': 'V'},
    'HV_BATTERY_VOLTAGE': {'min': 50, 'max': 800, 'unit': 'V'},
    
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
# Expected correlations based on either substrings or detected types
EXPECTED_CORRELATIONS = {
    ('RPM', 'ENGINE_SPEED'): 'Same signal, different names',
    ('SPEED', 'VEHICLE_SPEED'): 'Same signal, different names',
    ('RPM', 'VEHICLE_SPEED'): 'Engine speed typically correlates with vehicle speed',
    ('THROTTLE', 'ENGINE_LOAD'): 'Throttle position affects engine load',
    ('FUEL_FLOW', 'ENGINE_LOAD'): 'Higher load requires more fuel',
    ('ENGINE_TEMP', 'COOLANT_TEMP'): 'Engine and coolant temperatures are related',
    ('WHL_SPD', 'SPEED'): 'Individual wheel speeds correlate with vehicle speed',
    ('WHL_SPD', 'WHL_SPD'): 'Wheel speeds correlate with each other',
    ('VEH_SPD', 'WHEEL_SPEED'): 'Wheel speeds correlate with vehicle speed',
}

# Expected correlations defined at the signal-type level (order-insensitive)
EXPECTED_CORRELATION_TYPES = {
    frozenset({'RPM', 'ENGINE_SPEED'}): 'Same signal, different names',
    frozenset({'SPEED', 'VEHICLE_SPEED'}): 'Same signal, different names',
    frozenset({'RPM', 'SPEED'}): 'Engine speed typically correlates with vehicle speed',
    frozenset({'THROTTLE', 'ENGINE_LOAD'}): 'Throttle position affects engine load',
    frozenset({'FUEL_FLOW', 'ENGINE_LOAD'}): 'Higher load requires more fuel',
    frozenset({'ENGINE_TEMP', 'COOLANT_TEMP'}): 'Engine and coolant temperatures are related',
    frozenset({'WHEEL_SPEED', 'SPEED'}): 'Wheel speeds correlate with vehicle speed',
}


class AutomotiveDataQualityResults:
    """Container for automotive data quality assessment results."""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.basic_stats = {}
        self.signal_quality = {}
        self.signal_dictionary = {}  # Add signal dictionary information
        self.range_violations = {}
        self.temporal_consistency = {}
        self.conditional_signals = {}
        self.correlations = {}
        self.cross_signal = {}
        self.priority_issues = []
        self.recommendations = []
        self.overall_score = 0.0


def get_signal_info_from_dictionary(signal_name: str) -> Dict[str, str]:
    """
    Get signal information from the signal dictionary.
    
    Parameters
    ----------
    signal_name : str
        Name of the signal
        
    Returns
    -------
    Dict[str, str]
        Signal information with expanded name, description, units, and domain
    """
    # Check exact mappings first
    if signal_name in EXACT_MAP:
        return EXACT_MAP[signal_name]
    
    # Try pattern-based rules
    info = apply_rules(signal_name)
    if info:
        expanded, description, units, domain = info
        return {
            "expanded": expanded,
            "description": description,
            "units": units,
            "domain": domain,
        }
    
    # Fallback to heuristic
    return {
        "expanded": expand_heuristic(signal_name),
        "description": "Signal not in dictionary; description inferred heuristically.",
        "units": "",
        "domain": classify_domain(signal_name),
    }


def detect_automotive_signal_type_enhanced(column_name: str, data: pd.Series) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Enhanced signal type detection using signal dictionary.
    
    Parameters
    ----------
    column_name : str
        Name of the column to analyze
    data : pd.Series
        The data series
        
    Returns
    -------
    Tuple[Optional[str], Dict[str, str]]
        Detected signal type and signal information from dictionary
    """
    # Get signal info from dictionary
    signal_info = get_signal_info_from_dictionary(column_name)
    
    # Use existing detection logic
    signal_type = detect_automotive_signal_type(column_name, data)
    
    return signal_type, signal_info


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
    if any(term in column_upper for term in ['RPM', 'ENGINE_SPEED', 'ENG_SPEED', 'OBD_ENGRPM']):
        return 'RPM'
    elif any(term in column_upper for term in ['SPEED', 'SPD', 'VELOCITY', 'VEL', 'KPH', 'MPH', 'VEH_SPD']):
        return 'SPEED'
    elif any(term in column_upper for term in ['THROTTLE', 'TPS', 'ACCEL_PEDAL']):
        return 'THROTTLE'
    elif any(term in column_upper for term in ['ACCELERATOR_PEDAL', 'ACC_PEDAL', 'APP']):
        return 'ACCELERATOR_PEDAL'
    elif any(term in column_upper for term in ['ENGINE_LOAD', 'LOAD']):
        return 'ENGINE_LOAD'
    elif any(term in column_upper for term in ['TEMP', 'TEMPERATURE']):
        if any(term in column_upper for term in ['ENGINE', 'COOLANT', 'WATER']):
            return 'ENGINE_TEMP'
        elif any(term in column_upper for term in ['OIL']):
            return 'OIL_TEMP'
        elif any(term in column_upper for term in ['INTAKE', 'AIR']):
            return 'INTAKE_TEMP'
        elif any(term in column_upper for term in ['AMBIENT', 'OUTSIDE', 'EXTERNAL']):
            return 'AMBIENT_TEMP'
        elif any(term in column_upper for term in ['CAT', 'CATALYST']):
            return 'CAT_TEMP'
        elif any(term in column_upper for term in ['ATF', 'TRANS', 'GEARBOX']):
            return 'ATF_TEMP'
    elif any(term in column_upper for term in ['PRESSURE', 'PRESS']):
        if any(term in column_upper for term in ['OIL']):
            return 'OIL_PRESSURE'
        elif any(term in column_upper for term in ['FUEL']):
            if 'RAIL' in column_upper:
                return 'FUEL_RAIL_PRESSURE'
            return 'FUEL_PRESSURE'
        elif any(term in column_upper for term in ['BOOST', 'TURBO', 'MANIFOLD']):
            return 'BOOST_PRESSURE'
        elif any(term in column_upper for term in ['TIRE', 'TPMS']):
            return 'TIRE_PRESSURE_KPA'
    elif any(term in column_upper for term in ['VOLTAGE', 'VOLT']):
        if any(term in column_upper for term in ['BATTERY', 'BATT']):
            if any(term in column_upper for term in ['KEYBATT', 'HV', 'PACK']):
                return 'HV_BATTERY_VOLTAGE'
            return 'BATTERY_VOLTAGE'
        elif any(term in column_upper for term in ['ALTERNATOR', 'ALT']):
            return 'ALTERNATOR_VOLTAGE'
    elif any(term in column_upper for term in ['FUEL_LEVEL', 'FUEL_TANK', 'TANK_LEVEL']):
        return 'FUEL_LEVEL'
    elif any(term in column_upper for term in ['LATITUDE', 'LAT']):
        return 'LATITUDE'
    elif any(term in column_upper for term in ['LONGITUDE', 'LON', 'LNG']):
        return 'LONGITUDE'
    elif any(term in column_upper for term in ['WHL_SPD', 'WHEEL_SPEED', 'WHLSPD', 'WHL_SPD']):
        return 'WHEEL_SPEED'
    elif any(term in column_upper for term in ['STEERING_ANGLE', 'SAS_ANGL', 'MDPS_ESTSTRANGL', 'STR_ANGLE']):
        return 'STEERING_ANGLE'
    elif any(term in column_upper for term in ['SAS_SPD', 'STR_SPD', 'STEERING_SPEED']):
        return 'STEERING_SPEED'
    elif any(term in column_upper for term in ['GEAR', 'GEAR_STA', 'CURRGEAR', 'TRGTGEAR']):
        return 'GEAR'
    elif any(term in column_upper for term in ['MAF']):
        return 'MAF'
    elif any(term in column_upper for term in ['MAP', 'MAN_ABS_PRS', 'MANIFOLD_ABS']):
        return 'MAP'
    elif any(term in column_upper for term in ['LAMBDA', 'AFR']):
        return 'AFR_LAMBDA'
    elif any(term in column_upper for term in ['O2S', 'O2_SENSOR', 'OXYGEN_SENSOR']):
        return 'O2_VOLTAGE'
    
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
        Range violation analysis results with signal dictionary information
    """
    signal_type, signal_info = detect_automotive_signal_type_enhanced(column_name, data)
    
    if signal_type is None or signal_type not in AUTOMOTIVE_SIGNAL_RANGES:
        return {
            'signal_type': signal_type,
            'signal_info': signal_info,
            'has_violations': False,
            'violation_count': 0,
            'violation_percentage': 0.0,
            'violations': [],
            'expected_range': None,
            'actual_range': {'min': float(data.min()), 'max': float(data.max())},
            'summary': (
                f"Signal identified as {signal_type or 'unknown'} - no range validation available. "
                f"Dictionary info: {signal_info.get('expanded', 'N/A')}"
            )
        }
    
    expected_range = AUTOMOTIVE_SIGNAL_RANGES[signal_type]
    min_val, max_val = expected_range['min'], expected_range['max']
    
    # Find violations
    violations = data[(data < min_val) | (data > max_val)].dropna()
    violation_count = len(violations)
    violation_percentage = (violation_count / len(data.dropna())) * 100 if len(data.dropna()) > 0 else 0
    
    return {
        'signal_type': signal_type,
        'signal_info': signal_info,
        'has_violations': violation_count > 0,
        'violation_count': violation_count,
        'violation_percentage': round(violation_percentage, 2),
        'violations': violations.tolist()[:10],  # Limit to first 10 violations for output
        'expected_range': expected_range,
        'actual_range': {'min': float(data.min()), 'max': float(data.max())},
        'summary': (
            f"Signal '{signal_info.get('expanded', column_name)}' identified as {signal_type}. "
            f"{violation_count} values ({violation_percentage:.1f}%) outside expected range "
            f"[{min_val}, {max_val}] {expected_range['unit']}. "
            f"Domain: {signal_info.get('domain', 'Unknown')}"
        )
    }


def check_low_variance_and_saturation(data: pd.Series, bins: int = 50) -> Dict[str, Any]:
    """Detect near-constant sensors and saturation at min/max bounds."""
    result = {
        'is_constant': False,
        'constant_value': None,
        'unique_ratio': None,
        'saturation': {
            'min_hits_pct': 0.0,
            'max_hits_pct': 0.0,
            'is_saturated': False,
        },
    }
    non_null = data.dropna()
    if non_null.empty:
        return result
    unique_count = non_null.nunique()
    result['unique_ratio'] = round(unique_count / len(non_null), 4)
    if unique_count == 1:
        result['is_constant'] = True
        result['constant_value'] = float(non_null.iloc[0])
        return result
    # Saturation detection: percentage at min or max values
    dmin, dmax = float(non_null.min()), float(non_null.max())
    min_hits = (non_null == dmin).sum()
    max_hits = (non_null == dmax).sum()
    min_pct = 100 * min_hits / len(non_null)
    max_pct = 100 * max_hits / len(non_null)
    result['saturation'] = {
        'min_hits_pct': round(min_pct, 2),
        'max_hits_pct': round(max_pct, 2),
        'is_saturated': (min_pct > 30.0) or (max_pct > 30.0),
    }
    return result


def validate_binary_like_signal(column_name: str, data: pd.Series) -> Dict[str, Any]:
    """Validate status/flag signals expected to be binary or small-enum.

    Heuristic: if the column name contains STA/STATUS/SW/IND/LMP/LAMP or ends with _STA,
    and unique numeric values <= 5, report the set of observed states.
    """
    name = column_name.upper()
    if not pd.api.types.is_numeric_dtype(data):
        return {'is_flag_like': False}
    if not any(key in name for key in ['_STA', 'STATUS', 'SW', 'SWSTA', 'IND', 'LMP', 'LAMP', 'WRNG', 'WARN']):
        return {'is_flag_like': False}
    non_null = data.dropna()
    if non_null.empty:
        return {'is_flag_like': True, 'states': [], 'analysis': 'All values null'}
    states = sorted(pd.unique(non_null))
    is_binary = set(states).issubset({0, 1})
    small_enum = len(states) <= 5
    analysis = 'binary' if is_binary else ('small-enum' if small_enum else 'many-states')
    issues = []
    if not small_enum:
        issues.append('Too many distinct states for a status-like signal')
    return {
        'is_flag_like': True,
        'state_count': int(len(states)),
        'states': [int(s) if isinstance(s, (int, np.integer)) else float(s) for s in states[:10]],
        'analysis': analysis,
        'issues': issues,
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


def cross_signal_plausibility_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Run simple cross-signal plausibility checks.

    - Wheel speeds vs vehicle speed
    - Speed > 0 while RPM == 0 (impossible)
    """
    results: Dict[str, Any] = {
        'wheel_vs_vehicle_speed': [],
        'speed_positive_rpm_zero_count': 0,
    }
    # Identify columns
    cols = list(df.columns)
    speed_cols = [c for c in cols if detect_automotive_signal_type(c, df[c]) in ('SPEED', 'VEHICLE_SPEED')]
    rpm_cols = [c for c in cols if detect_automotive_signal_type(c, df[c]) in ('RPM', 'ENGINE_SPEED')]
    wheel_cols = [c for c in cols if detect_automotive_signal_type(c, df[c]) == 'WHEEL_SPEED']

    # Wheel vs vehicle speed
    if speed_cols and wheel_cols:
        vcol = speed_cols[0]
        v = pd.to_numeric(df[vcol], errors='coerce')
        for wcol in wheel_cols:
            w = pd.to_numeric(df[wcol], errors='coerce')
            both = pd.concat([v, w], axis=1).dropna()
            if both.empty:
                continue
            diff = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
            tol = 20.0  # km/h absolute tolerance
            bad = (diff > tol).sum()
            bad_pct = 100 * bad / len(both)
            results['wheel_vs_vehicle_speed'].append({
                'vehicle_speed_col': vcol,
                'wheel_speed_col': wcol,
                'mismatch_pct': round(bad_pct, 2),
                'tolerance_kph': tol,
            })

    # Speed > 0 with RPM == 0
    if speed_cols and rpm_cols:
        s = pd.to_numeric(df[speed_cols[0]], errors='coerce')
        r = pd.to_numeric(df[rpm_cols[0]], errors='coerce')
        mask = (s > 1) & (r < 1)
        results['speed_positive_rpm_zero_count'] = int(mask.sum())

    return results


def analyze_correlations_automotive(
    df: pd.DataFrame,
    threshold: float = 0.95,
    signal_dict: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Analyze correlations with automotive context and signal dictionary.
    
    Parameters
    ----------
    df : pd.DataFrame
        The telemetry data
    threshold : float
        Correlation threshold for reporting
    signal_dict : Optional[Dict[str, Dict[str, str]]]
        Signal dictionary for enhanced analysis
        
    Returns
    -------
    Dict[str, Any]
        Correlation analysis results with automotive context and signal dictionary info
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
                
                # Get signal dictionary info if available
                col1_info = signal_dict.get(col1, {}) if signal_dict else {}
                col2_info = signal_dict.get(col2, {}) if signal_dict else {}
                
                high_corr_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': round(corr_val, 3),
                    'column1_info': col1_info,
                    'column2_info': col2_info
                })
    
    # Classify correlations as expected or unexpected
    expected_correlations = []
    unexpected_correlations = []
    
    for pair in high_corr_pairs:
        col1_name, col2_name = pair['column1'], pair['column2']
        col1, col2 = col1_name.upper(), col2_name.upper()
        is_expected = False
        explanation = ""

        # 1) Check if both signals are from the same domain (likely correlated)
        if signal_dict:
            domain1 = pair['column1_info'].get('domain', '')
            domain2 = pair['column2_info'].get('domain', '')
            if domain1 and domain2 and domain1 == domain2:
                if domain1 in ['Driveline/AWD', 'Chassis/Tires', 'Steering']:
                    is_expected = True
                    explanation = f"Both signals from {domain1} domain"

        # 2) Prefer detection via signal types
        type1 = detect_automotive_signal_type(col1_name, df[col1_name])
        type2 = detect_automotive_signal_type(col2_name, df[col2_name])
        if type1 and type2 and not is_expected:
            type_pair = frozenset({type1, type2})
            if type_pair in EXPECTED_CORRELATION_TYPES:
                is_expected = True
                explanation = EXPECTED_CORRELATION_TYPES[type_pair]

        # 3) Fallback to substring heuristics
        if not is_expected:
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
    
    # Build signal dictionary for all columns
    column_names = [col for col in df.columns]
    header_str = ",".join(column_names)
    signal_dict = build_signal_dictionary(header_str)
    results.signal_dictionary = signal_dict

    # Optionally set datetime index for temporal checks
    time_col_candidates = [c for c in df.columns if c.upper() in ['SIGNAL EVENT TIME', 'TIMESTAMP', 'TIME']]
    dt_index = None
    if time_col_candidates:
        try:
            tcol = time_col_candidates[0]
            dt_index = pd.to_datetime(df[tcol], errors='coerce')
        except Exception:
            dt_index = None

    # Analyze each column
    for column in df.columns:
        if column.upper() in ['SIGNAL EVENT TIME', 'TRIP_ID', 'TRIP_COLOR']:
            continue  # Skip metadata columns
            
        data = df[column]
        
        # Get signal dictionary information
        signal_info = signal_dict.get(column, {})
        
        # Basic column stats
        col_analysis = {
            'data_type': str(data.dtype),
            'signal_info': signal_info,  # Add signal dictionary info
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
            
            # Low variance and saturation
            col_analysis['stability'] = check_low_variance_and_saturation(data)
            
            # Status/flag validation
            flag_validation = validate_binary_like_signal(column, data)
            if flag_validation.get('is_flag_like'):
                col_analysis['flag_validation'] = flag_validation
            
            # Basic numeric stats
            col_analysis.update({
                'min': float(data.min()) if not data.empty else None,
                'max': float(data.max()) if not data.empty else None,
                'mean': float(data.mean()) if not data.empty else None,
                'std': float(data.std()) if not data.empty else None
            })

            # Temporal consistency for selected signals when time available
            if dt_index is not None:
                sig_type = range_analysis.get('signal_type')
                if sig_type in {'SPEED', 'RPM', 'ENGINE_TEMP', 'BATTERY_VOLTAGE'}:
                    ts = pd.Series(data.values, index=dt_index)
                    col_analysis['temporal_consistency'] = check_temporal_consistency(ts, sig_type)
        
        results.signal_quality[column] = col_analysis
    
    # Correlation analysis
    correlation_analysis = analyze_correlations_automotive(df, correlation_threshold, results.signal_dictionary)
    results.correlations = correlation_analysis

    # Cross-signal plausibility
    results.cross_signal = cross_signal_plausibility_checks(df)
    
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
        unexpected_count = len(correlation_analysis['unexpected_correlations'])
        priority_issues.append(f"High number of unexpected correlations: {unexpected_count}")
    
    results.priority_issues = priority_issues
    
    # Generate recommendations
    recommendations = []
    if results.basic_stats['missing_percentage'] > 20:
        recommendations.append("Consider investigating high missing value rates across the dataset")
    
    for column, analysis in results.signal_quality.items():
        conditional = analysis.get('conditional_signal', {})
        is_conditional = conditional.get('is_conditional_signal', False)
        low_zero_pct = conditional.get('zero_percentage', 0) < 50
        if is_conditional and low_zero_pct:
            recommendations.append(
                f"Verify signal '{column}' behavior - lower than expected zero percentage for conditional signal"
            )
    
    if len(correlation_analysis.get('unexpected_correlations', [])) > 5:
        recommendations.append(
            "Review unexpected high correlations for potential data quality issues or feature redundancy"
        )
    
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
    missing_total = results.basic_stats['total_missing_values']
    missing_pct = results.basic_stats['missing_percentage']
    
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
        f"Missing values: {missing_total:,} ({missing_pct:.1f}%)",
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
    
    # Include a brief stability and cross-signal summary
    constant_cols = [c for c, a in results.signal_quality.items() if a.get('stability', {}).get('is_constant')]
    
    def is_saturated(analysis):
        return analysis.get('stability', {}).get('saturation', {}).get('is_saturated')
    
    saturated_cols = [c for c, a in results.signal_quality.items() if is_saturated(a)]

    report_lines.extend([
        f"Automotive signals identified: {automotive_signals}",
        f"Signals with range violations: {range_violations}",
        f"Conditional signals identified: {conditional_signals}",
        f"Near-constant sensors: {len(constant_cols)}",
        f"Saturated sensors (min/max clipping): {len(saturated_cols)}",
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
            "(Expected correlations hidden - use include_all_correlations=True to show)",
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

    # Cross-signal checks in report (compact)
    if results.cross_signal.get('wheel_vs_vehicle_speed'):
        report_lines.extend([
            "PLAUSIBILITY CHECKS",
            "-" * 40,
        ])
        for item in results.cross_signal['wheel_vs_vehicle_speed'][:4]:
            wheel_col = item['wheel_speed_col']
            vehicle_col = item['vehicle_speed_col']
            mismatch_pct = item['mismatch_pct']
            tolerance = item['tolerance_kph']
            report_lines.append(
                f"Wheel vs vehicle speed mismatch: {wheel_col} vs {vehicle_col} -> "
                f"{mismatch_pct}% (> {tolerance} kph)"
            )
        spd_rpm = results.cross_signal.get('speed_positive_rpm_zero_count', 0)
        if spd_rpm:
            report_lines.append(f"Speed>0 with RPM≈0 occurrences: {spd_rpm}")
        report_lines.append("")
    
    # Add signal dictionary summary to report
    if results.signal_dictionary:
        report_lines.extend([
            "SIGNAL DICTIONARY SUMMARY",
            "-" * 40,
        ])
        
        domain_counts: Dict[str, int] = {}
        for signal_info in results.signal_dictionary.values():
            domain = signal_info.get('domain', 'Unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        report_lines.append("Signals by domain:")
        for domain, count in sorted(domain_counts.items()):
            report_lines.append(f"  {domain}: {count} signals")
        
        # Show some example signal mappings
        report_lines.extend([
            "",
            "Example signal mappings:",
        ])
        
        shown_count = 0
        for signal_name, signal_info in results.signal_dictionary.items():
            if shown_count >= 5:
                break
            if signal_info.get('expanded') != signal_name:  # Only show interesting mappings
                expanded = signal_info.get('expanded', 'N/A')
                units = signal_info.get('units', '')
                unit_str = f" ({units})" if units else ""
                report_lines.append(f"  {signal_name} -> {expanded}{unit_str}")
                shown_count += 1
        
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
        'signal_dictionary': results.signal_dictionary,  # Add signal dictionary
        'signal_quality': results.signal_quality,
        'correlations': results.correlations,
        'cross_signal': results.cross_signal,
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
