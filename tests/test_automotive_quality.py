import pandas as pd
import tempfile
import os

from data_analysis_agent.automotive_data_quality import (
    detect_automotive_signal_type,
    generate_automotive_quality_report,
)


def test_detect_signal_types_basic():
    assert detect_automotive_signal_type('OBD_EngRpmVal', pd.Series([0, 1])) == 'RPM'
    assert detect_automotive_signal_type('OBD_VehSpdSnsrVal', pd.Series([0, 1])) == 'SPEED'
    assert detect_automotive_signal_type('WHL_SpdFLVal', pd.Series([0, 1])) == 'WHEEL_SPEED'
    # HV battery voltage detection
    assert detect_automotive_signal_type('EMSV_KeyBattVolt', pd.Series([120.0])) == 'HV_BATTERY_VOLTAGE'


def test_generate_report_cross_signal_and_temporal():
    # Build a small dataframe
    df = pd.DataFrame({
        'Signal Event Time': pd.date_range('2025-01-01', periods=5, freq='s'),
        'OBD_EngRpmVal': [0, 1000, 2000, 3000, 4000],
        'OBD_VehSpdSnsrVal': [0, 10, 20, 30, 40],
        'WHL_SpdFLVal': [0, 12, 18, 32, 38],
        'ENG_EngClntTempVal': [20, 21, 22, 23, 24],
        'ENG_MilSta': [0, 0, 1, 1, 0],
    })

    text, json_data = generate_automotive_quality_report(df, correlation_threshold=0.9)
    # Basic JSON structure
    assert 'basic_stats' in json_data
    assert json_data['basic_stats']['shape'][0] == 5
    assert 'signal_quality' in json_data and len(json_data['signal_quality']) >= 5
    # Cross-signal checks present
    assert 'cross_signal' in json_data
    assert 'wheel_vs_vehicle_speed' in json_data['cross_signal']
    # Temporal consistency computed for speed or rpm
    rpm_analysis = json_data['signal_quality']['OBD_EngRpmVal'].get('temporal_consistency')
    assert rpm_analysis is None or rpm_analysis.get('has_consistency_check') in (True, False)
    # Flag validation recognized
    fl = json_data['signal_quality']['ENG_MilSta'].get('flag_validation')
    assert fl is None or fl.get('is_flag_like') in (True, False)


def test_silent_mode_auto_detection():
    """Test automatic silent mode when output files are specified."""
    df = pd.DataFrame({
        'OBD_EngRpmVal': [800, 1000, 1200],
        'OBD_VehSpdSnsrVal': [0, 10, 20],
    })
    
    # Test with JSON output file (should auto-activate silent mode)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name
    try:
        result = generate_automotive_quality_report(df, json_output_file=json_file)
        assert result is None, "Should return None in auto-silent mode with JSON output"
        assert os.path.exists(json_file), "JSON file should be created"
    finally:
        if os.path.exists(json_file):
            os.unlink(json_file)
    
    # Test with text output file (should auto-activate silent mode)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        text_file = f.name
    try:
        result = generate_automotive_quality_report(df, output_file=text_file)
        assert result is None, "Should return None in auto-silent mode with text output"
        assert os.path.exists(text_file), "Text file should be created"
    finally:
        if os.path.exists(text_file):
            os.unlink(text_file)


def test_silent_mode_explicit():
    """Test explicit silent mode control."""
    df = pd.DataFrame({
        'OBD_EngRpmVal': [800, 1000, 1200],
        'OBD_VehSpdSnsrVal': [0, 10, 20],
    })
    
    # Test explicit silent=True
    result = generate_automotive_quality_report(df, silent=True)
    assert isinstance(result, tuple), "Should return tuple in explicit silent mode"
    assert result[0] is None, "Text report should be None in silent mode"
    assert isinstance(result[1], dict), "JSON data should be available"
    
    # Test explicit silent=False (normal mode)
    result = generate_automotive_quality_report(df, silent=False)
    assert isinstance(result, tuple), "Should return tuple in normal mode"
    assert isinstance(result[0], str), "Text report should be string"
    assert isinstance(result[1], dict), "JSON data should be available"


def test_silent_mode_override():
    """Test silent mode override behavior."""
    df = pd.DataFrame({
        'OBD_EngRpmVal': [800, 1000, 1200],
        'OBD_VehSpdSnsrVal': [0, 10, 20],
    })
    
    # Test silent=False override with JSON output (should force output)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name
    try:
        result = generate_automotive_quality_report(
            df,
            json_output_file=json_file,
            silent=False
        )
        assert isinstance(result, tuple), "Should return tuple when silent=False override"
        assert isinstance(result[0], str), "Text report should be available"
        assert isinstance(result[1], dict), "JSON data should be available"
        assert os.path.exists(json_file), "JSON file should still be created"
    finally:
        if os.path.exists(json_file):
            os.unlink(json_file)


def test_normal_mode_behavior():
    """Test normal mode returns expected tuple structure."""
    df = pd.DataFrame({
        'OBD_EngRpmVal': [800, 1000, 1200],
        'OBD_VehSpdSnsrVal': [0, 10, 20],
    })
    
    # Test normal mode (no files, no silent parameter)
    result = generate_automotive_quality_report(df)
    assert isinstance(result, tuple), "Should return tuple in normal mode"
    assert isinstance(result[0], str), "Text report should be string"
    assert isinstance(result[1], dict), "JSON data should be dict"
    assert len(result[0]) > 0, "Text report should not be empty"
    assert 'basic_stats' in result[1], "JSON should contain basic_stats"
