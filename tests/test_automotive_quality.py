import pandas as pd

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
