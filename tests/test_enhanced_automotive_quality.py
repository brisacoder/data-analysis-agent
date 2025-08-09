#!/usr/bin/env python3
"""
Test suite for enhanced automotive data quality assessment with signal dictionary integration.

This test suite validates the improvements made to the automotive_data_quality.py module:
1. Signal dictionary integration for better signal identification
2. Enhanced correlation analysis using signal domains
3. Improved reporting with signal metadata
4. Better range validation with signal context
"""

import json
import pandas as pd
import pytest
from pathlib import Path

from data_analysis_agent.automotive_data_quality import (
    generate_automotive_quality_report,
    get_signal_info_from_dictionary,
    detect_automotive_signal_type_enhanced,
    analyze_correlations_automotive
)


class TestEnhancedAutomotiveQuality:
    """Test suite for enhanced automotive data quality assessment."""

    @pytest.fixture
    def sample_automotive_data(self):
        """Load sample automotive data for testing."""
        # Try to load the real automotive data first
        data_path = Path(__file__).parent.parent / "data" / "tables" / "kona-obd-signals-0605.csv"
        if data_path.exists():
            return pd.read_csv(data_path)
        else:
            # Create minimal test data if real data doesn't exist
            return pd.DataFrame({
                'Signal Event Time': pd.date_range('2025-01-01', periods=100, freq='s'),
                'Trip_ID': ['TRIP_001'] * 100,
                'Trip_Color': ['blue'] * 100,
                'ENG_EngSpdVal': [800 + i * 10 for i in range(100)],
                'MDPS_EstStrAnglVal': [-90 + i * 1.8 for i in range(100)],
                'SAS_AnglVal': [-90 + i * 1.8 for i in range(100)],
                'TPMS_FLTirePrsrVal': [220 + (i % 20) for i in range(100)],
                'CLU_OdoVal': [1000 + i for i in range(100)],
                'WHL_SpdFLVal': [0 + i * 0.5 for i in range(100)],
                'OBD_EngRpmVal': [800 + i * 10 for i in range(100)],
                'Unknown_Signal': [i for i in range(100)]
            })

    def test_signal_dictionary_integration(self, sample_automotive_data):
        """Test that signal dictionary is properly integrated."""
        df = sample_automotive_data
        
        # Generate enhanced report
        text_report, json_data = generate_automotive_quality_report(
            df,
            correlation_threshold=0.95,
            include_all_correlations=False
        )
        
        # Verify signal dictionary is included
        assert 'signal_dictionary' in json_data
        signal_dict = json_data['signal_dictionary']
        
        # Should have entries for all columns
        assert len(signal_dict) == len(df.columns)
        
        # Verify signal dictionary structure
        for signal_name, signal_info in signal_dict.items():
            assert 'expanded' in signal_info
            assert 'description' in signal_info
            assert 'units' in signal_info
            assert 'domain' in signal_info
            
        # Test specific signal mappings
        if 'ENG_EngSpdVal' in signal_dict:
            eng_speed = signal_dict['ENG_EngSpdVal']
            assert 'Engine Speed' in eng_speed['expanded']
            assert eng_speed['units'] == 'rpm'
            assert 'Powertrain/Engine' in eng_speed['domain']

    def test_enhanced_signal_detection(self):
        """Test the enhanced signal type detection with dictionary info."""
        # Test with a known engine speed signal
        test_data = pd.Series([800, 1000, 1500, 2000, 2500])
        signal_type, signal_info = detect_automotive_signal_type_enhanced('ENG_EngSpdVal', test_data)
        
        assert signal_type == 'RPM'
        assert signal_info['expanded'] == 'Engine Speed'
        assert signal_info['units'] == 'rpm'
        assert signal_info['domain'] == 'Powertrain/Engine'
        
        # Test with steering angle
        signal_type, signal_info = detect_automotive_signal_type_enhanced('MDPS_EstStrAnglVal', test_data)
        assert signal_type == 'STEERING_ANGLE'
        assert 'Steering' in signal_info['expanded']
        assert signal_info['units'] == 'deg'

    def test_enhanced_correlation_analysis(self, sample_automotive_data):
        """Test enhanced correlation analysis with signal dictionary."""
        df = sample_automotive_data
        
        # Build signal dictionary
        column_names = list(df.columns)
        header_str = ",".join(column_names)
        from data_analysis_agent.build_signal_dictionary import build_signal_dictionary
        signal_dict = build_signal_dictionary(header_str)
        
        # Test correlation analysis
        corr_results = analyze_correlations_automotive(df, threshold=0.8, signal_dict=signal_dict)
        
        assert 'high_correlations_count' in corr_results
        assert 'expected_correlations' in corr_results
        assert 'unexpected_correlations' in corr_results
        assert 'analysis' in corr_results
        
        # Verify enhanced correlation info includes signal dictionary data
        all_correlations = corr_results['expected_correlations'] + corr_results['unexpected_correlations']
        for corr in all_correlations:
            assert 'column1_info' in corr
            assert 'column2_info' in corr
            if corr['column1_info']:
                assert 'domain' in corr['column1_info']
                assert 'expanded' in corr['column1_info']

    def test_enhanced_range_validation(self, sample_automotive_data):
        """Test enhanced range validation with signal context."""
        df = sample_automotive_data
        
        text_report, json_data = generate_automotive_quality_report(df)
        
        # Check that signal quality includes enhanced information
        signal_quality = json_data['signal_quality']
        
        for signal_name, quality_info in signal_quality.items():
            if signal_name not in ['Signal Event Time', 'Trip_ID', 'Trip_Color']:
                # Should have signal_info from dictionary
                assert 'signal_info' in quality_info
                signal_info = quality_info['signal_info']
                assert 'expanded' in signal_info
                assert 'domain' in signal_info
                
                # If range validation was performed, should include signal_info
                if 'range_validation' in quality_info:
                    range_val = quality_info['range_validation']
                    assert 'signal_info' in range_val
                    assert 'summary' in range_val

    def test_domain_classification(self, sample_automotive_data):
        """Test that signals are properly classified into domains."""
        df = sample_automotive_data
        
        text_report, json_data = generate_automotive_quality_report(df)
        signal_dict = json_data['signal_dictionary']
        
        # Count domains
        domain_counts = {}
        for signal_info in signal_dict.values():
            domain = signal_info.get('domain', 'Unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Should have multiple domains
        assert len(domain_counts) > 1
        
        # Should have expected automotive domains
        expected_domains = [
            'Powertrain/Engine', 'Steering', 'Chassis/Tires',
            'Cluster', 'Trip/Position'
        ]
        
        found_domains = set(domain_counts.keys())
        automotive_domains_found = sum(1 for domain in expected_domains if domain in found_domains)
        assert automotive_domains_found >= 3  # Should find at least 3 automotive domains

    def test_signal_info_function(self):
        """Test the get_signal_info_from_dictionary function."""
        # Test with known signals
        test_signals = [
            'ENG_EngSpdVal',
            'MDPS_EstStrAnglVal',
            'TPMS_FLTirePrsrVal',
            'Unknown_Custom_Signal'
        ]
        
        for signal_name in test_signals:
            signal_info = get_signal_info_from_dictionary(signal_name)
            
            assert isinstance(signal_info, dict)
            assert 'expanded' in signal_info
            assert 'description' in signal_info
            assert 'units' in signal_info
            assert 'domain' in signal_info
            
            # Known signals should have meaningful expansions
            if signal_name == 'ENG_EngSpdVal':
                assert 'Engine Speed' in signal_info['expanded']
                assert signal_info['units'] == 'rpm'

    def test_report_structure_enhancements(self, sample_automotive_data, tmp_path):
        """Test that enhanced reports have the expected structure."""
        df = sample_automotive_data
        
        # Generate report with output files
        output_txt = tmp_path / "test_report.txt"
        output_json = tmp_path / "test_report.json"
        
        text_report, json_data = generate_automotive_quality_report(
            df,
            output_file=str(output_txt),
            json_output_file=str(output_json),
            silent=False  # Force output to test the return values
        )
        
        # Verify enhanced JSON structure
        expected_keys = [
            'timestamp', 'basic_stats', 'signal_dictionary', 'signal_quality',
            'correlations', 'cross_signal', 'priority_issues', 'recommendations', 'overall_score'
        ]
        for key in expected_keys:
            assert key in json_data
        
        # Verify files were created
        assert output_txt.exists()
        assert output_json.exists()
        
        # Verify text report contains signal dictionary section
        assert "SIGNAL DICTIONARY SUMMARY" in text_report
        assert "Signals by domain:" in text_report
        assert "Example signal mappings:" in text_report
        
        # Verify JSON file is valid and contains expected structure
        with open(output_json, 'r') as f:
            loaded_json = json.load(f)
            # Check that the loaded JSON has the same keys as the original
            assert set(loaded_json.keys()) == set(json_data.keys())
            # Verify specific important fields are preserved
            assert loaded_json['timestamp'] == json_data['timestamp']
            assert len(loaded_json['signal_quality']) == len(json_data['signal_quality'])
            assert 'signal_dictionary' in loaded_json

    def test_automotive_specific_insights(self, sample_automotive_data):
        """Test that automotive-specific insights are generated."""
        df = sample_automotive_data
        
        text_report, json_data = generate_automotive_quality_report(df)
        
        # Should identify automotive signals
        automotive_signals = 0
        for signal_name, quality_info in json_data['signal_quality'].items():
            range_val = quality_info.get('range_validation', {})
            if range_val.get('signal_type'):
                automotive_signals += 1
        
        assert automotive_signals > 0
        
        # Should have correlation analysis
        correlations = json_data['correlations']
        assert isinstance(correlations['expected_correlations'], list)
        assert isinstance(correlations['unexpected_correlations'], list)
        
        # Should have cross-signal checks
        assert 'cross_signal' in json_data
