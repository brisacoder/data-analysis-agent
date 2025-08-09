"""
Tests for the Automotive Data Quality API.

This module provides comprehensive test coverage for the automotive_data_quality_api module,
including all methods, error handling, and edge cases.
"""

import json
import os
import tempfile
import pytest
import pandas as pd

from data_analysis_agent.automotive_data_quality_api import (
    AutomotiveDataQualityAPI,
    load_quality_report,
    get_column_quality_summary
)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return pd.DataFrame({
        'speedAvg': [50.0, 60.0, 70.0, 80.0, None],
        'soc': [85, 80, 75, 70, 65],
        'soh': [98.5, 98.5, 98.5, 98.5, 98.5],
        'longitude': [-2.5, -2.4, -2.3, -2.2, -2.1],
        'latitude': [43.2, 43.21, 43.22, 43.23, 43.24],
        'wind_mph': [10.0, 12.0, 8.0, 15.0, 11.0],
        'wind_kph': [16.09, 19.31, 12.87, 24.14, 17.70],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
        'car_id': [6, 6, 7, 7, 6],
        'route_code': ['R001', 'R001', 'R002', 'R002', 'R001']
    })


@pytest.fixture
def sample_quality_report():
    """Create sample quality report data for testing."""
    return {
        'timestamp': '2024-01-01T10:00:00.000000',
        'basic_stats': {
            'shape': [5, 10],
            'memory_usage_mb': 0.01,
            'numeric_columns': 7,
            'non_numeric_columns': 3,
            'total_missing_values': 1,
            'missing_percentage': 2.0
        },
        'signal_dictionary': {
            'speedAvg': {
                'expanded': 'Average Speed',
                'description': 'Average vehicle speed over measurement interval',
                'units': 'km/h',
                'domain': 'Vehicle Dynamics'
            },
            'soc': {
                'expanded': 'State of Charge',
                'description': 'Battery state of charge percentage',
                'units': '%',
                'domain': 'Battery'
            },
            'soh': {
                'expanded': 'State of Health',
                'description': 'Battery state of health percentage',
                'units': '%',
                'domain': 'Battery'
            },
            'longitude': {
                'expanded': 'Longitude',
                'description': 'GPS longitude coordinate',
                'units': 'degrees',
                'domain': 'GPS'
            },
            'latitude': {
                'expanded': 'Latitude',
                'description': 'GPS latitude coordinate',
                'units': 'degrees',
                'domain': 'GPS'
            }
        },
        'signal_quality': {
            'speedAvg': {
                'data_type': 'float64',
                'missing_count': 1,
                'missing_percentage': 20.0,
                'unique_count': 4,
                'zero_count': 0,
                'min': 50.0,
                'max': 80.0,
                'mean': 65.0,
                'std': 12.91,
                'range_validation': {
                    'signal_type': 'SPEED',
                    'signal_info': {'expanded': 'Average Speed', 'domain': 'Vehicle Dynamics'},
                    'has_violations': False,
                    'soft_violations': 0,
                    'hard_violations': 0,
                    'soft_violation_percentage': 0.0,
                    'hard_violation_percentage': 0.0,
                    'expected_range': {'unit': 'km/h', 'soft_min': 0, 'soft_max': 250},
                    'actual_range': {'min': 50.0, 'max': 80.0},
                    'summary': 'Signal within expected range'
                },
                'conditional_signal': {
                    'is_conditional_signal': False,
                    'analysis': 'Not identified as a conditional signal'
                },
                'stability': {
                    'is_constant': False,
                    'unique_ratio': 0.8,
                    'saturation': {
                        'min_hits_pct': 0.0,
                        'max_hits_pct': 0.0,
                        'is_saturated': False
                    }
                }
            },
            'soc': {
                'data_type': 'int64',
                'missing_count': 0,
                'missing_percentage': 0.0,
                'unique_count': 5,
                'zero_count': 0,
                'min': 65.0,
                'max': 85.0,
                'mean': 75.0,
                'std': 7.91,
                'range_validation': {
                    'signal_type': None,
                    'signal_info': {'expanded': 'State of Charge', 'domain': 'Battery'},
                    'has_violations': False,
                    'summary': 'No range validation available'
                },
                'conditional_signal': {
                    'is_conditional_signal': False,
                    'analysis': 'Not identified as a conditional signal'
                },
                'stability': {
                    'is_constant': False,
                    'unique_ratio': 1.0,
                    'saturation': {'is_saturated': False}
                }
            },
            'soh': {
                'data_type': 'float64',
                'missing_count': 0,
                'missing_percentage': 0.0,
                'unique_count': 1,
                'zero_count': 0,
                'min': 98.5,
                'max': 98.5,
                'mean': 98.5,
                'std': 0.0,
                'range_validation': {
                    'signal_type': None,
                    'signal_info': {'expanded': 'State of Health', 'domain': 'Battery'},
                    'has_violations': False,
                    'summary': 'No range validation available'
                },
                'conditional_signal': {
                    'is_conditional_signal': False,
                    'analysis': 'Not identified as a conditional signal'
                },
                'stability': {
                    'is_constant': True,
                    'constant_value': 98.5,
                    'unique_ratio': 0.2,
                    'saturation': {'is_saturated': False}
                }
            },
            'longitude': {
                'data_type': 'float64',
                'missing_count': 0,
                'missing_percentage': 0.0,
                'unique_count': 5,
                'zero_count': 0,
                'range_validation': {
                    'signal_type': 'LONGITUDE',
                    'signal_info': {'expanded': 'Longitude', 'domain': 'GPS'},
                    'has_violations': False,
                    'soft_violations': 0,
                    'hard_violations': 0,
                    'summary': 'Signal within expected range'
                },
                'conditional_signal': {'is_conditional_signal': False},
                'stability': {'is_constant': False, 'saturation': {'is_saturated': False}}
            },
            'wind_mph': {
                'data_type': 'float64',
                'missing_count': 0,
                'missing_percentage': 0.0,
                'unique_count': 5,
                'zero_count': 0,
                'stability': {
                    'is_constant': False,
                    'saturation': {'is_saturated': True}  # Example saturated signal
                }
            }
        },
        'correlations': {
            'high_correlations_count': 2,
            'expected_correlations': [
                {
                    'column1': 'wind_mph',
                    'column2': 'wind_kph',
                    'correlation': 1.000,
                    'explanation': 'Unit conversion correlation'
                }
            ],
            'unexpected_correlations': [
                {
                    'column1': 'soc',
                    'column2': 'speedAvg',
                    'correlation': -0.956,
                    'explanation': 'Unexpected correlation'
                }
            ],
            'analysis': 'Found 2 high correlations. 1 expected, 1 unexpected.'
        },
        'cross_signal': {
            'wheel_vs_vehicle_speed': [],
            'speed_positive_rpm_zero_count': 0
        },
        'priority_issues': [
            "Column 'speedAvg': 20.0% missing values",
            "Column 'soh': Signal has constant value"
        ],
        'recommendations': [
            "Investigate high missing value rates in speedAvg column",
            "Verify soh sensor functionality - showing constant value"
        ],
        'overall_score': 0.75
    }


@pytest.fixture
def temp_report_file(sample_quality_report):
    """Create a temporary JSON report file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_quality_report, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_csv_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


class TestAutomotiveDataQualityAPI:
    """Test class for AutomotiveDataQualityAPI."""
    
    def test_init_with_report_data(self, sample_quality_report, sample_csv_data):
        """Test initialization with pre-loaded data."""
        api = AutomotiveDataQualityAPI(
            report_data=sample_quality_report,
            csv_data=sample_csv_data
        )
        
        assert api.report_data == sample_quality_report
        assert api.csv_data is not None
        assert len(api.csv_data) == 5
    
    def test_init_with_file_paths(self, temp_report_file, temp_csv_file):
        """Test initialization with file paths."""
        api = AutomotiveDataQualityAPI(
            report_path=temp_report_file,
            csv_path=temp_csv_file
        )
        
        assert api.report_data is not None
        assert api.csv_data is not None
        assert 'basic_stats' in api.report_data
    
    def test_init_with_report_path_only(self, temp_report_file):
        """Test initialization with report path only."""
        api = AutomotiveDataQualityAPI(report_path=temp_report_file)
        
        assert api.report_data is not None
        assert api.csv_data is None
    
    def test_init_no_data_raises_error(self):
        """Test that initialization without data raises ValueError."""
        with pytest.raises(ValueError, match="Either report_data or report_path must be provided"):
            AutomotiveDataQualityAPI()
    
    def test_init_invalid_json_file(self):
        """Test initialization with invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                AutomotiveDataQualityAPI(report_path=temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_get_column_info_valid_column(self, sample_quality_report):
        """Test getting information for a valid column."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        info = api.get_column_info('speedAvg')
        
        assert info['data_type'] == 'float64'
        assert info['missing_percentage'] == 20.0
        assert info['signal_dictionary']['expanded'] == 'Average Speed'
        assert 'correlations' in info
        assert 'quality_assessment' in info
        assert 'priority_issues' in info
    
    def test_get_column_info_invalid_column(self, sample_quality_report):
        """Test getting information for invalid column raises KeyError."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        with pytest.raises(KeyError, match="Column 'nonexistent' not found"):
            api.get_column_info('nonexistent')
    
    def test_get_signal_dictionary_info(self, sample_quality_report):
        """Test getting signal dictionary information."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        info = api.get_signal_dictionary_info('speedAvg')
        
        assert info['expanded'] == 'Average Speed'
        assert info['domain'] == 'Vehicle Dynamics'
        assert info['units'] == 'km/h'
    
    def test_get_signal_dictionary_info_missing_column(self, sample_quality_report):
        """Test getting signal dictionary info for missing column."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        info = api.get_signal_dictionary_info('nonexistent')
        
        assert info == {}
    
    def test_get_correlations_all(self, sample_quality_report):
        """Test getting all correlations."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        correlations = api.get_correlations()
        
        assert len(correlations) == 2  # 1 expected + 1 unexpected
        assert any(corr['column1'] == 'wind_mph' for corr in correlations)
        assert any(corr['column1'] == 'soc' for corr in correlations)
    
    def test_get_correlations_expected_only(self, sample_quality_report):
        """Test getting expected correlations only."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        correlations = api.get_correlations(correlation_type='expected')
        
        assert len(correlations) == 1
        assert correlations[0]['column1'] == 'wind_mph'
        assert correlations[0]['column2'] == 'wind_kph'
    
    def test_get_correlations_unexpected_only(self, sample_quality_report):
        """Test getting unexpected correlations only."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        correlations = api.get_correlations(correlation_type='unexpected')
        
        assert len(correlations) == 1
        assert correlations[0]['column1'] == 'soc'
        assert correlations[0]['correlation'] == -0.956
    
    def test_get_correlations_for_specific_column(self, sample_quality_report):
        """Test getting correlations for specific column."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        correlations = api.get_correlations(column_name='wind_mph')
        
        assert len(correlations) == 1
        assert correlations[0]['column2'] == 'wind_kph'
    
    def test_get_correlations_with_threshold(self, sample_quality_report):
        """Test getting correlations with threshold filter."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        correlations = api.get_correlations(threshold=0.99)
        
        assert len(correlations) == 1  # Only perfect correlation
        assert correlations[0]['correlation'] == 1.000
    
    def test_get_columns_by_signal_type(self, sample_quality_report):
        """Test getting columns by signal type."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        speed_columns = api.get_columns_by_signal_type('SPEED')
        longitude_columns = api.get_columns_by_signal_type('LONGITUDE')
        
        assert 'speedAvg' in speed_columns
        assert 'longitude' in longitude_columns
        assert len(api.get_columns_by_signal_type('NONEXISTENT')) == 0
    
    def test_get_columns_by_domain(self, sample_quality_report):
        """Test getting columns by domain."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        battery_columns = api.get_columns_by_domain('Battery')
        gps_columns = api.get_columns_by_domain('GPS')
        
        assert 'soc' in battery_columns
        assert 'soh' in battery_columns
        assert 'longitude' in gps_columns
        assert 'latitude' in gps_columns
        assert len(api.get_columns_by_domain('NONEXISTENT')) == 0
    
    def test_get_problematic_columns_default_criteria(self, sample_quality_report):
        """Test getting problematic columns with default criteria."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        problems = api.get_problematic_columns()
        
        assert 'speedAvg' in problems['high_missing']  # 20% missing
        assert 'soh' in problems['constant_values']
        assert 'wind_mph' in problems['saturated']
    
    def test_get_problematic_columns_custom_criteria(self, sample_quality_report):
        """Test getting problematic columns with custom criteria."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        problems = api.get_problematic_columns(
            min_missing_pct=50.0,  # Higher threshold
            include_constant=False,
            include_saturated=False
        )
        
        assert len(problems['high_missing']) == 0  # No columns > 50% missing
        assert len(problems['constant_values']) == 0
        assert len(problems['saturated']) == 0
    
    def test_get_basic_stats(self, sample_quality_report):
        """Test getting basic statistics."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        stats = api.get_basic_stats()
        
        assert stats['shape'] == [5, 10]
        assert stats['missing_percentage'] == 2.0
        assert stats['numeric_columns'] == 7
    
    def test_get_overall_quality_score(self, sample_quality_report):
        """Test getting overall quality score."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        score = api.get_overall_quality_score()
        
        assert score == 0.75
    
    def test_get_priority_issues(self, sample_quality_report):
        """Test getting priority issues."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        issues = api.get_priority_issues()
        
        assert len(issues) == 2
        assert any('speedAvg' in issue for issue in issues)
        assert any('soh' in issue for issue in issues)
    
    def test_get_recommendations(self, sample_quality_report):
        """Test getting recommendations."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        recommendations = api.get_recommendations()
        
        assert len(recommendations) == 2
        assert any('speedAvg' in rec for rec in recommendations)
        assert any('soh' in rec for rec in recommendations)
    
    def test_list_all_columns(self, sample_quality_report):
        """Test listing all columns."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        columns = api.list_all_columns()
        
        expected_columns = ['speedAvg', 'soc', 'soh', 'longitude', 'wind_mph']
        for col in expected_columns:
            assert col in columns
    
    def test_search_columns_by_name(self, sample_quality_report):
        """Test searching columns by name."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        speed_columns = api.search_columns('speed')
        wind_columns = api.search_columns('wind')
        
        assert 'speedAvg' in speed_columns
        assert 'wind_mph' in wind_columns
    
    def test_search_columns_by_expanded_name(self, sample_quality_report):
        """Test searching columns by expanded name."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        state_columns = api.search_columns('state', search_in='expanded')
        
        assert 'soc' in state_columns
        assert 'soh' in state_columns
    
    def test_search_columns_by_domain(self, sample_quality_report):
        """Test searching columns by domain."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        battery_columns = api.search_columns('battery', search_in='domain')
        
        assert 'soc' in battery_columns
        assert 'soh' in battery_columns
    
    def test_get_column_summary(self, sample_quality_report):
        """Test getting column summary."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        summary = api.get_column_summary('speedAvg')
        
        assert 'Column: speedAvg' in summary
        assert 'Average Speed' in summary
        assert 'Vehicle Dynamics' in summary
        assert 'float64' in summary
        assert '20.0%' in summary  # Missing percentage
    
    def test_get_column_summary_invalid_column(self, sample_quality_report):
        """Test getting summary for invalid column."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        summary = api.get_column_summary('nonexistent')
        
        assert "not found in the quality report" in summary
    
    def test_assess_column_quality_high_quality(self, sample_quality_report):
        """Test quality assessment for high quality column."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        # Mock a high quality column info
        col_info = {
            'missing_percentage': 5.0,
            'range_validation': {'has_violations': False},
            'stability': {'is_constant': False, 'saturation': {'is_saturated': False}}
        }
        
        assessment = api._assess_column_quality('test_col', col_info)
        
        assert assessment['overall_score'] > 0.8
        assert len(assessment['issues']) == 0
        assert len(assessment['strengths']) > 0
    
    def test_assess_column_quality_low_quality(self, sample_quality_report):
        """Test quality assessment for low quality column."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        # Mock a low quality column info
        col_info = {
            'missing_percentage': 70.0,
            'range_validation': {
                'has_violations': True,
                'hard_violation_percentage': 15.0,
                'soft_violation_percentage': 5.0
            },
            'stability': {
                'is_constant': True,
                'saturation': {'is_saturated': True}
            }
        }
        
        assessment = api._assess_column_quality('test_col', col_info)
        
        assert assessment['overall_score'] < 0.3
        assert len(assessment['issues']) >= 3
    
    def test_get_column_correlations_private_method(self, sample_quality_report):
        """Test private method for getting column correlations."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        correlations = api._get_column_correlations('wind_mph')
        
        assert len(correlations['expected']) == 1
        assert len(correlations['unexpected']) == 0
        
        correlations = api._get_column_correlations('soc')
        assert len(correlations['unexpected']) == 1
    
    def test_get_column_cross_signal_info_private_method(self, sample_quality_report):
        """Test private method for getting cross signal info."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        cross_info = api._get_column_cross_signal_info('speedAvg')
        
        # Should be empty in our sample data
        assert cross_info == {}
    
    def test_get_column_priority_issues_private_method(self, sample_quality_report):
        """Test private method for getting column priority issues."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        issues = api._get_column_priority_issues('speedAvg')
        
        assert len(issues) == 1
        assert 'speedAvg' in issues[0]
        assert '20.0% missing values' in issues[0]
    
    def test_get_column_recommendations_private_method(self, sample_quality_report):
        """Test private method for getting column recommendations."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        recommendations = api._get_column_recommendations('speedAvg')
        
        assert len(recommendations) >= 0  # May or may not have recommendations
        # Check that if there are recommendations, they mention the column
        for rec in recommendations:
            assert 'speedAvg' in rec


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_quality_report_with_csv(self, temp_report_file, temp_csv_file):
        """Test load_quality_report convenience function with CSV."""
        api = load_quality_report(temp_report_file, temp_csv_file)
        
        assert isinstance(api, AutomotiveDataQualityAPI)
        assert api.report_data is not None
        assert api.csv_data is not None
    
    def test_load_quality_report_without_csv(self, temp_report_file):
        """Test load_quality_report convenience function without CSV."""
        api = load_quality_report(temp_report_file)
        
        assert isinstance(api, AutomotiveDataQualityAPI)
        assert api.report_data is not None
        assert api.csv_data is None
    
    def test_get_column_quality_summary_convenience(self, temp_report_file):
        """Test get_column_quality_summary convenience function."""
        summary = get_column_quality_summary(temp_report_file, 'speedAvg')
        
        assert 'Column: speedAvg' in summary
        assert 'Average Speed' in summary


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_report_data(self):
        """Test with empty report data."""
        api = AutomotiveDataQualityAPI(report_data={})
        
        assert api.get_basic_stats() == {}
        assert api.get_overall_quality_score() == 0.0
        assert api.get_priority_issues() == []
        assert api.list_all_columns() == []
    
    def test_missing_report_sections(self):
        """Test with missing sections in report data."""
        minimal_report = {'signal_quality': {'test_col': {'data_type': 'float64'}}}
        api = AutomotiveDataQualityAPI(report_data=minimal_report)
        
        # Should not raise errors
        assert api.get_correlations() == []
        assert api.get_basic_stats() == {}
        assert api.get_problematic_columns() == {
            'high_missing': [], 'range_violations': [],
            'constant_values': [], 'saturated': []
        }
    
    def test_search_columns_empty_pattern(self, sample_quality_report):
        """Test searching with empty pattern."""
        api = AutomotiveDataQualityAPI(report_data=sample_quality_report)
        
        # Should return all columns when pattern is empty
        columns = api.search_columns('')
        assert len(columns) == len(api.list_all_columns())
    
    def test_get_columns_by_signal_type_no_range_validation(self, sample_quality_report):
        """Test getting columns by signal type when no range validation exists."""
        # Test with None signal type - columns without range validation won't have signal_type
        columns_without_signal_type = []
        for col_name, col_info in sample_quality_report['signal_quality'].items():
            range_val = col_info.get('range_validation', {})
            if not range_val.get('signal_type'):
                columns_without_signal_type.append(col_name)
        
        # Should find columns that don't have signal_type set
        assert len(columns_without_signal_type) > 0  # soc, soh, wind_mph don't have signal_type
    
    def test_nonexistent_file_paths(self):
        """Test with nonexistent file paths."""
        with pytest.raises(FileNotFoundError):
            AutomotiveDataQualityAPI(report_path='/nonexistent/path.json')
    
    def test_invalid_csv_file(self, temp_report_file):
        """Test with invalid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\nwith,wrong,format,too,many,columns")
            temp_csv_path = f.name
        
        try:
            # Should still work, might just have issues with CSV loading
            api = AutomotiveDataQualityAPI(
                report_path=temp_report_file,
                csv_path=temp_csv_path
            )
            assert api.report_data is not None
        finally:
            os.unlink(temp_csv_path)


class TestReportStructureVariations:
    """Test different variations of report structure."""
    
    def test_missing_correlations_section(self, sample_quality_report):
        """Test report without correlations section."""
        report = sample_quality_report.copy()
        del report['correlations']
        
        api = AutomotiveDataQualityAPI(report_data=report)
        
        correlations = api.get_correlations()
        assert correlations == []
    
    def test_missing_signal_dictionary_section(self, sample_quality_report):
        """Test report without signal dictionary section."""
        report = sample_quality_report.copy()
        del report['signal_dictionary']
        
        api = AutomotiveDataQualityAPI(report_data=report)
        
        info = api.get_signal_dictionary_info('speedAvg')
        assert info == {}
    
    def test_incomplete_signal_quality_data(self, sample_quality_report):
        """Test with incomplete signal quality data."""
        report = sample_quality_report.copy()
        # Remove some fields from signal quality
        report['signal_quality']['speedAvg'] = {'data_type': 'float64'}
        
        api = AutomotiveDataQualityAPI(report_data=report)
        
        info = api.get_column_info('speedAvg')
        assert info['data_type'] == 'float64'
        # Missing fields should be handled gracefully
        assert 'quality_assessment' in info


if __name__ == "__main__":
    pytest.main([__file__])
