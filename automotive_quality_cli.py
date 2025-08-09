#!/usr/bin/env python3
"""
Automotive Telemetry Data Quality Assessment CLI

Command-line interface for automotive-specific data quality assessment.
Provides both text and JSON output formats with configurable parameters.

Usage:
    python automotive_quality_cli.py input.csv
    python automotive_quality_cli.py input.csv --output report.txt --json-output report.json
    python automotive_quality_cli.py input.csv --correlation-threshold 0.9 --include-all-correlations

Author: GitHub Copilot
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_analysis_agent.automotive_data_quality import generate_automotive_quality_report


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Automotive Telemetry Data Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s telemetry_data.csv
  %(prog)s data.csv --output quality_report.txt --json-output quality_data.json
  %(prog)s data.csv --correlation-threshold 0.9 --include-all-correlations
  %(prog)s data.csv --help-automotive

This tool is specifically designed for automotive telemetry data and provides:
- Automotive signal validation (RPM, speed, temperature ranges)
- Conditional signal analysis (signals only active under certain conditions)
- High correlation analysis with automotive context
- JSON and text output formats
- Priority issue identification
        """
    )
    
    parser.add_argument(
        "input_file",
        type=Path,
        nargs='?',  # Make input_file optional when using --help-automotive
        help="Path to the CSV file containing automotive telemetry data"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to save the text report (optional)"
    )
    
    parser.add_argument(
        "--json-output", "-j",
        type=Path,
        help="Path to save the JSON report (optional)"
    )
    
    parser.add_argument(
        "--correlation-threshold", "-c",
        type=float,
        default=0.95,
        help="Correlation threshold for reporting (default: 0.95). "
             "Lower values report more correlations."
    )
    
    parser.add_argument(
        "--include-all-correlations",
        action="store_true",
        help="Include all high correlations in the report, including expected ones "
             "(default: only show unexpected correlations)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output (only save to files)"
    )
    
    parser.add_argument(
        "--help-automotive",
        action="store_true",
        help="Show automotive-specific help and signal types"
    )
    
    args = parser.parse_args()
    
    if args.help_automotive:
        show_automotive_help()
        return
    
    # Check if input_file is provided (required unless --help-automotive)
    if not args.input_file:
        print("Error: input_file is required", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not args.input_file.suffix.lower() == '.csv':
        print(f"Warning: Input file '{args.input_file}' is not a CSV file.", file=sys.stderr)
    
    # Validate correlation threshold
    if not 0.0 <= args.correlation_threshold <= 1.0:
        print(f"Error: Correlation threshold must be between 0.0 and 1.0, got {args.correlation_threshold}",
              file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load data
        if not args.quiet:
            print(f"Loading data from {args.input_file}...")
        
        df = pd.read_csv(args.input_file)
        
        if not args.quiet:
            print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
            print("Analyzing automotive telemetry data quality...")
        
        # Generate quality report
        text_report, json_data = generate_automotive_quality_report(
            df,
            output_file=args.output,
            json_output_file=args.json_output,
            correlation_threshold=args.correlation_threshold,
            include_all_correlations=args.include_all_correlations
        )
        
        # Display summary
        if not args.quiet:
            if args.output or args.json_output:
                print("\nReport generation completed!")
                if args.output:
                    print(f"Text report saved to: {args.output}")
                if args.json_output:
                    print(f"JSON report saved to: {args.json_output}")
                
                print("\nQuality Summary:")
                print(f"  Overall Score: {json_data['overall_score']:.2f}/1.0")
                print(f"  Priority Issues: {len(json_data['priority_issues'])}")
                print(f"  Missing Data: {json_data['basic_stats']['missing_percentage']:.1f}%")
                automotive_signals = sum(
                    1 for analysis in json_data['signal_quality'].values()
                    if analysis.get('range_validation', {}).get('signal_type')
                )
                print(f"  Automotive Signals Identified: {automotive_signals}")
                
                if json_data['priority_issues']:
                    print("\nTop Priority Issues:")
                    for i, issue in enumerate(json_data['priority_issues'][:3], 1):
                        print(f"  {i}. {issue}")
                    if len(json_data['priority_issues']) > 3:
                        print(f"  ... and {len(json_data['priority_issues']) - 3} more (see full report)")
                        
            else:
                print("\n" + text_report)
                
    except FileNotFoundError:
        print(f"Error: Could not read file '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{args.input_file}' is empty or has no valid data", file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error: Could not parse CSV file '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def show_automotive_help():
    """Show automotive-specific help information."""
    help_text = """
AUTOMOTIVE TELEMETRY DATA QUALITY ASSESSMENT - HELP

This tool is specifically designed for automotive telemetry data and provides 
specialized analysis beyond generic data quality tools.

AUTOMOTIVE SIGNAL TYPES RECOGNIZED:
Engine Signals:
  - RPM, ENGINE_SPEED: 0-8000 rpm
  - THROTTLE, THROTTLE_POSITION: 0-100%
  
Vehicle Dynamics:
  - SPEED, VEHICLE_SPEED: 0-300 km/h
  - ACCELERATION: -20 to 20 m/s²
  
Temperatures:
  - ENGINE_TEMP, COOLANT_TEMP: -40 to 150°C
  - OIL_TEMP: -40 to 200°C
  - INTAKE_TEMP: -40 to 100°C
  - AMBIENT_TEMP: -50 to 60°C
  
Pressures:
  - OIL_PRESSURE: 0-10 bar
  - FUEL_PRESSURE: 0-10 bar
  - BOOST_PRESSURE: -1 to 3 bar
  
Electrical:
  - BATTERY_VOLTAGE: 9-16V
  - ALTERNATOR_VOLTAGE: 12-15V
  
Others:
  - FUEL_LEVEL: 0-100%
  - GPS coordinates (LATITUDE, LONGITUDE)

CONDITIONAL SIGNALS (Expected to be mostly zero):
  - BRAKE_PRESSURE, BRAKE_PEDAL
  - TURN_SIGNAL_LEFT, TURN_SIGNAL_RIGHT
  - ABS_ACTIVE, ESP_ACTIVE
  - CRUISE_CONTROL, LANE_ASSIST
  - etc.

AUTOMOTIVE-SPECIFIC FEATURES:
1. Range Validation: Checks if signal values are within physically possible ranges
2. Conditional Signal Analysis: Understands that some signals are normally zero
3. Expected Correlations: Knows that RPM/SPEED correlation is normal
4. Missing Data Context: Telemetry often has missing data during certain conditions
5. Temporal Consistency: Checks for physically impossible changes

CORRELATION THRESHOLD GUIDANCE:
- 0.99: Very strict (almost identical signals only)
- 0.95: Default (strong correlations)
- 0.90: Moderate (catch more potential issues)
- 0.80: Loose (many correlations will be reported)

For automotive data, 0.90-0.95 is often appropriate since vehicle signals
naturally correlate (engine RPM vs speed, temperatures, etc.).

OUTPUT FORMATS:
- Text Report: Human-readable summary for quick review
- JSON Report: Machine-readable data for integration/analysis
  Contains: signal_quality, correlations, priority_issues, recommendations

EXAMPLE USAGE:
  # Basic analysis
  python automotive_quality_cli.py car_data.csv
  
  # Save reports and lower correlation threshold
  python automotive_quality_cli.py car_data.csv \\
    --output report.txt \\
    --json-output data.json \\
    --correlation-threshold 0.9
  
  # Include all correlations (not just unexpected ones)
  python automotive_quality_cli.py car_data.csv \\
    --include-all-correlations
"""
    print(help_text)


if __name__ == "__main__":
    main()
