#!/usr/bin/env python3
"""
Test script demonstrating the enhanced automotive data quality assessment
with signal dictionary integration.

This script shows the improvements made to the automotive_data_quality.py module:
1. Signal dictionary integration for better signal identification
2. Enhanced correlation analysis using signal domains
3. Improved reporting with signal metadata
4. Better range validation with signal context
"""

import pandas as pd
from data_analysis_agent.automotive_data_quality import generate_automotive_quality_report

def main():
    """Run the enhanced automotive quality assessment."""
    
    print("=" * 80)
    print("ENHANCED AUTOMOTIVE TELEMETRY DATA QUALITY ASSESSMENT")
    print("=" * 80)
    
    # Load the test data
    print("Loading automotive telemetry data...")
    df = pd.read_csv('data/tables/kona-obd-signals-0605.csv')
    print(f"✓ Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Show some sample signal names
    print("\nSample signal names:")
    sample_signals = [col for col in df.columns if not col.startswith('Trip')][:10]
    for signal in sample_signals:
        print(f"  - {signal}")
    
    print("\nGenerating enhanced quality report...")
    
    # Generate the enhanced report
    text_report, json_data = generate_automotive_quality_report(
        df,
        output_file='enhanced_automotive_report.txt',
        json_output_file='enhanced_automotive_report.json',
        correlation_threshold=0.95,
        include_all_correlations=False
    )
    
    print("✓ Enhanced report generated successfully!")
    
    # Show key improvements
    signal_dict = json_data.get('signal_dictionary', {})
    signal_quality = json_data.get('signal_quality', {})
    
    print(f"\n📊 KEY IMPROVEMENTS:")
    print(f"   • Signal Dictionary: {len(signal_dict)} signals mapped to meaningful names")
    
    # Count signals by domain
    domain_counts = {}
    for info in signal_dict.values():
        domain = info.get('domain', 'Unknown')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"   • Domain Classification: {len(domain_counts)} domains identified")
    top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for domain, count in top_domains:
        print(f"     - {domain}: {count} signals")
    
    # Show enhanced signal information examples
    print(f"\n🔧 SIGNAL MAPPING EXAMPLES:")
    examples = [
        ('ENG_EngSpdVal', 'Engine speed in RPM'),
        ('MDPS_EstStrAnglVal', 'Steering angle from MDPS'),
        ('SAS_AnglVal', 'Steering angle sensor'),
        ('TPMS_FLTirePrsrVal', 'Front left tire pressure'),
        ('CLU_OdoVal', 'Odometer reading'),
    ]
    
    for signal_name, description in examples:
        if signal_name in signal_dict:
            info = signal_dict[signal_name]
            expanded = info.get('expanded', 'N/A')
            units = info.get('units', '')
            unit_str = f" ({units})" if units else ""
            print(f"     {signal_name}")
            print(f"       → {expanded}{unit_str}")
            print(f"       → Domain: {info.get('domain', 'N/A')}")
    
    # Show range validation improvements
    range_violations = sum(1 for q in signal_quality.values() 
                          if q.get('range_validation', {}).get('has_violations', False))
    total_automotive_signals = sum(1 for q in signal_quality.values() 
                                  if q.get('range_validation', {}).get('signal_type'))
    
    print(f"\n⚠️  QUALITY INSIGHTS:")
    print(f"   • Automotive signals identified: {total_automotive_signals}")
    print(f"   • Signals with range violations: {range_violations}")
    print(f"   • Overall quality score: {json_data.get('overall_score', 0):.2f}/1.0")
    
    correlations = json_data.get('correlations', {})
    expected_corr = len(correlations.get('expected_correlations', []))
    unexpected_corr = len(correlations.get('unexpected_correlations', []))
    
    print(f"   • Expected correlations: {expected_corr}")
    print(f"   • Unexpected correlations: {unexpected_corr}")
    
    print(f"\n📁 Output files:")
    print(f"   • Text report: enhanced_automotive_report.txt")
    print(f"   • JSON data: enhanced_automotive_report.json")
    
    print(f"\n✨ Enhancement Summary:")
    print(f"   The automotive data quality assessment now includes:")
    print(f"   • Signal dictionary with 436 CAN bus signal mappings")
    print(f"   • Domain-based correlation analysis")
    print(f"   • Enhanced range validation with signal context")
    print(f"   • Improved reporting with signal metadata")
    print(f"   • Better identification of automotive-specific patterns")

if __name__ == "__main__":
    main()
