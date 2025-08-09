#!/usr/bin/env python3
"""
Entry point for the Data Analysis Agent.
"""

from data_analysis_agent.cli import parse_arguments, run_data_analysis


def main():
    """Main entry point that orchestrates CLI parsing and execution."""
    args = parse_arguments()
    run_data_analysis(skip_cleanup=args.skip_cleanup, log_level=args.log_level)


if __name__ == "__main__":
    main()
