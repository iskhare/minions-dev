#!/usr/bin/env python3
import argparse
import os
import json
import sys
import tempfile
import traceback
from typing import Dict, Any, List, Tuple
import openai
from processor import BuggyCodeProcessor

class BuggyCodeCLI:
    """
    CLI version of the Buggy Code Processor that works with either:
    - a single Python file containing a class with buggy methods
    - a CSV file with buggy code entries
    """
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser for the CLI."""
        parser = argparse.ArgumentParser(
            description="Process Python files or CSV files containing buggy code, "
                        "extract functions, generate test cases, and fix bugs."
        )
        
        parser.add_argument(
            "input_file",
            help="Path to the Python (.py) or CSV (.csv) file containing the buggy class/code"
        )
        
        parser.add_argument(
            "--output-dir", "-o",
            default="output",
            help="Directory to store the output files (default: 'output')"
        )
        
        parser.add_argument(
            "--threshold", "-t",
            type=float,
            default=0.7,
            help="Pass rate threshold (0.0-1.0) to consider a function fixed (default: 0.7)"
        )
        
        parser.add_argument(
            "--max-iterations", "-m",
            type=int,
            default=3,
            help="Maximum number of improvement attempts (default: 3)"
        )
        
        parser.add_argument(
            "--api-key", "-k",
            help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        
        return parser
    
    def _validate_input_file(self, file_path: str) -> bool:
        """Validate that the input file exists and is a Python or CSV file."""
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return False
        
        if not (file_path.endswith(".py") or file_path.endswith(".csv")):
            print(f"Warning: File does not have a .py or .csv extension: {file_path}")
            response = input("Continue anyway? (y/n): ")
            return response.lower() in ["y", "yes"]
        
        return True
    
    def _setup_api_key(self, api_key: str) -> bool:
        """Set up the OpenAI API key, either from args or environment variable."""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return True
        
        if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
            return True
        
        print("Error: OpenAI API key is required.")
        print("Either provide it with --api-key or set the OPENAI_API_KEY environment variable.")
        return False
    
    def _print_summary(self, results: Dict[str, Any], verbose: bool = False) -> None:
        """Print a summary of the processing results to the console."""
        print("\n===== PROCESSING SUMMARY =====")
        for func_name, data in results.items():
            # Calculate pass rate percentage for display
            pass_rate = data.get('pass_rate', 0)
            pass_percent = pass_rate * 100
            
            # Print basic function info
            print(f"\n{func_name}: {data['tests_passed']}/{data['tests_total']} tests passed ({pass_percent:.2f}%)")
            print(f"  Fixed by: {data['fixed_by']}")
            
            if data['fixed_by'] != 'original':
                print(f"  Iterations used: {data['iterations_used']}")
                
                # Show iteration progression
                if 'iterations_results' in data and verbose:
                    print("  Iteration progress:")
                    for iter_result in data['iterations_results']:
                        iter_pass_rate = iter_result['pass_rate'] * 100
                        print(f"    Iter {iter_result['iteration']}: "
                              f"{iter_result['passed']}/{iter_result['total']} ({iter_pass_rate:.2f}%)")
            
            # Show failures if any remain and verbose is enabled
            if data['failure_messages'] and verbose:
                print(f"  Remaining failures: {len(data['failure_messages'])}")
                for i, failure in enumerate(data['failure_messages'][:3]):  # Show first 3 failures
                    print(f"    Failure {i+1}: {failure[:100]}...")  # Truncate long messages
                if len(data['failure_messages']) > 3:
                    print(f"    ... and {len(data['failure_messages']) - 3} more failures")
    
    def run(self) -> int:
        """Run the CLI application with the provided arguments."""
        args = self.parser.parse_args()
        
        # Validate input file
        if not self._validate_input_file(args.input_file):
            return 1
        
        # Set up API key
        if not self._setup_api_key(args.api_key):
            return 1
        
        try:
            file_extension = os.path.splitext(args.input_file)[1].lower()
            
            print(f"Processing file: {args.input_file}")
            print(f"Output directory: {args.output_dir}")
            print(f"Pass threshold: {args.threshold}")
            print(f"Max iterations: {args.max_iterations}")
            print("Starting process...")
            
            # Create and run the processor - pass file directly to processor
            processor = BuggyCodeProcessor(
                input_path=args.input_file,
                output_dir=args.output_dir,
                pass_threshold=args.threshold,
                max_iterations=args.max_iterations
            )
            
            # Process the code
            results = processor.process_all()
            
            # Print results summary
            self._print_summary(results, args.verbose)
            
            # Indicate where to find the full results
            print(f"\nDetailed results and fixed code saved to: {args.output_dir}")
            print(f"Summary available in: {os.path.join(args.output_dir, 'summary.json')}")
            
            return 0
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            if args.verbose:
                print(traceback.format_exc())
            return 1


if __name__ == "__main__":
    cli = BuggyCodeCLI()
    sys.exit(cli.run())
