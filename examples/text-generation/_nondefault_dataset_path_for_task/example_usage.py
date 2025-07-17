#!/usr/bin/env python3
"""
Example usage of the enhanced run_lm_eval.py script with dataset path overrides.

This script demonstrates how to use the new syntax 'task:dataset_path' to override
the default dataset path for specific tasks while keeping all other task configuration intact.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle its output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)

def main():
    """Demonstrate usage examples."""
    
    # Base directory
    base_dir = Path(__file__).parent.parent
    run_lm_eval_script = base_dir / "run_lm_eval.py"
    
    print("Enhanced run_lm_eval.py with Dataset Path Override Support")
    print("=" * 60)
    print()
    print("This script demonstrates the new syntax for overriding dataset paths:")
    print("  --tasks task_name               # Use default dataset")
    print("  --tasks task_name:dataset_path  # Override dataset path")
    print()
    
    # Example 1: Standard usage (no overrides)
    print("Example 1: Standard usage with default datasets")
    example1_cmd = [
        "python3", str(run_lm_eval_script),
        "--model_name_or_path", "gpt2",  # Small model for testing
        "--tasks", "hellaswag", "piqa",  # Standard tasks
        "--output_file", "/tmp/test_results_standard.json",
        "--batch_size", "1",
        "--limit_iters", "5",  # Just a few iterations for testing
    ]
    run_command(example1_cmd, "Standard task evaluation")
    
    # Example 2: Usage with dataset overrides
    print("\nExample 2: Using dataset path overrides")
    example2_cmd = [
        "python3", str(run_lm_eval_script),
        "--model_name_or_path", "gpt2",
        "--tasks", "piqa", "piqa:karol/piqa",  # One standard, one overridden
        "--output_file", "/tmp/test_results_override.json",
        "--batch_size", "1",
        "--limit_iters", "5",
    ]
    run_command(example2_cmd, "Task evaluation with dataset path override")
    
    # Example 3: Multiple overrides
    print("\nExample 3: Multiple dataset path overrides")
    example3_cmd = [
        "python3", str(run_lm_eval_script),
        "--model_name_or_path", "gpt2",
        "--tasks", "hellaswag:custom/hellaswag", "piqa:another/piqa_dataset",
        "--output_file", "/tmp/test_results_multi_override.json",
        "--batch_size", "1",
        "--limit_iters", "5",
    ]
    run_command(example3_cmd, "Multiple dataset path overrides")

if __name__ == "__main__":
    main()
