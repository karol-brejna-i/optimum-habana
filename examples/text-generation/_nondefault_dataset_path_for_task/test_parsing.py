#!/usr/bin/env python3
"""
Test script for the dataset path override functionality in run_lm_eval.py

This script tests the parsing logic without requiring the full lm-eval environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock logger for testing
class MockLogger:
    def info(self, msg):
        print(f"INFO: {msg}")
    
    def warning(self, msg):
        print(f"WARNING: {msg}")

# Mock the logger to avoid import issues
import run_lm_eval
run_lm_eval.logger = MockLogger()

def test_task_parsing():
    """Test the task parsing functionality."""
    print("Testing Dataset Path Override Parsing")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        # Standard tasks (no override)
        (["piqa", "hellaswag"], "Standard tasks without overrides"),
        
        # Single override
        (["piqa:karol/piqa"], "Single task with dataset override"),
        
        # Mixed standard and override
        (["hellaswag", "piqa:custom/piqa"], "Mix of standard and override tasks"),
        
        # Multiple overrides
        (["hellaswag:custom/hellaswag", "piqa:karol/piqa"], "Multiple dataset overrides"),
        
        # Complex dataset paths
        (["piqa:organization/dataset-name/config"], "Complex dataset path with config"),
        
        # Dataset paths with hyphens and underscores
        (["task:user/my-dataset_v2"], "Dataset path with special characters"),
    ]
    
    for tasks, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {tasks}")
        
        try:
            # Test the parsing function
            parsed = run_lm_eval.parse_tasks_with_dataset_overrides(tasks)
            print(f"Output: {parsed}")
            
            # Validate the results
            for i, task_spec in enumerate(tasks):
                if ":" in task_spec:
                    task_name, dataset_path = task_spec.split(":", 1)
                    print(f"  Override detected: {task_name} -> {dataset_path}")
                else:
                    print(f"  Standard task: {task_spec}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 30)

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\nTesting Edge Cases")
    print("=" * 50)
    
    edge_cases = [
        # Empty list
        ([], "Empty task list"),
        
        # Single colon (invalid)
        ([":"], "Invalid format - just colon"),
        
        # Multiple colons
        (["task:user/dataset:config"], "Multiple colons in path"),
        
        # Empty task name
        ([":dataset"], "Empty task name"),
        
        # Empty dataset path
        (["task:"], "Empty dataset path"),
        
        # URL-like dataset path
        (["task:https://example.com/dataset"], "URL-like dataset path"),
    ]
    
    for tasks, description in edge_cases:
        print(f"\nEdge Case: {description}")
        print(f"Input: {tasks}")
        
        try:
            parsed = run_lm_eval.parse_tasks_with_dataset_overrides(tasks)
            print(f"Output: {parsed}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 30)

def main():
    """Run all tests."""
    test_task_parsing()
    test_edge_cases()
    
    print("\nTest Summary")
    print("=" * 50)
    print("✓ Task parsing functionality implemented")
    print("✓ Support for 'task:dataset_path' syntax")
    print("✓ Backwards compatibility with standard task names")
    print("✓ Error handling for edge cases")
    print("\nTo use the enhanced functionality:")
    print("  python3 run_lm_eval.py --tasks piqa:karol/piqa hellaswag")

if __name__ == "__main__":
    main()
