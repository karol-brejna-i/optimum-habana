# Dataset Path Override Feature for run_lm_eval.py

This document describes the enhanced `run_lm_eval.py` script that supports overriding dataset paths for specific tasks while preserving all other task configuration.

## Overview

The enhanced script supports a new syntax for the `--tasks` argument that allows you to specify a custom dataset path for individual tasks:

```bash
--tasks task_name:dataset_path
```

This enables you to run evaluations with custom datasets while keeping all other aspects of the task configuration (prompts, metrics, etc.) intact.

## Syntax

### Standard Usage (No Override)
```bash
--tasks piqa hellaswag
```
Uses the default dataset paths defined in the lm-eval task configurations.

### Dataset Path Override
```bash
--tasks piqa:karol/piqa
```
Uses the dataset `karol/piqa` from Hugging Face Hub instead of the default dataset for the `piqa` task.

### Mixed Usage
```bash
--tasks hellaswag piqa:custom/piqa_dataset
```
Uses the default dataset for `hellaswag` and a custom dataset for `piqa`.

### Multiple Overrides
```bash
--tasks hellaswag:custom/hellaswag piqa:another/piqa_v2
```
Overrides datasets for both tasks.

## Examples

### Example 1: Standard Evaluation
```bash
python3 run_lm_eval.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tasks hellaswag piqa winogrande \
    --output_file results.json \
    --batch_size 8
```

### Example 2: Single Dataset Override
```bash
python3 run_lm_eval.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tasks hellaswag piqa:karol/piqa winogrande \
    --output_file results_with_override.json \
    --batch_size 8
```

### Example 3: Multiple Dataset Overrides
```bash
python3 run_lm_eval.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tasks hellaswag:custom/hellaswag piqa:karol/piqa \
    --output_file results_multi_override.json \
    --batch_size 8
```

## Implementation Details

### How It Works

1. **Task Parsing**: The script parses task specifications and identifies those with `:` syntax
2. **Configuration Override**: For tasks with dataset overrides, it creates new task configurations that inherit from the base task but use the custom dataset
3. **Task Registration**: Custom task configurations are registered with the lm-eval TaskManager
4. **Evaluation**: The evaluation proceeds normally with the custom configurations

### Code Changes

The main changes to `run_lm_eval.py` include:

1. **Enhanced Argument Help**: Updated the `--tasks` argument description to document the new syntax
2. **Task Parsing Function**: Added `create_task_override_configs()` to handle the parsing logic
3. **Main Function Updates**: Modified the main evaluation loop to handle custom task configurations

### Key Functions

#### `create_task_override_configs(tasks)`
- Parses task specifications with dataset overrides
- Creates proper task configurations that inherit from base tasks
- Returns both task names and override configurations

#### `parse_tasks_with_dataset_overrides(tasks)`
- Legacy wrapper for backwards compatibility
- Simple parsing without TaskManager integration

## Supported Dataset Formats

The dataset path can be any format supported by Hugging Face `datasets.load_dataset()`:

- **Hub datasets**: `username/dataset_name`
- **Hub datasets with configs**: `username/dataset_name/config_name`
- **Local paths**: `/path/to/local/dataset`
- **Other formats**: Any valid dataset identifier

## Error Handling

- If a base task is not found in the lm-eval registry, the script falls back to creating a minimal task configuration
- Import errors (when lm-eval is not available) are handled gracefully with warnings
- Invalid dataset paths will be caught by the underlying lm-eval framework

## Compatibility

- **Backwards Compatible**: Existing scripts using standard task names will work unchanged
- **lm-eval Versions**: Tested with lm-evaluation-harness v0.4.x
- **Python**: Requires Python 3.8+

## Limitations

1. **Base Task Required**: The override feature works best when the base task exists in the lm-eval registry
2. **Configuration Inheritance**: Only dataset path is overridden; other dataset-specific configurations (like column mappings) may need manual adjustment
3. **Custom Tasks**: For completely custom tasks, consider creating proper YAML task definitions instead

## Troubleshooting

### Task Not Found Warning
```
WARNING: Task 'task_name' not found in registry, using fallback approach
```
This occurs when the base task doesn't exist in lm-eval. The script will create a minimal configuration, but you may need to specify additional task parameters.

### Import Errors
```
WARNING: lm_eval not available, task overrides may not work properly
```
This indicates that the lm-evaluation-harness package is not properly installed. Install it with:
```bash
pip install lm-eval
```

### Dataset Loading Errors
If the custom dataset fails to load, check:
1. Dataset exists and is accessible
2. Proper authentication for private datasets
3. Dataset format is compatible with the task's expected structure

## Related Documentation

- [lm-evaluation-harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Task Configuration Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)
