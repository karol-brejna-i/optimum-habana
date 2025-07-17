# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
# Copyright (C) 2020-2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import argparse
import json
import multiprocessing as mp
import os
from typing import Literal, Optional

import psutil
import torch
import torch.nn.functional as F
from lm_eval import evaluator, utils
from lm_eval.models.huggingface import HFLM, TemplateLM

# Local imports
from run_generation import setup_parser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from utils import finalize_quantization, initialize_model, save_model

from optimum.habana.utils import HabanaGenerationTime, get_hpu_memory_stats


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = utils.eval_logger

# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    pool_size = physical_cpu_count
    world_size = int(os.getenv("WORLD_SIZE", 1))
    pool_size //= max(world_size, 1)
    if (pool_size * world_size) != physical_cpu_count:
        pool_size -= 1
    return spawn_context.Pool(pool_size)


mp.Pool = LimitedSpawnPool


def setup_lm_eval_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Evaluation script for HPU"
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        help="Input length buckets to use with static_shapes",
        default=[16, 32, 64, 128, 189, 284, 384],
    )

    parser.add_argument(
        "--output_file", "-o", type=str, help="Output file with end results and runtime parameters", required=True
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to run. Support format 'task:dataset_path' to override dataset path, e.g., 'piqa:karol/piqa'",
        default=["hellaswag", "lambada_openai", "piqa", "winogrande"],
    )
    parser.add_argument("--limit_iters", type=int, help="limit examples to run that many iterations", default=None)
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument("--max_graphs", type=int, help="Maximum number of HPU graphs", default=None)
    args = setup_parser(parser)

    return args


class HabanaModelAdapter(HFLM):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        args: argparse.Namespace,
        options: GenerationConfig,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        logits_cache: bool = True,
        add_bos_token: Optional[bool] = True,
        prefix_token_id: Optional[int] = None,
        delta: Optional[str] = None,
        **kwargs,
    ) -> None:
        # To skip cuda code of the HFLM init
        TemplateLM.__init__(self)
        self.tokenizer = tokenizer
        self._model = model
        self._config = self._model.config
        self._batch_size = args.batch_size
        self.buckets: list[int] = sorted(args.buckets)
        self.options = options
        self.device_ = args.device
        self.pretrained = model
        self.peft = args.peft_model
        self.delta = delta
        self.custom_prefix_token_id = prefix_token_id
        # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(config=self._config, backend=backend, trust_remote_code=args.trust_remote_code)
        self.logits_cache = logits_cache
        self.add_bos_token = add_bos_token
        self._max_length = options.max_length
        self.batch_size_per_gpu = int(args.batch_size)
        self.revision = args.model_revision
        self.model_inputs = {"use_cache": self.options.use_cache}
        if self._model.config.model_type in [
            "llama",
            "mistral",
            "falcon",
            "phi",
            "mixtral",
            "qwen2",
            "gptj",
            "starcoder2",
            "gemma",
            "baichuan",
        ]:
            self.model_inputs.update(
                {
                    "reuse_cache": self.options.reuse_cache,
                }
            )

        if self.model.config.model_type in [
            "llama",
            "mistral",
            "qwen2",
            "falcon",
            "starcoder2",
            "gemma",
            "baichuan",
            "gpt_bigcode",
        ]:
            if self.model.config.model_type not in ["falcon", "gpt_bigcode"]:
                self.model_inputs.update(
                    {
                        "attn_softmax_bf16": self.options.attn_softmax_bf16,
                    }
                )
            self.model_inputs.update(
                {
                    "use_flash_attention": self.options.use_flash_attention,
                    "flash_attention_recompute": self.options.flash_attention_recompute,
                    "flash_attention_causal_mask": self.options.flash_attention_causal_mask,
                }
            )
            if self.model.config.model_type in ["llama", "qwen2", "baichuan", "gpt_bigcode"]:
                self.model_inputs.update({"flash_attention_fast_softmax": self.options.flash_attention_fast_softmax})
        if args.warmup:
            self.warm_up()

    def warm_up(self) -> None:
        for bucket_size in reversed(self.buckets):
            inps = torch.ones((self._batch_size, bucket_size), dtype=torch.int64)
            self._model_call(inps)

    @property
    def eot_token_id(self) -> int:
        return self._model.config.eos_token_id

    @property
    def max_length(self) -> int:
        return self.buckets[-1]

    @property
    def device(self):
        # We need to do padding ourselves, otherwise we'll end up with recompilations
        # Returning 'cpu' to keep tensors on CPU in lm_eval code
        return "cpu"

    def find_bucket(self, length: int) -> list[int]:
        return [b for b in self.buckets if b >= length][0]

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        bs, seq_length = inps.shape
        padding_length = 0
        if self.options.static_shapes:
            bucket_length = self.find_bucket(seq_length)
            if self.options.use_cache and self.options.reuse_cache:
                self._model.allocate_kv_cache(bs, bucket_length + 1, bucket_length)
            padding_length = bucket_length - seq_length
            inps = F.pad(inps, (0, padding_length), value=self._model.config.pad_token_id)
        logits = self._model(inps.to(self.device_), **self.model_inputs)["logits"].cpu()

        if self.options.static_shapes and padding_length > 0:
            logits = logits[:, :-padding_length, :]
        logits = logits.to(torch.float32)
        return logits

    def get_model_info(self) -> dict:
        """
        Patched method to get Hugging Face model information for experiment reproducibility.
        source: https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/models/huggingface.py/#L1375
        Remove from SynapseAI 1.21
        """

        def get_model_num_params(model) -> int:
            if hasattr(model, "num_parameters"):
                return model.num_parameters()
            if hasattr(model, "parameters"):
                return sum(p.numel() for p in model.parameters())
            else:
                return -1

        def get_model_dtype(model) -> str:
            if hasattr(model, "dtype"):
                return model.dtype
            else:
                return ""

        def get_model_sha(pretrained: str, revision: str) -> str:
            return ""

        model_info = {
            "model_num_parameters": get_model_num_params(self._model),
            "model_dtype": get_model_dtype(self._model),
            "model_revision": self.revision,
            "model_sha": get_model_sha(self.pretrained, self.revision),
        }
        if self.peft:
            model_info["peft_sha"] = get_model_sha(self.peft, self.revision)
        if self.delta:
            model_info["delta_sha"] = get_model_sha(self.delta, self.revision)
        return model_info


def create_task_override_configs(tasks):
    """Create task configuration overrides that can be used with lm_eval.
    
    This function handles the 'task_name:dataset_path' syntax to create task configuration
    dictionaries that override only the dataset path while preserving other task settings.
    
    Args:
        tasks: List of task specifications
        
    Returns:
        A list compatible with lm_eval.simple_evaluate tasks parameter
    """
    # Import here to avoid import errors in environments without lm_eval
    try:
        from lm_eval.tasks import TaskManager
    except ImportError:
        logger.warning("lm_eval not available, task overrides may not work properly")
        return tasks
    
    parsed_tasks = []
    override_configs = []
    
    # Initialize task manager to access base configurations
    task_manager = TaskManager()
    
    for task_spec in tasks:
        if ":" in task_spec:
            task_name, dataset_path = task_spec.split(":", 1)
            logger.info(f"Creating override for task '{task_name}' with dataset '{dataset_path}'")
            
            try:
                # Try to get the base task configuration
                if task_name in task_manager._all_tasks:
                    base_config = task_manager._get_config(task_name)
                    
                    # Create override config
                    override_name = f"{task_name}_override_{dataset_path.replace('/', '_').replace('-', '_')}"
                    override_config = base_config.copy()
                    override_config["task"] = override_name
                    override_config["dataset_path"] = dataset_path
                    
                    override_configs.append(override_config)
                    parsed_tasks.append(override_name)
                else:
                    logger.warning(f"Task '{task_name}' not found in registry, using fallback approach")
                    # Fallback: create a basic override config
                    override_config = {
                        "task": f"{task_name}_custom_{dataset_path.replace('/', '_').replace('-', '_')}",
                        "dataset_path": dataset_path,
                        "description": f"Custom dataset override for {task_name}",
                    }
                    override_configs.append(override_config)
                    parsed_tasks.append(override_config["task"])
                    
            except Exception as e:
                logger.warning(f"Failed to create override for '{task_name}': {e}")
                # Fall back to original task
                parsed_tasks.append(task_name)
        else:
            # Regular task without override
            parsed_tasks.append(task_spec)
    
    # Return both the task names and the override configs
    # The caller should pass both to the TaskManager
    return parsed_tasks, override_configs


def parse_tasks_with_dataset_overrides(tasks):
    """Legacy wrapper for backwards compatibility"""
    parsed_tasks, _ = create_task_override_configs(tasks)
    return parsed_tasks


def main() -> None:
    # Modified based on cli_evaluate function in https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.7/lm_eval/__main__.py/#L268
    args = setup_lm_eval_parser()
    model, _, tokenizer, generation_config = initialize_model(args, logger)

    # Parse tasks to handle dataset path overrides
    try:
        task_names, override_configs = create_task_override_configs(args.tasks)
        
        # If we have override configs, we need to create a custom TaskManager
        if override_configs:
            logger.info(f"Created {len(override_configs)} task configuration overrides")
            from lm_eval.tasks import TaskManager
            
            # Create task manager and register override configs
            task_manager = TaskManager()
            
            # Convert override configs to the format expected by lm_eval
            final_tasks = []
            for task_name in task_names:
                # Check if this task has an override config
                override_config = next(
                    (config for config in override_configs if config.get("task") == task_name),
                    None
                )
                if override_config:
                    final_tasks.append(override_config)
                else:
                    final_tasks.append(task_name)
            
            parsed_tasks = final_tasks
        else:
            parsed_tasks = args.tasks
            task_manager = None
    except ImportError:
        logger.warning("lm_eval TaskManager not available, using simple task parsing fallback")
        parsed_tasks = parse_tasks_with_dataset_overrides(args.tasks)
        task_manager = None

    with torch.no_grad():
        lm = HabanaModelAdapter(tokenizer, model, args, generation_config)

    with HabanaGenerationTime() as timer:
        with torch.no_grad():
            # Pass the task_manager if we have one
            eval_kwargs = {"lm": lm, "tasks": parsed_tasks, "limit": args.limit_iters, "log_samples": False}
            if task_manager is not None:
                eval_kwargs["task_manager"] = task_manager
            
            results = evaluator.simple_evaluate(**eval_kwargs)
        if args.device == "hpu":
            import habana_frameworks.torch.hpu as torch_hpu

            torch_hpu.synchronize()

    results["args"] = vars(args)
    results["duration"] = timer.last_duration

    if args.local_rank == 0:
        if args.device == "hpu":
            mem = get_hpu_memory_stats()
            for k, v in mem.items():
                print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))

        json_str = json.dumps(results, indent=2, default=utils.handle_non_serializable, ensure_ascii=False)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        if args.show_config:
            print(json_str)

    if args.quant_config:
        finalize_quantization(model)
    if args.save_quantized_model_with_inc:
        save_model(model, tokenizer, args.saved_model_path)
    if args.pt2e_save and args.pt2e_path:
        from quantization_tools.pt2e import pt2e_save

        pt2e_save(model)

    if args.const_serialization_path and os.path.isdir(args.const_serialization_path):
        import shutil

        shutil.rmtree(args.const_serialization_path)


if __name__ == "__main__":
    main()
