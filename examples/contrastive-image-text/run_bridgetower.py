#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Team All rights reserved.
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
"""
Training BridgeTower with a contrastive text-image loss.
"""

import io
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from habana_dataloader_trainer import HabanaDataloaderTrainer
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode, to_grayscale, to_tensor
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.models.bridgetower.modeling_bridgetower import BridgeTowerForContrastiveLearning
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers and Optimum Habana are not installed. Remove at your own risks.
check_min_version("4.51.0")
check_optimum_habana_min_version("1.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the vision model parameters or not."}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "Whether to freeze the text model parameters or not."}
    )
    freeze_text_pooler: bool = field(
        default=True, metadata={"help": "Whether to freeze the text pooler parameters or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_revision: str = field(
        default="main",
        metadata={"help": "The specific dataset version to use (can be a branch name, tag name or commit id)."},
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mediapipe_dataloader: bool = field(
        default=False, metadata={"help": "Use gaudi2/gaudi3 HW mediapipe over regular dataloader."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


dataset_name_mapping = {
    "image_caption_dataset.py": ("image_path", "caption"),
}


# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GaudiTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_bridgetower", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    gaudi_config = GaudiConfig.from_pretrained(
        training_args.gaudi_config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    # Log on each process the small summary:
    mixed_precision = training_args.bf16 or gaudi_config.use_torch_autocast
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        + f"mixed-precision training: {mixed_precision}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    # captions (unless you specify column names for this with the `image_column` and `caption_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            token=model_args.token,
            revision=data_args.dataset_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.mediapipe_dataloader and "image_path" not in dataset["train"].column_names:
        dataset = dataset.cast_column(data_args.image_column, datasets.Image(decode=False))

    # 5. Load pretrained model, tokenizer, and image processor
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load image_processor, in this script we only use this to get the mean and std for normalization.
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    model = BridgeTowerForContrastiveLearning.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    config = model.config

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_text_pooler:
        model.bridgetower.text_model.pooler = None

    if model_args.freeze_vision_model:
        _freeze_params(model.vision_model)

    if model_args.freeze_text_model:
        _freeze_params(model.text_model)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Create validation and test splits if they don't exist yet
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.4, shuffle=True, seed=training_args.seed)
        if "validation" not in dataset:
            buffer_dataset = dataset["test"].train_test_split(test_size=0.5, shuffle=False)
            dataset["validation"] = buffer_dataset["train"]
            dataset["test"] = buffer_dataset["test"]

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(data_args.dataset_name, None)
    if data_args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = data_args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{data_args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = data_args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # 7. Preprocessing the datasets.
    # Initialize torchvision transforms and jit it for faster processing.
    image_transformations = Transform(
        config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = list(examples[caption_column])
        text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples

    def get_image(image_or_path):
        if isinstance(image_or_path, str):
            # If the argument is a path to an image file, read it
            return read_image(image_or_path, mode=ImageReadMode.RGB)
        elif isinstance(image_or_path, Image.Image):
            if len(image_or_path.getbands()) == 1:
                image_or_path = to_grayscale(image_or_path, num_output_channels=3)
            return to_tensor(image_or_path)

        return None

    def transform_images(examples):
        images = []

        for item in examples[image_column]:
            # Manage the case where images are a dictionary with keys 'bytes' and 'path'
            if isinstance(item, dict):
                encoding = "ISO-8859-1"
                s = item["bytes"].decode(encoding)
                b = bytearray(s, encoding)
                image = Image.open(io.BytesIO(b)).convert("RGB")
                images.append(to_tensor(image))
            else:
                images.append(get_image(item))

        examples["pixel_values"] = [image_transformations(image) for image in images]

        return examples

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.map(
            function=tokenize_captions,
            batched=True,
            remove_columns=[col for col in column_names if col != image_column],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        if data_args.mediapipe_dataloader:
            train_dataset.image_mean = image_processor.image_mean
            train_dataset.image_std = image_processor.image_std
            train_dataset.text_max_length = data_args.max_seq_length
            train_dataset.image_resize = config.vision_config.image_size
            train_dataset.transform_func = transform_images
        else:
            # Transform images on the fly as doing it on the whole dataset takes too much time.
            train_dataset.set_transform(transform_images)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        eval_dataset = eval_dataset.map(
            function=tokenize_captions,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != image_column],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

        if data_args.mediapipe_dataloader:
            eval_dataset.image_mean = image_processor.image_mean
            eval_dataset.image_std = image_processor.image_std
            eval_dataset.text_max_length = data_args.max_seq_length
            eval_dataset.image_resize = config.vision_config.image_size
            eval_dataset.transform_func = transform_images
        else:
            # Transform images on the fly as doing it on the whole dataset takes too much time.
            eval_dataset.set_transform(transform_images)

    # 8. Initialize our trainer
    trainer_cls = HabanaDataloaderTrainer if data_args.mediapipe_dataloader else GaudiTrainer
    trainer = trainer_cls(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        image_processor.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("validation", metrics)
        trainer.save_metrics("validation", metrics)

    # 11. Write Training Stats and push to hub.
    finetuned_from = model_args.model_name_or_path
    # If from a local directory, don't set `finetuned_from` as this is required to be a valid repo. id on the Hub.
    if os.path.isdir(finetuned_from):
        finetuned_from = None
    kwargs = {"finetuned_from": finetuned_from, "tasks": "contrastive-image-text-modeling"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
