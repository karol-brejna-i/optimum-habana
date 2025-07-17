#!/bin/bash

cd optimum-habana;
pip install .;
cd examples/text-generation;
pip install -r requirements.txt;
pip install -r requirements_lm_eval.txt;

echo "Example 1: Standard usage with default datasets"
PT_HPU_LAZY_MODE=1 \
HF_DATASETS_TRUST_REMOTE_CODE=true \
QUANT_CONFIG=quantization_config/maxabs_measure.json \
TQDM_DISABLE=1 \
python3 run_lm_eval.py \
    -o /root/logs/test_results_standard.json \
    --model_name_or_path /mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/ \
    --warmup 0 \
    --attn_softmax_bf16 \
    --use_hpu_graphs \
    --trim_logits \
    --use_kv_cache \
    --bf16 \
    --batch_size 1 \
    --bucket_size=128 \
    --bucket_internal \
    --trust_remote_code \
    --tasks hellaswag piqa \
    --use_flash_attention \
    --flash_attention_recompute

echo "Example 2: Using dataset path override for piqa task"
PT_HPU_LAZY_MODE=1 \
HF_DATASETS_TRUST_REMOTE_CODE=true \
QUANT_CONFIG=quantization_config/maxabs_measure.json \
TQDM_DISABLE=1 \
python3 run_lm_eval.py \
    -o /root/logs/test_results_override.json \
    --model_name_or_path /mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/ \
    --warmup 0 \
    --attn_softmax_bf16 \
    --use_hpu_graphs \
    --trim_logits \
    --use_kv_cache \
    --bf16 \
    --batch_size 1 \
    --bucket_size=128 \
    --bucket_internal \
    --trust_remote_code \
    --tasks hellaswag piqa:karol/piqa \
    --use_flash_attention \
    --flash_attention_recompute

echo "Example 3: Multiple dataset path overrides"
PT_HPU_LAZY_MODE=1 \
HF_DATASETS_TRUST_REMOTE_CODE=true \
QUANT_CONFIG=quantization_config/maxabs_measure.json \
TQDM_DISABLE=1 \
python3 run_lm_eval.py \
    -o /root/logs/test_results_multi_override.json \
    --model_name_or_path /mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/ \
    --warmup 0 \
    --attn_softmax_bf16 \
    --use_hpu_graphs \
    --trim_logits \
    --use_kv_cache \
    --bf16 \
    --batch_size 1 \
    --bucket_size=128 \
    --bucket_internal \
    --trust_remote_code \
    --tasks hellaswag:custom/hellaswag piqa:custom/piqa_v2 \
    --use_flash_attention \
    --flash_attention_recompute