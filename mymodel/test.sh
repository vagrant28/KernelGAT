#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/root/autodl-tmp/KernelGAT/mymodel"

# 定义变量
program="/root/autodl-tmp/KernelGAT/mymodel/test.py"
cwd="/root/autodl-tmp/KernelGAT/mymodel"
outdir="../logs/gumbel"
test_path="../data/gumbel_test.jsonl"
test_batch_size="200"
prefix="DEBUG_predict_test"
encoder_name="/root/autodl-tmp/KernelGAT/hg_models/roberta-large"
seed="42"
checkpoint="/root/autodl-tmp/KernelGAT/logs/gumbel/04-03-2024/official_basemodel_3_roberta_large-seed42-epoch10.0-bsz4-lr2e-05/checkpoint_best_10394.pt"

# 启动程序
python3 "$program" \
    --outdir "$outdir" \
    --test_path "$test_path" \
    --test_batch_size "$test_batch_size" \
    --prefix "$prefix" \
    --encoder_name "$encoder_name" \
    --seed "$seed" \
    --checkpoint "$checkpoint"
