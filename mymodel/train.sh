#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/root/autodl-tmp/KernelGAT/mymodel"

# 运行训练脚本
python3 /root/autodl-tmp/KernelGAT/mymodel/train.py \
    --outdir "../logs/gumbel" \
    --train_path "../data/gumbel_train.jsonl" \
    --valid_path "../data/gumbel_dev.jsonl" \
    --train_batch_size "4" \
    --valid_batch_size "4" \
    --eval_step "1000" \
    --prefix "official_basemodel_3_roberta_large" \
    --encoder_name "/root/autodl-tmp/KernelGAT/hg_models/roberta-large" \
    --num_train_epochs "10" \
    --gradient_accumulation_steps "13"

# 1000 5
    
