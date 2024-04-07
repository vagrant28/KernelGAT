# python train.py --outdir ../checkpoint/kgat \
# --train_path ../data/bert_train.json \
# --valid_path ../data/bert_dev.json \
# --bert_pretrain ../bert_base \
# --postpretrain ../pretrain/save_model/model.best.pt

export CUDA_VISIBLE_DEVICES="1"

python train.py \
    --outdir ../logs/kgat \
    --train_path ../data/bert_train.json \
    --valid_path ../data/bert_dev.json \
    --bert_pretrain ../bert_base \
    --train_batch_size 16 \
    --valid_batch_size 16 \
    --prefix "official_baseline"
    # --postpretrain ../pretrain/save_model/model.best.pt \
    # --gradient_accumulation_steps 1 \
    # --eval_step 32 \
    