from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from gumbel_dataset import FeverDataset, collate
from gumbel_model import BaseModel, BaseModel_2, BaseModel_3
import argparse
import random
import numpy as np
import os
import json


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument("--test_batch_size", default=10, type=int, help="Total batch size for test.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evid_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--max_intra_len", default=200, type=int)
    parser.add_argument("--max_inter_len", default=300, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="test")
    parser.add_argument("--encoder_name", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = load_args()

    # # 设置随机数种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 加载config, tokenizer, model
    config = AutoConfig.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = BaseModel_3(config=config, args=args)
    model.load_state_dict(torch.load(args.checkpoint))

    test_dataset = FeverDataset(
                    file_path=args.test_path,
                    tokenizer=tokenizer,
                    max_intra_len=args.max_intra_len,
                    max_inter_len=args.max_inter_len,
                    evid_num=args.evid_num,
                    train=False
                    )
    collate_fc = partial(collate, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fc, pin_memory=True, num_workers=10)
    print(f"Num of train batches: {len(test_dataloader)}")

    model.cuda()
    model.eval()
    label2id = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    id2label = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
    with torch.no_grad():
        all_ids = []
        all_veri_labels = []
        for batch in tqdm(test_dataloader):
            b_ids = batch['b_ids'].tolist()
            all_ids.extend(b_ids)
            for k, v in batch.items():
                batch[k] = v.cuda()
            veri_label = batch["b_labels"]
            chosen_label = batch["b_sent_labels"].view(-1)

            outputs = model.infer(batch, temperature=0.5, hard_gumbel=False)
            pred_veri_logits = outputs["pred_veri_logits"]
            pred_veri_label = torch.argmax(pred_veri_logits, dim=-1).tolist()
            all_veri_labels.extend(pred_veri_label)
            # break
    with open('gumbel_test_pred_id_label.jsonl', 'w') as wf:
        for sid, label in zip(all_ids, all_veri_labels):
            res = {
                "id": sid,
                "predicted_label": id2label[label]
            }
            # print(res)
            wf.write(json.dumps(res) + '\n')