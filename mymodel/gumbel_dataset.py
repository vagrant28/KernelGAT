import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json
import os
import sys
import re
import unicodedata
import random
from functools import partial
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_path', help='train path', default="/root/autodl-tmp/KernelGAT/data/gumbel_train.jsonl")
    parser.add_argument('--valid_path', help='valid path', default="/root/autodl-tmp/KernelGAT/data/gumbel_dev.jsonl")
    parser.add_argument('--test_path', help='test path', default="/root/autodl-tmp/KernelGAT/data/gumbel_test.jsonl")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    # parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evid_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--max_intra_len", default=200, type=int)
    parser.add_argument("--max_inter_len", default=300, type=int)
    parser.add_argument("--eval_step", default=500, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="test")
    parser.add_argument("--encoder_name", type=str, default="/root/autodl-tmp/KernelGAT/hg_models/bigbird-roberta-base")
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    args = parser.parse_args()

    return args




def process_wiki_title(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    string = re.sub("_", " ", string)
    return string

def process_sent(sentence):
    sentence = re.sub(" LSB.*?RSB", "", sentence)
    sentence = re.sub("LRB RRB ", "", sentence)
    sentence = re.sub("LRB", "(", sentence)
    sentence = re.sub("RRB", ")", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)

    return sentence

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')

def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string


class FeverDataset(Dataset):
    def __init__(self, 
                 file_path,
                 tokenizer,
                 max_intra_len,
                 max_inter_len,
                 evid_num=5,
                 train=False):
        self.tokenizer = tokenizer
        self.max_intra_len = max_intra_len
        self.max_inter_len = max_inter_len
        self.evid_num = evid_num
        self.train = train
        self.label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
        self.data = self.read_file(file_path)

    def __len__(self):
        return len(self.data)
    
    def compare_origin_and_chosen_tokens(self, orgin, tokens, chosen_mask):
        print(orgin)
        chosen_tokens = []
        # print(tokens)
        # print(chosen_mask)
        for msk, token in zip(chosen_mask, tokens):
            if msk == 1:
                chosen_tokens.append(token)
        chosen_tokens = self.tokenizer.convert_tokens_to_string(chosen_tokens)
        print(chosen_tokens)

    def shorten_sentences(self, tokenized_sentences, max_length):
        total_len = sum([len(sent) for sent in tokenized_sentences])
        while total_len > max_length:
            sorted_indices = sorted(range(len(tokenized_sentences)), key=lambda i: len(tokenized_sentences[i]), reverse=True)
            longest_ind = sorted_indices[0]
            tokenized_sentences[longest_ind] = tokenized_sentences[longest_ind][:-1]
            total_len = sum([len(sent) for sent in tokenized_sentences])
        return tokenized_sentences


    def inter_tokens_and_masks(self, claim, evidences):
        # inter_sents = [claim, ]
        # inter_sents = claim + self.tokenizer.sep_token
        # for i in range(len(evidences)):
        #     evid = evidences[i]
        #     evid_title, evid_sent = evid[0], evid[2]
        #     inter_sents += process_wiki_title(evid_title) + self.tokenizer.unk_token + evid_sent + self.tokenizer.sep_token
        # inter_sents_tokens = self.tokenizer.tokenize(inter_sents)
        inter_sents_tokens = []
        inter_sents_tokens.append(self.tokenizer.tokenize(claim))
        inter_sents_tokens.append([self.tokenizer.sep_token])
        for i in range(len(evidences)):
            evid = evidences[i]
            evid_title, evid_sent = evid[0], evid[2]
            evid_tokens = self.tokenizer.tokenize(process_wiki_title(evid_title) + self.tokenizer.unk_token + evid_sent)
            inter_sents_tokens.append(evid_tokens)
            inter_sents_tokens.append([self.tokenizer.sep_token])
        inter_sents_tokens = self.shorten_sentences(inter_sents_tokens, self.max_inter_len)
        concat_inter_sent_tokens = []
        for item in inter_sents_tokens:
            concat_inter_sent_tokens.extend(item)
        inter_sents_tokens = concat_inter_sent_tokens

        SEP_indices = [index for index, value in enumerate(inter_sents_tokens) if value == self.tokenizer.sep_token]
        # print("inter:\n", inter_sents)
        # print("inter:\n", len(inter_sents_tokens))

        inter_sents_tokens_masks = []
        s = 0
        prefix_len = 0
        token_len = len(inter_sents_tokens)
        for i in range(len(SEP_indices)):
            if i > 0:
                s = SEP_indices[i - 1] + 1
            e = SEP_indices[i]
            chosen_len = e - s
            cur_mask = prefix_len * [0] + [1] * chosen_len + [0] * (token_len - prefix_len - chosen_len)
            inter_sents_tokens_masks.append(cur_mask)
            prefix_len = prefix_len + chosen_len + 1
        # origins = [claim] + [(evid[0], evid[2]) for evid in evidences]
        # for i in range(len(origins)):
        #     origin = origins[i]
        #     cur_mask = inter_sents_tokens_masks[i]
        #     cur_tokens = inter_sents_tokens
        #     self.compare_origin_and_chosen_tokens(orgin=origin, tokens=cur_tokens, chosen_mask=cur_mask)
        inter_sents_tokens = self.tokenizer.convert_tokens_to_ids(inter_sents_tokens)
        inter_sents_attention_mask = [1] * len(inter_sents_tokens)
        return inter_sents_tokens, inter_sents_tokens_masks, inter_sents_attention_mask

    def intra_tokens_and_masks(self, claim, evidences, cand_docs):
        intra_sents_tokens_list = []
        intra_sents_tokens_masks = []
        for evid in evidences:
            doc_id, sent_id, sent_text, _ = evid
            prev, post = [], []
            idx = sent_id - 1
            while idx >= 0 and len(prev) < 1:
                text = cand_docs[doc_id][idx]
                if text != "":
                    prev.insert(0, text)
                idx -= 1
            idx = sent_id + 1
            while idx < len(cand_docs[doc_id]) and len(post) < 1:
                text = cand_docs[doc_id][idx]
                if text != "":
                    post.append(text)
                idx += 1
            intra_sents = [
                            self.tokenizer.tokenize(claim), 
                            [self.tokenizer.sep_token],
                            self.tokenizer.tokenize(process_wiki_title(doc_id)),
                            [self.tokenizer.unk_token]
                        ]
            for text in prev:
                intra_sents.append(self.tokenizer.tokenize(text))
                intra_sents.append([self.tokenizer.unk_token])
            intra_sents.append(self.tokenizer.tokenize(sent_text))
            intra_sents.append([self.tokenizer.unk_token])
            for text in post:
                intra_sents.append(self.tokenizer.tokenize(text))
                intra_sents.append([self.tokenizer.unk_token])
            intra_sents = self.shorten_sentences(intra_sents, self.max_intra_len)
            # intra_sents = [process_wiki_title(doc_id)] + prev + [sent_text] + post
            # intra_sents = claim + self.tokenizer.sep_token + self.tokenizer.unk_token.join(intra_sents) + self.tokenizer.unk_token
            # intra_sents_tokens = self.tokenizer.tokenize(intra_sents)
            intra_sents_tokens = []
            for item in intra_sents:
                intra_sents_tokens.extend(item)

            intra_sents_tokens_list.append(intra_sents_tokens)
            UNK_indices = [index for index, value in enumerate(intra_sents_tokens) if value == self.tokenizer.unk_token]
            SEP_indice = intra_sents_tokens.index(self.tokenizer.sep_token)
            sent_id = len(prev)
            # token:    x [SEP] x x x [UNK] x x [UNK] x x  x  x  [UNK] 
            # id:       0   1   2 3 4   5   6 7   8   9 10 11 12  13
            # highlight:_   _   * _ _   *   _ _   _   * _  _  _   *
            title_s, title_e = SEP_indice + 1, UNK_indices[0]
            sent_s, sent_e = UNK_indices[sent_id] + 1, UNK_indices[sent_id + 1]
            title_len = title_e - title_s
            sent_len = sent_e - sent_s
            gap_len = sent_s - title_e
            cur_mask = [0] * title_s + [1] * title_len + [0] * gap_len + [1] * sent_len
            cur_mask += [0] * (len(intra_sents_tokens) - len(cur_mask))
            intra_sents_tokens_masks.append(cur_mask)
            assert len(cur_mask) == len(intra_sents_tokens)
            # self.compare_origin_and_chosen_tokens(orgin=(doc_id, sent_text), tokens=intra_sents_tokens, chosen_mask=cur_mask)
        # print(intra_sents_tokens_list)
        intra_sents_tokens_list = [self.tokenizer.convert_tokens_to_ids(item) for item in intra_sents_tokens_list]
        intra_sents_attention_masks = [[1] * len(item) for item in intra_sents_tokens_list]

        return intra_sents_tokens_list, intra_sents_tokens_masks, intra_sents_attention_masks

    def pad_placeholder_evidence(self, evidences):
        pad_num = self.evid_num - len(evidences)
        for _ in range(pad_num):
            doc_id = self.tokenizer.pad_token
            sent_id = 0
            sent_text = self.tokenizer.pad_token
            sent_label = 0
            evid = [doc_id, sent_id, sent_text, sent_label]
            evidences.append(evid)
        return evidences


    def pad_placeholder_cand_docs(self, cand_docs):
        doc_id = self.tokenizer.pad_token
        sents = [self.tokenizer.pad_token]
        cand_docs[doc_id] = sents
        return cand_docs


    def read_file(self, file_path):
        res = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if self.train:
                lines = lines
            else:
                lines = lines
            for line in lines:
                sample = json.loads(line)
                if 'test' in file_path:
                    sample['label'] = 0
                else:
                    sample['label'] = self.label_map[sample['label']]
                sample['claim'] = process_sent(sample['claim'])
                for evid in sample['evidence']:
                    evid[2] = process_sent(evid[2])
                sample['evidence'] = self.pad_placeholder_evidence(sample['evidence'])
                sample['cand_docs'] = self.pad_placeholder_cand_docs(sample['cand_docs'])
                res.append(sample)
        return res
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sid = sample['id']
        claim = sample['claim']
        label = sample['label']
        if self.train:
            random.shuffle(sample['evidence'])
        # print(json.dumps(sample, indent=4))
        sent_label = [1 if evid[3] == 1 else 0 for evid in sample['evidence']]
        intra_sents_tokens_list, intra_sents_tokens_masks, intra_sents_attention_masks = self.intra_tokens_and_masks(claim, sample['evidence'], sample['cand_docs'])
        # # print("=====" * 20)
        inter_sents_tokens, inter_sents_tokens_masks, inter_sents_attention_mask = self.inter_tokens_and_masks(claim, sample['evidence'])

        sample = {
            "id": sid,
            "claim": claim,
            "label": label,
            "sent_label": sent_label,
            "intra_sents_tokens_list": intra_sents_tokens_list,
            "intra_sents_tokens_masks": intra_sents_tokens_masks,
            "intra_sents_attention_masks": intra_sents_attention_masks,
            "inter_sents_tokens": inter_sents_tokens,
            "inter_sents_tokens_masks": inter_sents_tokens_masks,
            "inter_sents_attention_mask": inter_sents_attention_mask,
        }
        

        return sample

def pad_to_max_len(b_lst, pad, max_length):
    lens = [len(lst) for lst in b_lst]
    assert max_length >= max(lens), "max_length必须大于等于token序列长度"
    # print("max_len: ", max_len, lens)
    for i in range(len(b_lst)):
        b_lst[i] = b_lst[i] +  [pad] * (max_length - len(b_lst[i]))
    return b_lst


def get_ids_by_mask(token_ids, masks):
    res = []
    for token_id, msk in zip(token_ids, masks):
        if msk == 1:
            res.append(token_id)
    return res


def collate(samples, tokenizer, intra_max_len=200, inter_max_len=300):
    if len(samples) == 0:
        return {}
    b_ids = [sample['id'] for sample in samples]
    b_labels = [sample['label'] for sample in samples]
    b_sent_labels = [sample['sent_label'] for sample in samples]

    # 计算token长度
    # intra_lens = [len(sent) for sample in samples for sent in sample['intra_sents_tokens_list']]
    # inter_lens = [len(sample['inter_sents_tokens']) for sample in samples]

    # intra
    b_intra_sents_tokens = []
    for sample in samples:
        b_intra_sents_tokens.extend(sample['intra_sents_tokens_list'])
    b_intra_sents_tokens = pad_to_max_len(b_intra_sents_tokens, tokenizer.pad_token_id, intra_max_len)

    b_intra_sents_tokens_masks = []
    for sample in samples:
        b_intra_sents_tokens_masks.extend(sample['intra_sents_tokens_masks'])
    b_intra_sents_tokens_masks = pad_to_max_len(b_intra_sents_tokens_masks, 0, intra_max_len)

    b_intra_sents_attention_masks = []
    for sample in samples:
        b_intra_sents_attention_masks.extend(sample['intra_sents_attention_masks'])
    b_intra_sents_attention_masks = pad_to_max_len(b_intra_sents_attention_masks, 0, intra_max_len)

    # inter
    b_inter_sents_tokens = [sample['inter_sents_tokens'] for sample in samples]
    b_inter_sents_tokens = pad_to_max_len(b_inter_sents_tokens, tokenizer.pad_token_id, inter_max_len)

    b_inter_sents_tokens_masks = []
    for sample in samples:
        b_inter_sents_tokens_masks.extend(sample['inter_sents_tokens_masks'])
    b_inter_sents_tokens_masks = pad_to_max_len(b_inter_sents_tokens_masks, 0, inter_max_len)

    b_inter_sents_attention_mask = [sample['inter_sents_attention_mask'] for sample in samples]
    b_inter_sents_attention_mask = pad_to_max_len(b_inter_sents_attention_mask, 0, inter_max_len)

    res = {
        "b_ids": b_ids,
        "b_labels": b_labels,
        "b_sent_labels": b_sent_labels,
        "b_intra_sents_tokens": b_intra_sents_tokens,
        "b_intra_sents_tokens_masks": b_intra_sents_tokens_masks,
        "b_intra_sents_attention_masks": b_intra_sents_attention_masks,
        "b_inter_sents_tokens": b_inter_sents_tokens,
        "b_inter_sents_tokens_masks": b_inter_sents_tokens_masks,
        "b_inter_sents_attention_mask": b_inter_sents_attention_mask,
        # "intra_lens": intra_lens,
        # "inter_lens": inter_lens
    }

    # for b_id in range(len(samples)):
    #     intra_tokens = res['b_intra_sents_tokens'][b_id * 5: (b_id + 1) * 5]
    #     token_mask = res['b_intra_sents_tokens_masks'][b_id * 5: (b_id + 1) * 5]
    #     att_mask = res['b_intra_sents_attention_masks'][b_id * 5: (b_id + 1) * 5]
    #     print("## INTRA:")
    #     for i in range(len(intra_tokens)):
    #         print(f"all_token: {i}\n", tokenizer.decode(intra_tokens[i][:200]))
    #         print(f"evid_mask: {i}\n", tokenizer.decode(get_ids_by_mask(intra_tokens[i], token_mask[i])))
    #         print(f"att_mask: {i}\n", tokenizer.decode(get_ids_by_mask(intra_tokens[i], att_mask[i])))

    #     print("******" * 20)

    #     inter_tokens = res['b_inter_sents_tokens'][b_id]
    #     token_mask = res['b_inter_sents_tokens_masks'][b_id * 6: (b_id + 1) * 6]
    #     att_mask = res['b_inter_sents_attention_mask'][b_id]
    #     print("## INTER:")
    #     print(f"all_token: \n", tokenizer.decode(inter_tokens[:300]))
    #     print(f"att_mask: \n", tokenizer.decode(get_ids_by_mask(inter_tokens, att_mask)))
    #     for i in range(len(token_mask)):
    #         print(f"evid_mask: {i}\n", tokenizer.decode(get_ids_by_mask(inter_tokens, token_mask[i])))

    # print("=====" * 20)
    # print("=====" * 20)
    # print("=====" * 20)
    # print("=====" * 20)
    # print("=====" * 20)
    # print("=====" * 20)

    # print(json.dumps(res, indent=4))

    for k, v in res.items():
        res[k] = torch.tensor(v)
        # print(k, res[k].shape)
    return res


def cal_len(dataloader):
    all_intra_lens = []
    all_inter_lens = []
    for batch in tqdm(dataloader):
        intra_lens = batch['intra_lens']
        inter_lens = batch['inter_lens']
        all_intra_lens.extend(intra_lens)
        all_inter_lens.extend(inter_lens)
        # break
    print("max =", max(all_intra_lens), "90_percentile =", np.percentile(all_intra_lens, 90), "99_percentile =", np.percentile(all_intra_lens, 99))
    print("max =", max(all_inter_lens), "90_percentile =", np.percentile(all_inter_lens, 90), "99_percentile =", np.percentile(all_inter_lens, 99))
    return all_intra_lens, all_inter_lens
    


if __name__ == "__main__":
    args = load_args()
    print(args)
    config = AutoConfig.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    train_dataset = FeverDataset(
                        file_path=args.train_path,
                        tokenizer=tokenizer,
                        max_intra_len=args.max_intra_len,
                        max_inter_len=args.max_inter_len,
                        evid_num=args.evid_num,
                        train=True
                        )
    dev_dataset = FeverDataset(
                    file_path=args.valid_path,
                    tokenizer=tokenizer,
                    max_intra_len=args.max_intra_len,
                        max_inter_len=args.max_inter_len,
                    evid_num=args.evid_num,
                    train=False
                    )
    test_dataset = FeverDataset(
                    file_path=args.test_path,
                    tokenizer=tokenizer,
                    max_intra_len=args.max_intra_len,
                    max_inter_len=args.max_inter_len,
                    evid_num=args.evid_num,
                    train=False
                    )
    collate_fc = partial(collate, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fc, pin_memory=True, num_workers=10)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fc, pin_memory=True, num_workers=10)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fc, pin_memory=True, num_workers=10)

    cal_len(train_dataloader)
    cal_len(dev_dataloader)
    cal_len(test_dataloader)
    