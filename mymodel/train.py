from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from functools import partial
from gumbel_dataset import FeverDataset, collate
from gumbel_model import BaseModel, BaseModel_2, BaseModel_3
import argparse
import random
import numpy as np
from apex import amp
from datetime import date
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evid_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--max_intra_len", default=200, type=int)
    parser.add_argument("--max_inter_len", default=300, type=int)
    parser.add_argument("--eval_step", default=500, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
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
    parser.add_argument("--encoder_name", type=str, default="test")
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    args = parser.parse_args()

    return args



def eval_model(model, dataloader, temperature):
    model.eval()
    all_veri_label = []
    all_chosen_label = []
    all_veri_pred = []
    all_chosen_pred = []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.cuda()
        veri_label = batch["b_labels"]
        chosen_label = batch["b_sent_labels"].view(-1)

        outputs = model(batch, temperature)
        pred_veri_logits = outputs["pred_veri_logits"]
        chosen_logits = outputs["chosen_logits"].view(-1, 2)

        veri_pred = torch.argmax(pred_veri_logits, dim=-1)
        chosen_pred = torch.argmax(chosen_logits, dim=-1)

        all_veri_label.extend(veri_label.tolist())
        all_chosen_label.extend(chosen_label.tolist())
        all_veri_pred.extend(veri_pred.tolist())
        all_chosen_pred.extend(chosen_pred.tolist())

    veri_acc = accuracy_score(all_veri_label, all_veri_pred)
    chosen_acc = accuracy_score(all_chosen_label, all_chosen_pred)
    chosen_pr, chosen_re, chosen_f1, _ = precision_recall_fscore_support(all_chosen_label, all_chosen_pred, zero_division=np.nan)
    chosen_pr = chosen_pr[1]
    chosen_re = chosen_re[1]
    chosen_f1 = chosen_f1[1]

    veri_class_report = classification_report(all_veri_label, all_veri_pred, target_names=['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'], zero_division=np.nan)
    veri_confusion_mat = confusion_matrix(all_veri_label, all_veri_pred)

    print(f"Eval chosen precion={chosen_pr}  recall={chosen_re}  f1={chosen_f1}")
    print("Eval veri class report:\n", veri_class_report)
    print("Eval veri confusion mat:\n", veri_confusion_mat)


    model.train()

    return {
        "eval_veri_acc": veri_acc,
        "eval_chosen_acc": chosen_acc,
        "eval_chosen_precison": chosen_pr,
        "eval_chosen_recall": chosen_re,
        "eval_chosen_f1": chosen_f1
    }


if __name__ == '__main__':
    # 读取命令行参数
    args = load_args()

    # # 设置存储路径
    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-epoch{args.num_train_epochs}-bsz{args.train_batch_size}-lr{args.learning_rate}"
    args.outdir = os.path.join(args.outdir, date_curr, model_name)
    tb_logger = SummaryWriter(os.path.join(args.outdir.replace("logs","tflogs")))

    # # 设置日志格式
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    # # 设置随机数种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 加载config, tokenizer, model
    config = AutoConfig.from_pretrained(args.encoder_name)
    print(config)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    model = BaseModel_3(config=config, args=args)
    logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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
    logger.info(f"Num of train dataset: {len(train_dataset)}")
    logger.info(f"Num of dev dataset: {len(dev_dataset)}")
    collate_fc = partial(collate, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fc, pin_memory=True, num_workers=10)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=collate_fc, pin_memory=True, num_workers=10)
    logger.info(f"Num of train batches: {len(train_dataloader)}")
    logger.info(f"Num of dev batches: {len(dev_dataloader)}")

    model.cuda()
    
    encoder_named_paras = [(n, p) for n, p in model.named_parameters() if 'encoder' in n]
    other_named_paras = [(n, p) for n, p in model.named_parameters() if 'encoder' not in n]
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in encoder_named_paras if not any(nd in n for nd in no_decay)], 
            'weight_decay': args.weight_decay,
            'lr': args.learning_rate
        },
        {
            'params': [p for n, p in encoder_named_paras if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0,
            'lr': args.learning_rate
        },
        {
            'params': [p for n, p in other_named_paras if not any(nd in n for nd in no_decay)], 
            'weight_decay': args.weight_decay,
            'lr': args.learning_rate * 10
        },
        {
            'params': [p for n, p in other_named_paras if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0,
            'lr': args.learning_rate * 10
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    decrease_steps = total_steps * 0.4
    logger.info(f"total steps: {total_steps}")
    assert decrease_steps >= 7000, "temperature下降的步数必须大于等于7000，请调大epoch或者调小train_batch"
    warmup_steps = total_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)



    model.train()
    logger.info('Start training!')
    global_step = 0
    best_accuracy = 0.0
    tau0 = 1.0  # init temperature
    temperature = tau0
    ANNEAL_RATE=0.0001
    MIN_TEMP=0.5
    for epoch in range(int(args.num_train_epochs)):
        train_veri_meter = AverageMeter()
        train_chosen_meter = AverageMeter()
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            for k, v in batch.items():
                batch[k] = v.cuda()
            veri_label = batch["b_labels"]
            sent_label = batch["b_sent_labels"]
            sent_mask = sent_label == 1
            one_num = torch.sum(sent_label, dim=-1) * 2
            one_num[one_num == 0] = 2

            ascend_sorted_indices = torch.argsort(sent_label, dim=-1, descending=False)
            for i in range(veri_label.shape[0]):
                selected_indices = ascend_sorted_indices[i, :one_num[i]]
                sent_mask[i, selected_indices] = True
            # print("sent_label:\n", sent_label)
            # print("sent_mask:\n", sent_mask)
            sent_label = sent_label.view(-1)
            sent_mask = sent_mask.view(-1)
            # print("sent_label:\n", sent_label)
            # print("sent_mask:\n", sent_mask)

            


            outputs = model(batch, temperature)
            pred_chosen = torch.argmax(outputs["chosen_logits"], dim=-1)
            pred_chosen[batch["b_sent_labels"] == 0] = 0
            pred_chosen_num = torch.sum(pred_chosen, dim=-1)
            pred_veri_label = veri_label.clone()
            pred_veri_label[pred_chosen_num == 0] = 2


            pred_veri_logits = outputs["pred_veri_logits"]
            ground_veri_logits = outputs["ground_veri_logits"]
            chosen_logits = outputs["chosen_logits"].view(-1, 2)
            pred_veri_loss = F.cross_entropy(pred_veri_logits, pred_veri_label)
            ground_veri_loss = F.cross_entropy(ground_veri_logits, veri_label)
            chosen_loss = F.nll_loss(torch.log(chosen_logits[sent_mask]), sent_label[sent_mask], reduction='mean')
            loss = pred_veri_loss + chosen_loss + ground_veri_loss
            # print(f"## batch_idx={batch_idx}, loss = veri + chosen = {veri_loss} + {chosen_loss} = {loss}")
            # print(f"## batch_idx={batch_idx}, ground={sent_label}")
            # print(f"## batch_idx={batch_idx}, pred={torch.argmax(chosen_logits, dim=-1)}")
            # print(f"## batch_idx={batch_idx}, ground={veri_label}")
            # print(f"## batch_idx={batch_idx}, pred={torch.argmax(veri_logits, dim=-1)}")
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            train_veri_correct_num = torch.sum(torch.argmax(pred_veri_logits, dim=-1) == veri_label).item()
            train_chosen_correct_num = torch.sum(torch.argmax(chosen_logits, dim=-1) == sent_label).item()
            train_veri_meter.update(train_veri_correct_num)
            train_chosen_meter.update(train_chosen_correct_num)

            if (batch_idx > 0 and batch_idx % args.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                tb_logger.add_scalar('train_loss', loss.item(), global_step)
                tb_logger.add_scalar('train_pred_veri_loss', pred_veri_loss.item(), global_step)
                tb_logger.add_scalar('train_ground_veri_loss', ground_veri_loss.item(), global_step)
                tb_logger.add_scalar('train_chosen_loss', chosen_loss.item(), global_step)
                accumulate_batch_num = args.gradient_accumulation_steps * args.train_batch_size
                tb_logger.add_scalar('train_veri_acc', train_veri_meter.sum / accumulate_batch_num, global_step)
                tb_logger.add_scalar('train_chosen_acc', train_chosen_meter.sum / (accumulate_batch_num * 5), global_step)
                # logger.info(f"global step:{global_step}, train loss = veri + chosen = {veri_loss.item()} + {chosen_loss.item()} = {veri_loss.item() + chosen_loss.item()}")
                # logger.info(f"global step:{global_step}, train veri acc: {train_veri_meter.sum} / {accumulate_batch_num} = {train_veri_meter.sum / accumulate_batch_num}")
                # logger.info(f"global step:{global_step}, train chosen acc: {train_chosen_meter.sum} / {accumulate_batch_num * 5} = {train_chosen_meter.sum / (accumulate_batch_num * 5)}")
                train_veri_meter.reset()
                train_chosen_meter.reset()

                if global_step > 0 and global_step % 500 == 0:
                    temperature = np.maximum(tau0 * np.exp(- ANNEAL_RATE * global_step), MIN_TEMP)
                    tb_logger.add_scalar('temperature', temperature, global_step)
                    # logger.info(f"Epoch:{epoch}\tglobal_step:{global_step}\ttemperature:{temperature}")
            
            if (batch_idx > 0 and batch_idx % (args.eval_step * args.gradient_accumulation_steps) == 0) or (batch_idx + 1 == len(train_dataloader)):
                logger.info('Start eval!')
                with torch.no_grad():
                    eval_outputs = eval_model(model, dev_dataloader, temperature)
                    for k, v in eval_outputs.items():
                        tb_logger.add_scalar(k, v, global_step)
                        logger.info(f"Epoch={epoch}, gloabl step: {global_step}\t{k} : {v}\ttempature: {temperature}")
                    veri_accuracy = eval_outputs['eval_veri_acc']
                    if veri_accuracy > best_accuracy:
                        best_accuracy = veri_accuracy

                        torch.save(model.state_dict(), os.path.join(args.outdir, f"checkpoint_best_{global_step}.pt"))
                        logger.info("Saved best global step {0}, best accuracy {1}".format(global_step, best_accuracy))
            


            # break

    logger.info(f"best eval veri acc: {best_accuracy}")
    logger.info("Training finished!")