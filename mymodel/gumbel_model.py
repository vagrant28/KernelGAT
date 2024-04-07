
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BaseModel(nn.Module):
    def __init__(self, config=None, args=None):
        super().__init__()
        self.config = config
        self.args = args
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name, attention_type="original_full")
        self.merge_intra_and_inter_proj = nn.Sequential(nn.Linear(768 * 2, 768), nn.ReLU(True))
        self.claim_proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU(True))
        self.propagation_1 = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.propagation_2 = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.veri_proj = nn.Linear(768, 3)
        self.chosen_proj = nn.Linear(768, 2)

        self.placeholder = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(1, 768)))


    def moe_encoding(self, batch):
        pass
        
    def get_claim_and_evid_embs_by_token_avg(self, batch):
        # intra embed
        intra_input_ids = batch['b_intra_sents_tokens']
        intra_attention_masks = batch['b_intra_sents_attention_masks']
        intra_token_masks = batch['b_intra_sents_tokens_masks']
        intra_embs = self.encoder(input_ids=intra_input_ids, attention_mask=intra_attention_masks)[0]
        intra_embs = intra_embs * intra_token_masks.unsqueeze(dim=-1)
        intra_embs = torch.mean(intra_embs, dim=1)
        intra_embs = intra_embs.view(-1, 5, 768)    # [bz, evid_num, hidden_dim]

        # inter embed
        inter_input_ids = batch['b_inter_sents_tokens']
        inter_attention_mask = batch['b_inter_sents_attention_mask']
        inter_token_masks = batch['b_inter_sents_tokens_masks']
        inter_embs = self.encoder(input_ids=inter_input_ids, attention_mask=inter_attention_mask)[0]
        inter_embs = inter_embs.repeat_interleave(repeats=5 + 1, dim=0)
        inter_embs = inter_embs * inter_token_masks.unsqueeze(dim=-1)
        inter_embs = torch.mean(inter_embs, dim=1)
        inter_embs = inter_embs.view(-1, 5 + 1, 768)

        claim_embs = inter_embs[:, 0, :]    # [bz, hidden_dim]
        inter_embs = inter_embs[:, 1:, :]   # [bz, evid_num, hidden_dim]

        return claim_embs, intra_embs, inter_embs
    

    def evid_gate(self, claim_embs, evid_embs):
        evid_embs = evid_embs.view(-1, 768)
        claim_embs = claim_embs.repeat_interleave(repeats=5, dim=0)
        claim_evid_embs = torch.concat([claim_embs, evid_embs], dim=-1)
        evid_chosen_logits = self.evid_gate_proj(claim_evid_embs)
        evid_chosen_logits = F.gumbel_softmax(evid_chosen_logits, tau=1, dim=-1)
        evid_chosen_logits = evid_chosen_logits.view(-1, 5, 2)
        return evid_chosen_logits


    def forward(self, batch, temperature):
        claim_embs, intra_embs, inter_embs = self.get_claim_and_evid_embs_by_token_avg(batch)

        evid_embs = torch.concat([intra_embs, inter_embs], dim=-1)
        evid_embs = self.merge_intra_and_inter_proj(evid_embs)
        claim_embs = self.claim_proj(claim_embs)


        seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), evid_embs], dim=1)
        seq_embs = self.propagation_1(seq_embs)
        claim_embs = seq_embs[:, 0, :]
        evid_embs = seq_embs[:, 1:, :]
        chosen_logits = self.chosen_proj(evid_embs)
        chosen_logits = F.gumbel_softmax(chosen_logits, tau=temperature, dim=-1)
        evid_embs = chosen_logits[:, :, :1] * self.placeholder + chosen_logits[:, :, 1:] * evid_embs
        seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), evid_embs], dim=1)
        seq_embs = self.propagation_2(seq_embs)
        claim_embs = seq_embs[:, 0, :]
        veri_logits = self.veri_proj(claim_embs)

        return {
            "veri_logits": veri_logits, 
            "chosen_logits": chosen_logits
            }
    

class BaseModel_2(nn.Module):
    def __init__(self, config=None, args=None):
        super().__init__()
        self.config = config
        self.args = args
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name, attention_type="original_full")
        self.merge_intra_and_inter_proj = nn.Sequential(nn.Linear(768 * 2, 768), nn.LeakyReLU(True))
        self.claim_proj = nn.Sequential(nn.Linear(768, 768), nn.LeakyReLU(True))
        self.propagation_1 = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.propagation_2 = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.veri_proj = nn.Linear(768, 3)
        self.chosen_proj = nn.Linear(768, 2)

        self.placeholder = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(1, 768)))

        
    def get_claim_and_evid_embs_by_token_avg(self, batch):
        # intra embed
        intra_input_ids = batch['b_intra_sents_tokens']
        intra_attention_masks = batch['b_intra_sents_attention_masks']
        intra_token_masks = batch['b_intra_sents_tokens_masks']
        intra_embs = self.encoder(input_ids=intra_input_ids, attention_mask=intra_attention_masks)[0]
        intra_embs = intra_embs * intra_token_masks.unsqueeze(dim=-1)
        intra_embs = torch.mean(intra_embs, dim=1)
        intra_embs = intra_embs.view(-1, 5, 768)    # [bz, evid_num, hidden_dim]

        # inter embed
        inter_input_ids = batch['b_inter_sents_tokens']
        inter_attention_mask = batch['b_inter_sents_attention_mask']
        inter_token_masks = batch['b_inter_sents_tokens_masks']
        inter_embs = self.encoder(input_ids=inter_input_ids, attention_mask=inter_attention_mask)[0]
        inter_embs = inter_embs.repeat_interleave(repeats=5 + 1, dim=0)
        inter_embs = inter_embs * inter_token_masks.unsqueeze(dim=-1)
        inter_embs = torch.mean(inter_embs, dim=1)
        inter_embs = inter_embs.view(-1, 5 + 1, 768)

        claim_embs = inter_embs[:, 0, :]    # [bz, hidden_dim]
        inter_embs = inter_embs[:, 1:, :]   # [bz, evid_num, hidden_dim]

        return claim_embs, intra_embs, inter_embs
    

    def evid_gate(self, claim_embs, evid_embs):
        evid_embs = evid_embs.view(-1, 768)
        claim_embs = claim_embs.repeat_interleave(repeats=5, dim=0)
        claim_evid_embs = torch.concat([claim_embs, evid_embs], dim=-1)
        evid_chosen_logits = self.evid_gate_proj(claim_evid_embs)
        evid_chosen_logits = F.gumbel_softmax(evid_chosen_logits, tau=1, dim=-1)
        evid_chosen_logits = evid_chosen_logits.view(-1, 5, 2)
        return evid_chosen_logits


    def forward(self, batch, temperature):
        claim_embs, intra_embs, inter_embs = self.get_claim_and_evid_embs_by_token_avg(batch)

        evid_embs = torch.concat([intra_embs, inter_embs], dim=-1)
        evid_embs = self.merge_intra_and_inter_proj(evid_embs)
        claim_embs = self.claim_proj(claim_embs)


        seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), evid_embs], dim=1)
        seq_embs = self.propagation_1(seq_embs)
        chosen_logits = self.chosen_proj(seq_embs[:, 1:, :])
        chosen_logits = F.gumbel_softmax(chosen_logits, tau=temperature, dim=-1)
        # chosen_logits = F.softmax(chosen_logits, dim=-1)
        evid_embs = chosen_logits[:, :, :1] * self.placeholder + chosen_logits[:, :, 1:] * evid_embs
        seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), evid_embs], dim=1)
        seq_embs = self.propagation_2(seq_embs)
        claim_embs = seq_embs[:, 0, :]
        veri_logits = self.veri_proj(claim_embs)

        return {
            "veri_logits": veri_logits, 
            "chosen_logits": chosen_logits
            }
    

class BaseModel_3(nn.Module):
    def __init__(self, config=None, args=None):
        super().__init__()
        self.config = config
        self.args = args
        # self.encoder = AutoModel.from_pretrained(self.args.encoder_name, attention_type="original_full")
        self.encoder = AutoModel.from_pretrained(self.args.encoder_name)
        self.merge_intra_and_inter_proj = nn.Sequential(nn.Linear(self.config.hidden_size * 2, self.config.hidden_size), nn.LeakyReLU(True))
        self.claim_proj = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.LeakyReLU(True))
        self.propagation_1 = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=self.config.num_attention_heads, batch_first=True)
        self.propagation_2 = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=self.config.num_attention_heads, batch_first=True)
        self.veri_proj = nn.Linear(self.config.hidden_size, 3)
        self.chosen_proj = nn.Linear(self.config.hidden_size, 2)

        self.placeholder = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(1, self.config.hidden_size)))

        
    def get_claim_and_evid_embs_by_token_avg(self, batch):
        # intra embed
        intra_input_ids = batch['b_intra_sents_tokens']
        intra_attention_masks = batch['b_intra_sents_attention_masks']
        intra_token_masks = batch['b_intra_sents_tokens_masks']
        intra_embs = self.encoder(input_ids=intra_input_ids, attention_mask=intra_attention_masks)[0]
        intra_embs = intra_embs * intra_token_masks.unsqueeze(dim=-1)
        intra_embs = torch.mean(intra_embs, dim=1)
        intra_embs = intra_embs.view(-1, 5, self.config.hidden_size)    # [bz, evid_num, hidden_dim]

        # inter embed
        inter_input_ids = batch['b_inter_sents_tokens']
        inter_attention_mask = batch['b_inter_sents_attention_mask']
        inter_token_masks = batch['b_inter_sents_tokens_masks']
        inter_embs = self.encoder(input_ids=inter_input_ids, attention_mask=inter_attention_mask)[0]
        inter_embs = inter_embs.repeat_interleave(repeats=5 + 1, dim=0)
        inter_embs = inter_embs * inter_token_masks.unsqueeze(dim=-1)
        inter_embs = torch.mean(inter_embs, dim=1)
        inter_embs = inter_embs.view(-1, 5 + 1, self.config.hidden_size)

        claim_embs = inter_embs[:, 0, :]    # [bz, hidden_dim]
        inter_embs = inter_embs[:, 1:, :]   # [bz, evid_num, hidden_dim]

        return claim_embs, intra_embs, inter_embs
    

    def evid_gate(self, claim_embs, evid_embs):
        evid_embs = evid_embs.view(-1, self.config.hidden_size)
        claim_embs = claim_embs.repeat_interleave(repeats=5, dim=0)
        claim_evid_embs = torch.concat([claim_embs, evid_embs], dim=-1)
        evid_chosen_logits = self.evid_gate_proj(claim_evid_embs)
        evid_chosen_logits = F.gumbel_softmax(evid_chosen_logits, tau=1, dim=-1)
        evid_chosen_logits = evid_chosen_logits.view(-1, 5, 2)
        return evid_chosen_logits


    def forward(self, batch, temperature):
        claim_embs, intra_embs, inter_embs = self.get_claim_and_evid_embs_by_token_avg(batch)

        evid_embs = torch.concat([intra_embs, inter_embs], dim=-1)
        evid_embs = self.merge_intra_and_inter_proj(evid_embs)
        claim_embs = self.claim_proj(claim_embs)


        seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), evid_embs], dim=1)
        seq_embs = self.propagation_1(seq_embs)
        chosen_logits = self.chosen_proj(seq_embs[:, 1:, :])
        chosen_logits = F.gumbel_softmax(chosen_logits, tau=temperature, dim=-1, hard=False)

        sent_label = batch['b_sent_labels']
        ground_chosen_logits = torch.zeros(sent_label.size(0), sent_label.size(1), 2).cuda()
        ground_chosen_logits[..., 0] = 1 - sent_label  # 在第一个通道上，将原来为1的位置变为0
        ground_chosen_logits[..., 1] = sent_label      # 在第二个通道上，将原来为1的位置标记为1

        pred_evid_embs = chosen_logits[:, :, :1] * self.placeholder + chosen_logits[:, :, 1:] * evid_embs
        ground_evid_embs = ground_chosen_logits[:, :, :1] * self.placeholder + ground_chosen_logits[:, :, 1:] * evid_embs

        pred_seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), pred_evid_embs], dim=1)
        ground_seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), ground_evid_embs], dim=1)

        pred_seq_embs = self.propagation_2(pred_seq_embs)
        ground_seq_embs = self.propagation_2(ground_seq_embs)
        pred_claim_embs = pred_seq_embs[:, 0, :]
        ground_claim_embs = ground_seq_embs[:, 0, :]
        pred_veri_logits = self.veri_proj(pred_claim_embs)
        ground_veri_logits = self.veri_proj(ground_claim_embs)

        return {
            "pred_veri_logits": pred_veri_logits,
            "ground_veri_logits": ground_veri_logits, 
            "chosen_logits": chosen_logits
            }
    

    def infer(self, batch, temperature, hard_gumbel=False):
        claim_embs, intra_embs, inter_embs = self.get_claim_and_evid_embs_by_token_avg(batch)

        evid_embs = torch.concat([intra_embs, inter_embs], dim=-1)
        evid_embs = self.merge_intra_and_inter_proj(evid_embs)
        claim_embs = self.claim_proj(claim_embs)


        seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), evid_embs], dim=1)
        seq_embs = self.propagation_1(seq_embs)
        chosen_logits = self.chosen_proj(seq_embs[:, 1:, :])
        chosen_logits = F.gumbel_softmax(chosen_logits, tau=temperature, dim=-1, hard=hard_gumbel)

        pred_evid_embs = chosen_logits[:, :, :1] * self.placeholder + chosen_logits[:, :, 1:] * evid_embs

        pred_seq_embs = torch.concat([claim_embs.unsqueeze(dim=1), pred_evid_embs], dim=1)

        pred_seq_embs = self.propagation_2(pred_seq_embs)
        pred_claim_embs = pred_seq_embs[:, 0, :]
        pred_veri_logits = self.veri_proj(pred_claim_embs)

        return {
            "pred_veri_logits": pred_veri_logits,
            "chosen_logits": chosen_logits
            }