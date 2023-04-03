import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import AutoModel, AutoTokenizer
from utils.data_utils import DualEncoderBatch
from utils.train_utils import log_tensor


class DualEncoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.q_encoder, self.q_tokenizer, self.q_hidden_size = self.get_q_encoder()
        self.p_encoder, self.p_tokenizer, self.p_hidden_size = self.get_p_encoder()
        self.q_proj = nn.Linear(self.q_hidden_size, args.output_size)
        self.p_proj = nn.Linear(self.p_hidden_size, args.output_size)
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.p_proj.weight)

        self.tanh = nn.Tanh()

    def get_q_encoder(self):
        if self.args.q_encoder_type == "hugging_face":
            q_encoder = AutoModel.from_pretrained(self.args.q_encoder_name)
            q_tokenizer = AutoTokenizer.from_pretrained(self.args.q_encoder_name)
            q_hidden_size = q_encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported model type {self.args.q_encoder_type} "
                             f"with model name {self.args.q_encoder_name}")
        return q_encoder, q_tokenizer, q_hidden_size

    def get_p_encoder(self):
        if self.args.p_encoder_type == "hugging_face":
            p_encoder = AutoModel.from_pretrained(self.args.p_encoder_name)
            p_tokenizer = AutoTokenizer.from_pretrained(self.args.p_encoder_name)
            p_hidden_size = p_encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported model type {self.args.p_encoder_type} "
                             f"with model name {self.args.p_encoder_name}")
        return p_encoder, p_tokenizer, p_hidden_size

    def encode_q(self, qp_batch: DualEncoderBatch):
        q_inputs = self.q_tokenizer(
            qp_batch.qs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = q_inputs["input_ids"].to(self.device)
        attention_mask = q_inputs["attention_mask"].to(self.device)

        q_encodings = self.q_encoder(                                   # (b, t, h)
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        q_mask = attention_mask.unsqueeze(-1)                           # (b, t) -> (b, t, 1)
        q_encodings = q_encodings * q_mask
        q_encodings = q_encodings.sum(dim=1)                            # (b, t, h) -> (b, h)

        if self.args.q_pool_type == "sum":
            pass
        elif self.args.q_pool_type == "mean":
            q_lens = attention_mask.sum(dim=1)                          # (b, t) -> b
            q_encodings = q_encodings / q_lens.unsqueeze(-1)

        q_encodings = self.tanh(self.q_proj(q_encodings))               # (b, h_proj)

        return q_encodings

    def encode_p(self, qp_batch: DualEncoderBatch):
        p_inputs = self.p_tokenizer(
            qp_batch.ps,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = p_inputs["input_ids"].to(self.device)
        attention_mask = p_inputs["attention_mask"].to(self.device)

        # log_tensor(input_ids, "input_ids", shape_only=True)
        # log_tensor(attention_mask, "attention_mask", shape_only=True)

        p_encodings = self.p_encoder(                                   # (b, t, h)
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        p_mask = attention_mask.unsqueeze(-1)                           # (b, t) -> (b, t, 1)
        p_encodings = p_encodings * p_mask
        p_encodings = p_encodings.sum(dim=1)                            # (b, t, h) -> (b, h)

        if self.args.p_pool_type == "sum":
            pass
        elif self.args.p_pool_type == "mean":
            p_lens = attention_mask.sum(dim=1)                          # (b, t) -> b
            p_encodings = p_encodings / p_lens.unsqueeze(-1)

        p_encodings = self.tanh(self.p_proj(p_encodings))               # (b, h_proj)

        return p_encodings

    def forward(self, qp_batch: DualEncoderBatch, mode="train"):
        if mode == "train":
            batch_size = self.args.train_batch_size
        elif mode == "val":
            batch_size = self.args.val_batch_size
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        q = self.encode_q(qp_batch)
        p = self.encode_p(qp_batch)

        s_pos = (q * p).sum(1)                                          # (b, h_proj) -> b
        q_exp = q.unsqueeze(1).expand(-1, batch_size, -1)               # (b, h_proj) -> (b, b, h_proj)
        p_exp = p.unsqueeze(0)                                          # (b, h_proj) -> (1, b, h_proj)
        s_pos_neg = q_exp * p_exp
        s_pos_neg = s_pos_neg.sum(-1)                                   # (b, b, h_proj) -> (b, b)

        losses = -(s_pos - torch.logsumexp(s_pos_neg, dim=-1))
        loss = losses.mean()

        accs = torch.as_tensor(s_pos.unsqueeze(1) >= s_pos_neg)
        accs = torch.all(accs, dim=1)
        acc = accs.float().mean()

        return loss, acc
