from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import os

import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers import AutoModel, AutoConfig
from transformers import AutoTokenizer

from src.dataset import (
    Infer_QA_Dataset
)

AUTH_TOKEN = "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"

class Cross_Model(nn.Module):
    def __init__(self,max_length=352, batch_size=16, device="cpu", model_name="xlmr"):
        super(Cross_Model, self).__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        
        if model_name == "envibert" :
            pretrained='../pretrained'
            self.tokenizer = SourceFileLoader(
                "envibert.tokenizer", 
                os.path.join(pretrained,'envibert_tokenizer.py')) \
                    .load_module().RobertaTokenizer(pretrained)
            self.model = RobertaModel.from_pretrained(pretrained)
            
        elif model_name == "xlmr":
            self.tokenizer = AutoTokenizer.from_pretrained('nguyenvulebinh/vi-mrc-base', cache_dir="pretrained", use_auth_token=AUTH_TOKEN)

            self.model = AutoModel.from_pretrained(
                "nguyenvulebinh/vi-mrc-base", cache_dir="pretrained", use_auth_token=AUTH_TOKEN)

            modules = [self.model.embeddings, self.model.encoder.layer[:6]]
            
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1).to(self.device)

    def forward(self, ids, masks):
        # out = self.model(input_ids=ids,
        #                  attention_mask=masks)
        # out = out.last_hidden_state[:, 0]
        out = torch.randn((ids.shape[0], 768))
            
        embedding = self.dropout(out)
        score = self.fc(embedding)
        return score, embedding
    
    @torch.no_grad()
    def ranking(self, question, texts):
        df = texts.rename(columns={"passage_text":"text"})
        df["question"] = question
        valid_dataset = Infer_QA_Dataset(
            df, self.tokenizer, self.max_length)

        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=valid_dataset.infer_collate_fn,
            num_workers=0, shuffle=False, pin_memory=True)
        scores = []
        embeddings = []
        with torch.no_grad():
            bar = enumerate(tqdm(valid_loader))
            for step, data in bar:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                score, embedding = self(ids, masks)
                
                embeddings.append(embedding)
                scores.append(score)
                
            scores = torch.concat(scores).squeeze(-1)
            embeddings = torch.concat(embeddings, dim=0)
            ranks = scores.argsort(descending=True)
        df["reranking_rank"] = [rank.item() for rank in ranks]
        df["embedding"] = [embedding.numpy() for embedding in embeddings.cpu()]
        df["reranking_score"] = [score.numpy() for score in scores.cpu()]
        df["reranking_rank"] = [score.numpy() for score in scores.cpu().argsort(descending=True)]
        
        return df

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        
        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        
        self.d_k = d_model
        self.d_v = d_model
        
        self.W_Q = nn.Linear(d_model, d_model * num_attention_heads)
        self.W_K = nn.Linear(d_model, d_model * num_attention_heads)
        self.W_V = nn.Linear(d_model, d_model * num_attention_heads)
        
        self.ffw = nn.Linear(d_model * num_attention_heads, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, attention_masks, K=None, V=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads, self.d_v).transpose(1, 2)
        
        attention_masks = attention_masks.unsqueeze(1).expand(batch_size, Q.size(1), Q.size(1))
        attention_masks = attention_masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
        
        context, attention_weights = ScaleDotProductAttention()(
            q=q_s, k=k_s, v=v_s,
            mask=attention_masks,
        )
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, 
            self.num_attention_heads * self.d_v)
        context = self.ffw(context)
        return context

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(
            d_model, n_head, drop_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffw = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(Q=x, K=x, V=x, attention_masks=s_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x_ = x
        x = self.ffw(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x_)

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.d_model = d_model
        if d_model % 2 == 1:
            d_model += 1
            
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :self.d_model]
        
        return self.dropout(x)

class Transformer_Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(Transformer_Encoder, self).__init__()
        
        
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        

    def forward(self, inputs, attention_masks, device="cpu"):
        outputs = inputs
        
        if attention_masks is None:
            attention_masks = torch.ones(inputs.shape[0:2]).to(device)
                
        for layer in self.layers:
            outputs = layer(outputs, attention_masks)            

        return outputs


class HLATR(nn.Module):
    def __init__(self):
        super(HLATR, self).__init__()        
        self.linear = nn.Linear(768, 256)
        self.retrieval_rank_embedding = nn.Embedding(
            num_embeddings=101, 
            embedding_dim=256,
            padding_idx=0)
        
        self.reranking_rank_embedding = nn.Embedding(
            num_embeddings=101, 
            embedding_dim=256,
            padding_idx=0)
        
        self.model = Transformer_Encoder(
            d_model=256, 
            ffn_hidden=768, 
            n_head=4, 
            n_layers=4, 
            drop_prob=0.1
        )
        self.drop_out = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(256)
        self.cls_head = nn.Linear(256, 1)

    def forward(self, inputs, masks, retrieval_ranks, reranking_ranks):
        retrieval_rank_embedding = self.retrieval_rank_embedding(retrieval_ranks)
        reranking_rank_embedding = self.reranking_rank_embedding(reranking_ranks)
        
        inputs = self.linear(inputs)
        
        inputs = inputs + retrieval_rank_embedding + reranking_rank_embedding
        inputs = self.layer_norm(inputs)
        
        output = self.model(inputs=inputs, attention_masks=masks)
        
        logits = self.cls_head(output)
        logits = logits.squeeze(-1)
        
        return logits
    
if __name__ == "__main__":
    model = HLATR()
    inputs = torch.randn(4, 8, 768)
    ranks = torch.randint(0, 7, size=(4, 8))
    masks = torch.ones((4, 8))
    
    output = model(inputs, masks, ranks)
    print(output.shape)