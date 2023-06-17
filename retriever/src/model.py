from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import norm_text
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers import AutoModel, AutoConfig
from transformers import AutoTokenizer
from torchmetrics.functional import pairwise_cosine_similarity

from src.dataset import QA_Dataset
from src.dataset import (
    Infer_Dual_Dataset,
    Infer_Pairwise_Dataset
)

class Dual_Model(nn.Module):
    def __init__(self, model, tokenizer, max_length=384, droprate=0.1, device="cpu"):
        super(Dual_Model, self).__init__()
        self.max_length = max_length
        self.device = device
        self.tokenizer = tokenizer
        
        self.model = model
        # self.freeze(n_layer=12)
        self.linear = nn.Linear(768, 768)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def freeze(self, n_layer):
        num_freeze_layer = n_layer
        print(f"freezing {num_freeze_layer} layer")
        modules = [self.model.embeddings, self.model.encoder.layer[:num_freeze_layer]]
        
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        # for name, state_dict in self.model.named_parameters():
        #     print(name, state_dict.requires_grad)
            
    def encode_query(self, query_ids, query_masks):
        output = self.model(input_ids=query_ids, attention_mask=query_masks)
        output = output.last_hidden_state[:, 0]
        output = self.linear(output)
        
        return output
    
    def encode_passage(self, contexts_ids, context_masks):
        output = self.model(input_ids=contexts_ids, attention_mask=context_masks)
        output = output.last_hidden_state[:, 0]
        output = self.linear(output)
        
        return output
    
    def forward(self, contexts_ids, context_masks, query_ids, query_masks, labels, masks):
        query_embeddings = self.encode_query(query_ids=query_ids, query_masks=query_masks)
        context_embeddings = self.encode_passage(contexts_ids=contexts_ids, context_masks=context_masks)
        
        loss, logits, labels = self.dot_product(
            labels=labels,
            query_embeddings=query_embeddings,
            context_embeddings=context_embeddings,
            masks=masks,
            temperature=8
        )
        
        return loss, logits, labels

    def dot_product(self, labels, query_embeddings, context_embeddings, masks, temperature=8):
        batch_size, n_passage_per_query = labels.shape[0], labels.shape[1]
        assert labels[:,0].sum() == batch_size
        
        logits = torch.matmul(query_embeddings, context_embeddings.transpose(0, 1))
        labels = torch.arange(0, batch_size, device=logits.device) * n_passage_per_query
        
        loss = self.contrastive_loss(labels, logits, masks, temperature=temperature) 
        labels = torch.nn.functional.one_hot(labels, num_classes=n_passage_per_query*batch_size)
        
        return loss, logits, labels
    
    def pairwise_cosine(self, labels, query_embeddings, context_embeddings, masks, temperature=1):
        batch_size, n_passage_per_query = labels.shape[0], labels.shape[1]
        assert labels[:,0].sum() == batch_size

        logits = pairwise_cosine_similarity(query_embeddings, context_embeddings)
        labels = torch.arange(0, batch_size, device=logits.device) * n_passage_per_query
        
        loss = self.contrastive_loss(labels, logits, masks, temperature=temperature) 
        labels = torch.nn.functional.one_hot(labels, num_classes=n_passage_per_query*batch_size)
        
        return loss, logits, labels
    
    def contrastive_loss(self, labels, logits, masks, temperature=1):
        logits = logits/temperature
        logits = torch.masked_fill(input=logits, mask=~masks.flatten(), value=-1000)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss

    
    @torch.no_grad()
    def extract_query_embedding(self, ids, masks):
        query = self.model(input_ids=ids, attention_mask=masks,
            output_hidden_states=False).last_hidden_state
        query = query[:, 0].squeeze(0)
        return query
    
    @torch.no_grad()
    def extract_embeddings(self, texts, device="cpu"):    
        self.model.eval()
            
        df = pd.DataFrame()
        df["text"] = texts
        _dataset = Infer_Dual_Dataset(df, self.tokenizer, self.max_length)
        _dl = torch.utils.data.DataLoader(
            _dataset, batch_size=1, 
            shuffle=False, pin_memory=True, drop_last=False,
            collate_fn=_dataset.infer_dual_collate_fn)
        
        bar = tqdm(enumerate(_dl), total=len(_dl))
        embeddings = []
        for _, data in bar:
            ids = data["ids"].to(device)
            masks = data["masks"].to(device)
            
            out = self.model(input_ids=ids, attention_mask=masks,
                output_hidden_states=False).last_hidden_state
            out = out[:, 0]
            embeddings.append(out)
        
        return embeddings