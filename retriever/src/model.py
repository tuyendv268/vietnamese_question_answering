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

from src.dataset import QA_Dataset
from src.dataset import (
    Infer_Dual_Dataset,
    Infer_Pairwise_Dataset
)

class Dual_Model(nn.Module):
    def __init__(self, model, tokenizer, max_length=384, droprate=0.1, batch_size=16, device="cpu"):
        super(Dual_Model, self).__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        
        self.model = model
        
        total_layer = len(self.model.encoder.layer)
        num_freeze_layer = int(2*total_layer/3)
        print(f"freezing {num_freeze_layer} layer")
        modules = [self.model.embeddings, self.model.encoder.layer[:num_freeze_layer]]
        
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
                
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(droprate)
        
    def forward(self, ids, masks):
        output = self.model(input_ids=ids, attention_mask=masks)
        
        output = output.last_hidden_state[:, 0]
        output = self.dropout(output)
        
        return output
    
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