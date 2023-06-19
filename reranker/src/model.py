from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from src.utils import norm_text

from src.dataset import (
    Infer_Pairwise_Dataset
)

AUTH_TOKEN = "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"

class Cross_Model(nn.Module):
    def __init__(self, model, tokenizer, max_length=512, droprate=0.1, batch_size=16, device="cpu"):
        super(Cross_Model, self).__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.model = model
        self.device = device
        
        total_layer = len(self.model.encoder.layer)
        num_freeze_layer = int(total_layer/2)
        print(f"freezing {num_freeze_layer} layer")
        
        modules = [self.model.embeddings, self.model.encoder.layer[:num_freeze_layer]]
        
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
                
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(droprate)
        self.fc = nn.Linear(768, 1)
        self.cre = torch.nn.CrossEntropyLoss()

    def forward(self, ids, masks, labels=None, context_masks=None):
        out = self.model(input_ids=ids, attention_mask=masks)
        out = out.last_hidden_state[:, 0]
        
        embedding = self.dropout(out)
        logits = self.fc(embedding)
        if labels is not None:
            logits = logits.view(labels.size(0), labels.size(1))
            return logits, self.loss(labels=labels, logits=logits, context_masks=context_masks)
        
        return logits
    
    def loss(self, labels, logits, context_masks):
        exp = torch.exp(logits)
        exp = torch.masked_fill(input=exp, mask=~context_masks, value=0)
        loss = -torch.log(torch.sum(torch.mul(exp, labels), dim=1) / torch.sum(exp, dim=1))
        loss = torch.mean(loss)
        
        return loss
    
    @torch.no_grad()
    def ranking(self, query, texts):
        tmp = pd.DataFrame()
        tmp["text"] = [norm_text(x) for x in texts]
        tmp["query"] = norm_text(query)
        
        valid_dataset = Infer_Pairwise_Dataset(
            tmp, self.tokenizer, self.max_length)
        
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=valid_dataset.infer_collate_fn,
            num_workers=0, shuffle=False, pin_memory=True)
        preds = []
        with torch.no_grad():
            bar = tqdm(enumerate(valid_loader))
            for step, data in bar:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                preds.append(torch.sigmoid(self(ids, masks)).view(-1))
            preds = torch.concat(preds)
        
        scores = preds.cpu()
        ranks = scores.argsort(descending=True)
        print("model score: ", preds)
        print("rank: ", ranks)
        return scores, ranks