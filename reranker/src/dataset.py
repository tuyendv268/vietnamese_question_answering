import numpy as np
from importlib.machinery import SourceFileLoader
import os
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.utils import norm_text
from transformers import AutoModel, AutoConfig
from src.indexed_datasets import IndexedDataset
import json
    
class Infer_Pairwise_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.questions = tokenizer.batch_encode_plus(
            list(df["query"].values), max_length=max_length, truncation=True)[
            "input_ids"]
        
        self.contexts = tokenizer.batch_encode_plus(
            list(df["text"].values), max_length=max_length, truncation=True)[
            "input_ids"]
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            'ids1': torch.tensor(self.questions[index][:-1], dtype=torch.long),
            'ids2': torch.tensor(self.contexts[index][1:], dtype=torch.long)
        }
                
    def infer_collate_fn(self, batch):
        ids = [torch.cat([x["ids1"],torch.tensor([self.tokenizer.sep_token_id]), x["ids2"]]) for x in batch]
        max_len = np.max([len(x) for x in ids])
        masks = []
        for i in range(len(ids)):
            if len(ids[i]) < max_len:
                ids[i] = torch.cat((ids[i], torch.tensor([self.tokenizer.pad_token_id, ] * (max_len - len(ids[i])), dtype=torch.long)))
            masks.append(ids[i] != self.tokenizer.pad_token_id)
        outputs = {
            "ids": torch.vstack(ids),
            "masks": torch.vstack(masks)
        }
        return outputs

class Infer_Dual_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
            
    def __len__(self):
        return self.df.shape[0]
    
    def _parse_sample(self, text):   
        text_ids = self.tokenizer.encode_plus(
            text, max_length=self.max_length, truncation=True, 
            return_tensors="pt")["input_ids"][0]
        
        return {
            "ids": text_ids
        }
       
    def __getitem__(self, index):
        text = self.df.iloc[index]["text"]
        
        return self._parse_sample(text=text)
    
    def infer_dual_collate_fn(self, batch):
        ids = [sample["ids"] for sample in batch]
        lengths = [sample.shape[0] for sample in ids]
        max_length = max(lengths)    
        masks = []
        
        for i in range(len(ids)):
            if len(ids[i]) < max_length:
                ids[i] = torch.cat((ids[i], torch.tensor([self.tokenizer.pad_token_id, ] * (max_length - len(ids[i])), dtype=torch.long)))
            masks.append(ids[i] != self.tokenizer.pad_token_id)
        
        return {
            "ids":torch.vstack(ids),
            "masks":torch.vstack(masks)
        }
        
class QA_Dataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512, mode="train", mask_percent=0.15):
        self.data = IndexedDataset(path)
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.mask_percent = mask_percent
        self.mode = mode

    def __len__(self):
        return len(self.data)
    
    def _parse_sample(self, query, positive_index, contexts):  
        normed_query = norm_text(query) 
        normed_contexts = [norm_text(text) for text in contexts]
        
        positive_index = torch.tensor(positive_index)
        
        contexts = self.tokenizer(
            normed_contexts, max_length=self.max_length, truncation=True)["input_ids"]
        
        query = self.tokenizer(
            normed_query, max_length=self.max_length, truncation=True)["input_ids"]
        
        return {
            "positive_index": positive_index,
            "contexts":contexts,
            "query": query
        }
       
    def __getitem__(self, index):
        sample = self.data[index]      
        query = sample["query"]
        
        contexts = []
        positive_index = None
        for index, context in enumerate(sample["passages"]):
            if context["is_selected"] == 1:
                assert positive_index is None
                positive_index=index
            contexts.append(context["passage_text"])
                    
        return self._parse_sample(
            query=query,
            positive_index=positive_index,
            contexts=contexts
        )
        
    def cross_collate_fn(self, batch):
        ids = [
            [
                torch.cat([
                    torch.Tensor(sample["query"][:-1]),
                    torch.Tensor([self.tokenizer.sep_token_id]), 
                    torch.Tensor(context[1:])
                    ]) 
                for context in sample["contexts"]
            ]
            for sample in batch 
        ]
        
        labels = torch.tensor([sample["positive_index"] for sample in batch])
        
        max_length = max([len(x) for sample in ids for x in sample])
        max_context = max([len(sample) for sample in ids])
        
        inputs_ids, masks, context_masks = [], [], []
        for sample in ids:
            _temp_ids, _temp_masks = [], []
            for i in range(len(sample)):
                if len(sample[i]) < max_length:
                    sample[i] = torch.cat(
                        (
                            sample[i], 
                            torch.Tensor([self.tokenizer.pad_token_id, ]*(max_length-len(sample[i])))
                            )
                        )
                _temp_ids.append(sample[i])
                _temp_masks.append(sample[i] != self.tokenizer.pad_token_id)
            
            _context_masks = [1]*len(_temp_ids) + [0]*(max_context-len(_temp_ids))
            _temp_masks += [torch.zeros_like(_temp_masks[0])]*(max_context-len(_temp_masks))
            _temp_ids += [self.tokenizer.pad_token_id*torch.ones_like(_temp_ids[0]), ]*(max_context-len(_temp_ids))
            
            masks.append(torch.stack(_temp_masks, dim=0))
            inputs_ids.append(torch.stack(_temp_ids, dim=0))
            context_masks.append(_context_masks)
            
        masks = torch.vstack(masks)
        inputs_ids = torch.vstack(inputs_ids).long()
        labels = torch.nn.functional.one_hot(labels, max_context)
        context_masks = torch.tensor(context_masks, dtype=torch.bool)
        
        return {
            "inputs_ids": inputs_ids,
            "masks": masks,
            "labels": labels,
            "context_masks": context_masks
        }
    
if __name__ == "__main__":                
    tokenizer = SourceFileLoader(
            "envibert.tokenizer", 
            os.path.join("pretrained",'envibert_tokenizer.py')) \
                .load_module().RobertaTokenizer("pretrained")
                
    with open("data/test_hidden_v1.1-translated.json", "r", encoding="utf-8") as f:
        data = f.readlines()

    dataset = QA_Dataset(data=data, tokenizer=tokenizer)
    dl = DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset._collate_fn)
    for i in dl:
        print(i)
        break