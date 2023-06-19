import numpy as np
from importlib.machinery import SourceFileLoader
import os
import random
from pandarallel import pandarallel
from typing import List

import torch
from torch.utils.data import Dataset

class HLTAR_Dataset(Dataset):
    def __init__(self, data, max_doc):
        self.data_path = data
        self.max_doc = max_doc
                    
    def __len__(self):
        return len(self.data_path)
    
    def _load_data(self, path):
        return np.load(path, allow_pickle=True).item()

    def __getitem__(self, index):
        sample = self._load_data(self.data_path[index])
        
        return {
            'embedding': sample["embedding"],
            "label": sample["label"],
            "retrieval_rank": sample["retrieval_rank"],
            "reranking_rank": sample["reranking_rank"],
        }
                
    def collate_fn(self, batch):
        embeddings = [torch.tensor(x["embedding"]) for x in batch]
        labels = [torch.tensor(x["label"]) for x in batch]
        max_len = max([len(x) for x in embeddings])
        retrieval_ranks = [torch.tensor(x["retrieval_rank"]) for x in batch]
        reranking_ranks = [torch.tensor(x["reranking_rank"]) for x in batch]
        
        masks = []
        for i in range(len(embeddings)):
            mask = [1]*len(embeddings[i]) + [0]*(max_len-len(embeddings[i]))
            masks.append(torch.tensor(mask))
            if len(embeddings[i]) < max_len:
                labels[i] = torch.cat((labels[i], torch.zeros(max_len-labels[i].shape[0])))
            
            retrieval_ranks[i] += 1
            if len(retrieval_ranks[i]) < max_len:
                retrieval_ranks[i] = torch.cat((retrieval_ranks[i], torch.zeros(max_len-retrieval_ranks[i].shape[0])))
            
            reranking_ranks[i] += 1
            if len(reranking_ranks[i]) < max_len:
                reranking_ranks[i] = torch.cat((reranking_ranks[i], torch.zeros(max_len-reranking_ranks[i].shape[0])))

            if len(embeddings[i]) < max_len:
                embeddings[i] = torch.cat((embeddings[i], torch.zeros((max_len-embeddings[i].shape[0], embeddings[i].shape[1]))))
        
        outputs = {
            "embeddings": torch.stack(embeddings, dim=0),
            "masks": torch.vstack(masks),
            "labels":torch.stack(labels, dim=0),
            "retrieval_ranks":torch.stack(retrieval_ranks, dim=0).long(),
            "reranking_ranks":torch.stack(reranking_ranks, dim=0).long()
        }
        return outputs
        
class Infer_QA_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.questions = tokenizer.batch_encode_plus(
            list(df.question.values), max_length=max_length, truncation=True)[
            "input_ids"]
        self.contexts = tokenizer.batch_encode_plus(
            list(df.text.values), max_length=max_length, truncation=True)[
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

class Infer_Dual_QA_Dataset(Dataset):
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

class QA_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, batch_size=32, mode="train"):
        self.df = df
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
    
    def _convert_text_to_ids(self, text):
        ids = self.tokenizer.batch_encode_plus(text, max_length=self.max_length, 
                                            truncation=True)["input_ids"]
        return ids
    
    def __len__(self):
        return self.df.shape[0]
    
    def _parse_sample(self, positive_sample: str, negative_samples: List[str], question: str):   
        if len(negative_samples) <= 1:
            positive_index = 0
        else:
            positive_index = random.randint(0, len(negative_samples) - 1)
        contexts = negative_samples.copy()
        contexts.insert(positive_index, positive_sample)
        
        # contexts_ids = list(map(lambda context: self.text2ids[context], contexts))
        contexts_ids = self._convert_text_to_ids(contexts)
        contexts_ids = [torch.tensor(context) for context in contexts_ids]
        question_ids = self.tokenizer.encode_plus(question, max_length=self.max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        
        if self.mode == "train":
            num_mask = 4 if len(question_ids) > 16 else 2
            for _ in range(num_mask):
                mask_idx = random.randint(1, len(question_ids) - 1)
                question_ids[mask_idx] = self.tokenizer.mask_token_id
            
            # print(self.tokenizer.decode(question_ids))
            num_mask = 12 if len(question_ids) > 96 else 6
            for index in range(len(contexts_ids)):
                for _ in range(num_mask):
                    mask_idx = random.randint(1, len(contexts_ids[index]) - 1)
                    contexts_ids[index][mask_idx] = self.tokenizer.mask_token_id
                    # print(self.tokenizer.decode(contexts_ids[index]))
                
        return {
            "positive_index": positive_index,
            "contexts_ids":contexts_ids,
            "question_ids": question_ids
        }
       
    def __getitem__(self, index):
        positive_sample = self.df.iloc[index]["positive_sample"]
        negative_samples = self.df.iloc[index]["negative_samples"]
        if self.batch_size < len(negative_samples):
            negative_samples = negative_samples[:self.batch_size+1]
            
        question = self.df.iloc[index]["question"]
        
        return self._parse_sample(
            positive_sample=positive_sample,
            negative_samples=negative_samples,
            question=question
        )
        
    def cross_collate_fn(self, batch):  
        ids = [torch.cat([x["question_ids"][:-1], torch.tensor([self.tokenizer.sep_token_id,]), context[1:]]) for x in batch for context in x["contexts_ids"]]
        positive_indexs = [x["positive_index"] for x in batch]
        max_len = np.max([len(x) for x in ids])
        masks = []
        for i in range(len(ids)):
            if len(ids[i]) < max_len:
                ids[i] = torch.cat((ids[i], torch.tensor([self.tokenizer.pad_token_id, ] * (max_len - len(ids[i])), dtype=torch.long)))
            masks.append(ids[i] != self.tokenizer.pad_token_id)
            
        outputs = {
            "ids": torch.vstack(ids).long(),
            "masks": torch.vstack(masks),
            "labels": torch.tensor(positive_indexs)
        }
        
        return outputs

    def dual_collate_fn(self, batch):
        ids = [batch[0]["question_ids"], ] + batch[0]["contexts_ids"]
        # print("dataset decode 1: ", tokenizer.decode(ids[0]))
        positive_indexs = batch[0]["positive_index"]
        
        max_len = np.max([len(x) for x in ids])
        masks = []
        for i in range(len(ids)):
            if len(ids[i]) < max_len:
                ids[i] = torch.cat((ids[i], torch.tensor([self.tokenizer.pad_token_id, ] * (max_len - len(ids[i])), dtype=torch.long)))
            masks.append(ids[i] != self.tokenizer.pad_token_id)
            
        # print("dataset decode 2: ", tokenizer.decode(ids[0]))
        
        sample = {
            "labels": torch.tensor(positive_indexs),
            "ids": torch.vstack(ids),
            "masks":torch.vstack(masks)
        }
            
        return sample