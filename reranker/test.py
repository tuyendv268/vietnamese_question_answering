import os
from omegaconf import OmegaConf
from importlib.machinery import SourceFileLoader
from tqdm import tqdm
import numpy as np
import json

import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel
from transformers import AutoModel
from transformers import AutoTokenizer


from src.dataset import (
    QA_Dataset
    )

from src.model import Cross_Model
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model_and_tokenizer(config):
    AUTH_TOKEN = "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"
    
    if config.general.plm == "envibert":
        tokenizer = SourceFileLoader(
            "envibert.tokenizer", 
            os.path.join(config.path.pretrained_dir,'envibert_tokenizer.py')) \
                .load_module().RobertaTokenizer(config.path.pretrained_dir)
        plm = RobertaModel.from_pretrained(config.path.pretrained_dir)
        
    elif config.general.plm == "xlm-roberta-base":
        tokenizer = AutoTokenizer.from_pretrained(
            'xlm-roberta-base', cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)
        plm = AutoModel.from_pretrained(
            "xlm-roberta-base", cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)
        
    elif config.general.plm == "vi-mrc-base":
        tokenizer = AutoTokenizer.from_pretrained(
            'nguyenvulebinh/vi-mrc-base', cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)
        plm = AutoModel.from_pretrained(
            "nguyenvulebinh/vi-mrc-base", cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)
            
    model = Cross_Model(
        max_length=config.general.max_length, 
        batch_size=config.general.batch_size,
        device=config.general.device,
        tokenizer=tokenizer, model=plm).to(device)
    return model, tokenizer

def prepare_dataloader(config, tokenizer):
    dataset = QA_Dataset(
        config.path.test_data, mode="val",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    
    loader = DataLoader(
        dataset, batch_size=config.general.batch_size, 
        collate_fn=dataset.cross_collate_fn, 
        num_workers=0, shuffle=False, pin_memory=False, drop_last=False)
    
    return loader
        
def load(path, model, optimizer=None):
    state_dict = torch.load(path, map_location="cpu")
    
    model_state_dict = state_dict["model"]
    optimizer_state_dict = state_dict["optimizer"]
    
    model.load_state_dict(model_state_dict)
    print(f"loaded model and optimizer state dict from {path}")
    
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state_dict)    
    return model, optimizer

@torch.no_grad()
def test(config):
    model, tokenizer = init_model_and_tokenizer(config)
    loader = prepare_dataloader(config=config, tokenizer=tokenizer)
    
    if os.path.exists(config.path.warm_up):
        model, _ = load(path=config.path.warm_up, model=model)
    model.eval()
    mrrs = []
    tqdm_loader = tqdm(enumerate(loader), total=len(loader))
    for index, batch in tqdm_loader:
        inputs_ids = batch["inputs_ids"].to(device)
        masks = batch["masks"].to(device)
        labels = batch["labels"].to(device)
        context_masks = batch["context_masks"].to(device)
        
        logits, loss = model(
            ids=inputs_ids, 
            context_masks=context_masks,
            masks=masks, 
            labels=labels)

        y_pred = torch.softmax(logits, dim=0).squeeze(1)
        y_true = labels

        pair = [[label, pred] for label, pred in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
        mrrs += pair  
        
    mrrs = list(map(calculate_mrr, mrrs))
    mrr = np.array(mrrs).mean()
    
    print("mrr_test: ", mrr)

def calculate_mrr(pair):
    return mrr_score(*pair)

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

if __name__ == "__main__":
    path = "config.yaml"
    config = OmegaConf.load(path)
    test(config)