import os
from omegaconf import OmegaConf
from importlib.machinery import SourceFileLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel
from transformers import AutoModel
from transformers import AutoTokenizer


from src.utils import (
    load_data,
    optimizer_scheduler
    )

from src.dataset import (
    QA_Dataset
    )

from src.model import Cross_Model
from datetime import datetime
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_directories_and_logger(config):
    if not os.path.exists(config.path.ckpt):
        os.makedirs(config.path.ckpt)
        
    if not os.path.exists(config.path.log):
        os.makedirs(config.path.log)
        
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")
    
    log_dir = f"{config.path.log}/{current_time}"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print(f"logging into {log_dir}")
    else:
        raise Exception("current log dir is exist !!!")

    writer = SummaryWriter(
        log_dir=log_dir
    )
    
    return writer
def init_model_and_tokenizer(config):
    AUTH_TOKEN = "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"
    
    if config.general.plm == "envibert":
        tokenizer = SourceFileLoader(
            "envibert.tokenizer", 
            os.path.join(config.path.pretrained_dir,'envibert_tokenizer.py')) \
                .load_module().RobertaTokenizer(config.path.pretrained_dir)
        plm = RobertaModel.from_pretrained(config.path.pretrained_dir)
    elif config.general.plm == "xlmr":
        tokenizer = AutoTokenizer.from_pretrained(
            'nguyenvulebinh/vi-mrc-base', cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)
        plm = AutoModel.from_pretrained(
            "nguyenvulebinh/vi-mrc-base", cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)
    
    model = Cross_Model(
        max_length=config.general.max_length, 
        batch_size=config.general.batch_size,
        device=config.general.device,
        tokenizer=tokenizer, model=plm).to(device)
    
    if os.path.exists(config.path.warm_up):
        model.load_state_dict(torch.load(config.path.warm_up, map_location="cpu"))
        print(f"load model state dict from {config.path.warm_up}")
        
    return model, tokenizer

def prepare_dataloader(config, tokenizer):
    # train_data = load_data(config.path.train_data)
    test_data = config.path.test_data
    train_data = config.path.train_data
    val_data = config.path.val_data
    
    # train_data, val_data = train_test_split(train_data, test_size=config.general.valid_size, random_state=42)
    
    train_dataset = QA_Dataset(
        train_data, mode="train",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    
    if config.general.model_type =="cross" :
        collate_fn = train_dataset.cross_collate_fn
    elif config.general.model_type =="dual" :
        collate_fn = train_dataset.dual_collate_fn
        
    train_loader = DataLoader(
        train_dataset, batch_size=config.general.batch_size,
        collate_fn=collate_fn,
        num_workers=config.general.n_worker, shuffle=True, pin_memory=True, drop_last=True)
    
    valid_dataset = QA_Dataset(
        val_data, mode="val",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.general.batch_size, 
        collate_fn=collate_fn,
        num_workers=0, shuffle=False, pin_memory=True)
    
    test_dataset = QA_Dataset(
        test_data, mode="val",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    
    test_loader = DataLoader(
        test_dataset, batch_size=config.general.batch_size, 
        collate_fn=collate_fn, 
        num_workers=0, shuffle=False, pin_memory=True, drop_last=False)
    
    return train_loader, valid_loader, test_loader
        

def train(config):
    writer = init_directories_and_logger(config)        
    model, tokenizer = init_model_and_tokenizer(config)

    train_loader, valid_loader, test_loader = prepare_dataloader(config=config, tokenizer=tokenizer)

    total = len(train_loader)
    num_train_steps = int(len(train_loader) * config.general.epoch / config.general.accumulation_steps)
    optimizer, scheduler = optimizer_scheduler(model, num_train_steps)
    
    print("### start training")
    step = 0
    for epoch in range(config.general.epoch):
        model.train()
        train_losses = []
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, data in bar:
            inputs_ids = data["inputs_ids"].to(device)
            masks = data["masks"].to(device)
            labels = data["labels"].to(device)
            context_masks = data["context_masks"].to(device)
            
            logits, loss = model(
                ids=inputs_ids, 
                context_masks=context_masks,
                masks=masks, 
                labels=labels)
                            
            train_losses.append(loss.item())
            
            loss /= config.general.accumulation_steps
            loss.backward()

            if (step + 1) % config.general.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            bar.set_postfix(loss=loss.item(), epoch=epoch, lr=scheduler.get_last_lr())
            if step % config.general.save_per_steps == 0:
                torch.save(model.state_dict(), f"{config.path.ckpt}/{config.general.model_type}_epoch={epoch}_step={step}.bin")
                
            if step % config.general.logging_per_steps == 0:            
                print("### start validate ")
                valid_mrrs, valid_losses = [], []

                with torch.no_grad():
                    model.eval()
                    # bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
                    for _, data in enumerate(valid_loader):
                        inputs_ids = data["inputs_ids"].to(device)
                        masks = data["masks"].to(device)
                        labels = data["labels"].to(device)
                        context_masks = data["context_masks"].to(device)
                        
                        logits, loss = model(
                            ids=inputs_ids, 
                            context_masks=context_masks,
                            masks=masks, 
                            labels=labels)

                        y_pred = torch.softmax(logits, dim=0).squeeze(1)
                        y_true = labels
                        # y_true, y_score
                        pair = [[label, pred] for label, pred in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
                        valid_mrrs += pair  
                        valid_losses.append(loss.item())
                        
                        # bar.set_postfix(loss=loss.item(), epoch=epoch)
                        
                print("### start testing ")
                with torch.no_grad():
                    test_mrrs, test_losses = [], []
                    model.eval()
                    # bar = tqdm(enumerate(test_loader), total=len(test_loader))
                    for _, data in enumerate(test_loader):
                        inputs_ids = data["inputs_ids"].to(device)
                        masks = data["masks"].to(device)
                        labels = data["labels"].to(device)
                        context_masks = data["context_masks"].to(device)
                        
                        logits, loss = model(
                            ids=inputs_ids, 
                            context_masks=context_masks,
                            masks=masks, 
                            labels=labels)
                                
                        y_pred = torch.softmax(logits, dim=0).squeeze(1)
                        y_true = labels
                        
                        pair = [[label, pred] for label, pred in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
                        test_mrrs += pair
                        test_losses.append(loss.item())
                        # bar.set_postfix(loss=loss.item(), epoch=epoch)
                    
                valid_mrrs = list(map(calculate_mrr, valid_mrrs))
                valid_mrrs = np.array(valid_mrrs).mean()
                        
                test_mrrs = list(map(calculate_mrr, test_mrrs))
                test_mrrs = np.array(test_mrrs).mean()
                
                print(f"mrr_valid: {valid_mrrs} mrr_test: {test_mrrs}")
                print(f"train_loss: {np.mean(np.array(train_losses))}, valid_loss: {np.mean(np.array(valid_losses))} test_losses: {np.mean(np.array(test_losses))}")
                writer.add_scalars(
                    "mrr",
                    {
                        "test_mrrs": test_mrrs,
                        "valid_mrrs":valid_mrrs},
                    global_step=step
                )
                
                writer.add_scalars(
                    "loss",
                    {
                        "train_loss": np.mean(np.array(train_losses)),
                        "test_losses": np.mean(np.array(test_losses)),
                        "valid_loss": np.mean(np.array(valid_losses))},
                    global_step=step
                )
                model.train()
            step += 1
        
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
    print(json.dumps(OmegaConf.to_container(config), indent=4, ensure_ascii=False))
    train(config)