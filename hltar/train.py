import os
from omegaconf import OmegaConf
from importlib.machinery import SourceFileLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json

import torch
from torch.utils.data import DataLoader
from glob import glob


from src.utils import (
    optimizer_scheduler
    )

from src.dataset import (
    HLTAR_Dataset
    )

from src.models import HLATR
from src.loss import HLTAR_Loss
from datetime import datetime

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_directories_and_logger(config):
    if not os.path.exists(config.path.ckpt):
        os.mkdir(config.path.ckpt)
        
    if not os.path.exists(config.path.log):
        os.mkdir(config.path.log)
        
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
def init_model(config):
    model = HLATR()
    
    if os.path.exists(config.path.warm_up):
        model.load_state_dict(torch.load(config.path.warm_up, map_location="cpu"))
        print(f"load model state dict from {config.path.warm_up}")
        
    return model

def prepare_dataloader(config):
    train_df = glob(f'{config.path.train_data}/*.npy')
    val_df = glob(f'{config.path.val_data}/*.npy')
    test_df = glob(f'{config.path.test_data}/*.npy')
        
    train_dataset = HLTAR_Dataset(
        data=train_df,
        max_doc=100
    )
            
    train_loader = DataLoader(
        train_dataset, batch_size=config.general.batch_size, 
        collate_fn=train_dataset.collate_fn, 
        num_workers=config.general.n_worker, shuffle=True, pin_memory=True, drop_last=True)
    
    test_dataset = HLTAR_Dataset(
        data=test_df,
        max_doc=100
    )
            
    test_loader = DataLoader(
        test_dataset, batch_size=config.general.batch_size, 
        num_workers=0,
        collate_fn=train_dataset.collate_fn, 
        shuffle=True, pin_memory=True, drop_last=True)

    val_dataset = HLTAR_Dataset(
        data=val_df,
        max_doc=100
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.general.batch_size, 
        num_workers=0,
        collate_fn=val_dataset.collate_fn, 
        shuffle=True, pin_memory=True, drop_last=True)
        
    return train_loader, test_loader, val_loader
        

def train(config):
    writer = init_directories_and_logger(config)
    hltar_loss = HLTAR_Loss()
        
    model = init_model(config)

    train_loader, test_loader, val_loader = prepare_dataloader(config=config)

    num_train_steps = len(train_loader) * config.general.epoch
    optimizer, scheduler = optimizer_scheduler(model, num_train_steps)
    
    step = 0
    print("### start training")
    for epoch in range(config.general.epoch):
        model.train()
        train_losses, train_mrrs = [], []
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, data in bar:
            embeddings = data["embeddings"].to(device)
            masks = data["masks"].to(device)
            labels = data["labels"].to(device)
            retrieval_ranks = data["retrieval_ranks"].to(device)
            reranking_ranks = data["reranking_ranks"].to(device)
                                    
            output = model(embeddings, masks, retrieval_ranks, reranking_ranks)
            
            loss = hltar_loss(labels, output)
            
            y_true = labels
            y_pred = output
            
            pairs = [[label, pred] for label, pred in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
            train_mrrs += pairs
            train_losses.append(loss.item())
            
            loss /= config.general.accumulation_steps
            loss.backward()

            if (step + 1) % config.general.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            step += 1
            
            bar.set_postfix(loss=loss.item(), epoch=epoch)
        if (epoch + 1) % config.general.save_ckpt_per_n_epoch == 0:
            torch.save(model.state_dict(), f"{config.path.ckpt}/ckpt_{epoch}.bin")
        
        print("### start validate")
        valid_mrrs, valid_losses = [], []

        with torch.no_grad():
            model.eval()
            bar = tqdm(enumerate(val_loader), total=len(val_loader))
            for _, data in bar:
                embeddings = data["embeddings"].to(device)
                masks = data["masks"].to(device)
                labels = data["labels"].to(device)
                retrieval_ranks = data["retrieval_ranks"].to(device)
                reranking_ranks = data["reranking_ranks"].to(device)
                            
                output = model(embeddings, masks, retrieval_ranks, reranking_ranks)
                
                loss = hltar_loss(labels, output)
                
                y_true = labels
                y_pred = output
                
                pairs = [[label, pred] for label, pred in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
                valid_mrrs += pairs
                valid_losses.append(loss.item())
                
                bar.set_postfix(loss=loss.item(), epoch=epoch)
                                    
        print("### start testing")
        test_mrrs, test_losses = [], []

        with torch.no_grad():
            model.eval()
            bar = tqdm(enumerate(test_loader), total=len(test_loader))
            for _, data in bar:
                embeddings = data["embeddings"].to(device)
                masks = data["masks"].to(device)
                labels = data["labels"].to(device)
                retrieval_ranks = data["retrieval_ranks"].to(device)
                reranking_ranks = data["reranking_ranks"].to(device)
                            
                output = model(embeddings, masks, retrieval_ranks, reranking_ranks)
                
                loss = hltar_loss(labels, output)
                
                y_true = labels
                y_pred = output
                
                pairs = [[label, pred] for label, pred in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
                test_mrrs += pairs
                test_losses.append(loss.item())
                
                bar.set_postfix(loss=loss.item(), epoch=epoch)
        test_mrrs = list(map(calculate_mrr, test_mrrs))
        test_mrrs = np.array(test_mrrs).mean()
        
        valid_mrrs = list(map(calculate_mrr, valid_mrrs))
        valid_mrrs = np.array(valid_mrrs).mean()
        
        train_mrrs = list(map(calculate_mrr, train_mrrs))
        train_mrrs = np.array(train_mrrs).mean()
        
        print(f"mrr_train: {train_mrrs}, mrr_valid: {valid_mrrs}, test_mrrs: {test_mrrs}")
        print(f"train_loss: {np.mean(np.array(train_losses))}, valid_loss: {np.mean(np.array(valid_losses))}, test_losses: {np.mean(np.array(test_losses))}")
        writer.add_scalars(
            "mrr",
            {
                "train_mrrs": train_mrrs,
                "train_mrrs": train_mrrs,
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