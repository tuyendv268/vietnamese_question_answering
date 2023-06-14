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
import torch.distributed as dist
import torch 

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed():
    dist_url = "env://"
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    torch.cuda.set_device(local_rank)
    dist.barrier()
    setup_for_distributed(rank == 0)

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
    init_distributed()
    if is_main_process():
        writer = init_directories_and_logger(config)        
    hltar_loss = HLTAR_Loss()
        
    model = init_model(config)
    model = model.cuda()
    
    local_rank = int(os.environ['LOCAL_RANK'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    train_loader, test_loader, val_loader = prepare_dataloader(config=config)
    print("num_train_sample: ", len(train_loader))
    print("num_valid_sample: ", len(val_loader))
    print("num_test_sample: ", len(test_loader))
    
    total = len(train_loader)
    num_train_steps = len(train_loader) * config.general.epoch
    optimizer, scheduler = optimizer_scheduler(model, num_train_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    step = 0
    print("### start training")
    for epoch in range(config.general.epoch):
        model.train()
        train_losses, train_mrrs = [], []
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, data in bar:
            embeddings = data["embeddings"].cuda()
            masks = data["masks"].cuda()
            labels = data["labels"].cuda()
            retrieval_ranks = data["retrieval_ranks"].cuda()
            reranking_ranks = data["reranking_ranks"].cuda()
            
            with torch.cuda.amp.autocast(dtype=torch.float16):            
                output = model(embeddings, masks, retrieval_ranks, reranking_ranks)
                loss = hltar_loss(labels, output)
                loss /= config.general.accumulation_steps
            
            scaler.scale(loss).backward()
            
            y_true = labels
            y_pred = output
            
            pairs = [[label, pred] for label, pred in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
            train_mrrs += pairs
            train_losses.append(loss.item())

            if (step + 1) % config.general.accumulation_steps == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scheduler.step()
                scaler.update()
            step += 1
            
            bar.set_postfix(loss=loss.item(), epoch=epoch)
        
            if is_main_process() and step % config.general.logging_per_steps == 0:
                message = {
                    "loss":round(np.mean(np.array(train_losses)), 3),
                    "step":step,
                    "learning_rate":scheduler.get_last_lr(),
                    "gpu_id": get_rank(),
                    "total":total,
                }
                print("log: ", message)

            if is_main_process() and (step + 1) % config.general.evaluate_per_step == 0:
                print("### start evaluate")
                torch.save(model.state_dict(), f"{config.path.ckpt}/{config.general.model_type}_{epoch}.bin")
                valid_mrrs, valid_losses = [], []

                with torch.no_grad():
                    model.eval()
                    bar = tqdm(enumerate(val_loader), total=len(val_loader))
                    for _, data in bar:
                        embeddings = data["embeddings"].cuda()
                        masks = data["masks"].cuda()
                        labels = data["labels"].cuda()
                        retrieval_ranks = data["retrieval_ranks"].cuda()
                        reranking_ranks = data["reranking_ranks"].cuda()
                        with torch.cuda.amp.autocast(dtype=torch.float16):
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
                        embeddings = data["embeddings"].cuda()
                        masks = data["masks"].cuda()
                        labels = data["labels"].cuda()
                        retrieval_ranks = data["retrieval_ranks"].cuda()
                        reranking_ranks = data["reranking_ranks"].cuda()
                        
                        with torch.cuda.amp.autocast(dtype=torch.float16):   
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
                
                print(f"### mrr_train: {train_mrrs}, mrr_valid: {valid_mrrs}, test_mrrs: {test_mrrs}")
                print(f"### train_loss: {np.mean(np.array(train_losses))}, valid_loss: {np.mean(np.array(valid_losses))}, test_losses: {np.mean(np.array(test_losses))}")
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