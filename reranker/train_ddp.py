import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from omegaconf import OmegaConf
from importlib.machinery import SourceFileLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json

from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel
from transformers import AutoModel
from transformers import AutoTokenizer
from torch import nn

from utils import (
    load_data,
    optimizer_scheduler
    )

from dataset import (
    QA_Dataset
    )

from model import Cross_Model
from datetime import datetime
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig
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
        os.makedirs(config.path.ckpt)
        
    if not os.path.exists(config.path.log):
        os.makedirs(config.path.log)
        
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")
    
    log_dir = f"{config.path.log}/{current_time}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
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
        tokenizer=tokenizer, model=plm)
    
    if os.path.exists(config.path.warm_up):
        model.load_state_dict(torch.load(config.path.warm_up, map_location="cpu"))
        print(f"load model state dict from {config.path.warm_up}")
        
    return model, tokenizer

# def prepare_dataloader(config, tokenizer):
#     train_data = load_data(config.path.train_data)
#     test_data = load_data(config.path.test_data)
    
#     train_data, val_data = train_test_split(train_data, test_size=config.general.valid_size, random_state=42)
    
#     train_dataset = QA_Dataset(
#         train_data, mode="train",
#         tokenizer=tokenizer, 
#         max_length=config.general.max_length)
    
#     if config.general.model_type =="cross" :
#         collate_fn = train_dataset.cross_collate_fn
#     elif config.general.model_type =="dual" :
#         collate_fn = train_dataset.dual_collate_fn
    
#     sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
#     train_loader = DataLoader(
#         train_dataset, batch_size=config.general.batch_size,
#         collate_fn=collate_fn,
#         sampler=sampler,
#         num_workers=config.general.n_worker, drop_last=True)
    
    
#     valid_dataset = QA_Dataset(
#         val_data, mode="val",
#         tokenizer=tokenizer, 
#         max_length=config.general.max_length)
#     sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)
#     valid_loader = DataLoader(
#         valid_dataset, batch_size=config.general.batch_size, 
#         sampler=sampler,
#         collate_fn=collate_fn,
#         num_workers=0, drop_last=False)
    
#     test_dataset = QA_Dataset(
#         test_data, mode="val",
#         tokenizer=tokenizer, 
#         max_length=config.general.max_length)
#     sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
#     test_loader = DataLoader(
#         test_dataset, batch_size=config.general.batch_size,
#         sampler=sampler, 
#         collate_fn=collate_fn, 
#         num_workers=0, drop_last=False)
    
#     return train_loader, valid_loader, test_loader


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
        
    sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config.general.batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        num_workers=config.general.n_worker, drop_last=True)
    
    
    valid_dataset = QA_Dataset(
        val_data, mode="val",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.general.batch_size, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0, drop_last=False)
    
    test_dataset = QA_Dataset(
        test_data, mode="val",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config.general.batch_size,
        sampler=sampler, 
        collate_fn=collate_fn, 
        num_workers=0, drop_last=False)
    
    return train_loader, valid_loader, test_loader

def train(config):    
    init_distributed()
    if is_main_process():
        writer = init_directories_and_logger(config)        
    model, tokenizer = init_model_and_tokenizer(config)
    
    model = model.cuda()
    
    local_rank = int(os.environ['LOCAL_RANK'])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    train_loader, valid_loader, test_loader = prepare_dataloader(config=config, tokenizer=tokenizer)
    print("num_train_sample: ", len(train_loader))
    print("num_valid_sample: ", len(valid_loader))
    print("num_test_sample: ", len(test_loader))
    
    total = len(train_loader)
    num_train_steps = int(len(train_loader) * config.general.epoch / config.general.accumulation_steps)
    optimizer, scheduler = optimizer_scheduler(model, num_train_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    print("start training")
    step = 0
    for epoch in range(config.general.epoch):
        model.train()
        train_losses = []
        
        train_loader.sampler.set_epoch(epoch)
        # bar = tqdm(enumerate(train_loader), total=len(train_loader), position=get_rank())
        for _, data in enumerate(train_loader):
            try:
                inputs_ids = data["inputs_ids"].cuda()
                masks = data["masks"].cuda()
                labels = data["labels"].cuda()
                context_masks = data["context_masks"].cuda()
                
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits, loss = model(
                        ids=inputs_ids, 
                        context_masks=context_masks,
                        masks=masks, 
                        labels=labels)
                    
                    loss /= config.general.accumulation_steps
                scaler.scale(loss).backward()
                train_losses.append(loss.item())
            except RuntimeError as e:
                torch.cuda.empty_cache()
                print("Error in training: ", e)
                continue
            
            if (step + 1) % config.general.accumulation_steps == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scheduler.step()
                scaler.update()
            step += 1
            
            if is_main_process() and step % config.general.logging_per_steps == 0:
                message = {
                    "loss":round(np.mean(np.array(train_losses)), 3),
                    "step":step,
                    "learning_rate":scheduler.get_last_lr(),
                    "gpu_id": get_rank(),
                    "total":total,
                }
                print("log: ", message)
            # bar.set_postfix(loss=loss.item(), epoch=epoch, id=get_rank(), lr=scheduler.get_last_lr())
            
            if is_main_process() and (step + 1) % config.general.evaluate_per_step == 0:
                print("start evaluate")
                torch.save(model.state_dict(), f"{config.path.ckpt}/{config.general.model_type}_{epoch}.bin")
                
                valid_mrrs, valid_losses = [], []

                with torch.no_grad():
                    model.eval()
                    # val_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), position=1)
                    for _, data in enumerate(valid_loader):
                        try:
                            inputs_ids = data["inputs_ids"].cuda()
                            masks = data["masks"].cuda()
                            labels = data["labels"].cuda()
                            context_masks = data["context_masks"].cuda()
                            
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                logits, loss = model(
                                    ids=inputs_ids, 
                                    context_masks=context_masks,
                                    masks=masks, 
                                    labels=labels)
                                    
                            y_pred = torch.softmax(logits, dim=0).squeeze(1)
                            y_true = labels
                            pair = [[pred, label] for pred, label in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
                            valid_mrrs += pair  
                            valid_losses.append(loss.item())
                        except RuntimeError as e:
                            torch.cuda.empty_cache()
                            print("Error in validation: ", e)
                            continue
                        # val_bar.set_postfix(loss=loss.item(), epoch=epoch)
                        
                with torch.no_grad():
                    test_mrrs, test_losses = [], []
                    model.eval()
                    # test_bar = tqdm(enumerate(test_loader), total=len(test_loader), position=2)
                    for _, data in enumerate(test_loader):
                        try:
                            inputs_ids = data["inputs_ids"].cuda()
                            masks = data["masks"].cuda()
                            labels = data["labels"].cuda()
                            context_masks = data["context_masks"].cuda()
                            
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                logits, loss = model(
                                    ids=inputs_ids, 
                                    context_masks=context_masks,
                                    masks=masks, 
                                    labels=labels)
                            y_pred = torch.softmax(logits, dim=0).squeeze(1)
                            y_true = labels
                            
                            pair = [[pred, label] for pred, label in zip(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())]
                            test_mrrs += pair
                            test_losses.append(loss.item())
                        except RuntimeError as e:
                            torch.cuda.empty_cache()
                            print("Error in testing: ", e)
                            continue
                        # test_bar.set_postfix(loss=loss.item(), epoch=epoch)
                    
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
    if is_main_process():
        print(json.dumps(OmegaConf.to_container(config), indent=4, ensure_ascii=False))
    train(config)