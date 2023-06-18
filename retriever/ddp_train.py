import os
from omegaconf import OmegaConf
from importlib.machinery import SourceFileLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json

from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel
from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import torch.distributed as dist
from torch import nn

from src.utils import (
    load_data,
    optimizer_scheduler
    )

from src.dataset import (
    QA_Dataset
    )

from src.model import Dual_Model
from datetime import datetime
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig

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
    
    model = Dual_Model(
        max_length=config.general.max_length, 
        device=config.general.device,
        tokenizer=tokenizer, model=plm)
    
    if os.path.exists(config.path.warm_up):
        model.load_state_dict(torch.load(config.path.warm_up, map_location="cpu"))
        print(f"load model state dict from {config.path.warm_up}")
        
    return model, tokenizer

def prepare_dataloader(config, tokenizer):
    test_data = config.path.test_data
    train_data = config.path.train_data
    val_data = config.path.val_data
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
        collate_fn=collate_fn, sampler=sampler,
        num_workers=config.general.n_worker, 
        shuffle=False, pin_memory=True, drop_last=True)
    
    
    valid_dataset = QA_Dataset(
        val_data, mode="val",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    
    sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.general.batch_size, 
        collate_fn=collate_fn, sampler=sampler,
        num_workers=0, shuffle=False, pin_memory=True)
    
    test_dataset = QA_Dataset(
        test_data, mode="val",
        tokenizer=tokenizer, 
        max_length=config.general.max_length)
    
    sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config.general.batch_size, 
        collate_fn=collate_fn, sampler=sampler,
        num_workers=0, shuffle=False, pin_memory=True, drop_last=False)
    
    return train_loader, valid_loader, test_loader

def train(config):
    init_distributed()
    if is_main_process():
        writer = init_directories_and_logger(config)   
             
    model, tokenizer = init_model_and_tokenizer(config)
    train_loader, valid_loader, test_loader = prepare_dataloader(config=config, tokenizer=tokenizer)
    print("num_train_sample: ", len(train_loader))
    print("num_valid_sample: ", len(valid_loader))
    print("num_test_sample: ", len(test_loader))
    
    total = len(train_loader)
    num_train_steps = int(len(train_loader) * config.general.epoch / config.general.accumulation_steps)
    
    model = model.cuda()
    optimizer, scheduler = optimizer_scheduler(model, num_train_steps)
    if os.path.exists(config.path.warm_up):
        state_dict = torch.load(config.path.warm_up)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        print(f"loaded model and optimizer state dict from {config.path.warm_up}")

    model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    print("### start training")
    step = 0
    for epoch in range(config.general.epoch):
        model.train()
        train_losses = []
        # bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, data in enumerate(train_loader):
            contexts_ids = data["context_ids"].cuda()
            query_ids = data["query_ids"].cuda()
            query_masks = data["query_masks"].cuda()
            masks = data["masks"].cuda()
            labels = data["labels"].cuda()
            context_masks = data["context_masks"].cuda()
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss, logits, labels = model(
                    contexts_ids=contexts_ids,
                    query_ids=query_ids,
                    query_masks=query_masks,
                    context_masks=context_masks,
                    labels=labels, 
                    masks=masks,
                    )
                loss /= config.general.accumulation_steps
                
            scaler.scale(loss).backward()

            train_losses.append(loss.item() * config.general.accumulation_steps)
            _temp = np.mean(np.array(train_losses))
            # bar.set_postfix(mean_loss=_temp, loss=loss.item()*config.general.accumulation_steps, epoch=epoch, lr=scheduler.get_last_lr())
            if (step + 1) % config.general.accumulation_steps == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scheduler.step()
                scaler.update()
                
            if is_main_process() and step % config.general.logging_per_steps == 0:
                message = {
                    "loss":round(np.mean(np.array(train_losses)), 3),
                    "step":step,
                    "learning_rate":scheduler.get_last_lr(),
                    "gpu_id": get_rank(),
                    "total":total,
                }
                print("log: ", message)
            if is_main_process() and step % config.general.save_per_steps == 0:
                path = f"{config.path.ckpt}/{config.general.model_type}_epoch={epoch}_step={step}.bin"
                save(path=path, optimizer=optimizer, model=model)
   
            if is_main_process() and (step + 1) % config.general.evaluate_per_steps == 0:
                print("### start validate ")
                val_preds, val_labels, val_losses = [], [], []

                with torch.no_grad():
                    model.eval()
                    # _bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
                    for _, data in enumerate(valid_loader):
                        contexts_ids = data["context_ids"].cuda()
                        query_ids = data["query_ids"].cuda()
                        query_masks = data["query_masks"].cuda()
                        masks = data["masks"].cuda()
                        labels = data["labels"].cuda()
                        context_masks = data["context_masks"].cuda()
                        
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            val_loss, logits, labels = model(
                                contexts_ids=contexts_ids,
                                query_ids=query_ids,
                                query_masks=query_masks,
                                context_masks=context_masks,
                                labels=labels, 
                                masks=masks,
                                )
                        val_preds.append(logits)
                        val_labels.append(labels)
                        val_losses.append(val_loss.item())
                    model.train()
                    
                val_preds = torch.cat(val_preds, dim=0).argmax(dim=1)
                val_labels = torch.cat(val_labels, dim=0).argmax(dim=1)
                val_acc = f1_score(val_labels.numpy(), val_preds.numpy())
                    
                print(f"#### val_acc: {val_acc}, false: {torch.sum(val_labels==0)}, true: {torch.sum(val_labels==1)}")
                print("### start testing ")
                with torch.no_grad():
                    test_preds, test_labels, test_losses = [], [], []
                    model.eval()
                    # _bar = tqdm(enumerate(test_loader), total=len(test_loader))
                    for _, data in enumerate(test_loader):
                        contexts_ids = data["context_ids"].cuda()
                        query_ids = data["query_ids"].cuda()
                        query_masks = data["query_masks"].cuda()
                        masks = data["masks"].cuda()
                        labels = data["labels"].cuda()
                        context_masks = data["context_masks"].cuda()
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            test_loss, logits, labels = model(
                                contexts_ids=contexts_ids,
                                query_ids=query_ids,
                                query_masks=query_masks,
                                context_masks=context_masks,
                                labels=labels, 
                                masks=masks,
                                )
                            
                        test_preds.append(logits)
                        test_labels.append(labels)
                        test_losses.append(test_loss.item())
                        # _bar.set_postfix(loss=test_loss.item(), epoch=epoch)
                    model.train()
                    
                valid_mrrs = list(map(calculate_mrr, valid_mrrs))
                test_preds = torch.cat(test_preds, dim=0).argmax(dim=1)
                test_labels = torch.cat(test_labels, dim=0).argmax(dim=1)
                test_acc = f1_score(test_labels.numpy(), test_preds.numpy())
                
                print(f"#### test_acc: {test_acc}, false: {torch.sum(test_labels==0)}, true: {torch.sum(test_labels==1)}")

                print(f"#### train_loss: {np.mean(np.array(train_losses))}")
                print(f"#### val_loss: {np.mean(np.array(val_losses))}")
                print(f"#### test_loss: {np.mean(np.array(test_losses))}")
                
                writer.add_scalars(
                    "mrr",
                    {
                        "test_acc": test_acc,
                        "val_acc": val_acc},
                    global_step=step
                )
                
                writer.add_scalars(
                    "loss",
                    {
                        "train_loss": np.mean(np.array(train_losses)),
                        "test_losses": np.mean(np.array(test_losses)),
                        "valid_loss": np.mean(np.array(val_losses))},
                    global_step=step
                )
            step += 1

def save(path, optimizer, model):
    state_dict = {
        "model":model.module.state_dict(),
        "optimizer":optimizer.state_dict()
    }
    
    torch.save(state_dict ,path)
    print(f'saved state dict to {path}')

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