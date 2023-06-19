from tqdm.auto import tqdm
tqdm.pandas()
import re
from torch.optim import Adam 
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from pandarallel import pandarallel
import pandas as pd
from glob import glob
import random
import json

def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    return set(lines)

def load_file(path, max_context=7):
    with open(path, "r", encoding="utf-8") as f:
        data = []
        for line in f.readlines():
            json_obj = json.loads(line.strip())
            query = json_obj["query"]
            passages = []
            mark, count = 0, 0
            
            for passage in json_obj["passages"]:
                if count > max_context:
                    break
                if passage["is_selected"] == 1:
                    if mark == 1:
                        continue
                    passages.append(passage)
                    mark = 1
                else:
                    if len(passages) == max_context and mark == 0:
                        continue
                    passages.append(passage)
                count+=1
                                
            random.shuffle(passages)
                
            if mark == 1:
                sample = {
                    "query":query,
                    "passages":passages
                }
                sample = json.dumps(sample, ensure_ascii=False)
                data.append(sample)
    return data

def save_data(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for sample in data:
            json_obj = json.dumps(sample, ensure_ascii=False)
            f.write(json_obj+"\n")

def load_data(path):
    data = []
    for _file in glob(path+"/*.json"):
        data += load_file(_file)
      
    return tuple(data)

def optimizer_scheduler(model, num_train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    optimizer = Adam(optimizer_parameters, lr=4e-5, betas=(0.9, 0.999))
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_train_steps)
    return optimizer, scheduler

def norm_text(text):
    text = text.lower()
    text = re.sub(
        r'[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ_0-9\s]+', 
        ' ', text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    return text


def norm_question(text):
    text = text.lower()
    text = re.sub(
        r'[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ0-9\;\-\,\.\?\!\/\\\:\(\)+\s]+', 
        ' ', text)
    text = re.sub("\n+", "\n", text)
    text = re.sub("\s+", " ", text)
    text = re.sub("(?<=[a-zA-Z])\.(?=[a-zA-Z])", " ", text)
    text = re.sub("(?<=[a-zA-Z])\,(?=[a-zA-Z])", " ", text)
    return text


def norm_negative_sample(negative_samples):
    normed_negative_samples = []
    for sample in negative_samples:
        normed_negative_samples.append(norm_text(sample))
    return normed_negative_samples