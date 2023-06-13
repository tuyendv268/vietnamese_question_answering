from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm.auto import tqdm
tqdm.pandas()
import json
import pandas as pd
import re

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

    opt = AdamW(optimizer_parameters, lr=3e-5, no_deprecation_warning=True)
    sch = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.05*num_train_steps),
        num_training_steps=num_train_steps,
        last_epoch=-1,
    )
    return opt, sch

def norm_text(text):
    text = text.lower()
    text = re.sub(
        r'[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ0-9\s]+', 
        ' ', text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    return text

def load_chunks(path):
    chunks = pd.read_csv(path, index_col=0, dtype={"chunk_id":str})
    
    return chunks

def save_json(path, df):
    with open(path, "w", encoding="utf-8") as f:
        json_obj = json.dumps(df.to_dict(), ensure_ascii=False, indent=4)
        f.write(json_obj)
    
    print("saved: ", path)

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

def load_json(path):
    data = pd.read_json(path, dtype={"id":str, "question":str, "positive_sample":str})
    
    return data

# def load_data(data_dir):
#     question_path = f"{data_dir}/questions.json"
#     question = load_json(question_path)
    
#     context_path = f"{data_dir}/contexts.json"
#     context = load_json(context_path)
    
#     data_path = f"{data_dir}/datas.json"
#     data = load_json(data_path)
    
#     return data, question, context

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data, passages = [], []
        for line in f.readlines():
            json_obj = json.loads(line.strip())
            data.append(json_obj)
            for passage in json_obj["passages"]:
                passage_text = passage["passage_text"]
                passages.append(passage_text)
    return data, passages


def norm_negative_sample(negative_samples):
    normed_negative_samples = []
    for sample in negative_samples:
        normed_negative_samples.append(norm_text(sample))
    return normed_negative_samples