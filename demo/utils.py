from tqdm.auto import tqdm
tqdm.pandas()
import re
import pandas as pd
from glob import glob
import random
import json

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

def norm_text(text):
    text = text.lower()
    text = re.sub(
        r'[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ_0-9\s]+', 
        ' ', text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    return text