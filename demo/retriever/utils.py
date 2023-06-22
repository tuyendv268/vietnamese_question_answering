import re
import os
import numpy as np
import torch

def preprocess_text(text):
    text = re.sub(
        r'[^A-ZAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴa-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ_0-9,\.\!\?\:\;\\\/\-\s]+', 
        ' ', text)
    text = re.sub("\n+", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " . ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\,", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\:", " : ", text)
    text = re.sub("\s+", " ", text)
    return text

def norm_text(text):
    text = text.lower()
    text = re.sub(
        r'[^A-ZAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴa-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ_0-9\s]+', 
        ' ', text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    return text

def save_index(output_dir, texts, embeddings):
    text_path = os.path.join(output_dir, "passages.txt")
    embedding_path = os.path.join(output_dir, "embeddings.npy")
    
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    
    np.save(embedding_path, embeddings)
    print("saved texts and embeddings to: ", output_dir)

def load_index(path):
    text_path = os.path.join(path, "passages.txt")
    with open(text_path, "r", encoding="utf-8") as f:
        passages = [passage.strip() for passage in f.readlines()]
        
    embedding_path = os.path.join(path, "embeddings.npy")
    embeddings = np.load(embedding_path)
    embeddings = torch.from_numpy(embeddings)
    
    return passages, embeddings
