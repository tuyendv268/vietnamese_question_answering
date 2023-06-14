from retriever.model import Dual_Model
from retriever.src.dataset import Infer_Dual_Dataset
from omegaconf import OmegaConf
import torch
import pandas as pd
import json
import os

from retriever.model import Dual_Model
from transformers import AutoTokenizer
from transformers import AutoModel
from importlib.machinery import SourceFileLoader
from transformers import RobertaModel

from retriever.src.utils import norm_text

path = "configs/config.yaml"
config = OmegaConf.load(path)

device = "cpu" if not torch.cuda.is_available() else "cuda"
AUTH_TOKEN = "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"
    
tokenizer = SourceFileLoader(
    "envibert.tokenizer", 
    os.path.join(config.path.pretrained_dir,'envibert_tokenizer.py')) \
        .load_module().RobertaTokenizer(config.path.pretrained_dir)
plm = RobertaModel.from_pretrained(config.path.pretrained_dir)

# tokenizer = AutoTokenizer.from_pretrained(
#     'nguyenvulebinh/vi-mrc-base', cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)
# plm = AutoModel.from_pretrained(
#     "nguyenvulebinh/vi-mrc-base", cache_dir=config.path.pretrained_dir, use_auth_token=AUTH_TOKEN)

model = Dual_Model(
    max_length=config.general.max_length, 
    batch_size=config.general.batch_size,
    device=config.general.device,
    tokenizer=tokenizer, model=plm).to(device)

if os.path.exists(config.path.embedd_model):
    model.load_state_dict(torch.load(config.path.embedd_model, map_location="cpu"))
    print(f"load model state dict from {config.path.embedd_model}")
        
path = "outputs/bm25/chunks.csv"
df = pd.read_csv(path)
df["text"] = df.chunk.apply(lambda x: norm_text(x))

embeddings = model.extract_embeddings(df["text"].tolist(), device="cpu")
embeddings = torch.vstack(embeddings)

docs = []

for id, (text, embedding) in enumerate(zip(df["chunk"].tolist(), embeddings)):
    sample = {
        "doc_id": id,
        "text": text,
        "embedding": embedding.tolist()
    }
    
    docs.append(sample)
    
with open("outputs/docs.json", "w", encoding="utf-8") as f:
    json_obj = json.dumps(docs, indent=4, ensure_ascii=False)
    f.write(json_obj)