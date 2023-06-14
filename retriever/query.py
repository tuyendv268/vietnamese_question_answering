from retriever.model import Dual_Model
from retriever.src.dataset import Infer_Dual_Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
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

path = "outputs/docs.json"

docs = json.load(open(path, "r", encoding="utf-8"))

with torch.no_grad():
    query = """giới thiệu về giải pháp MobiFone eHRM """
    query = norm_text(query)
    query_ids = model.tokenizer.encode_plus(query, max_length=384, truncation=True, return_tensors="pt")
    query_embedding = model.extract_query_embedding(query_ids["input_ids"], query_ids["attention_mask"])
    
scores = []
for doc in tqdm(docs):
    doc_embedding = torch.tensor(doc["embedding"])
    score = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0)
    scores.append(score)
    
scores = torch.tensor(scores)
print(scores)
document_indexs = torch.topk(scores, k=20).indices
print(document_indexs)

documents = [docs[index]["text"] for index in document_indexs]
print("#################")
print(f"QUERY: {query}")
for top_i, doc in enumerate(documents):
    print(f"###TOP_{top_i}###")
    print(doc)
    print("#################")