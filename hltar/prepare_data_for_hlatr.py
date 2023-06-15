from src.utils import load_data
from src.utils import norm_text
from src.models import Cross_Model
from bm25 import BM25
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import json
from glob import glob
from transformers import RobertaModel
from transformers import AutoModel
from transformers import AutoTokenizer
from importlib.machinery import SourceFileLoader
import torch
from omegaconf import OmegaConf

def gen_hltar_data(in_path, out_path):
    data, passages = load_data(in_path)

    retriever_model = BM25()
    passages = list(set(passages))
    passages = [[str(index).zfill(6), passage] for index, passage in enumerate(passages)]
    passages_df = pd.DataFrame(passages, columns=["id", "passage_text"])

    passage2id = passages_df.set_index("passage_text")["id"].to_dict()
    retriever_model.train(passages_df)

    index = len(os.listdir(out_path))
    for sample in tqdm(data):
        query = sample["query"]
        passages = sample["passages"]
        
        positive_passage = None
        for passage in passages:
            if passage["is_selected"] == 1:
                positive_passage =  passage["passage_text"]
        if positive_passage is None:
            continue
                
        positive_passage = passage2id[positive_passage]
        
        query = norm_text(query)
        retrieval_result = retriever_model.ranking(query, topk=100)
        temps = retrieval_result[["id", "text"]].rename(columns={"text":"passage_text"})
        reranking_result = reranker_model.ranking(query, texts=temps)
        
        merged = pd.merge(retrieval_result, reranking_result, on="id", how="inner")
        
        _contexts = merged["id"].tolist()
        if positive_passage not in set(_contexts):
            print("Warning !!!")
            continue
        else:
            positive_index = _contexts.index(positive_passage)
            labels = [0] * len(_contexts)
            labels[positive_index] = 1
        
        sample = {
            # "qid": qid,
            # "pid": np.array(merged.pid.tolist()),
            # "retrieval_text": np.array(merged.retrieval_text.tolist()),
            "reranking_rank": np.array(merged.reranking_rank.tolist(), dtype=np.int8),
            "retrieval_rank": np.array(merged.retrieval_rank.tolist(), dtype=np.int8),
            "label": np.array(labels, dtype=np.bool_),
            "embedding": np.array([embedding.tolist() for embedding in merged.embedding.tolist()], dtype=np.float32),
            # "retrieval_score": np.array(merged.retrieval_score.tolist(), dtype=np.float16),
            # "reranking_score": np.array([reranking_score.tolist() for reranking_score in merged.reranking_score.tolist()], dtype=np.float16),
            }
        
        _path = os.path.join(out_path, str(index).zfill(6)+".npy")
        np.save(_path, sample, allow_pickle=True)

        index += 1
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

if __name__ == "__main__":
    input_dir = "/home/tuyendv/Desktop/mbf_ir/data/hltar-raw-data/test"
    output_dir = "/home/tuyendv/Desktop/mbf_ir/data/hltar-data/test"
    config_path = "config.yaml"
    
    config = OmegaConf.load(config_path)
    reranker_model, tokenizer = init_model_and_tokenizer(config)
    reranker_model.eval()
    
    for input_file in glob(f"{input_dir}/*.json"):
        gen_hltar_data(
            in_path=input_file, 
            out_path=output_dir)

        break