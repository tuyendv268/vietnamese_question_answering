from src.utils import load_chunks, norm_text, save_json
from src.models import Cross_Model
from bm25 import BM25
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import json
from src.utils import load_data

data_dir = "../data/processed-ir-data/test/mbf_data.json"
out_dir = "data/test"
data, passages = load_data(data_dir)
reranker_model = Cross_Model(model_name="envibert")

retriever_model = BM25()
passages = list(set(passages))
passages = [[str(index).zfill(6), passage] for index, passage in enumerate(passages)]
passages_df = pd.DataFrame(passages, columns=["id", "passage_text"])

passage2id = passages_df.set_index("passage_text")["id"].to_dict()
retriever_model.train(passages_df)

hlatr_data = []

index = len(os.listdir(out_dir))
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
    
    out_path = os.path.join(out_dir, str(index).zfill(6)+".npy")
    np.save(out_path, sample, allow_pickle=True)

    index += 1