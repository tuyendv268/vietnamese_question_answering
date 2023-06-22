from tqdm import tqdm

import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
from retriever.model import Dual_Model
from reranker.utils import load_index, norm_text, preprocess_text
import torch

class Retriever():
    def __init__(self, index_dir, pretrained_dir, vncorenlp_dir) -> None:
        self.pretrained_dir = pretrained_dir
        self.vncorenlp_dir = vncorenlp_dir
        self.model, self.tokenizer, self.segmenter= self.init_model_and_tokenizer() 
        self.model.eval()
        self.passages, self.embeddings = load_index(index_dir)

    def init_model_and_tokenizer(self):    
        tokenizer = AutoTokenizer.from_pretrained(
            'keepitreal/vietnamese-sbert', cache_dir=self.pretrained_dir)
        plm = AutoModel.from_pretrained(
            'keepitreal/vietnamese-sbert', cache_dir=self.pretrained_dir)
        
        segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=self.vncorenlp_dir)
        
        model = Dual_Model(tokenizer=tokenizer, model=plm, segmenter=segmenter)
        
        return model, tokenizer, segmenter

    @torch.no_grad()
    def get_topk_passages(self, query, topk=2):        
        preprocessed_query = preprocess_text(query)
        segmented_query = " ".join(self.segmenter.word_segment(preprocessed_query))
        normed_query = norm_text(segmented_query)
            
        query_ids = self.tokenizer(
            normed_query, max_length=256, 
            return_tensors="pt",
            truncation=True)
        
        query_embedding = self.model.extract_embedding(
            input_ids=query_ids["input_ids"],
            attention_mask=query_ids["attention_mask"]
            )
        
        scores = []
        for index in range(self.embeddings.shape[0]):
            score = torch.nn.functional.cosine_similarity(query_embedding[0], self.embeddings[index], dim=-1)
            
            scores.append(score)
        scores = torch.vstack(scores).view(-1)
        
        ranked = torch.topk(scores, k=topk)
        topk_index = ranked.indices
        topk_scores = ranked.values
        
        topk_text = [self.passages[rank] for rank in topk_index]
        
        return topk_text, topk_scores
    

# if __name__ == "__main__":
    # index_dir = "/home/tuyendv/Desktop/retriever/demo/outputs"
    # pretrained_dir = "/home/tuyendv/Desktop/retriever/pretrained"
    # vncorenlp_dir = "/home/tuyendv/Desktop/retriever/vncorenlp"
    
    # model, tokenizer, segmenter= init_model_and_tokenizer(
    #     pretrained_dir=pretrained_dir,
    #     vncorenlp_dir=vncorenlp_dir
    #     ) 
    # model.eval()
    # passages, embeddings = load_index(index_dir)
    
    # query = "mobifone eoffice là giải pháp gì"
    # topk_passage, topk_scores = get_topk_passages(query, top_k=10)
    # for index, (passage, score) in enumerate(zip(topk_passage, topk_scores)):
    #     print(f"TOP_{index} score={score}: {passage}\n")
