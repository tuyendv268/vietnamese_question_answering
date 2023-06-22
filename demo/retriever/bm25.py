import numpy as np
import pandas as pd
import os

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
from retriever.utils import norm_text

class BM25:
    def load(self, path):
        self.dictionary = Dictionary.load(f"{path}/dict")
        self.bm25_index = SparseMatrixSimilarity.load(f"{path}/bm25_index")
        self.tfidf_model = TfidfModel.load(f"{path}/tfidf")
        self.passages = pd.read_csv(f"{path}/passage.txt", index_col=0)
        
    def save(self, path):
        self.dictionary.save(f"{path}/dict")
        self.tfidf_model.save(f"{path}/tfidf")
        self.bm25_index.save(f"{path}/bm25_index")
        self.passages.to_csv(f"{path}/passage.txt")
                    
    def train(self, passages):
        self.passages = passages
        
        passages['bm25_text'] = passages["passage_text"].apply(lambda x: norm_text(x))
        corpus = [x.split() for x in passages['bm25_text'].values]
                
        self.dictionary = Dictionary(corpus)
        bm25_model = OkapiBM25Model(dictionary=self.dictionary)
        bm25_corpus = bm25_model[list(map(self.dictionary.doc2bow, corpus))]

        self.bm25_index = SparseMatrixSimilarity(
            bm25_corpus, num_docs=len(corpus), num_terms=len(self.dictionary),
            normalize_queries=False, normalize_documents=False)

        self.tfidf_model = TfidfModel(dictionary=self.dictionary, smartirs='bnn')
        
        
    def get_topk(self, query, topk: int=10):
        query = norm_text(query)
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query.split())]
        scores = self.bm25_index[tfidf_query]
        
        if topk == None:
            top_n = np.argsort(scores)[::-1]
        else:
            top_n = np.argsort(scores)[::-1][:topk]
            
        return top_n, scores[top_n]
    
    def search(self, query, topk: int=10):
        query = norm_text(query)
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query.split())]
        scores = self.bm25_index[tfidf_query]
        
        top_n = np.argsort(scores)[::-1][:topk]
        
        result = []
        for rank, index in enumerate(top_n):
            _score = scores[index]
            _text = self.passages.passage_text.values[index]
            
            result.append(_text)
        return result

    
    def ranking(self, query, topk: int=100):
        query = norm_text(query)
        tfidf_query = self.tfidf_model[self.dictionary.doc2bow(query.split())]
        scores = self.bm25_index[tfidf_query]
        
        top_n = np.argsort(scores)[::-1][:topk]
        
        result = []
        for rank, index in enumerate(top_n):
            _id = self.passages.id.values[index]
            _score = scores[index]
            _rank = rank
            _text = self.passages.passage_text.values[index]
            
            result.append([_id, _score, _rank, _text])
            
        result = pd.DataFrame(result, columns=["id", "retrieval_score", "retrieval_rank", "retrieval_text"])
        return result
    
    
if __name__ == "__main__":
    bm25 = BM25()
    path = "/mnt/sda2/datas/mbf_ir/demo/inputs/passage.txt"
    with open(path, "r", encoding="utf-8") as f:
        passages = f.readlines()
        passages = pd.DataFrame(passages, columns=["passage_text"])
    bm25.train(passages)
    bm25.save("/mnt/sda2/datas/mbf_ir/demo/checkpoints/bm25")