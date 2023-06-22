import os
import gradio as gr
import torch
from omegaconf import OmegaConf

from retriever.bm25 import BM25
from retriever.retriever import Retriever
from reranker.reranker import Reranker

def infer(query, topk=20):
    bm25_passages = bm25_model.search(query=query, topk=topk)
    
    # print("###### BM25 RESULT: \n")
    # for index, passage in enumerate(bm25_passages):
    #     print(f"###TOP_{index}: {passage}")
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    # print("###### DUAL RESULT: \n")
    retriever_passages, retriever_scores = retriever.get_topk_passages(query=query, topk=topk)
    # for index, passage in enumerate(retriever_passages):
    #     print(f"###TOP_{index}: {passage}")
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    retrieval_result = []
    
    for i in bm25_passages:
        if i.strip("\n") not in retrieval_result:
            retrieval_result.append(i.strip("\n"))
            
    for i in retriever_passages:
        if i.strip("\n") not in retrieval_result:
            retrieval_result.append(i.strip("\n"))
            
    print("TOTAL PASSAGES: ", len(retrieval_result))
    scores, ranks = reranker.reranking(query=query, passages=retrieval_result)

    top_k = [retrieval_result[i] for i in ranks]
    results = ""
    print("###### CROSS RESULT: \n")
    print("QUERY: ", query)    
    print("MODEL_SCORES: ", scores)
    for index, context in enumerate(top_k):
        print(f"###TOP_{index}: {context}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        results = results + f"###TOP_{index}: {context}\n"
                
    return results


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    
    bm25_model = BM25()
    bm25_model.load(config.bm25.ckpt_path)
    
    retriever = Retriever(
        index_dir=config.retriever.index_dir,
        pretrained_dir=config.retriever.pretrained_dir,
        vncorenlp_dir=config.retriever.vncorenlp_dir
        )
    
    reranker = Reranker(
        ckpt_path=config.reranker.ckpt_path,
        pretrained_dir=config.reranker.pretrained_dir
    )
    
    gr.Interface(
        fn=infer,
        inputs=[
            gr.components.Textbox(
                lines=1,
                label="Query",
                placeholder="giới thiệu về mobifone.",
            )
        ], 
        outputs=[
            gr.components.Textbox(
                minimum=1,
                maximum=2000,
                label="Docs",
            )
        ],
        title="ChatBot Demo",
        description="Information Retrieval",
    ).launch(share=True)