import os
from tqdm.auto import tqdm
tqdm.pandas()
from argparse import ArgumentParser
import openai
import gradio as gr
import torch
from omegaconf import OmegaConf
from model import Cross_Model
from importlib.machinery import SourceFileLoader
from transformers import RobertaModel
from transformers import AutoModel
from transformers import AutoTokenizer
from bm25 import BM25

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
        device="cuda",
        tokenizer=tokenizer, model=plm)
    
    if os.path.exists(config.path.warm_up):
        state_dict = torch.load(config.path.warm_up, map_location="cpu")
        # state_dict = {"module.".join(key.split("module.")[1:]):value for key, value in state_dict.items()}
        model.load_state_dict(state_dict["model"])
        print(f"load model state dict from {config.path.warm_up}")
        
    return model, tokenizer

config = OmegaConf.load("config.yaml")
model, tokenizer = init_model_and_tokenizer(config)
model = model.cuda()
model.eval()

bm25_model = BM25()
bm25_model.load("checkpoints/bm25")

def infer(query):
    bm25_result = bm25_model.search(query=query, topk=20)
    docs = [sample[1] for sample in bm25_result]
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        scores, ranks = model.ranking(query=query, texts=docs)

    print("QUERY: ", query)    
    print("MODEL_SCORES: ", scores)

    top_k = [docs[i] for i in ranks]
    results = ""
    for index, context in enumerate(top_k):
        print(f"###TOP_{index}: {context}")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        results = results + f"###TOP_{index}: {context}\n"
                
    return results


if __name__ == "__main__":
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