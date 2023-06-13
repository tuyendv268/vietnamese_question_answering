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
        device=config.general.device,
        tokenizer=tokenizer, model=plm).to(device)
    
    if os.path.exists(config.path.warm_up):
        model.load_state_dict(torch.load(config.path.warm_up, map_location="cpu"))
        print(f"load model state dict from {config.path.warm_up}")
        
    return model, tokenizer


parser = ArgumentParser()
parser.add_argument("--config", default='config.yaml', type=str)
parser.add_argument("--bm25_dir", default='bm25', type=str)

args = parser.parse_args()
config = OmegaConf.load(args.config)
model, tokenizer = init_model_and_tokenizer(config)

device = "cpu" if not torch.cuda.is_available() else "cuda"
model = Cross_Model(
    model=model,
    tokenizer=tokenizer,
    max_length=config.general.max_length,
    device="cpu")
if os.path.exists(config.path.warm_up):
    state_dict = torch.load(config.path.warm_up, map_location="cpu")
    state_dict = {"module.".join(key.split("module.")[1:]):value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"load state dict from {config.path.warm_up}")
model.eval()

bm25_model = BM25(f"{args.bm25}")


def infer(query):
    question            = norm_question(query)
    tokenized_query     = norm_text(query)
    top_n, bm25_scores  = bm25_model.get_text_topk(tokenized_query, topk=32)
    scores              = model.ranking(question, top_n)

    print("QUESTIONS:         ", question)    
    print("TOKENIZED QUESTION:", tokenized_query)
    print("TOP_N:             ", top_n[0])
    print("BM25_SCORE", bm25_scores)
    print("MODEL_SCORES:            ", scores)

    best_chunks         = [top_n[i] for i in scores.argsort()[-1:]]
    for b in best_chunks:
        print(b)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # best_chunks         = [top_n[i] for i in scores.argsort()[-1:]]
    context             = " ".join(best_chunks[-1:])
    prompt              = (
        "Below is an instruction that describes a task, paired with an input that provides further context. \n"
        "Write a response in vietnamese that appropriately completes the request. \n"
        "### Instruction: \n"
        f"{question} \n"
        f"### Input: \n"
        f"{context}"
        "### Response: \n")
     
    messages = []
    messages.append({"role": "user", "content": prompt})

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages,
    #     # stream=True,
    #     temperature=0.3
    # )
    # print("ANSWERS: ", response.choices[0].message.content)
    semantic_search = ""
    # best_chunks.reverse()
    # for i, context in enumerate(best_chunks):
    #     semantic_search += f"#TOP_{i} : " + context + "\n"
        
    return context, None


if __name__ == "__main__":
    gr.Interface(
        fn=infer,
        inputs=[
            gr.components.Textbox(
                lines=1,
                label="Question",
                placeholder="giới thiệu về mobifone.",
            )
        ], 
        outputs=[
            gr.components.Textbox(
                minimum=1,
                maximum=2000,
                label="Answer",
            ),
            gr.components.Textbox(
                minimum=1,
                maximum=2000,
                label="Context",
            )
        ],
        title="ChatBot Demo",
        description="test",
    ).launch(share=True)