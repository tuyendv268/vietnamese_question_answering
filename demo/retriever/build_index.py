from tqdm import tqdm

from torch.utils.data import DataLoader
from utils import save_index
import py_vncorenlp
from transformers import AutoTokenizer, AutoModel
from dataset import Dual_Infer_Dataset
from model import Dual_Model
import torch

def init_model_and_tokenizer(pretrained_dir, vncorenlp_dir):    
    tokenizer = AutoTokenizer.from_pretrained(
        'keepitreal/vietnamese-sbert', cache_dir=pretrained_dir)
    plm = AutoModel.from_pretrained(
        'keepitreal/vietnamese-sbert', cache_dir=pretrained_dir)
    
    segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], save_dir=vncorenlp_dir)
    
    model = Dual_Model(
        tokenizer=tokenizer, model=plm, segmenter=segmenter)
    
    return model, tokenizer, segmenter

@torch.no_grad()
def build_index(passage_path, pretrained_dir, vncorenlp_dir, output_dir):
    with open(passage_path, "r", encoding="utf-8") as f:
        passages = [passage.strip() for passage in f.readlines()]
        
    model, tokenizer, segmenter= init_model_and_tokenizer(
        pretrained_dir=pretrained_dir,
        vncorenlp_dir=vncorenlp_dir
        )
    model.eval()
    
    dataset = Dual_Infer_Dataset(
        texts=passages, 
        tokenizer=tokenizer, 
        segmenter=segmenter
        )
    
    dataloader = DataLoader(
        dataset=dataset, 
        collate_fn=dataset.dual_collate_fn, 
        batch_size=2, 
        shuffle=False, 
        drop_last=False)

    texts, embeddings = [], []
    for batch in tqdm(dataloader, desc="build index"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        _texts = batch["texts"]
        _embeddings = model.extract_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        texts += _texts
        embeddings.append(_embeddings)

    embeddings = torch.vstack(embeddings)
    
    save_index(
        output_dir=output_dir,
        texts=texts,
        embeddings=embeddings.detach().numpy()
    )

if __name__ == "__main__":
    passage_path = "inputs/passage.txt"
    pretrained_dir = "/home/tuyendv/Desktop/retriever/pretrained"
    vncorenlp_dir = "/home/tuyendv/Desktop/retriever/vncorenlp"
    output_dir = "/home/tuyendv/Desktop/retriever/demo/outputs"
    
    build_index(
        passage_path=passage_path,
        pretrained_dir=pretrained_dir,
        vncorenlp_dir=vncorenlp_dir,
        output_dir=output_dir
    )
        
        
    
