from reranker.model import Cross_Model
from transformers import RobertaModel, AutoModel, AutoTokenizer
import torch
AUTH_TOKEN = "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"

class Reranker():
    def __init__(self, pretrained_dir, ckpt_path) -> None:
        self.ckpt_path = ckpt_path
        self.pretrained_dir = pretrained_dir
        
        self.model, self.tokenizer = self.init_models_and_tokenizers()
        self.model.eval()
    
    def init_models_and_tokenizers(self):
        tokenizer = AutoTokenizer.from_pretrained(
            'nguyenvulebinh/vi-mrc-base', 
            cache_dir=self.pretrained_dir, use_auth_token=AUTH_TOKEN)
        
        plm = AutoModel.from_pretrained(
            "nguyenvulebinh/vi-mrc-base",
            cache_dir=self.pretrained_dir, use_auth_token=AUTH_TOKEN)
        
        model = Cross_Model(
            max_length=512, batch_size=4, device="cpu",
            model=plm, tokenizer=tokenizer, droprate=0)
        
        state_dict = torch.load(self.ckpt_path, map_location="cpu")        
        model.load_state_dict(state_dict["model"])

        return model, tokenizer
    
    def reranking(self, query, passages):
        scores, ranks = self.model.ranking(query=query, texts=passages)
        
        return scores, ranks
