import torch

class Dual_Model(torch.nn.Module):
    def __init__(self, model, tokenizer, segmenter):
        super(Dual_Model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        
    def mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def extract_embedding(self, input_ids, attention_mask, token_type_ids=None):
        last_hidden_state = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask)[0]
        
        embedding = self.mean_pooling(
            last_hidden_state=last_hidden_state,
            attention_mask=attention_mask)
        
        return embedding
    