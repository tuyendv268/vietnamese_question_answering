import torch
from torch.nn import Module
from torchmetrics.functional import pairwise_cosine_similarity

class Loss(Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cre = torch.nn.CrossEntropyLoss()
        
    def contrastive_loss(self, labels, logits, masks, temperature=1):
        logits = logits/temperature
        logits = torch.masked_fill(input=torch.exp(logits), mask=~masks.flatten(), value=-1000)
        loss = self.cre(logits, labels)
        
        return loss
    
    def caculate_cosin_loss(self, labels, query_embeddings, context_embeddings, masks, temperature=1):
        context_embeddings = context_embeddings.reshape(labels.size(0), labels.size(1), 768)
        query_embeddings = query_embeddings.unsqueeze(1)
        logits = [pairwise_cosine_similarity(x, y) for x, y in zip(query_embeddings, context_embeddings)]            
        logits = torch.cat(logits, dim=0)
        loss = self.contrastive_loss(labels, logits, masks, temperature=temperature) 
        
        return loss, logits

    def caculate_dot_product_loss(self, labels, query_embeddings, context_embeddings, masks, temperature=4):
        batch_size, n_passage_per_query = labels.shape[0], labels.shape[1]
        assert labels[:,0].sum() == batch_size

        logits = torch.matmul(query_embeddings, context_embeddings.transpose(0, 1)).squeeze(-1)
        labels = torch.arange(0, batch_size, device=logits.device) * n_passage_per_query
        
        loss = self.contrastive_loss(labels, logits, masks, temperature=temperature) 
        
        return loss, logits
