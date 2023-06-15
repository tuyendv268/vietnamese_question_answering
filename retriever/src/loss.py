import torch
from torch.nn import Module
from torchmetrics.functional import pairwise_cosine_similarity

class Loss(Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cre = torch.nn.CrossEntropyLoss()
        
    def contrastive_loss(self, labels, logits, masks, temperature=1):
        # exp = torch.exp(logits/temperature)
        # exp = torch.masked_fill(input=exp, mask=~masks, value=0)
        # loss = -torch.log(torch.sum(torch.mul(exp, labels), dim=1) / torch.sum(exp, dim=1))
        # loss = torch.mean(loss)
        logits = logits/temperature
        logits = torch.masked_fill(input=logits, mask=~masks, value=-1000)
        loss = self.cre(logits, labels.float())
        
        return loss
    
    def caculate_cosin_loss(self, labels, query_embeddings, context_embeddings, masks, temperature=1):
        context_embeddings = context_embeddings.reshape(labels.size(0), labels.size(1), 768)
        query_embeddings = query_embeddings.unsqueeze(1)
        logits = [pairwise_cosine_similarity(x, y) for x, y in zip(query_embeddings, context_embeddings)]            
        logits = torch.cat(logits, dim=0)
        loss = self.contrastive_loss(labels, logits, masks, temperature=temperature) 
        
        return loss, logits

    def caculate_dot_product_loss(self, labels, query_embeddings, context_embeddings, masks, temperature=4):
        context_embeddings = context_embeddings.reshape(labels.size(0), labels.size(1), -1)
        query_embeddings = query_embeddings.unsqueeze(-1)
        
        logits = torch.matmul(context_embeddings, query_embeddings).squeeze(-1)
        loss = self.contrastive_loss(labels, logits, masks, temperature=temperature) 
        
        return loss, logits
