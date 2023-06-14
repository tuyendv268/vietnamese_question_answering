import torch
from torch import nn
from torchmetrics.functional import pairwise_cosine_similarity

device = "cpu" if not torch.cuda.is_available() else "cuda"

class QA_Loss(torch.nn.Module):
    def __init__(self, batch_size=32, mode="dual"):
        super(QA_Loss, self).__init__()
        
        self.batch_size = batch_size
        if mode == "dual":
            self.temperature = nn.Parameter(torch.ones(1)).to(device)
        elif mode == "cross":
            self.temperature = nn.Parameter(1*torch.ones(1)).to(device)
        
        assert mode in ["dual", "cross"]
        self.mode = mode
    
    def dual_forward(self, question_embedding, context_embeddings):
        # similarity = torch.matmul(question_embedding, context_embeddings.T)
        similarity = pairwise_cosine_similarity(question_embedding, context_embeddings)
        return similarity[0]
    
        
    def forward(self, labels, query_embedding=None, similarity=None, context_embedding=None):
        if self.mode == "dual":
            similarity = self.dual_forward(
                question_embedding=query_embedding, 
                context_embeddings=context_embedding)
            labels = torch.nn.functional.one_hot(labels, similarity.size(0))
        elif self.mode == "cross":
            similarity = similarity.transpose(0, 1)
            labels = torch.nn.functional.one_hot(labels, similarity.size(1))
            
        loss = similarity - torch.nn.functional.log_softmax(torch.mul(labels, similarity), dim=0)
        loss = torch.mean(loss)
                
        return loss, self.temperature
    