import torch

class HLTAR_Loss(torch.nn.Module):
    def __init__(self):
        super(HLTAR_Loss, self).__init__()
        self.cre = torch.nn.CrossEntropyLoss()
        
    def forward(self, labels, logits):
        loss = self.cre(logits, labels.float())
                
        return loss