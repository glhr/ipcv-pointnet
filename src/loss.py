import torch
from torch import nn

class PointNetLoss(nn.Module):
    def __init__(self, device, w=0.0001):
        super(PointNetLoss, self).__init__()
        self.w = w
        self.nll_loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, gt, pr, A_):
        A = A_.clone()
        #  Orthogonality constraint
        orth = torch.norm(torch.eye(A.shape[1]).to(self.device) - torch.matmul(A, A.transpose(1, 2)))
        loss = self.nll_loss(pr, gt) + self.w * orth
        return loss


