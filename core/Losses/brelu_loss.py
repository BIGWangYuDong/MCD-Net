import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
from Dehaze.core.Losses.builder import LOSSES



@LOSSES.register_module()
class BRELULoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(BRELULoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, img1):
        img_brelu = img1.clamp(0,1)
        bs,c,a,b=img1.shape
        loss = torch.log((img1 - img_brelu).abs() + 1).sum()/bs/c/a/b
        loss = self.loss_weight * loss
        return loss