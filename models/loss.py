from __future__ import print_function, division
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5, margin=2):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        # loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)

    # criterion = nn.BCELoss()
    # bce = criterion(prediction, target)
    # bce = F.binary_cross_entropy(prediction, target)
    # bce = torch.sum((1-target)*torch.pow(prediction,2 ) + \
    #                                    target * torch.pow(torch.clamp(margin - prediction, min=0.0),2))
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


class FocalLoss2d(nn.Module):

    def __init__(self,alpha=0.25, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)
        alpha = self.alpha

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.binary_cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -1 * alpha * ((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()