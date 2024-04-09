import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


# calculates the jaccard coefficient approximation using per-voxel probabilities
def jaccard_coeff(pred, target):
    """
    jaccard index = TP / (TP + FP + FN)
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    # smoothing parameter
    smooth = 1e-6
    
    # number of examples in the batch
    N = pred.size(0)

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(N,-1)
    tflat = target.contiguous().view(N,-1)
    intersection = (iflat * tflat).sum(1)
    jacc_index = (intersection / (iflat.sum(1) + tflat.sum(1) - intersection + smooth)).mean()

    return jacc_index
