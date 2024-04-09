import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def vae_loss(pred,target,mu,logvar,beta=1.0,latent_dim=5):
    #fig,ax=plt.subplots(1,2)
    #ax[0].imshow(pred[0,0].cpu().detach().numpy())
    #ax[1].imshow(target[0,0].cpu().detach().numpy())
    #plt.show()
    diff = pred-target

    #Compute reconstruction loss
    mse = nn.MSELoss()
    recons_loss = 0.5*(latent_dim*np.log(2*np.pi) + mse(pred,target))

    #Compute KL loss
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    #Compute total loss
    loss = recons_loss + beta*kld_loss
    return loss

def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)


    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


# create a 3D gaussian kernel
def GaussianKernel(shape=(7, 7, 7), sigma=1, normfactor=1):
    """
    3D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]) in 3D
    """
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    """
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
        h = h * normfactor
    """
    maxh = h.max()
    if maxh != 0:
        h /= maxh
        h = h * normfactor
    h = torch.from_numpy(h).type(torch.FloatTensor)
    h = h.unsqueeze(0)
    h = h.unsqueeze(1)
    return h


# define the 3D extended loss function from DeepSTORM

def KDE_loss3D(pred_bol, target_bol, factor=800):

    kernel = GaussianKernel()
    if pred_bol.is_cuda:
        kernel = kernel.cuda()
    # extract kernel dimensions
    N, C, D, H, W = kernel.size()
    
    # extend prediction and target to have a single channel
    target_bol = target_bol.unsqueeze(1)
    pred_bol = pred_bol.unsqueeze(1)

    # KDE for both input and ground truth spikes
    Din = F.conv3d(pred_bol, kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
    Dtar = F.conv3d(target_bol, factor*kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))

    # kde loss
    kde_loss = nn.MSELoss()(Din, Dtar)
    
    # final loss
    dice = dice_loss(pred_bol/factor, target_bol)

    final_loss = kde_loss + dice

    return final_loss



