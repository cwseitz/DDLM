import torch
import numpy as np
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import matplotlib.pyplot as plt
import os
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
from skimage.io import imsave
from skimage.filters import gaussian
from dataset import Dataset
from generators import *
from model.deep_storm import NeuralEstimator2D
from BaseSMLM.utils import BasicKDE
from psf import PipelineLocalize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    diffusion = Model.create_model(opt)
    encoder = NeuralEstimator2D(opt['deep_storm'])

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    nx = ny = 20
    radius = 7.0
    nspots = 10
    nsamples = 10
    
    disc2D = Disc2D(nx,ny)
    X,_,theta = disc2D.forward(radius,nspots,N0=1000,show=False)
    X = X[np.newaxis,np.newaxis,:,:]
    _,Z = encoder.forward(X)
    Z = Z[np.newaxis,np.newaxis,:,:]
    kde = BasicKDE(theta[:2,:].T)
    S = kde.forward(nx,sigma=1.5,upsample=4)
        
    data_dict = {}
    data_dict['HR'] = torch.from_numpy(Z)
    data_dict['SR'] = torch.from_numpy(Z)
    data_dict['LR'] = torch.from_numpy(X)
    data_dict['Index'] = torch.from_numpy(np.array([0]))
    
    stack = []
    for n in range(nsamples):
        diffusion.feed_data(data_dict)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=True)
        pred = np.squeeze(visuals['SR'])[-1].numpy()
        stack.append(pred)
        
    stack = np.array(stack)
    stack_avg = np.mean(stack,axis=0)
    stack_var = np.var(stack,axis=0)
    
    fig,ax=plt.subplots(1,3)
    rgb = np.zeros((80,80,3))
    stack_var = stack_var/stack_var.max()
    stack_var = gaussian(stack_var,sigma=1.0)
    Z = Z[0,0]; Z = Z/Z.max()
    
    rgb[:,:,0] = stack_var
    rgb[:,:,2] = Z
    ax[0].imshow(stack_avg,cmap='gray')
    ax[1].imshow(stack_var,cmap='coolwarm')
    ax[2].imshow(rgb)
    ax[2].scatter(4*theta[1,:],4*theta[0,:],marker='x',color='lime',s=5)
    plt.show()

    #pipe = PipelineLocalize(stack)
    #spots = pipe.localize(threshold=0.2,plot_spots=True,fit=True)
        
    





