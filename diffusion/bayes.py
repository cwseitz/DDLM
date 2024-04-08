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
    disc2D = Disc2D(nx,ny)
    X,_,theta = disc2D.forward(radius,nspots,N0=200,show=True)
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
    nsamples=100
    
    stack = []
    for n in range(nsamples):
        diffusion.feed_data(data_dict)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=True)
        pred = np.squeeze(visuals['SR'])[-1].numpy()
        pred = pred - pred.min()
        stack.append(pred)
        
    stack = np.array(stack)
    pipe = PipelineLocalize(stack)
    spots = pipe.localize(threshold=0.2,plot_spots=False,fit=True)
    fig,ax=plt.subplots()
    ax.imshow(pred,cmap='gray')
    ax.scatter(spots['y_lsq'],spots['x_lsq'],marker='x',color='blue',s=10)
    plt.show()
    print(spots[['x_lsq','y_lsq','sigma','N0','conv']])




