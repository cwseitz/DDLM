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
from make.kde import BasicKDE
from core.wandb_logger import WandbLogger
from skimage.io import imsave
from skimage.filters import gaussian
from generators import *
from encode.localize import NeuralEstimator2D
from matplotlib.ticker import LogFormatter 


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
    
def ddpm(X,Z,nsamples=10):
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
        pred = np.squeeze(visuals['SR'].numpy())
        nt,nx,ny = pred.shape
        out = pred[-1]
        stack.append(out)
        
    stack = np.array(stack)
    stack_avg = np.mean(stack,axis=0)
    stack_var = np.var(stack,axis=0)  
    return stack_avg,stack_var

nsamples = 100
path = '/research3/shared/cwseitz/Data/240821/'
filename = '240821_Hela-BRD4-AF488-20mW-40ms-2-snip3.tif'
X1 = imread(path+filename)
X1[X1 < 10] = 0
X1 = 0.1*X1

"""
X1 = X1[np.newaxis,np.newaxis,:,:]
Z1 = encoder.forward(X1)
Z1 = Z1/Z1.max()
Z1 = Z1[np.newaxis,np.newaxis,:,:]
kde1 = BasicKDE(theta1[:2,:].T)
S1 = kde1.forward(20,sigma=1.5,upsample=4)
S1 = S1/S1.max()
S1 = S1[np.newaxis,np.newaxis,:,:]
stack_avg1,stack_var1 = ddpm(X1,S1,nsamples=nsamples)
"""

