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
from skimage.feature import blob_log
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
from skimage.io import imsave
from dataset import Dataset
from generators import *
from encode.localize import NeuralEstimator2D
from BaseSMLM.utils import BasicKDE
from utils.errors import errors2d

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
        pred = np.squeeze(visuals['SR'])[-1].numpy()
        stack.append(pred)
        
    stack = np.array(stack)
    stack_avg = np.mean(stack,axis=0)
    stack_var = np.var(stack,axis=0)  
    return stack_avg,stack_var

diffusion = Model.create_model(opt)
encoder = NeuralEstimator2D(opt['deep_storm'])
diffusion.set_new_noise_schedule(
    opt['model']['beta_schedule']['val'], schedule_phase='val')


niters = 1000
nsamples = 1
nspots = 10
plot = False
gen = Disc2D(20,20)

all_x_err = []; all_y_err = []
all_fp = []; all_fn = []

for n in range(niters):
    X,_,theta = gen.forward(7.0,nspots,N0=200,show=False)
    X = X[np.newaxis,np.newaxis,:,:]
    _,Z = encoder.forward(X)
    Z = Z[np.newaxis,np.newaxis,:,:]
    kde = BasicKDE(theta[:2,:].T)
    S = kde.forward(20,sigma=1.5,upsample=4)
    coords = blob_log(Z[0,0],min_sigma=2,max_sigma=3,
                      num_sigma=5,threshold=0.2,exclude_border=5)
    #stack_avg,stack_var = ddpm(X,Z)
    
    coords = coords[:,:2]
    xerr,yerr,inter,union,fp,fn = errors2d(4*theta[:2,:].T,coords)
    all_x_err += list(xerr); all_y_err += list(yerr)
    all_fp.append(fp); all_fn.append(fn)
    
    if plot:
        fig,ax=plt.subplots(1,2)
        ax[0].imshow(X[0,0],cmap='gray')
        ax[1].imshow(Z[0,0],cmap='gray')
        ax[1].scatter(4*theta[1,:],4*theta[0,:],marker='x',color='red',s=5)
        ax[1].scatter(coords[:,1],coords[:,0],marker='x',color='blue',s=5)
        plt.show()
        
x_std = np.std(np.array(all_x_err))
y_std = np.std(np.array(all_y_err))
tp = niters*nspots
precision = tp / (tp + sum(all_fp))
recall = tp / (tp + sum(all_fn))
print(x_std,y_std,precision,recall)

sigma_crlb = 10.4
x = np.linspace(-3*sigma_crlb, 3*sigma_crlb, 1000)
gaussian_pdf = np.exp(-0.5 * (x / sigma_crlb) ** 2)

all_x_err = 27*np.array(all_x_err)
all_y_err = 27*np.array(all_y_err)

hist_x, bins_x = np.histogram(all_x_err, bins=50, density=False)
hist_y, bins_y = np.histogram(all_y_err, bins=50, density=False)
hist_x = hist_x/hist_x.max()
hist_y = hist_y/hist_y.max()

fig, ax = plt.subplots(1,2)
ax[0].bar(bins_x[:-1], hist_x, width=np.diff(bins_x), color='white', edgecolor='black')
ax[1].bar(bins_y[:-1], hist_y, width=np.diff(bins_x), color='white', edgecolor='black')
ax[0].plot(x,gaussian_pdf,color='red')
ax[1].plot(x,gaussian_pdf,color='red')
ax[0].set_xlabel('x error')
ax[1].set_xlabel('y error')
ax[0].set_ylabel('Density')
ax[1].set_ylabel('Density')
plt.tight_layout()
plt.show()

