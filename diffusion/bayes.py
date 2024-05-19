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
from tensorboardX import SummaryWriter
from skimage.io import imsave
from skimage.filters import gaussian
from dataset import Dataset
from generators import *
from encode.localize import NeuralEstimator2D
from psf import PipelineLocalize
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

disc2D = Disc2D(20,20)
nsamples = 100

X1,_,theta1 = disc2D.forward(7.0,10,sigma=0.92,N0=200,show=False)
X1 = X1[np.newaxis,np.newaxis,:,:]
_,Z1 = encoder.forward(X1)
Z1 = Z1/Z1.max()
Z1 = Z1[np.newaxis,np.newaxis,:,:]
kde1 = BasicKDE(theta1[:2,:].T)
S1 = kde1.forward(20,sigma=1.5,upsample=4)
S1 = S1/S1.max()
S1 = S1[np.newaxis,np.newaxis,:,:]

X2,_,theta2 = disc2D.forward(7.0,10,sigma=0.92,N0=200,show=False)
X2 = X2[np.newaxis,np.newaxis,:,:]
_,Z2 = encoder.forward(X2)
Z2 = Z2/Z2.max()
Z2 = Z2[np.newaxis,np.newaxis,:,:]
kde2 = BasicKDE(theta2[:2,:].T)
S2 = kde2.forward(20,sigma=1.5,upsample=4)
S2 = S2/S2.max()
S2 = S2[np.newaxis,np.newaxis,:,:]

X3,_,theta3 = disc2D.forward(7.0,10,sigma=0.92,N0=200,show=False)
X3 = X3[np.newaxis,np.newaxis,:,:]
_,Z3 = encoder.forward(X3)
Z3 = Z3/Z3.max()
Z3 = Z3[np.newaxis,np.newaxis,:,:]
kde3 = BasicKDE(theta3[:2,:].T)
S3 = kde3.forward(20,sigma=1.5,upsample=4)
S3 = S3/S3.max()
S3 = S3[np.newaxis,np.newaxis,:,:]

#fig,ax=plt.subplots()
#ax.imshow(S1[0,0],cmap='gray')
#ax.set_xticks([]); ax.set_yticks([])
#plt.savefig('/home/cwseitz/Desktop/KDE.png',dpi=300)
#plt.show()

stack_avg1,stack_var1 = ddpm(X1,S1,nsamples=nsamples)
stack_avg2,stack_var2 = ddpm(X2,S2,nsamples=nsamples)
stack_avg3,stack_var3 = ddpm(X3,S3,nsamples=nsamples)

fig,ax=plt.subplots(3,3,figsize=(5,5))

#stack_var1 = gaussian(stack_var1,sigma=0.5)
#stack_var2 = gaussian(stack_var2,sigma=0.5)
#stack_var3 = gaussian(stack_var3,sigma=0.5)

ax[0,0].imshow(X1[0,0],cmap='gray')
ax[0,0].set_xticks([0,10]); ax[0,0].set_yticks([0,10])
ax[1,0].imshow(X2[0,0],cmap='gray')
ax[1,0].set_xticks([0,10]); ax[1,0].set_yticks([0,10])
ax[2,0].imshow(X3[0,0],cmap='gray')
ax[2,0].set_xticks([0,10]); ax[2,0].set_yticks([0,10])


ax[0,1].imshow(S1[0,0],cmap='gray')
ax[0,1].set_xticks([0,40]); ax[0,1].set_yticks([0,40])
ax[1,1].imshow(S2[0,0],cmap='gray')
ax[1,1].set_xticks([0,40]); ax[1,1].set_yticks([0,40])
ax[2,1].imshow(S3[0,0],cmap='gray')
ax[2,1].set_xticks([0,40]); ax[2,1].set_yticks([0,40])


vmin = 0.0
#vmax = 0.025
vmax = 1.0
im = ax[0,2].imshow(stack_avg1,cmap='plasma',
                    vmin=vmin,vmax=vmax)
formatter = LogFormatter(10, labelOnlyBase=True) 
plt.colorbar(im,ax=ax[0,2],fraction=0.046,pad=0.04,label=r'$\sqrt{Var(y_k)}$',format=formatter)

ax[1,2].imshow(stack_avg2,cmap='plasma',
                    vmin=vmin,vmax=vmax)
ax[2,2].imshow(stack_avg3,cmap='plasma',
                    vmin=vmin,vmax=vmax)

ax[0,2].set_xticks([0,40]); ax[0,2].set_yticks([0,40])
ax[1,2].set_xticks([0,40]); ax[1,2].set_yticks([0,40])
ax[2,2].set_xticks([0,40]); ax[2,2].set_yticks([0,40])

plt.subplots_adjust(hspace=0.0,wspace=0.37)
plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/Bayes.png',dpi=300)
plt.show()








