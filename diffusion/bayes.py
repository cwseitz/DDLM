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
from encode.localize import NeuralEstimator2D
from BaseSMLM.utils import BasicKDE
from psf import PipelineLocalize


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
        #fig,ax=plt.subplots(1,2)
        #ax[0].imshow(pred[0])
        #ax[1].imshow(pred[-10]); plt.show()
        stack.append(out)
        
    stack = np.array(stack)
    stack_avg = np.mean(stack,axis=0)
    stack_var = np.var(stack,axis=0)  
    return stack_avg,stack_var

disc2D = Disc2D(20,20)

X1,_,theta1 = disc2D.forward(7.0,1,sigma=0.92,N0=200,show=False)
X1 = X1[np.newaxis,np.newaxis,:,:]
_,Z1 = encoder.forward(X1)
Z1 = Z1/Z1.max()
Z1 = Z1[np.newaxis,np.newaxis,:,:]
kde1 = BasicKDE(theta1[:2,:].T)
S1 = kde1.forward(20,sigma=1.5,upsample=4)
S1 = S1/S1.max()
S1 = S1[np.newaxis,np.newaxis,:,:]
stack_avg1,stack_var1 = ddpm(X1,S1)

X2,_,theta2 = disc2D.forward(7.0,10,sigma=0.92,N0=200,show=False)
X2 = X2[np.newaxis,np.newaxis,:,:]
_,Z2 = encoder.forward(X2)
Z2 = Z2/Z2.max()
Z2 = Z2[np.newaxis,np.newaxis,:,:]
kde2 = BasicKDE(theta2[:2,:].T)
S2 = kde2.forward(20,sigma=1.5,upsample=4)
S2 = S2/S2.max()
S2 = S2[np.newaxis,np.newaxis,:,:]
stack_avg2,stack_var2 = ddpm(X2,S2)

X3,_,theta3 = disc2D.forward(7.0,10,sigma=0.92,N0=200,show=False)
X3 = X3[np.newaxis,np.newaxis,:,:]
_,Z3 = encoder.forward(X3)
Z3 = Z3/Z3.max()
Z3 = Z3[np.newaxis,np.newaxis,:,:]
kde3 = BasicKDE(theta3[:2,:].T)
S3 = kde3.forward(20,sigma=1.5,upsample=4)
S3 = S3/S3.max()
S3 = S3[np.newaxis,np.newaxis,:,:]
stack_avg3,stack_var3 = ddpm(X3,S3)

X4,_,theta4 = disc2D.forward(7.0,10,sigma=0.92,N0=200,show=False)
X4 = X4[np.newaxis,np.newaxis,:,:]
_,Z4 = encoder.forward(X4)
Z4 = Z4/Z4.max()
Z4 = Z4[np.newaxis,np.newaxis,:,:]
kde4 = BasicKDE(theta4[:2,:].T)
S4 = kde4.forward(20,sigma=1.5,upsample=4)
S4 = S4/S4.max()
S4 = S4[np.newaxis,np.newaxis,:,:]
stack_avg4,stack_var4 = ddpm(X4,S4)

fig,ax=plt.subplots(4,3,figsize=(5,7))

stack_var1 = gaussian(stack_var1,sigma=1.0)
stack_var2 = gaussian(stack_var2,sigma=1.0)
stack_var3 = gaussian(stack_var3,sigma=1.0)
stack_var4 = gaussian(stack_var4,sigma=1.0)

bins = np.linspace(0.001,0.03,100)
vals1,bins1 = np.histogram(np.sqrt(stack_var1).flatten(),
                           bins=bins,density=True)
vals2,bins2 = np.histogram(np.sqrt(stack_var2).flatten(),
                           bins=bins,density=True)
vals3,bins3 = np.histogram(np.sqrt(stack_var3).flatten(),
                           bins=bins,density=True)

ax[0,0].imshow(X1[0,0],cmap='gray')
ax[1,0].imshow(X2[0,0],cmap='gray')
ax[2,0].imshow(X3[0,0],cmap='gray')
ax[3,0].imshow(X4[0,0],cmap='gray')

ax[0,1].imshow(Z1[0,0],cmap='gray')
ax[1,1].imshow(Z2[0,0],cmap='gray')
ax[2,1].imshow(Z3[0,0],cmap='gray')
ax[3,1].imshow(Z4[0,0],cmap='gray')

vmin = 0.0; vmax = 0.025
im = ax[0,2].imshow(np.sqrt(stack_var1),cmap='coolwarm',vmin=vmin,vmax=vmax)
ax[1,2].imshow(np.sqrt(stack_var2),cmap='coolwarm',vmin=vmin,vmax=vmax)
ax[2,2].imshow(np.sqrt(stack_var3),cmap='coolwarm',vmin=vmin,vmax=vmax)
ax[3,2].imshow(np.sqrt(stack_var4),cmap='coolwarm',vmin=vmin,vmax=vmax)
plt.colorbar(im, ax=ax[0,2], location='top')

for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])
    
plt.subplots_adjust(hspace=0.05,wspace=0.05)
plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/Bayes.png',dpi=300)
plt.show()








