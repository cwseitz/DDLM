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
from model.deep_storm import NeuralEstimator2D
from BaseSMLM.utils import BasicKDE
from utils.errors import errors2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
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
    niters = 500
    nsamples = 1
    plot = False

    all_x_err = []; all_y_err = []; all_inter = []; all_union = []
    all_x_terr = []; all_y_terr = []; all_tinter = []; all_tunion = []
    all_fp = []; all_fn = []; all_tfp = []; all_tfn = []
    for n in range(niters):
    
        disc2D = Disc2D(nx,ny)
        X,_,theta = disc2D.forward(radius,nspots,N0=1000,show=False)
        X = X[np.newaxis,np.newaxis,:,:]
        tcoords,Z = encoder.forward(X)
        tcoords = tcoords[['x','y']].values
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

        fig,ax=plt.subplots(1,2)
        ax[0].imshow(stack_avg,cmap='gray')
        ax[1].imshow(Z[0,0],cmap='gray')
        plt.show()

        """
        coords = blob_log(stack_avg,min_sigma=2,max_sigma=3,
                          num_sigma=5,threshold=0.2,exclude_border=5)
        tcoords = blob_log(Z[0,0],min_sigma=2,max_sigma=3,
                          num_sigma=5,threshold=0.2,exclude_border=5)
        print(coords,tcoords)
                          
        coords = coords[:,:2]
        xerr,yerr,inter,union,fp,fn = errors2d(4*theta[:2,:].T,coords)
        txerr,tyerr,tinter,tunion,tfp,tfn = errors2d(4*theta[:2,:].T,4*tcoords)
        all_x_err += list(xerr); all_y_err += list(yerr)
        all_inter.append(inter); all_union.append(union)
        all_x_terr += list(txerr); all_y_terr += list(tyerr)
        all_tinter.append(tinter); all_tunion.append(tunion)  
        all_fp.append(fp); all_fn.append(fn)
        all_tfp.append(tfp); all_tfn.append(tfn)
        """
        
        """
        if plot:
            fig,ax=plt.subplots(1,3)
            ax[0].imshow(np.squeeze(visuals['LR']),cmap='gray')
            ax[0].scatter(theta[1,:],theta[0,:],marker='x',color='red',s=5)
            ax[1].imshow(np.squeeze(visuals['HR']),cmap='gray')
            ax[1].scatter(4*theta[1,:],4*theta[0,:],marker='x',color='red',s=5)
            ax[2].imshow(pred,cmap='gray')
            ax[2].scatter(4*theta[1,:],4*theta[0,:],marker='x',color='red',s=5)
            ax[2].scatter(coords[:,1],coords[:,0],marker='x',color='blue',s=5)
            ax[2].scatter(4*tcoords[:,1],4*tcoords[:,0],marker='x',color='green',s=5)
            plt.show()
        """


        
fig,ax=plt.subplots(1,2)

x_std = np.std(np.array(all_x_err))
y_std = np.std(np.array(all_y_err))
precision = niters*nspots / (niters*nspots + sum(all_fp))
recall = niters*nspots / (niters*nspots + sum(all_fn))
#jacc = np.sum(np.array(all_inter))/np.sum(np.array(all_union))

x_tstd = np.std(np.array(all_x_terr))
y_tstd = np.std(np.array(all_y_terr))
tprecision = niters*nspots / (niters*nspots + sum(all_tfp))
trecall = niters*nspots / (niters*nspots + sum(all_tfn))
#tjacc = np.sum(np.array(all_tinter))/np.sum(np.array(all_tunion))

print(x_std,y_std,precision,recall)
print(x_tstd,y_tstd,tprecision,trecall)

ax[0].hist(all_x_err,bins=20,color='black',alpha=0.3)
ax[1].hist(all_y_err,bins=20,color='black',alpha=0.3)
ax[0].hist(all_x_terr,bins=20,color='red',alpha=0.3)
ax[1].hist(all_y_terr,bins=20,color='red',alpha=0.3)
plt.show()

