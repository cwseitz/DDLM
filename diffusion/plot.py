import torch
import numpy as np
import diffusion.data as Data
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
from BaseSMLM.generators import *


class CRB2D:
    def __init__(self,config):
        self.config = config
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]  
             
    def forward(self,N0space,npix):
        crlb_n0 = np.zeros((len(N0space),4))
        for i,n0 in enumerate(N0space):
            theta0 = np.array([3.5,3.5,0.92,n0])
            crlb_n0[i] = crlb2d(theta0,self.cmos_params)
        return crlb_n0

class Figure_1:
    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='config/ddpm.json',
                            help='JSON file for configuration')
        parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
        parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
        parser.add_argument('-debug', '-d', action='store_true')
        parser.add_argument('-enable_wandb', action='store_true')
        parser.add_argument('-log_infer', action='store_true')
        args = parser.parse_args()
        opt = Logger.parse(args)
        opt = Logger.dict_to_nonedict(opt)
        return opt
        
    def __init__(self):
        opt = self.setup()
        diffusion = Model.create_model(opt)
        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')

    def plot(self,nx=20,ny=20,radius=0.92,nspots=3,
             N0=1000,sigma_ring=0.1,rand_center=True):

        args = [radius,nspots]
        kwargs = {'N0':N0,'sigma_ring':sigma_ring,'rand_center':rand_center}
        generator = GaussianRing2D(nx,ny)
        dataset = Dataset(1)
        X,Y,Z,S = dataset.make_dataset(generator,args,kwargs,show=True,
                                       interpolate=False,upsample=4,
                                       sigma_kde=1.5,sigma_gauss=0.5)
        data_dict = {}
        X = np.expand_dims(X,0)
        Y = np.expand_dims(Y,0)
        Z = np.expand_dims(Z,0)
        data_dict['HR'] = torch.from_numpy(Z)
        data_dict['SR'] = torch.from_numpy(Y)
        data_dict['LR'] = torch.from_numpy(X)
        data_dict['Index'] = torch.from_numpy(np.array([0]))
        nsamples=1
        for n in range(nsamples):
            diffusion.feed_data(data_dict)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals(need_LR=True)
            fig,ax=plt.subplots(1,3)
            ax[0].imshow(np.squeeze(visuals['LR']),cmap='gray')
            ax[1].imshow(np.squeeze(visuals['SR'])[-1],cmap='gray')
            ax[2].imshow(np.squeeze(visuals['HR']),cmap='gray')
            plt.show()
 

figure = Figure_1()

