import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from qSMLM.localize import LoGDetector
from qSMLM.utils import *
from numpy.linalg import inv
from skimage.filters import median

class PipelineG22D:
    def __init__(self,config,dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)     
    def localize(self,cov_tmax,plot_spots=False,peak_thresh=0.1,plot_cov=False,plot_ts=False):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        summed = np.sum(self.stack,axis=0)
        summed = summed/summed.max()
        med = median(summed)
        threshold = self.config['thresh_log']
        log = LoGDetector(med,min_sigma=1,max_sigma=3,threshold=threshold)
        spots = log.detect()
        if plot_spots:
            log.show(X=summed); plt.show()
        covs = self.compute_cov(spots,cov_tmax,plot=plot_cov,plot_ts=plot_ts)
        return covs
    def compute_cov(self,spots,tmax,plot=False,plot_ts=False):
        config = self.config
        patchw = self.config['patchw']
        nt,nx,ny = self.stack.shape
        covsize = 2*(2*patchw+1) - 1
        covs = []
        for i in spots.index:
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            patch = self.stack[:tmax,x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            if plot_ts:
                fig,ax=plt.subplots()
                ax.plot(np.sum(patch,axis=(1,2)),color='black')
                plt.show()
            Exy = Exy_em(patch); ExEy = ExEy_em(patch)
            X = np.nan_to_num(Exy/ExEy); covs.append(X)
            if plot:
                fig,ax=plt.subplots(1,4)
                ax[0].imshow(Exy,cmap='coolwarm')
                ax[1].imshow(ExEy,cmap='coolwarm')
                ax[2].imshow(X,cmap='coolwarm')
                ax[3].imshow(np.sum(patch,axis=0),cmap='gray')
                plt.show()
        covs = np.array(covs)
        return covs
        
class PipelineG22D_Win:
    def __init__(self,config,dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)     
    def localize(self,win_size,plot_spots=False,peak_thresh=0.1,plot_cov=False,plot_ts=False):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        summed = np.sum(self.stack,axis=0)
        summed = summed/summed.max()
        med = median(summed)
        threshold = self.config['thresh_log']
        log = LoGDetector(med,min_sigma=1,max_sigma=3,threshold=threshold)
        spots = log.detect()
        if plot_spots:
            log.show(X=summed); plt.show()
        covs = self.compute_cov(spots,win_size,plot=plot_cov,plot_ts=plot_ts)
        return covs
    def compute_cov(self,spots,win_size,plot=False,plot_ts=False):
        config = self.config
        patchw = self.config['patchw']
        nt,nx,ny = self.stack.shape
        covsize = 2*(2*patchw+1) - 1
        covs = []
        for i in spots.index:
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            patch = self.stack[:,x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            if plot_ts:
                fig,ax=plt.subplots()
                ax.plot(np.sum(patch,axis=(1,2)),color='black')
                plt.show()
            Exy = Exy_em_win(patch,win_size=win_size)
            ExEy = ExEy_em_win(patch,win_size=win_size)
            X = np.nan_to_num(Exy/ExEy); covs.append(X)
            if plot:
                fig,ax=plt.subplots(1,4)
                ax[0].imshow(Exy,cmap='coolwarm')
                ax[1].imshow(ExEy,cmap='coolwarm')
                ax[2].imshow(X,cmap='coolwarm')
                ax[3].imshow(np.sum(patch,axis=0),cmap='gray')
                plt.show()
        covs = np.array(covs)
        return covs  

