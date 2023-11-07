import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from qSMLM.localize import LoGDetector
from qSMLM.utils import computeCov
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
        self.dump_config()
    def dump_config(self):
        with open(self.analpath+self.dataset.name+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)        
    def localize(self,plot_spots=False,peak_thresh=0.1,plot_cov=False):
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
        covs = self.compute_cov(spots,plot=plot_cov)
        return covs
    def compute_cov(self,spots,plot=False):
        config = self.config
        patchw = self.config['patchw']
        nt,nx,ny = self.stack.shape
        covsize = 2*(2*patchw+1) - 1
        covs = []
        for i in spots.index:
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            patch = self.stack[:,x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            CovR,VarR = computeCov(patch)
            X = np.nan_to_num(CovR/VarR); covs.append(X)
            if plot:
                fig,ax=plt.subplots(1,4)
                ax[0].imshow(CovR,cmap='coolwarm')
                ax[1].imshow(VarR,cmap='coolwarm')
                ax[2].imshow(X,cmap='coolwarm')
                ax[3].imshow(np.sum(patch,axis=0),cmap='gray')
                plt.show()
        covs = np.array(covs)
        return covs
    

