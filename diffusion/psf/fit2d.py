import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from .log2d import LoGDetector
from .lsq2d import LSQ_BFGS
from psf.psf2d import *
from numpy.linalg import inv
             
class PipelineLocalize:
    def __init__(self,stack):
        self.stack = stack
      
    def localize(self,threshold=0.5,plot_spots=False,plot_fit=False,fit=True):
        nt,nx,ny = self.stack.shape
        spotst = []
        for n in range(nt):
            print(f'Det in frame {n}')
            framed = self.stack[n]
            log = LoGDetector(framed,threshold=threshold)
            spots = log.detect() #image coordinates
            if plot_spots:
                log.show(); plt.show()
            if fit:
                spots = self.fit(framed,spots,plot_fit=plot_fit)
            spots = spots.assign(frame=n)
            spotst.append(spots)
        spotst = pd.concat(spotst)
        return spotst
                
    def fit(self,frame,spots,patchw=3,N0=1.0,sigma=1.0,max_iters=1000,plot_fit=False):
        for i in spots.index:
            start = time.time()
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            theta0 = np.array([patchw,patchw,sigma,N0])
            opt = LSQ_BFGS(theta0,adu)
            theta_lsq, conv =\
            opt.optimize(max_iters=max_iters,plot_fit=plot_fit)
            dx = theta_lsq[1] - patchw; dy = theta_lsq[0] - patchw
            spots.at[i, 'x_lsq'] = x0 + dx
            spots.at[i, 'y_lsq'] = y0 + dy
            spots.at[i, 'sigma'] = theta_lsq[2]
            spots.at[i, 'N0'] = theta_lsq[3]
            spots.at[i, 'conv'] = conv
            end = time.time()
            elapsed = end-start
            print(f'Fit spot {i} in {elapsed} sec')
            
        return spots

