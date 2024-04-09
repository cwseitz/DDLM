import numpy as np
import json
import matplotlib.pyplot as plt
from generators import *
from make.kde import BasicKDE
from skimage.io import imsave
from skimage import exposure
from skimage.filters import gaussian
from scipy.ndimage import zoom
from skimage.exposure import rescale_intensity

class TrainDataset:
    """Training dataset object"""
    def __init__(self,ngenerate):
        self.ngenerate = ngenerate
        self.X_type = np.int16
        self.Z_type = np.float32
    def make_dataset(self,generator,args,kwargs,upsample=8,sigma_kde=3.0,sigma_gauss=1.0,show=False):
        pad = upsample // 2
        Xs = []; Zs = []; Ss = []
        for n in range(self.ngenerate):
            print(f'Generating sample {n}')
            G = generator.forward(*args,**kwargs)
            theta = G[2][:2,:].T
            S = G[1]; X = G[0]
            nx,ny = X.shape
            Z = BasicKDE(theta).forward(nx,upsample=upsample,sigma=sigma_kde)
            Z = rescale_intensity(Z,out_range=self.Z_type)
            Xs.append(X); Zs.append(Z); Ss.append(S)
            if show:
                self.show(X,Y,Z,S,theta)
        Ss = np.array(Ss,dtype=np.int16)
        return (np.array(Xs),np.array(Zs),Ss)
 

