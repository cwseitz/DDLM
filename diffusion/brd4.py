import numpy as np
import os
import re
import matplotlib.pyplot as plt

from encode.localize import NeuralEstimator2D
from skimage.io import imread, imsave
from glob import glob

path = '/research3/shared/cwseitz/Data/240821/'
filename = '240821_Hela-BRD4-AF488-20mW-40ms-2-snip2.tif'

stack = imread(path+filename)
#stack = stack[0]
stack[stack < 10] = 0
stack = 0.1*stack
stack = stack[np.newaxis,:,:]
nt,nx,ny = stack.shape

config = {
'modelpath': 'experiments/encoder/models/encoder/60x/100pix/',
'modelname': '0821_133904'
}

for n in range(nt):
    encoder = NeuralEstimator2D(config)
    lr = stack[n]
    lr = lr[np.newaxis,np.newaxis,:,:]
    outputs = encoder.forward(lr)
    outputs[outputs < 0.05] = 0
    fig,ax=plt.subplots(1,2,figsize=(6,3))
    ax[0].imshow(lr[0,0],cmap='gray')
    ax[1].imshow(outputs,cmap='gray',vmin=0.0,vmax=0.3)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].set_xticks([]); ax[1].set_yticks([])
    plt.tight_layout()
    plt.show()


