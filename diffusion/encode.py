import numpy as np
import os
import re

from encode.localize import NeuralEstimator2D
from skimage.io import imread, imsave
from glob import glob

savepath = 'dataset/high_snr/'
prefix = 'Diffusion'

os.makedirs(savepath+'sr_20_80',exist_ok=True)
lrs = sorted(glob(savepath+'lr_20/'+'*_lr*.tif'),key=lambda x: int(x.split("-")[-1].split(".")[0]))
nsamples = len(lrs)

config = {
'thresh_cnn': 30, 
'radius': 3, 
'pixel_size_lateral': 108.3,
'modelpath': 'experiments/encoder/high_snr/models/encoder/',
'modelname': '0413_125847'
}

for n,lr in enumerate(lrs):
    lr = imread(lr)
    estimator = NeuralEstimator2D(config)
    lr = lr[np.newaxis,np.newaxis,:,:]
    spots,outputs = estimator.forward(lr)
    imsave(savepath+'sr_20_80/'+prefix+f'_sr-{n}.tif',outputs)


