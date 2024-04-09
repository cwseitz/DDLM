import numpy as np
import os

from encode.localize import NeuralEstimator2D
from skimage.io import imread, imsave
from glob import glob

savepath = 'dataset/diffusion/'
prefix = 'Diffusion'

os.makedirs(savepath+'sr_20_80',exist_ok=True)
lrs = sorted(glob(savepath+'lr_20/'+'*_lr*.tif'))
nsamples = len(lrs)

config = {
'thresh_cnn': 30, 
'radius': 3, 
'pixel_size_lateral': 108.3,
'modelpath': 'experiments/encoder/models/encoder/',
'modelname': '0409_185447'
}

for n,lr in enumerate(lrs):
    lr = imread(lr)
    estimator = NeuralEstimator2D(config)
    lr = lr[np.newaxis,np.newaxis,:,:]
    spots,outputs = estimator.forward(lr)
    imsave(savepath+'sr_20_80/'+prefix+f'_sr-{n}.tif',outputs)


