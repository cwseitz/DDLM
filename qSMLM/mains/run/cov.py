import numpy as np
import json
import matplotlib.pyplot as plt
from qSMLM.utils import *
from skimage.io import imread, imsave
import napari
from PIL import Image

with open('cov.json', 'r') as f:
    config = json.load(f)

prefixes = [
'Stack-Crop-7'
]

for prefix in prefixes:
    path = config['datapath']+prefix+'.tif'
    stack = imread(path)
    csum = np.cumsum(stack,axis=0)
    #csum = csum/csum.max()
    csum = csum.astype(np.int16)
    imsave('/home/cwseitz/Desktop/test.tif',csum,imagej=True)
    fig,ax=plt.subplots(1,2)
    summed = np.sum(stack,axis=(1,2))
    ax[0].plot(summed,color='black')
    bins = np.arange(0,20,1)
    ax[1].hist(summed,bins=bins,color='black')
    plt.show()
    
    #ExyR = Exy_em_win(stack,win_size=100)
    #ExEyR = ExEy_em_win(stack,win_size=100)
    #X = np.nan_to_num(ExyR/ExEyR)
    #viewer = napari.Viewer()
    #viewer.add_image(X, colormap='gray', name='win100')
    #napari.run()
    
    #ExyR = Exy_em(stack); ExEyR = ExEy_em(stack)
    #fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
    #ax[0].imshow(ExyR,cmap='coolwarm')
    #ax[1].imshow(ExEyR,cmap='coolwarm')
    #ax[2].imshow(X,cmap='coolwarm')
    #ax[0].set_title(prefix)
#plt.show()

