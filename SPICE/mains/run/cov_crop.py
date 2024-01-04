import numpy as np
import json
import matplotlib.pyplot as plt
from SPICE.utils import *
from skimage.io import imread, imsave
import napari
from PIL import Image

with open('cov_crop.json', 'r') as f:
    config = json.load(f)

prefixes = [
'Stack-Crop-6'
]

for prefix in prefixes:
    path = config['datapath']+prefix+'.tif'
    stack = imread(path)
    
    fig,ax=plt.subplots(1,2)
    summed = np.sum(stack,axis=(1,2))
    ax[0].plot(summed,color='black')
    bins = np.arange(0,20,1)
    ax[1].hist(summed,bins=bins,color='black')
        
    Exy,ExEy = Double(stack)
    X = np.nan_to_num(Exy[0]/ExEy)
    
    fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
    ax[0].imshow(Exy[0],cmap='coolwarm')
    ax[1].imshow(ExEy,cmap='coolwarm')
    ax[2].imshow(X,cmap='coolwarm')
    ax[0].set_title(prefix)
    plt.show()

