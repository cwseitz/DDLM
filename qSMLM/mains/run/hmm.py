import numpy as np
import json
import matplotlib.pyplot as plt
from qSMLM.utils import *
from qSMLM.hmm import *
from skimage.io import imread, imsave
import napari
from PIL import Image

with open('hmm.json', 'r') as f:
    config = json.load(f)

n_steps = 100
hmm = MyFullDiscreteFactorialHMM(n_steps)

"""
prefixes = [
'Stack-Crop-7'
]

for prefix in prefixes:
    path = config['datapath']+prefix+'.tif'
    stack = imread(path)
    print(stack.shape)
    fig,ax=plt.subplots(1,2)
    summed = np.sum(stack,axis=(1,2))
    ax[0].plot(summed,color='black')
    bins = np.arange(0,20,1)
    ax[1].hist(summed,bins=bins,color='black')
    plt.show()
"""
