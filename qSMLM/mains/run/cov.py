import numpy as np
import json
import matplotlib.pyplot as plt
from qSMLM.utils import computeCov
from skimage.io import imread, imsave

from PIL import Image

with open('cov.json', 'r') as f:
    config = json.load(f)

prefixes = [
'Stack-Crop-4',
'Stack-Crop-5',
'Stack-Crop-6',
'Stack-Crop-7'
]

for prefix in prefixes:
    path = config['datapath']+prefix+'.tif'
    stack = imread(path)
    ExyR, ExEyR = computeCov(stack)
    X = np.nan_to_num(ExyR/ExEyR)
    fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
    ax[0].imshow(ExyR,cmap='coolwarm')
    ax[1].imshow(ExEyR,cmap='coolwarm')
    ax[2].imshow(X,cmap='coolwarm')
    ax[0].set_title(prefix)
plt.show()

