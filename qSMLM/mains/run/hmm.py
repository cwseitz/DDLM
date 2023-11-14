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
hmm, params = MyFullDiscreteFactorialHMM(n_steps)
Z, X = hmm.Simulate(random_seed=None)
R = hmm.EM(X, likelihood_precision=0.1, n_iterations=1000, verbose=True, print_every=1, random_seed=None)

T = params['transition_matrices']
Te1, Te2, Te3, Te4 = R.transition_matrices_tensor[0]
print(T[0],Te1)

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
