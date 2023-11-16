import numpy as np
import json
import matplotlib.pyplot as plt
from SPICE.utils import *
from hmmlearn import hmm
from scipy.stats import poisson
from skimage.io import imread, imsave
import napari
from PIL import Image

with open('hmm.json', 'r') as f:
    config = json.load(f)

prefixes = [
'Stack-Crop-1'
]

def FitPoissonHMM(X,n_components):
    scores = list()
    models = list()
    for idx in range(10):
        model = hmm.PoissonHMM(n_components=n_components, random_state=idx,
                               n_iter=10)
        model.fit(X)
        models.append(model)
        scores.append(model.score(X))
        print(f'Converged: {model.monitor_.converged}\t\t'
              f'Score: {scores[-1]}')


for prefix in prefixes:
    path = config['datapath']+prefix+'.tif'
    stack = imread(path)
    X = np.sum(stack,axis=(1,2))
    fig,ax=plt.subplots(1,2)
    ax[0].plot(X,color='black')
    bins = np.arange(0,20,1)
    ax[1].hist(X,bins=bins,color='black')
    plt.show()
    X = X[:,np.newaxis]
    FitPoissonHMM(X[:500],7)



