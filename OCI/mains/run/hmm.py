import numpy as np
import json
import matplotlib.pyplot as plt
from oci.utils import *
from hmmlearn import hmm
from scipy.stats import poisson
from skimage.io import imread, imsave
import napari
from PIL import Image

with open('hmm.json', 'r') as f:
    config = json.load(f)

prefixes = [
'Stack-Crop-7'
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

    # get the best model
    model = models[np.argmax(scores)]
    print(f'The best model had a score of {max(scores)} and '
          f'{model.n_components} components')
          
    return model
              
time = np.linspace(0,10,10000)
for prefix in prefixes:
    path = config['datapath']+prefix+'.tif'
    stack = imread(path)
    X = np.sum(stack,axis=(1,2))

    X = X[:,np.newaxis]
    model = FitPoissonHMM(X[:500],1)    

    fig,ax=plt.subplots(1,3,figsize=(8,2.5))

    ax[0].imshow(np.sum(stack,axis=0),cmap='gray')
    ax[0].set_xticks([]); ax[0].set_yticks([]) 
   
    ax[1].plot(time,X,color='black')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Counts')
    bins = np.arange(0,20,1)
    hist,bins = np.histogram(X,bins=bins,density=True)
    width = 0.7 * (bins[1] - bins[0])
    
    prop_per_state = model.predict_proba(X).mean(axis=0)
    
    bins = np.arange(0,20,1)
    hist,bins = np.histogram(X,bins=bins,density=True)
    width = 0.7 * (bins[1] - bins[0])
    ax[2].bar(bins[:-1], hist, color='black',align='center', width=width)
    ax[2].plot(bins, poisson.pmf(bins, model.lambdas_).T @ prop_per_state,color='blue',label='HMM')
    ax[2].set_xlabel('Counts')
    ax[2].set_ylabel('Probability')
    ax[2].legend()    
    plt.tight_layout()
    plt.show()




