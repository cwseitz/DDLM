from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson, gaussian_kde
from skimage.io import imread
from scipy.optimize import curve_fit
from BaseSMLM.localize import LoGDetector

def fit_hmm(data,min_comp=1,max_comp=5):
    scores = list()
    models = list()
    for n_components in range(min_comp,max_comp):
        for idx in range(10):
            model = hmm.PoissonHMM(n_components=n_components, random_state=idx,
                                   n_iter=10)
            model.fit(data)
            models.append(model)
            scores.append(model.score(data))
            print(f'Converged: {model.monitor_.converged}\t\t'
                  f'Score: {scores[-1]}')

    model = models[np.argmax(scores)]
    print(f'The best model had a score of {max(scores)} and '
          f'{model.n_components} components')
    
    return model
    
def lifetime(states,state):
    indices = np.where(states[:-1] != states[1:])[0] + 1
    split = np.split(states,indices)
    lengths = [len(segment) for segment in split if np.all(segment==state)]
    return np.array(lengths)
    
def plot_hmm(data,states):
    time = np.arange(0,len(data),1)*0.01
    fig, ax = plt.subplots(1,3,figsize=(10,3))
    ax[0].plot(time,model.lambdas_[states], ".-", color='cyan')
    ax[0].plot(time,data,color='gray',alpha=0.5)
    ax[0].set_xlabel('Time (sec)')
    ax[0].set_ylabel('ADU')
    ax[0].set_xlim([0,5.0])
    unique, counts = np.unique(states, return_counts=True)
    counts = counts/len(states)
    rates = model.lambdas_.flatten()
    ax[1].bar(unique,counts,color='blue')
    ax[1].set_xlabel('State')
    ax[1].set_ylabel('Proportion')
    ax[2].bar(unique,rates,color='red')
    ax[2].set_xlabel('State')
    ax[2].set_ylabel('Rate (ADU/frame)')
    plt.tight_layout()
    plt.show()
    
f = '240110_Control_JF646_4pm_overnight_L640_30mW_10ms____10_MMStack_Default.ome.tif'
stack = imread(f)
nt,nx,ny = stack.shape
log = LoGDetector(stack[0],threshold=0.0007)
spots = log.detect()
#log.show(); plt.show()
spots = spots[['x','y']].values.astype(np.int16)
data = stack[:,spots[:,0],spots[:,1]]
    
nt,nspots = data.shape
life0 = []; life1 = []
data0 = []; data1 = []
for n in range(nspots):
    this_data = data[:,n].astype(np.int16)
    this_data = this_data.reshape(-1,1)
    model = fit_hmm(this_data,min_comp=2,max_comp=3)
    states = model.predict(this_data)
    this_data0 = this_data[np.argwhere(states == 0)]
    this_data1 = this_data[np.argwhere(states == 1)]
    _life0 = lifetime(states,0)*0.01
    _life1 = lifetime(states,1)*0.01
    plot_hmm(this_data,states)
    life0.append(_life0); life1.append(_life1)
    data0.append(this_data0); data1.append(this_data1)

    #fig,ax=plt.subplots(1,2,figsize=(3,3))
    #bins = 20
    #vals0,bins0 = np.histogram(this_data0,bins=bins,density=True)
    #vals1,bins1 = np.histogram(this_data1,bins=bins,density=True)
    #ax[0].plot(bins0[:-1],vals0,color='red',label='ON')
    #ax[1].plot(bins1[:-1],vals1,color='blue',label='OFF')
    #ax.set_xlabel('Value')
    #ax.set_ylabel('Density')
    #ax.legend()
    #plt.tight_layout()
    #plt.show()
    
life0 = np.concatenate(life0,axis=0)
life1 = np.concatenate(life1,axis=0)
data0 = np.concatenate(data0,axis=0)
data1 = np.concatenate(data1,axis=0)

def func_doublexp(x, m, c0, n, d0):
    return c0 * np.exp(m*x) + d0 * np.exp(n*x)

fig,ax=plt.subplots(figsize=(3,3))
bins = np.arange(1,20,1)*0.01
vals0,bins0 = np.histogram(life0,bins=bins,density=True)
opt0, cov0 = curve_fit(func_doublexp,bins0[:-1],vals0)
vals1,bins1 = np.histogram(life1,bins=bins,density=True)
opt1, cov1 = curve_fit(func_doublexp,bins1[:-1],vals1)
ax.set_xlabel('Lifetime (sec)')
ax.set_ylabel('Density')
ax.scatter(bins0[:-1],vals0,color='red',s=3,label='ON')
ax.plot(bins0[:-1],func_powerlaw(bins0[:-1],*opt0),color='red')
ax.scatter(bins1[:-1],vals1,color='blue',s=3,label='OFF')
ax.plot(bins1[:-1],func_powerlaw(bins1[:-1],*opt1),color='blue')
ax.set_xscale('log'); ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.show()

