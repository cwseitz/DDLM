import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from BaseSMLM.localize import LoGDetector
from oci.utils import *
from scipy.optimize import curve_fit
from hmmlearn import hmm

class LifetimeHMM:
    def __init__(self,stack):
        self.stack=stack
        
    def fit_hmm(self,data,min_comp=1,max_comp=5):
        scores = list()
        bics = list()
        models = list()
        for n_components in range(min_comp,max_comp):
            for idx in range(10):
                model = hmm.PoissonHMM(n_components=n_components, random_state=idx,
                                       n_iter=10)
                model.fit(data)
                models.append(model)
                loglike = model.score(data)
                num_params = sum(model._get_n_fit_scalars_per_param().values())
                bic = -2*loglike + num_params * np.log(len(data))
                scores.append(loglike)
                bics.append(bic)
               
                print(f'Converged: {model.monitor_.converged}\t\t'
                      f'NLL: {scores[-1]},BIC: {bics[-1]}')

        model = models[np.argmin(bics)]
        print(f'The best model had a BIC of {min(bics)} and '
              f'{model.n_components} components')
        
        return model
        
    def lifetime(self,states,state):
        indices = np.where(states[:-1] != states[1:])[0] + 1
        split = np.split(states,indices)
        lengths = [len(segment) for segment in split if np.all(segment==state)]
        return np.array(lengths)
        
    def plot_hmm(self,data,states,model):
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
        
    def plot_hist(self,life0,life1,data0,data1):
        """currently only for two states system"""
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
        ax.plot(bins0[:-1],func_doublexp(bins0[:-1],*opt0),color='red')
        ax.scatter(bins1[:-1],vals1,color='blue',s=3,label='OFF')
        ax.plot(bins1[:-1],func_doublexp(bins1[:-1],*opt1),color='blue')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    def forward(self,show_spots=False,threshold=0.0007,fps=100,
                show_hmm=False,show_hist=False,min_comp=2,max_comp=3):
      
        stack = self.stack
        nt,nx,ny = stack.shape
        log = LoGDetector(stack[0],threshold=threshold)
        spots = log.detect()
        if show_spots:
            log.show(); plt.show()
        spots = spots[['x','y']].values.astype(np.int16)
        data = stack[:,spots[:,0],spots[:,1]]
            
        nt,nspots = data.shape
        life0 = []; life1 = []
        data0 = []; data1 = []
        for n in range(nspots):
            print(f'Fitting Poisson HMM for spot {n}')
            this_data = data[:,n].astype(np.int16)
            this_data = this_data.reshape(-1,1)
            model = self.fit_hmm(this_data,min_comp=min_comp,max_comp=max_comp)
            states = model.predict(this_data)
            this_data0 = this_data[np.argwhere(states == 0)]
            this_data1 = this_data[np.argwhere(states == 1)]
            _life0 = self.lifetime(states,0)/fps
            _life1 = self.lifetime(states,1)/fps
            if show_hmm:
                self.plot_hmm(this_data,states,model)
            life0.append(_life0); life1.append(_life1)
            data0.append(this_data0); data1.append(this_data1)

        life0 = np.concatenate(life0,axis=0)
        life1 = np.concatenate(life1,axis=0)
        data0 = np.concatenate(data0,axis=0)
        data1 = np.concatenate(data1,axis=0)
        
        if show_hist:
            self.plot_hist(life0,life1,data0,data1)





