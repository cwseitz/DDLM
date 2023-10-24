import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from SMLM.localize import LoGDetector
from SMLM.utils import RLDeconvolver
from SMLM.psf.psf2d import MLE2D, MLE2D_MCMC
from numpy.linalg import inv

class PipelineMLE2D_MCMC:
    """A collection of functions for maximum likelihood localization 
       and Metropolis-Hastings to estimate localization uncertainty"""
    def __init__(self,config,dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
        self.dump_config()
    def dump_config(self):
        with open(self.analpath+self.dataset.name+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)        
    def localize(self,plot_spots=False,plot_fit=False,plot_mcmc=False,tmax=None,run_deconv=False):
        self.lr = self.config['lr']
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        deconv = RLDeconvolver()
        if tmax is not None: nt = tmax
        threshold = self.config['thresh_log']
        spotst = []
        if not file.exists():
            for n in range(nt):
                print(f'Det in frame {n}')
                framed = self.stack[n]
                if run_deconv:
                    framed = deconv.deconvolve(framed,iters=5)
                log = LoGDetector(framed,threshold=threshold)
                spots = log.detect() #image coordinates
                if plot_spots:
                    log.show(); plt.show()
                spots = self.fit(framed,spots,plot_fit=plot_fit,plot_mcmc=plot_mcmc)
                spots = spots.assign(frame=n)
                spotst.append(spots)
            spotst = pd.concat(spotst)
            self.save(spotst)
        else:
            print('Spot files exist. Skipping')
        return spotst
                
    def fit(self,frame,spots,plot_fit=False,plot_mcmc=False):
        config = self.config
        patchw = self.config['patchw']
        for i in spots.index:
            start = time.time()
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.cmos_params[3]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,self.config['sigma'],self.config['N0']])
            opt = MLE2D_MCMC(theta0,adu,self.config) #cartesian coordinates with top-left origin
            
            theta_mle, loglike, post_samples = opt.optimize(max_iters=config['max_iters'],
                                                            mcmc_iters=config['mcmc_iters'],
                                                            tburn=config['tburn'],
                                                            prop_cov=config['prop_cov'],
                                                            beta = config['beta'],
                                                            plot_fit=plot_fit,
                                                            plot_mcmc=plot_mcmc,
                                                            tol=config['tol'],
                                                            lr=self.lr)
            
            dx = theta_mle[1] - patchw; dy = theta_mle[0] - patchw
            spots.at[i, 'x_mle'] = x0 + dx
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'N0_mle'] = theta_mle[3]
            spots.at[i, 'x_mcmc_avg'] = np.mean(post_samples[1,:])
            spots.at[i, 'y_mcmc_avg'] = np.mean(post_samples[0,:])
            spots.at[i, 's_mcmc_avg'] = np.mean(post_samples[2,:])
            spots.at[i, 'N0_mcmc_avg'] = np.mean(post_samples[3,:])
            spots.at[i, 'x_mcmc_std'] = np.std(post_samples[1,:])
            spots.at[i, 'y_mcmc_std'] = np.std(post_samples[0,:])
            spots.at[i, 's_mcmc_std'] = np.std(post_samples[2,:])
            spots.at[i, 'N0_mcmc_std'] = np.std(post_samples[3,:])
            
            end = time.time()
            elapsed = end - start
            print(f'Fit spot {i} in {elapsed} sec')
            
        return spots
    def save(self,spotst):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        spotst.to_csv(path)

class PipelineMLE2D:
    """A collection of functions for maximum likelihood localization"""
    def __init__(self,config,dataset):
        self.config = config
        self.analpath = config['analpath']
        self.datapath = config['datapath']
        self.dataset = dataset
        self.stack = dataset.stack
        Path(self.analpath+self.dataset.name).mkdir(parents=True, exist_ok=True)
        self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
        self.dump_config()
    def dump_config(self):
        with open(self.analpath+self.dataset.name+'/'+'config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)        
    def localize(self,plot_spots=False,plot_fit=False,tmax=None,run_deconv=False):
        self.lr = self.config['lr']
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        file = Path(path)
        nt,nx,ny = self.stack.shape
        deconv = RLDeconvolver()
        if tmax is not None: nt = tmax
        threshold = self.config['thresh_log']
        spotst = []
        if not file.exists():
            for n in range(nt):
                print(f'Det in frame {n}')
                framed = self.stack[n]
                if run_deconv:
                    print(f'Deconvolution frame {n}')
                    framed = deconv.deconvolve(framed,iters=5)
                log = LoGDetector(framed,threshold=threshold)
                spots = log.detect() #image coordinates
                if plot_spots:
                    log.show(); plt.show()
                spots = self.fit(framed,spots,plot_fit=plot_fit)
                spots = spots.assign(frame=n)
                spotst.append(spots)
            spotst = pd.concat(spotst)
            self.save(spotst)
        else:
            print('Spot files exist. Skipping')
        return spotst
                
    def fit(self,frame,spots,plot_fit=False):
        config = self.config
        patchw = self.config['patchw']
        for i in spots.index:
            start = time.time()
            x0 = int(spots.at[i,'x'])
            y0 = int(spots.at[i,'y'])
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = adu - self.cmos_params[3]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,self.config['sigma'],self.config['N0']])
            opt = MLE2D(theta0,adu,self.config) #cartesian coordinates with top-left origin
            
            theta_mle, loglike, conv = opt.optimize(max_iters=config['max_iters'],
                                                    plot_fit=plot_fit,
                                                    tol=config['tol'],
                                                    lr=self.lr)
            dx = theta_mle[1] - patchw; dy = theta_mle[0] - patchw
            spots.at[i, 'x_mle'] = x0 + dx #switch back to image coordinates
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'N0'] = theta_mle[3]
            spots.at[i, 'conv'] = conv
            end = time.time()
            elapsed = end-start
            print(f'Fit spot {i} in {elapsed} sec')
            
        return spots
    def save(self,spotst):
        path = self.analpath+self.dataset.name+'/'+self.dataset.name+'_spots.csv'
        spotst.to_csv(path)

