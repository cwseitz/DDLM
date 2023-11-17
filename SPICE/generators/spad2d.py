import numpy as np
import matplotlib.pyplot as plt
import os
import secrets
import string
import json
import scipy.sparse as sp
import torch

from skimage.io import imsave
from scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial

from ..utils import *
from BaseSMLM.psf.psf2d.psf2d import *

class SPAD2D:
    """Simulates a small ROI of a 2D spad array 
    (SPAD photon counting camera)"""
    def __init__(self,config):
        self.config = config

    def sample_uniform_circle(self, x0, y0, r, n_samples):
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        radius = np.sqrt(np.random.uniform(0, 1, n_samples)) * r
        x = x0 + radius * np.cos(theta)
        y = y0 + radius * np.sin(theta)
        return x, y

    def generate(self,r=4,plot=False):
        theta = np.zeros((3,self.config['particles']))
        nx,ny = self.config['nx'],self.config['ny']
        xsamp,ysamp = self.sample_uniform_circle(nx/2,ny/2,r,self.config['particles'])
        theta[0,:] = ysamp
        theta[1,:] = xsamp
        theta[2,:] = self.config['sigma']
        
        photons,probsum = self.get_counts(theta,lam0=self.config['lam0'])
        return photons, probsum
        
    def add_photons(self,prob,counts):
        """Distribute photons over space"""
        prob_flat = prob.flatten()
        rows,cols = prob.shape
        result = np.zeros((rows, cols))
        for n in range(counts):
            idx = np.random.choice(rows*cols,p=prob_flat)
            row = idx // cols
            col = idx % cols
            result[row,col] += 1
        return result
            
    
    def distribute(self,nphotons,nt):
        """Distribute photons over time"""
        vec = np.zeros(nt,dtype=int)
        indices = np.random.choice(nt,nphotons,replace=True)
        for idx in indices:
            vec[idx] += 1
        return vec
        

    def get_counts(self,theta,patch_hw=3,lam0=10.0):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        nt,nx,ny = self.config['nt'],self.config['nx'],self.config['ny']
        photons = np.zeros((nt,nx,ny))
        probsum = np.zeros((nx,ny))
        for n in range(self.config['particles']):
            nphotons = np.random.poisson(lam=lam0)
            countvec = self.distribute(nphotons,self.config['nt'])
            prob = np.zeros((self.config['nx'],self.config['ny']),dtype=np.float32)
            x0,y0,sigma = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,y0p,sigma)*lamy(Y,x0p,sigma)
            prob[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += lam
            probsum += prob
            for m in range(nt):
                photons[m,:,:] += self.add_photons(prob,countvec[m])
        return photons,probsum




