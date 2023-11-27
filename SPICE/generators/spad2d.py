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

class SPAD2D_Ring:
    """Simulates a small ROI of a 2D spad array 
    (SPAD photon counting camera)"""
    def __init__(self,config):
        self.config = config

    def ring(self,n,radius=3,phase=0):
        thetas = np.arange(0,n,1)*2*np.pi/n
        xs = radius*np.cos(thetas+phase)
        ys = radius*np.sin(thetas+phase)
        return xs,ys

    def generate(self,ring_radius=3,show=False):
        theta = np.zeros((3,self.config['particles']))
        nx,ny = self.config['nx'],self.config['ny']
        xsamp,ysamp = self.ring(self.config['particles'],radius=ring_radius)
        x0 = nx/2; y0 = ny/2
        theta[0,:] = ysamp + x0
        theta[1,:] = xsamp + y0
        theta[2,:] = self.config['sigma']
        counts,probsum = self.get_counts(theta,lam0=self.config['lam0'])
        if show:
            self.show(theta,counts)
        return counts, probsum, theta
        
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
            
    
    def tdistribute(self,nphotons,nt):
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
            countvec = self.tdistribute(nphotons,self.config['nt'])
            prob = np.zeros((self.config['nx'],self.config['ny']),dtype=np.float32)
            x0,y0,sigma = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,y0p,sigma)*lamy(Y,x0p,sigma)
            lam /= lam.sum()
            prob[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += lam
            probsum += prob
            for m in range(nt):
                photons[m,:,:] += self.add_photons(prob,countvec[m])
        probsum /= probsum.sum()
        return photons,probsum

    def get_spikes(self,xyz_np,upsample=4):
        gain, offset, var = self.cmos_params
        nx,ny = offset.shape
        grid_shape = (nx,ny,1)
        boolean_grid = batch_xyz_to_boolean_grid(xyz_np,
                                                 upsample,
                                                 self.config['pixel_size'],
                                                 1,
                                                 0,
                                                 grid_shape)
        return boolean_grid

    def show(self,theta,counts):
        fig,ax=plt.subplots(1,4)
        csum = np.sum(counts,axis=0)
        ax[0].scatter(theta[1,:],theta[0,:],color='black',s=5)
        ax[0].set_aspect(1.0)
        ax[1].imshow(csum,cmap=plt.cm.BuGn_r)
        ax[2].imshow(counts[0],cmap='gray')
        ax[3].plot(np.sum(counts,axis=(1,2)),color='black')
        ax[3].set_xlabel('Time'); ax[3].set_ylabel('Counts')
        ax[3].set_aspect(4.0)
        ax[0].set_xlim([0,self.config['nx']])
        ax[0].set_ylim([0,self.config['ny']])
        ax[0].invert_yaxis()
        plt.tight_layout()



