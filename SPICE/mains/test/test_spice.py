import numpy as np
import matplotlib.pyplot as plt
import torch
from BaseSMLM.generators import *
from SPICE import SPICE

def show(adu,spikes,theta):
    nx,ny = adu.shape
    fig,ax=plt.subplots(1,3)
    ax[0].scatter(theta[1,:],theta[0,:],color='black',s=5)
    ax[0].set_aspect(1.0)
    ax[0].invert_yaxis()
    ax[0].set_xlim([0,nx]); ax[0].set_ylim([0,ny])
    im = ax[1].imshow(adu,cmap=plt.cm.BuGn_r)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    ax[2].imshow(spikes,cmap='gray')
    plt.tight_layout()
    
    
nx = ny = 20
radius = 3.0
nspots = 5

brown2d = Brownian2D(nx,ny)
counts,spikes,theta_star = brown2d.forward(radius,nspots,N0=1,offset=0.0,var=0.0,nframes=1000)

adu = np.sum(counts,axis=0)
show(adu,spikes[0],theta_star)
plt.show()

#spice = SPICE()
#spice.forward(counts,theta0=theta_star,num_samples=100)
