import numpy as np
import matplotlib.pyplot as plt
import torch
from BaseSMLM.generators import *
from SPICE import SPICE
from SPICE.utils import Double

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
radius = 2.0
nspots = 1

disc2d = Disc2D_TwoState(nx,ny)
counts,spikes,theta_star = disc2d.forward(radius,nspots,N0=100,B0=0.0,offset=100.0,var=5.0,nframes=1000,show=False)
S = np.sum(counts,axis=0)

Exy,ExEy = Double(counts)
Exy = Exy[0]
G2 = Exy/(ExEy+1e-8)
fig,ax=plt.subplots(1,4,sharex=False,sharey=False)
ax[0].imshow(S)
ax[1].imshow(Exy)
ax[2].imshow(ExEy)
ax[3].imshow(G2)
plt.show()
#spice = SPICE()
#spice.forward(counts,theta0=theta_star,num_samples=100)
