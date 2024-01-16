import numpy as np
import matplotlib.pyplot as plt
import torch
import napari
from BaseSMLM.generators import *
from oci import oci
from oci.utils import Double

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
radius = 0.0001 #center it in the frame
nspots = 1

disc2d = Disc2D_TwoState(nx,ny)
counts,spikes,theta_star = disc2d.forward(radius,nspots,N0=10,B0=0.0,sigma=0.552,offset=0.0,var=0.0,show=False,nframes=1000)
S = np.sum(counts,axis=0)/np.sum(counts) #probabilities

viewer = napari.Viewer()
viewer.add_image(counts, colormap='gray', name='Stack')
viewer.add_image(S, colormap='gray', name='Sum')
napari.run()

Exy,ExEy = Double(counts)
Exy = Exy[0]
G2 = Exy/(ExEy+1e-14)
fig,ax=plt.subplots(1,4,sharex=False,sharey=False)
ax[0].imshow(S)
ax[0].scatter(theta_star[1,:],theta_star[0,:],s=3)
ax[1].imshow(Exy)
ax[2].imshow(ExEy)
ax[3].imshow(G2)
plt.show()
#spice = oci()
#spice.forward(counts,theta0=theta_star,num_samples=100)
