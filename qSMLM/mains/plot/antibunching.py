import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import json
from qSMLM.utils import *

def ring(n,radius=3,phase=np.pi):
    thetas = np.arange(0,n,1)*2*np.pi/n
    xs = radius*np.cos(thetas+phase)
    ys = radius*np.sin(thetas+phase)
    return xs,ys

with open('antibunching.json', 'r') as f:
    config = json.load(f)

def get_theta(config,ring_radius=3):
    theta = np.zeros((4,config['particles']))
    nx,ny = config['npixels'],config['npixels']
    xsamp,ysamp = ring(config['particles'],radius=ring_radius)
    x0 = nx/2; y0 = ny/2
    theta[0,:] = xsamp + x0
    theta[1,:] = ysamp + y0
    theta[2,:] = config['sigma']
    theta[3,:] = config['N0']
    return theta

radii = [3,4,7]

fig,ax=plt.subplots(2,len(radii))
for n,r in enumerate(radii):
    theta = get_theta(config,ring_radius=r)
    Mu,CovR,CovL = renderCov(theta,config['npixels'],config['bpath'])
    if n == 0:
        vmin = CovR.min(); vmax = CovR.max()
    im1 = ax[0,n].imshow(Mu,cmap='coolwarm')
    ax[0,n].set_xlabel('pixels'); ax[0,n].set_ylabel('pixels')
    cbar1 = plt.colorbar(im1,ax=ax[0,n],fraction=0.046, pad=0.04)
    cbar1.set_label(r'$\langle X_{i}\rangle/\eta N_{0}\Delta t$')
    im2 = ax[1,n].imshow(CovR,cmap='coolwarm',vmin=vmin,vmax=vmax)
    cbar2 = plt.colorbar(im2,ax=ax[1,n],fraction=0.046, pad=0.04)
    cbar2.set_label(r'$\langle X_{i}X_{j}\rangle$')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))  # Display only powers of 10, e.g., 10^0, 10^-1
    cbar1.formatter = formatter; cbar2.formatter = formatter
    cbar1.update_ticks(); cbar2.update_ticks()
    ax[1,n].set_xlabel('pixels'); ax[1,n].set_ylabel('pixels')
plt.tight_layout()
plt.show()
