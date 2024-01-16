import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter
import json
from qSMLM.utils import *

def ring(n,radius=3,phase=0):
    thetas = np.arange(0,n,1)*2*np.pi/n
    xs = radius*np.cos(thetas+phase)
    ys = radius*np.sin(thetas+phase)
    return xs,ys

with open('cov.json', 'r') as f:
    config = json.load(f)

def get_theta(config,nparticles=4,ring_radius=3):
    theta = np.zeros((4,nparticles))
    nx,ny = config['npixels'],config['npixels']
    xsamp,ysamp = ring(nparticles,radius=ring_radius)
    x0 = nx/2; y0 = ny/2
    theta[0,:] = xsamp + x0
    theta[1,:] = ysamp + y0
    theta[2,:] = config['sigma']
    theta[3,:] = config['N0']
    return theta
    
def g2(nparticles,radius,plot=False):
    theta = get_theta(config,nparticles=nparticles,ring_radius=r)
    Mu,ExyR,ExyL = Exy_th(theta,config['npixels'],config['bpath'])
    ExEyR = ExEy_th(theta,config['npixels'])
    X = np.nan_to_num(ExyR/ExEyR)
    if plot:
        fig,ax=plt.subplots(3,len(radii))
        if n == 0:
            vmin = ExyR.min(); vmax = ExyR.max()
        im1 = ax[0,n].imshow(Mu,cmap='coolwarm')
        ax[0,n].set_xticks([]); ax[0,n].set_yticks([])
        cbar1 = plt.colorbar(im1,ax=ax[0,n],fraction=0.046, pad=0.04)
        cbar1.set_label(r'$\langle X_{i}\rangle/\eta N_{0}\Delta t$')
        
        im2 = ax[1,n].imshow(ExyR,cmap='coolwarm')
        cbar2 = plt.colorbar(im2,ax=ax[1,n],fraction=0.046, pad=0.04)
        cbar2.set_label(r'$\langle X_{i}X_{j}\rangle$')
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # Display only powers of 10, e.g., 10^0, 10^-1
        cbar1.formatter = formatter; cbar2.formatter = formatter
        cbar1.update_ticks(); cbar2.update_ticks()
        ax[1,n].set_xticks([]); ax[1,n].set_yticks([])
        
        im3 = ax[2,n].imshow(ExEyR,cmap='coolwarm')
        cbar2 = plt.colorbar(im3,ax=ax[2,n],fraction=0.046, pad=0.04)
        cbar2.set_label(r'$\langle X_{i}\rangle\langle X_{j}\rangle$')
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # Display only powers of 10, e.g., 10^0, 10^-1
        cbar1.formatter = formatter; cbar2.formatter = formatter
        cbar1.update_ticks(); cbar2.update_ticks()
        ax[2,n].set_xticks([]); ax[2,n].set_yticks([])
        
    return X

nparticles = np.arange(4,8,1)
radii = np.arange(0,7,1)
g20 = np.zeros((len(radii),len(nparticles)))
for n,r in enumerate(radii):
    for m,npar in enumerate(nparticles):
        print(f'Radius: {r}')
        X = g2(npar,r)
        g20[n,m] = X[29,28]

fig,ax=plt.subplots()
ax.plot(nparticles,g20[0,:],color='black')
plt.show()





