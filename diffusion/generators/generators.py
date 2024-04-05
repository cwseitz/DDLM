import numpy as np
import matplotlib.pyplot as plt
from .base import *

class Disc2D(Generator):
    def __init__(self,nx,ny):
        super().__init__(nx,ny)
    def forward(self,radius,nspots,sigma=0.92,texp=1.0,N0=1.0,eta=1.0,gain=1.0,
                B0=None,nframes=1,offset=100.0,var=5.0,show=False):
        density = Disc(radius)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        x0 = self.nx/2; y0 = self.ny/2
        theta[0,:] = x + x0; theta[1,:] = y + y0
        theta[2,:] = sigma; theta[3,:] = N0
        adu,spikes = self.sample_frames(theta,nframes,texp,eta,N0,B0,gain,offset,var,show=show)
        return adu,spikes,theta


