import numpy as np
import matplotlib.pyplot as plt
from psf.psf2d.psf2d import *
from scipy.stats import multivariate_normal

class BasicKDE:
    def __init__(self,theta):
        self.theta = theta
    def forward(self,npixels,upsample=10,sigma=1.0):
        patchw = int(round(3*sigma))
        theta = upsample*(self.theta)
        kde = np.zeros((upsample*npixels,upsample*npixels),dtype=np.float32)
        ns,nd = theta.shape
        x = np.arange(0,2*patchw); y = np.arange(0,2*patchw)
        X,Y = np.meshgrid(x,y,indexing='ij')
        for n in range(ns):
            x0,y0 = theta[n,:]
            patchx, patchy = int(round(x0))-patchw, int(round(y0))-patchw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            kde_xmin = patchx; kde_xmax = patchx+2*patchw
            kde_ymin = patchy; kde_ymax = patchy+2*patchw
            kde[kde_xmin:kde_xmax,kde_ymin:kde_ymax] += lam
        return kde
        
class DistKDE:
    def __init__(self,theta):
        self.theta = theta
    def forward(self,npixels,upsample=10,sigma=1.0,xyvar=1.0,xylim=4.0):
        patchw = int(round(5*sigma))
        theta = upsample*(self.theta)
        kde = np.zeros((upsample*npixels,upsample*npixels),dtype=np.float32)
        ns,nd = theta.shape
        x = np.arange(0,2*patchw); y = np.arange(0,2*patchw)
        X,Y = np.meshgrid(x,y,indexing='ij')
        mu = [patchw, patchw]; cov = [[xyvar,0],[0,xyvar]]
        for n in range(ns):
            x0,y0 = theta[n,:]
            patchx, patchy = int(round(x0))-patchw, int(round(y0))-patchw
            x0p = x0-patchx; y0p = y0-patchy
            x0p_vec = np.linspace(x0p-xylim,x0p+xylim,20)
            y0p_vec = np.linspace(y0p-xylim,y0p+xylim,20)
            lams = []; plams = []
            for this_x0p in x0p_vec:
                for this_y0p in y0p_vec:
                    lam = lamx(X,this_x0p,sigma)*lamy(Y,this_y0p,sigma)
                    lams.append(lam)
                    r = [this_x0p,this_y0p]
                    plam = multivariate_normal.pdf(r,mean=mu,cov=cov)
                    plams.append(plam)
            lams = np.array(lams); plams = np.array(plams)
            avg = np.average(lams,axis=0,weights=plams)
            var = np.average((lams-avg)**2,axis=0,weights=plams)
            kde_xmin = patchx; kde_xmax = patchx+2*patchw
            kde_ymin = patchy; kde_ymax = patchy+2*patchw
            kde[kde_xmin:kde_xmax,kde_ymin:kde_ymax] += var
        return kde
