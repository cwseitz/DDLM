from scipy.optimize import minimize
from .psf2d.psf2d import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

class LSQ_BFGS:
    def __init__(self,theta0,adu):
       self.theta0 = theta0
       self.adu = adu

    def optimize(self,max_iters=1000,patchw=3,plot_fit=False):
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        thetat = []; loss = []

        def objective_function(theta):
            x0,y0,sigma,N0 = theta
            x = np.arange(0,2*patchw + 1)
            y = np.arange(0,2*patchw + 1)
            X,Y = np.meshgrid(x,y,indexing='ij')
            lam = N0*lamx(X,x0,sigma)*lamy(Y,y0,sigma)
            loss = np.mean((lam-self.adu)**2)
            return loss

        #original_stderr = sys.stderr
        #sys.stderr = open(os.devnull, 'w')
        bounds = [(0,2*patchw + 1),(0,2*patchw + 1),(0,5.0),(0,100.0)]
        result = minimize(
            objective_function,
            theta,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iters, 'disp': False},
            callback=lambda xk: thetat.append(xk.copy()),
        )
        #sys.stderr = original_stderr
        theta_opt = result.x
        converged = result.success
        niters = result.nit
            
        return theta_opt,converged
