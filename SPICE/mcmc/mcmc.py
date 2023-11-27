import pyro
import pyro.distributions as dist
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import exp
import warnings
warnings.filterwarnings('ignore')

def dudn0(X,Y,x0,y0,sigma):
    return _lamx(X,x0,sigma)*_lamy(Y,y0,sigma)
     
def dudx0(X,Y,x0,y0,sigma):
    A = 1/(torch.tensor(2*np.pi).sqrt()*sigma)
    return A*_lamy(Y,y0,sigma)*(exp(-(X-0.5-x0)**2/(2*sigma**2))-exp(-(X+0.5-x0)**2/(2*sigma**2)))
    
def dudy0(X,Y,x0,y0,sigma):
    A = 1/(torch.tensor(2*np.pi).sqrt()*sigma)
    return A*_lamx(X,x0,sigma)*(exp(-(Y-0.5-y0)**2/(2*sigma**2))-exp(-(Y+0.5-y0)**2/(2*sigma**2)))

def dudsx(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(torch.tensor(2*np.pi).sqrt()*sigma_x**2)
    return A*_lamy(Y,y0,sigma_y)*((X-x0-0.5)*exp(-(X-0.5-x0)**2/(2*sigma_x**2))-(X-x0+0.5)*exp(-(X+0.5-x0)**2/(2*sigma_x**2)))
    
def dudsy(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(torch.tensor(2*np.pi).sqrt()*sigma_y**2)
    return A*_lamx(X,x0,sigma_x)*((Y-y0-0.5)*exp(-(Y-0.5-y0)**2/(2*sigma_y**2))-(Y-y0+0.5)*exp(-(Y+0.5-y0)**2/(2*sigma_y**2)))

def duds0(X,Y,x0,y0,sigma):
    return dudsx(X,Y,x0,y0,sigma,sigma)+dudsy(X,Y,x0,y0,sigma,sigma)
    
def jac1(X,Y,theta,eta=0.8,gain=1.0,offset=0.0,var=0.0,sigma=0.92,N0=1000.0,texp=1.0):
    x0,y0,sigma,N0 = theta
    i0 = N0*eta*gain*texp
    j_x0 = i0*dudx0(X,Y,x0,y0,sigma)
    j_y0 = i0*dudy0(X,Y,x0,y0,sigma)
    j_s0 = i0*duds0(X,Y,x0,y0,sigma)
    j_n0 = (i0/N0)*dudn0(X,Y,x0,y0,sigma)
    jac = torch.cat([j_x0, j_y0, j_s0, j_n0])
    return jac

def jac1mix(x,y,theta,eta=0.8,gain=1.0,offset=0.0,var=0.0,sigma=0.92,N0=1000.0,texp=1.0):
    ntheta,nspots = theta.shape
    jacblock = [jac1(x,y,theta[:,n]) for n in range(nspots)]
    return torch.cat(jacblock)

def jac2mix(adu,X,Y,theta,eta=0.8,gain=1.0,offset=0.0,var=0.0,sigma=0.92,N0=1000.0,texp=1.0):
    ntheta,nspots = theta.shape
    nlam = torch.zeros_like(adu,dtype=torch.float64)
    i0 = gain*eta*texp*N0
    for n in range(nspots):
        x0,y0,sigma,N0 = theta[:,n]
        alpha = torch.tensor(2.0).sqrt() * sigma
        lamx = 0.5*(torch.erf((X+0.5-x0)/alpha)-torch.erf((X-0.5-x0)/alpha))
        lamy = 0.5*(torch.erf((Y+0.5-y0)/alpha)-torch.erf((Y-0.5-y0)/alpha))
        nlam += lamx*lamy
    mu = i0*nlam + var
    jac = 1 - torch.nan_to_num(adu/mu)
    return jac.flatten()

def jacmix(theta,adu,eta=0.8,gain=1.0,offset=0.0,var=0.0,sigma=0.92,N0=1000.0,texp=1.0):
    nx, ny = adu.shape
    ntheta,nspots = theta.shape
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny))
    X = torch.tensor(X); Y = torch.tensor(Y)
    J1 = jac1mix(X,Y,theta)
    J1 = J1.reshape((ntheta*nspots,nx**2))
    J2 = jac2mix(adu,X,Y,theta)
    J = J1 @ J2
    J = J.reshape((ntheta,nspots))
    return J

def _lamx(X, x0, sigma):
    alpha = torch.tensor(2.0).sqrt() * sigma
    X = torch.tensor(X)
    return 0.5 * (torch.erf((X + 0.5 - x0) / alpha) - torch.erf((X - 0.5 - x0) / alpha))

def _lamy(Y, y0, sigma):
    alpha = torch.tensor(2.0).sqrt() * sigma
    Y = torch.tensor(Y)
    return 0.5 * (torch.erf((Y + 0.5 - y0) / alpha) - torch.erf((Y - 0.5 - y0) / alpha))
    
def mixloglike(theta,adu,eta=0.8,gain=1.0,offset=0.0,var=0.0,sigma=0.92,N0=1000.0,texp=1.0):
   
    nx,ny = adu.shape
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny))
    mu = torch.zeros_like(adu,dtype=torch.float64)
    ntheta,nspots = theta.shape
    i0 = gain*eta*texp*N0
    for n in range(nspots):
        x0,y0,sigma,N0 = theta[:,n]
        lam = _lamx(X,x0,sigma)*_lamy(Y,y0,sigma)
        mu += i0*lam + var #muprm

    stirling = adu * torch.nan_to_num(torch.log(adu+1e-8)) - adu
    p = torch.nan_to_num(adu*torch.log(mu))
    nll = stirling + mu - p
    nll = torch.sum(nll)
    return nll

def langevin_update(current_params, grad_log_posterior, lr):
    return current_params + 0.5 * lr * grad_log_posterior + lr**0.5 * torch.randn_like(current_params)

def run_langevin_dynamics(adu,initial_params,num_samples=1000,lr=None, warmup_steps=500,eta=0.8,gain=1.0,offset=0.0,var=0.0,print_every=1):
    adu = torch.tensor(adu)
    trace = []; loglikes = []
    params = torch.tensor(initial_params, requires_grad=True)
    if lr is None:
        lr = 1e-6*torch.ones((4,))
    ntheta,nspots = initial_params.shape
    lr = lr[:,np.newaxis]
    lr = np.repeat(lr,nspots,axis=1)
    for _ in range(warmup_steps + num_samples):
        if _ % print_every == 0:
            print(f'MCMC step {_}')
        loglike = mixloglike(params,adu)
        grad_loglike = jacmix(params,adu)
        params = langevin_update(params, grad_loglike, lr)
        if _ < warmup_steps:
            continue

        trace.append(params.clone().detach().numpy())
        loglikes.append(loglike.clone().detach().numpy())

    return np.array(trace), np.array(loglikes)

