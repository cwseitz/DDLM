import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.special import comb
from ..psf.psf2d import *


def Pmatrix(theta,npixels,patch_hw=5):
    """Get the matrix of probabilities (rows are particles, cols are pixels)"""
    x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
    X,Y = np.meshgrid(x,y)
    _,nparticles = theta.shape
    P = np.zeros((nparticles, npixels*npixels))
    for n in range(nparticles):
        mu = np.zeros((npixels,npixels),dtype=np.float32)
        x0,y0,sigma,N0 = theta[:,n]
        patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
        x0p = x0-patchx; y0p = y0-patchy
        lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
        mu[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += lam
        P[n,:] = mu.flatten()
    return P


def Eind(npixels):
    """Get indexers for the E matrix"""
    x = np.ones((npixels**2,))
    mask = np.arange(len(x)) % npixels == 0
    x[mask] = 0; x = np.roll(x,-1)
    A = np.diag(x,k=1) #horizontally adjacent

    x = np.ones((npixels**2-npixels,))
    B = np.diag(x,k=npixels) #vertically adjacent

    x = np.ones((npixels**2-npixels,))
    mask = np.arange(len(x)) % npixels == 0
    x[mask] = 0; x = np.roll(x,-1)
    C = np.diag(x,k=npixels+1) #right diagonal

    x = np.ones((npixels**2-npixels,))
    mask = np.arange(len(x)) % npixels == 0
    x[mask] = 0
    D = np.diag(x,k=npixels-1) #left diagonal

    F = np.eye(npixels**2) #autocorrelation

    Aind = np.where(A > 0); Bind = np.where(B > 0)
    Cind = np.where(C > 0); Dind = np.where(D > 0)
    Find = np.where(F > 0)
    return Aind,Bind,Cind,Dind,Find

def Sind(npixels):
    """Get indexers for the covariance map"""
    checker = np.indices((2*npixels-1,2*npixels-1)).sum(axis=0) % 2
    checker = 1-checker
    checker[::2,:] *= 2
    checker[::2,:] += 2
    Vind = np.where(checker == 0); RLind = np.where(checker == 1)
    Hind = np.where(checker == 2); Dind = np.where(checker == 4)
    return Vind, RLind, Hind, Dind

def _Exy(theta,npixels,bpath,r=4,patch_hw=5,Kmax=2):
    """Matrix of covariances <XY>"""
    def _mu(theta,npixels,nparticles,patch_hw=5):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        mu = np.zeros((npixels,npixels),dtype=np.float32)
        for n in range(nparticles):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += lam
        return mu

    _,nparticles = theta.shape
    mu = _mu(theta,npixels,nparticles)
    #need object of shape (K,K,npixels**2,npixels**2)
    Omega = np.zeros((Kmax+1,Kmax+1,npixels**2,npixels**2))
    #nparticles, npixels**2 (columns are probs emitters emit into that pixel)
    P = Pmatrix(theta,npixels) 
    for i in range(Kmax+1):
        for j in range(Kmax+1):
            B = np.load(bpath+f'bin_{nparticles}{i}{j}.npz')['B']
            Nc,_,_ = B.shape
            alpha = B[:,0,:]; beta = B[:,1,:]
            #nparticles, npixels**2, Nc
            thisP = np.repeat(P[:,:,np.newaxis],Nc,axis=2)
            #nparticles, npixels**2, Nc
            alpha = np.repeat(alpha[:,np.newaxis,:],npixels**2,axis=1)
            #nparticles, npixels**2, Nc
            beta = np.repeat(beta[:,np.newaxis,:],npixels**2,axis=1) 
            alpha = np.swapaxes(alpha,0,2); beta = np.swapaxes(beta,0,2)
            P2alpha = np.power(thisP,alpha); P2beta = np.power(thisP,beta)
            R = np.sum(np.prod(P2alpha*P2beta,axis=0),axis=1)
            Omega[i,j,:,:] = R

    Chi = np.arange(0,Kmax+1,1)
    Chi = np.outer(Chi,Chi)
    Chi = Chi[:,:,np.newaxis,np.newaxis]
    Exy = np.sum(Chi*Omega,axis=(0,1))
    return Exy, mu

def _ExEy(theta,npixels,r=4,patch_hw=5,Kmax=2):
    """Matrix of product of averages <X><Y>"""
    def _mu(theta,npixels,nparticles,patch_hw=5):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        mu = np.zeros((npixels,npixels),dtype=np.float32)
        for n in range(nparticles):
            x0,y0,sigma,N0 = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += lam
        return mu

    def strings(num_ones,string_length):
        binary_array = np.array([1] * num_ones + [0] * (string_length - num_ones), dtype=int)
        permutations = np.array(list(set(itertools.permutations(binary_array, string_length))))
        binary_matrix = permutations.T
        return binary_matrix

    _,nparticles = theta.shape
    mu = _mu(theta,npixels,nparticles)
    #need object of shape (K,K,npixels**2,npixels**2)
    Omega = np.zeros((Kmax+1,Kmax+1,npixels**2,npixels**2))
    #nparticles, npixels**2 (columns are probs emitters emit into that pixel)
    P = Pmatrix(theta,npixels) 
    for i in range(Kmax+1):
        for j in range(Kmax+1):
            alpha = strings(i,nparticles); beta = strings(j,nparticles)
            _,Nc_alpha = alpha.shape; _,Nc_beta = beta.shape
            
            #nparticles, npixels**2, Nc
            thisP = np.repeat(P[:,:,np.newaxis],Nc_alpha,axis=2)
            #nparticles, npixels**2, Nc
            alpha = np.repeat(alpha[:,np.newaxis,:],npixels**2,axis=1)
            #alpha = np.swapaxes(alpha,0,2)
            P2alpha = np.power(thisP,alpha)

            #nparticles, npixels**2, Nc
            thisP = np.repeat(P[:,:,np.newaxis],Nc_beta,axis=2)
            #nparticles, npixels**2, Nc
            beta = np.repeat(beta[:,np.newaxis,:],npixels**2,axis=1)
            #beta = np.swapaxes(beta,0,2)
            P2beta = np.power(thisP,beta)
            
            Ralpha = np.sum(np.prod(P2alpha,axis=0),axis=1)
            Rbeta = np.sum(np.prod(P2beta,axis=0),axis=1)
            
            Omega[i,j,:,:] = Ralpha*Rbeta
        
    Chi = np.arange(0,Kmax+1,1)
    Chi = np.outer(Chi,Chi)
    Chi = Chi[:,:,np.newaxis,np.newaxis]
    ExEy = np.sum(Chi*Omega,axis=(0,1))

    return ExEy, mu


def Exy_th(theta,npixels,bpath):
    """Wrapper to calculate theoretical pixel covariance <XY>, params"""
    Exy,mu = _Exy(theta,npixels,bpath)
    Eh,Ev,Er,El,Ed = Eind(npixels)
    Vind, RLind, Hind, Dind = Sind(npixels)

    Ehvals = Exy[Eh]; Evvals = Exy[Ev]; Ervals = Exy[Er]
    Elvals = Exy[El]; Edvals = Exy[Ed]

    ExyR = np.zeros((2*npixels-1,2*npixels-1))
    ExyR[Vind] = Evvals; ExyR[RLind] = Ervals
    ExyR[Hind] = Ehvals
    ExyR[Dind] = Edvals

    ExyL = np.zeros((2*npixels-1,2*npixels-1))
    ExyL[Vind] = Evvals; ExyL[RLind] = Elvals
    ExyL[Hind] = Ehvals
    ExyL[Dind] = Edvals

    return mu, ExyR, ExyL
    
def ExEy_th(theta,npixels):
    """Wrapper to calculate theoretical product of averages <X><Y>, params"""
    ExEy,mu = _ExEy(theta,npixels)

    Eh,Ev,Er,El,Ed = Eind(npixels)
    Vind, RLind, Hind, Dind = Sind(npixels)

    Ehvals = ExEy[Eh]; Evvals = ExEy[Ev]; Ervals = ExEy[Er]
    Elvals = ExEy[El]; Edvals = ExEy[Ed]

    ExEyR = np.zeros((2*npixels-1,2*npixels-1))
    ExEyR[Vind] = Evvals; ExEyR[RLind] = Ervals
    ExEyR[Hind] = Ehvals; ExEyR[Dind] = Edvals

    return ExEyR
    
def Exy_em(stack):
    """Wrapper to calculate empirical pixel covariance <XY> and product of
       averages <X><Y>, no params"""
    nt,nx,ny = stack.shape
    stack = stack.reshape((nt,nx*ny))
    stack = stack[:,:,np.newaxis]
    F = stack*stack.transpose(0,2,1)
    Exy = np.mean(F,axis=0)

    npixels = nx
    Vind, RLind, Hind, Dind = Sind(npixels)
    Eh,Ev,Er,El,Ed = Eind(npixels)

    Ehvals = Exy[Eh]; Evvals = Exy[Ev]; Ervals = Exy[Er]
    Elvals = Exy[El]; Edvals = Exy[Ed]

    ExyR = np.zeros((2*npixels-1,2*npixels-1))
    ExyR[Vind] = Evvals; ExyR[RLind] = Ervals
    ExyR[Hind] = Ehvals; 
    #ExyR[Dind] = Edvals

    return ExyR
    
    
def ExEy_em(stack):
    """Wrapper to calculate empirical pixel covariance <XY> and product of
       averages <X><Y>, no params"""
    nt,nx,ny = stack.shape
    stack = stack.reshape((nt,nx*ny))
    Ei = np.mean(stack,axis=0)
    Ei = np.outer(Ei,Ei)

    npixels = nx
    Vind, RLind, Hind, Dind = Sind(npixels)
    Eh,Ev,Er,El,Ed = Eind(npixels)

    Ehvals = Ei[Eh]; Evvals = Ei[Ev]; Ervals = Ei[Er]
    Elvals = Ei[El]; Edvals = Ei[Ed]

    ExEyR = np.zeros((2*npixels-1,2*npixels-1))
    ExEyR[Vind] = Evvals; ExEyR[RLind] = Ervals
    ExEyR[Hind] = Ehvals; 
    #ExEyR[Dind] = Edvals

    return ExEyR
