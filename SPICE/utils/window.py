import numpy as np
import matplotlib.pyplot as plt
import itertools
from .cov import *

def Exy_em_win(stack,win_size=100):
    """Wrapper to calculate empirical pixel covariance <XY>, 
       over a sliding window"""
    nt,nx,ny = stack.shape
    npixels = nx
    stack = stack.reshape((nt,nx*ny))

    Vind, RLind, Hind, Dind = Sind(npixels)
    Eh,Ev,Er,El,Ed = Eind(npixels)
    num_win = nt-win_size+1
    ExyR = np.zeros((2*npixels-1,2*npixels-1))
    ExyRt = []
    
    for i in range(num_win):
        window = stack[i:i+win_size]
        window = window[:,:,np.newaxis]
        F = window*window.transpose(0,2,1)
        Exy_win = np.mean(F, axis=0)
        
        Ehvals = Exy_win[Eh]; Evvals = Exy_win[Ev]
        Ervals = Exy_win[Er]; Elvals = Exy_win[El]
        Edvals = Exy_win[Ed]
        
        ExyR = np.zeros((2*npixels-1,2*npixels-1))
        ExyR[Vind] = Evvals; ExyR[RLind] = Ervals
        ExyR[Hind] = Ehvals; #ExyR[Dind] = Edvals
        ExyRt.append(ExyR)

    return np.array(ExyRt)
    
    
def ExEy_em_win(stack,win_size=100):
    """Wrapper to calculate empirical product of
       averages <X><Y>, over a sliding window"""
    nt,nx,ny = stack.shape
    stack = stack.reshape((nt,nx*ny))
    npixels = nx
    Vind, RLind, Hind, Dind = Sind(npixels)
    Eh,Ev,Er,El,Ed = Eind(npixels)
    num_win = nt-win_size+1
    ExEyRt = []
    
    for i in range(num_win):
        window = stack[i:i+win_size]
        Ei = np.mean(window,axis=0)
        Ei = np.outer(Ei,Ei)
        
        Ehvals = Ei[Eh]; Evvals = Ei[Ev]
        Ervals = Ei[Er]; Elvals = Ei[El]
        Edvals = Ei[Ed]

        ExEyR = np.zeros((2*npixels-1,2*npixels-1))
        ExEyR[Vind] = Evvals; ExEyR[RLind] = Ervals
        ExEyR[Hind] = Ehvals; #ExyR[Dind] = Edvals
        ExEyRt.append(ExEyR)

    return np.array(ExEyRt)
