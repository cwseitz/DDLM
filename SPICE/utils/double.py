import numpy as np

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
    
def Double(adu):
    nt,nx,ny = adu.shape
    _fft = np.fft.fft(adu,axis=0)
    _fftc = _fft.conj()
    corr = np.fft.ifft(_fft*_fftc,axis=0)
    auto = np.real(corr[0])

    adur = adu.reshape((nt,nx*ny))
    fft = np.fft.fft(adur,axis=0)
    fftc = fft.conj()
    fft = fft[:,:,np.newaxis]
    fftc = fftc[:,np.newaxis,:]
    corr = np.fft.ifft(fft*fftc,axis=0)
    X = np.real(corr) #take zero lag

    Y = np.zeros((nt,2*nx-1,2*nx-1))

    Eh,Ev,Er,El,Ed = Eind(nx)
    Vind, RLind, Hind, Dind = Sind(nx)

    Ehvals = X[:,Eh[0],Eh[1]]
    Evvals = X[:,Ev[0],Ev[1]]
    Ervals = X[:,Er[0],Er[1]]
    Elvals = X[:,El[0],El[1]]
    Edvals = X[:,Ed[0],Ed[1]]

    r = Ehvals.max()/Edvals.max()
    Edvals = Edvals*r
    
    Y[:,Vind[0],Vind[1]] = Evvals
    Y[:,RLind[0],RLind[1]] = Ervals
    Y[:,Hind[0],Hind[1]] = Ehvals
    Y[:,Dind[0],Dind[1]] = Edvals
    
    return auto,Y
    
    
    
    
