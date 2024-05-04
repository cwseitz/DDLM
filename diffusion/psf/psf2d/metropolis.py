import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class Sampler:
    def __init__(self):
        pass

class Metropolis2D(Sampler):
    def __init__(self,mu,cov,negloglike):
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.prop = multivariate_normal(mu,cov)
        self.negloglike = negloglike
        
    def diagnostic(self,acc):
        acc = np.array(acc)
        acc = acc.astype(np.int16)
        f = np.cumsum(acc)
        f = f/np.arange(1,len(f)+1,1)
        fig, ax = plt.subplots()
        ax.plot(f,color='black')
        plt.tight_layout()
        plt.show()

        
    def sample(self,theta_old,data,like_old,beta,n):
        accept = True
        dtheta = self.prop.rvs()
        theta_new = theta_old + dtheta
        
        if np.any(theta_new < 0):
            accept = False
            return theta_old, like_old, accept

        like_new = self.negloglike(theta_new,data)
        a = np.exp(beta*(like_old-like_new))        
        u = np.random.uniform(0,1)
        if u <= a:
            theta = theta_new
            like = like_new
        else:
            accept = False
            theta = theta_old
            like = like_old
            
        return theta, like, accept
        
    def post_marginals(self,thetas,tburn=500,bins=10):
        ntheta,nsamples = thetas.shape
        fig, ax = plt.subplots(1,ntheta,figsize=(2*ntheta,2))
        for n in range(ntheta):
            ax[n].hist(thetas[n,tburn:],bins=bins,color='black',density=True)
            ax[n].set_title(np.round(np.std(thetas[n,tburn:]),2))
        plt.tight_layout()

    def run(self,data,theta0,iters=1000,skip=5,beta=1,diag=False):
        theta = theta0
        thetas = np.zeros((len(theta),iters))
        like = self.negloglike(theta0,data)
        acc = []
        for n in range(iters):
            theta, like, accept = self.sample(theta,data,like,beta,n)
            acc.append(accept)
            thetas[:,n] = theta
        if diag:
            self.diagnostic(acc)
        return thetas

