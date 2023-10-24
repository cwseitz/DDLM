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
        
    def summarize(self,like_new,like_old,theta_new,theta_old,ratio,accept,n):
        print(f'MCMC iteration: {n}')
        print(f'Like new {like_new}')
        print(f'Like old {like_old}')
        print(f'Theta new {theta_new}')
        print(f'Theta old {theta_old}')
        print(f'Acceptance ratio {ratio}')
        print(f'Accepted {accept}\n')

    def thin(self,thetas,skip=5,tburn=500):
        thetas = thetas[:,tburn:]
        thetas = thetas[:,::5]
        return thetas

    def diagnostic(self,thetas,acc,tburn=500):
        acc = np.array(acc)
        acc = acc.astype(np.int)
        f = np.cumsum(acc)
        f = f/np.arange(1,len(f)+1,1)
        fig, ax = plt.subplots()
        ax.plot(f,color='black')
        plt.tight_layout()
        fig, ax = plt.subplots(1,4,figsize=(8,2))
        k1t = thetas[0,tburn:]
        k2t = thetas[1,tburn:]
        k3t = thetas[2,tburn:]
        k4t = thetas[3,tburn:]
        ax[0].plot(k1t[::5],color='black')
        ax[1].plot(k2t[::5],color='black')
        ax[2].plot(k3t[::5],color='black')
        ax[3].plot(k4t[::5],color='black')
        self.post_marginals(thetas,tburn=tburn)
        plt.tight_layout()
        plt.show()
        
    def sample(self,theta_old,data,like_old,beta,n):
        accept = True
        dtheta = self.prop.rvs()
        theta_new = theta_old + dtheta
        
        #for 2d, none of the parameters should be negative
        if np.any(theta_new < 0):
            accept = False
            #self.summarize(None,like_old,theta_new,theta_old,None,accept,n)
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
            
        #self.summarize(like_new,like_old,theta_new,theta_old,a,accept,n)
        return theta, like, accept
        
    def post_marginals(self,thetas,tburn=500,bins=10):
        ntheta,nsamples = thetas.shape
        fig, ax = plt.subplots(1,ntheta,figsize=(2*ntheta,2))
        for n in range(ntheta):
            ax[n].hist(thetas[n,tburn:],bins=bins,color='black',density=True)
            ax[n].set_title(np.round(np.std(thetas[n,tburn:]),2))
        plt.tight_layout()

    def run(self,data,theta0,iters=1000,tburn=500,skip=5,beta=1,thin=False,diag=False):
        theta = theta0
        thetas = np.zeros((len(theta),iters))
        like = self.negloglike(theta0,data)
        acc = []
        for n in range(iters):
            theta, like, accept = self.sample(theta,data,like,beta,n)
            acc.append(accept)
            thetas[:,n] = theta
        if thin:
            thetas = self.thin(thetas,skip=skip,tburn=tburn)
        if diag:
            self.diagnostic(thetas,acc,tburn=tburn)
        return thetas
        

        return thetas
