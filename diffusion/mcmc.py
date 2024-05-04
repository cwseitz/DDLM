from generators import *
import matplotlib.pyplot as plt
from psf import MLE2D_MCMC

disc2D = Disc2D(20,20)
X,_,theta = disc2D.forward(0.0,1,sigma=0.92,N0=200)

mle2d_mcmc = MLE2D_MCMC(theta,X)
theta = np.squeeze(np.delete(theta,2,axis=0))
samples = mle2d_mcmc.metropolis(theta,iters=10000,beta=0.2,
                                prop_cov=0.05,diag=True)

tburn = 1000; bins = 100
samples = samples[:,tburn:]
sample_vars = np.var(samples,axis=1)
print(np.sqrt(sample_vars))

fig,ax=plt.subplots(2,1)
ax[0].hist(samples[0,:],bins)
ax[1].hist(samples[1,:],bins)
fig,ax=plt.subplots()
ax.scatter(samples[0,:],samples[1,:],color='black',s=3)
plt.tight_layout()
plt.show()

