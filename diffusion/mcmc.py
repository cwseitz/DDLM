from generators import *
from make import DistKDE
import matplotlib.pyplot as plt
from psf import MLE2D_MCMC
from matplotlib.ticker import LogFormatter 

disc2D = Disc2D(20,20)
X,_,theta = disc2D.forward(0.0,1,sigma=0.92,N0=200)

kde = DistKDE(theta[:2,:].T)
S = kde.forward(20,sigma=1.5,upsample=4,xyvar=1.0)
plt.imshow(S)
plt.show()

mle2d_mcmc = MLE2D_MCMC(theta,X)
theta = np.squeeze(np.delete(theta,2,axis=0))
samples = mle2d_mcmc.metropolis(theta,iters=10000,beta=0.2,
                                prop_cov=0.05,diag=True)

tburn = 1000; bins = 100
samples = samples[:,tburn:]
sample_vars = np.var(samples,axis=1)
print(np.sqrt(sample_vars))

def scatter_hist(x, y, ax_main, ax_histx, ax_histy):
    ax_main.scatter(x, y, color='black', s=3)
    ax_histx.hist(x, bins=30, color='blue', alpha=0.5,density=True)
    ax_histy.hist(y, bins=30, color='blue', alpha=0.5, orientation='horizontal',density=True)
    ax_histx.set_ylabel(r'$p(\theta_{x}|x)$')
    ax_histy.set_xlabel(r'$p(\theta_{y}|x)$')
    ax_histx.set_xticks([]); ax_histx.set_yticks([])
    ax_histy.set_xticks([]); ax_histy.set_yticks([])

fig,ax=plt.subplots(1,3,figsize=(8,2.7))
ax_histx = ax[1].inset_axes([0, 1.05, 1, 0.25])
ax_histy = ax[1].inset_axes([1.05, 0, 0.25, 1])
scatter_hist(samples[0,:], samples[1,:], ax[1], ax_histx, ax_histy)
ax[0].imshow(X, cmap='gray')
ax[0].scatter(theta[1],theta[0],color='red',marker='x',s=5)
ax[0].set_xticks([0,10]); ax[0].set_yticks([0,10])
ax[1].set_xticks([10]); ax[1].set_yticks([10])
ax[0].set_xlabel(r'$u$'); ax[0].set_ylabel(r'$v$')
im = ax[2].imshow(np.sqrt(S[30:50,30:50]), cmap='gray')
ax[2].set_xlabel(r'$u$'); ax[2].set_ylabel(r'$v$')
formatter = LogFormatter(10, labelOnlyBase=True) 
plt.colorbar(im,ax=ax[2],fraction=0.046, 
             pad=0.04, format=formatter, label=r'$\sqrt{Var(y)}$')
plt.tight_layout()
plt.show()


