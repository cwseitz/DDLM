from make.dataset import *
import os

savepath = 'dataset/diffusion/'
prefix = 'Diffusion'

os.makedirs(savepath,exist_ok=True)
os.makedirs(savepath+'lr_20')
os.makedirs(savepath+'hr_80')
os.makedirs(savepath+'spikes')

nx = ny = 20
radius = 7.0
nspots = 10
nsamples = 5000
args = [radius,nspots]

kwargs = {
'N0':500,
'B0':0,
'eta':0.8,
'sigma':0.92,
"gain": 1.0,
"offset": 100.0,
"var": 5.0
}

generator = Disc2D(nx,ny)
dataset = TrainDataset(nsamples)
X,Z,S = dataset.make_dataset(generator,args,kwargs,
                             show=False,upsample=4,
                             sigma_kde=1.5,sigma_gauss=0.5)
for n in range(nsamples):
    imsave(savepath+'lr_20/'+prefix+f'_lr-{n}.tif',X[n])
    imsave(savepath+'hr_80/'+prefix+f'_hr-{n}.tif',Z[n])
    imsave(savepath+'spikes/'+prefix+f'_spikes-{n}.tif',S[n])

