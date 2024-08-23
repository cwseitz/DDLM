from make.dataset import *
import os

savepath = 'dataset/diffusion/'
prefix = 'Diffusion'

os.makedirs(savepath,exist_ok=True)
os.makedirs(savepath+'lr_20',exist_ok=True)
os.makedirs(savepath+'hr_80',exist_ok=True)

nx = ny = 100
radius = 40.0
nspots = 180
nsamples = 1000
args = [radius,nspots]

kwargs = {
'N0':5000,
'B0':0,
'eta':1.0,
'sigma':0.92,
"gain": 1.0,
"offset": 0.0,
"var": 5.0,
"show": False
}

generator = Disc2D(nx,ny)
dataset = TrainDataset(nsamples)

X,Z,_ = dataset.make_dataset(generator,args,kwargs,
                             show=False,upsample=4,
                             sigma_kde=3.0,sigma_gauss=0.5)
for n in range(nsamples):
    imsave(savepath+'lr_20/'+prefix+f'_lr-{n}.tif',X[n])
    imsave(savepath+'hr_80/'+prefix+f'_hr-{n}.tif',Z[n])
    
