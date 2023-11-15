from pipes import PipelineG22D, PipelineG22D_Win
from qSMLM.utils import SMLMDataset
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'gq2d.json' #replace with path to your config
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = [
'Stack',
#'Stack6'
]

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineG22D_Win(config,dataset)
    print(dataset.stack.shape)

"""
bins = np.arange(0,10,1)
fig,ax=plt.subplots()
win_size = 50
for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineG22D_Win(config,dataset)
    covs = pipe.localize(win_size,plot_spots=False,plot_cov=False,plot_ts=False)
    print(covs.shape)
    covs = covs.ravel(); covs = covs[covs > 0]
    print(covs.shape)
ax.hist(covs,bins=bins,density=True,color='red',alpha=0.5)


cov_tmax = 10000
for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineG22D(config,dataset)
    covs = pipe.localize(cov_tmax,plot_spots=False,plot_cov=False,plot_ts=False)
    covs = covs.ravel(); covs = covs[covs > 0]
ax.hist(covs,bins=bins,density=True,color='blue',alpha=0.5)
plt.show()
"""
