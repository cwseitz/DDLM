from pipes import PipelineG22D
from miniSMLM.utils import SMLMDataset
from miniSMLM.utils import KDE, make_animation
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'gq2d.json' #replace with path to your config
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = [
'Stack6',
]

bins = np.linspace(0,5,50)
for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineG22D(config,dataset)
    covs = pipe.localize(plot_spots=True,plot_cov=True)
    covs = covs.ravel()
    covs = covs[covs > 0]
    plt.hist(covs,bins=bins,density=True,color='red')
    plt.show()
