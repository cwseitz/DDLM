from pipes import *
from oci.utils import SMLMDataset
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'peaks_spad.json' #replace with path to your config
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = [
'Stack',
'Stack0',
'Stack1',
'Stack3',
'Stack6',
'Stack7'
]


all_peaks = []
bins = np.linspace(50,600,15)
for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineSPAD(config,dataset)
    spots,peaks = pipe.localize(plot_spots=False)
    all_peaks += list(peaks)
all_peaks = np.array(all_peaks)

fig,ax=plt.subplots(figsize=(3,3))
hist,bins = np.histogram(all_peaks,bins=bins,density=False)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
ax.bar(center, hist, align='center', width=width)
ax.set_xlabel('Peak Photon Counts')
ax.set_ylabel('Frequency')
plt.show()
