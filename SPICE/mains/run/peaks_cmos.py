from pipes import *
from SPICE.utils import SMLMDataset
from skimage.io import imsave
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

config_path = 'peaks_cmos.json' #replace with path to your config
with open(config_path, 'r') as f:
    config = json.load(f)

prefixes = [
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_1',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_2',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_3',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_4',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_5',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_6',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_7',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_8',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_9',
'QD655_HILO_Continuous_20mW_100_frames_10ms_CMOS_10'
]


all_peaks = []
bins = np.linspace(10,600,15)
for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineCMOS(config,dataset)
    spots,peaks = pipe.localize(plot_spots=False)
    all_peaks += list(peaks)
all_peaks = np.array(all_peaks)

fig,ax=plt.subplots(figsize=(3,3))
hist,bins = np.histogram(all_peaks,bins=bins,density=False)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
ax.bar(center, hist, align='center', width=width)
ax.set_xlabel('Peak ADU')
ax.set_ylabel('Frequency')
plt.show()
