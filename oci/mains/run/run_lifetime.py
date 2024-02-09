import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from BaseSMLM.localize import LoGDetector
from pipes import LifetimeHMM


prefixes = [
'240108_QD_20mW_10ms_1-Crop'
]

prefixes = [
'240206_QD-10ms-488-5V_3-Crop'
]

prefixes = [
'240110_Control_JF646_4pm_overnight_L640_30mW_10ms____10'
]



with open('run_lifetime.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    stack = imread(config['path'] + prefix + '.tif')
    pipe = LifetimeHMM(stack)
    pipe.forward(show_spots=True,show_hmm=True,min_comp=2,max_comp=10,
                 show_hist=False,threshold=0.001)


