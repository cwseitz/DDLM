from pipes import PipelineMLE2D_MCMC
from SMLM.torch.dataset import SMLMDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_5'
]

with open('run_mle_2d_mcmc.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineMLE2D_MCMC(config,dataset)
    pipe.localize(plot_spots=False,plot_fit=False,plot_mcmc=False,tmax=2000,run_deconv=False)
