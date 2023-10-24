from pipes import PipelineMLE2D
from SMLM.torch.dataset import SMLMDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

"""
prefixes = [
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-1',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-3',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-4',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-8',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-10',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-12'
]

prefixes = [
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-1',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-3',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-5',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-8',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-9',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-10'
]
"""

prefixes = [
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_0min_5p-1_5',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_3min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_6min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_10min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_15min_5p-1_2',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_20min_5p-1_2'
]

prefixes = [
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_0min_5p-1_3',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_3min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_6min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_10min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_15min_5p-1_1',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_20min_5p-1_1'
]

prefixes = [
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_0min_5p-1_13',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_3min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_6min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_10min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_15min_5p-1_10',
'230913_Hela_j646_10pM_overnight_J549_5pm_2h_2000frames_20mW_20min_5p-1_10'
]

prefixes = [
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_0min_control_4',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_3min_control_4',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_6min_control_4',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_10min_control_4',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_15min_control_4',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_20min_control_4'
]

prefixes = [
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_0min_5p_9',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_3min_5p_9',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_6min_5p_9',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_10min_5p_9',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_15min_5p_9',
'230920_Hela_j646_9pm_overnight_200frames_20mW_10ms_20min_5p_9'
]

with open('run_mle_2d.json', 'r') as f:
    config = json.load(f)

for prefix in prefixes:
    print("Processing " + prefix)
    dataset = SMLMDataset(config['datapath']+prefix,prefix)
    pipe = PipelineMLE2D(config,dataset)
    pipe.localize(plot_spots=False,plot_fit=False,tmax=200,run_deconv=False)
