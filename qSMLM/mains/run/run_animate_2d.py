import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from SMLM.utils import make_animation
from tifffile import imread

with open('run_animate_2d.json', 'r') as f:
    config = json.load(f)


prefixes = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_11',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_19',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_21',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_25',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_28',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_29',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_5',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_7',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_8',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKa_9'
]


prefixes = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_11',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_12',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_13',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_14',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_15',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_16',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_17',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_20',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_6',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_AMPKi_9'

]


prefixes = [
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_12',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_13',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_17',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_18',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_1',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_2',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_3',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_5',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_7',
'230823_U2OS_j646_2pM_overnight_2000frames_20mW_Ctrl_8'
]


for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['analpath'] + prefix + '/' + prefix + '_spots.csv')
    stack = imread(config['datapath']+prefix+'.tif')
    make_animation(stack,spots)
    plt.show()
