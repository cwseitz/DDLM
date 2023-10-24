import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from miniSMLM.utils import KDE
from tifffile import imread
from skimage.io import imsave

with open('run_kde_2d.json', 'r') as f:
    config = json.load(f)

prefixes = [
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-1',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-3',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-5',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-8',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_fixed-9',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-1',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-3',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-4',
'230909_Hela_j646_10pM_overnight_2000frames_20mW_live-8'
]


plot=False
for n,prefix in enumerate(prefixes):
    print("Processing " + prefix)
    spots = pd.read_csv(config['ipath'] + prefix + '_spots.csv')
    spots = spots.loc[(spots['N0'] > 10) & (spots['N0'] < 5000) & (spots['x_mle'] > 0) & (spots['y_mle'] > 0)]
    spots = spots.sample(10000)
    render = KDE(spots).get_kde(sigma=2.0)
    
    if plot:
        fig,ax=plt.subplots()
        ax.imshow(render,cmap='gray',vmin=0,vmax=0.2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        fig,ax=plt.subplots()
        ax.scatter(spots['y_mle'],spots['x_mle'],s=1,color='black',marker='x')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        plt.show()
    
    imsave(config['opath']+prefix+'-kde.tif',render)
