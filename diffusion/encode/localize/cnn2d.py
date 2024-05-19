import argparse
import collections
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from ..torch.utils import prepare_device
from ..torch.models import LocalizationCNN
from torch.nn import Module, MaxPool2d, ConstantPad3d
from torch.nn.functional import conv3d
from skimage.feature import blob_log
from skimage.util import img_as_float

def tensor_to_np(x):
    return np.squeeze(x.cpu().detach().numpy())

class NeuralEstimator2D:
    def __init__(self,config):
        self.modelpath = config['modelpath']
        self.modelname = config['modelname']
        self.model,self.device = self.load_model()
        self.spots = pd.DataFrame(columns=['x','y'])
    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_config = self.modelpath+self.modelname+'/'+self.modelname+'.json'
        with open(train_config, 'r') as f:
            train_config = json.load(f)
        args = train_config['arch']['args']
        model = LocalizationCNN(args['nz'],args['scaling_factor'],dilation_flag=args['dilation_flag'])
        model = model.to(device=device)
        checkpoint = self.modelpath+self.modelname+'/'+self.modelname+'.pth'
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model,device
    def forward(self,stack,show=False):
        stack = stack.astype(np.float32)
        stack = torch.from_numpy(stack)
        stack = stack.to(self.device)
        nb,nc,nx,ny = stack.shape
        outputs = []
        for n in range(nb):
            print(f'Model forward on frame {n}')
            output = self.model(stack[n].unsqueeze(0))
            outputs.append(output.detach().cpu().numpy())
        outputs = np.squeeze(np.array(outputs))
        return outputs


        
        
        

