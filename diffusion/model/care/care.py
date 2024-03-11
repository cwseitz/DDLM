import math
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from inspect import isfunction

class CARE(nn.Module):
    def __init__(self):
        super(CARE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        #a = x[0,0].cpu().detach().numpy()
        #b = x[0,1].cpu().detach().numpy()
        #fig,ax=plt.subplots(1,2)
        #ax[0].imshow(a); ax[1].imshow(b)
        #plt.show()
        x = self.encoder(x)
        #x = self.decoder(x)
        return x
