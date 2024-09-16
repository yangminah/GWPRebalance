import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import count_params, init_xavier

class model(nn.Module):
    def __init__(self, n_ch, n_d, n_dilations, n_levels=40):
        super(model, self).__init__()
        # conv1d(input_channel, output_channels, kernel_size, stride; args)
        self.n_levels = n_levels # Input size. (length of column)
        self.n_ch = n_ch
        self.n_d = n_d
        self.kernel_size=3
        self.n_dilations = n_dilations
        self.n_flat_in = int(self.n_ch[3]*n_levels/2)+3
        self.n_flat_out = int(self.n_ch[3]*n_levels/2)
        self.cnn_encode = nn.Sequential(
            nn.Conv1d(self.n_ch[0], self.n_ch[1], self.kernel_size, 1, padding='same', dilation=n_dilations[0]), nn.ELU(),
            nn.Conv1d(self.n_ch[1], self.n_ch[2], self.kernel_size, 1, padding='same', dilation=n_dilations[1]), nn.ELU(),
            nn.Conv1d(self.n_ch[2], self.n_ch[3], self.kernel_size, 1, padding='same', dilation=n_dilations[2]), nn.ELU(),
            nn.MaxPool1d(kernel_size=2),            nn.Flatten()
        )
        self.dense = nn.Sequential(
            nn.Linear(self.n_flat_in, self.n_d),       nn.ELU(),
            nn.Linear(self.n_d,       self.n_d),       nn.ELU(),
            # nn.Linear(self.n_d,    self.n_d), nn.ELU(), nn.BatchNorm1d(100)
            # nn.Linear(self.n_d,    self.n_d), nn.ELU(), nn.BatchNorm1d(100)
            nn.Linear(self.n_d,       self.n_flat_out), nn.ELU()
            )
        self.cnn_decode = nn.Sequential(
            nn.ConvTranspose1d(self.n_ch[3], self.n_ch[2], self.kernel_size, 2, padding=1, output_padding=1, dilation=n_dilations[3]), nn.ELU(),
            nn.ConvTranspose1d(self.n_ch[2], self.n_ch[1], self.kernel_size, 1, padding=1, dilation=n_dilations[4]), nn.ELU(),
            nn.ConvTranspose1d(self.n_ch[1], 1,          self.kernel_size, 1, padding=1, dilation=n_dilations[5])
        )
        self.cnn_encode.apply(init_xavier)
        self.dense.apply(init_xavier)
        self.cnn_decode.apply(init_xavier)
    def forward(self, x):
        # Encode 3d variables
        x3, xloc = x
        z = self.cnn_encode(x3)#; print(z3.shape)

        # Concatenate with loc variables
        z = torch.cat((z,xloc),axis=1)
        
        # Dense it up.
        z = self.dense(z)

        # Reshape for convolutions.
        z = torch.reshape(z,(z.shape[0], self.n_ch[3], int(self.n_levels/2)))
        
        # Decode.
        gu = self.cnn_decode(z).squeeze()
        return gu
