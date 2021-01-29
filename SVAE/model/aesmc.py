import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from .svo import SVO


class AESMC(SVO):
    def __init__(self, x_dim, config, device):
        super(AESMC, self).__init__(x_dim, config, device)
        self.smooth_obs = False
        
        # networks
        self.q0_tran = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU())
        
        self.q_tran = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())