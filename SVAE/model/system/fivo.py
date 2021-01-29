import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from .base import BaseSystem


class FIVO(BaseSystem):
    def __init__(self, x_dim, config, device):
        super(FIVO, self).__init__(x_dim, config, device)
    
    
    def _resample_particles(self, X, logW, sample_size=None):
        """resample particles following their likelihood"""
        # X:(np,bs,Dx)/(M,np,bs,Dx), logW:(np,bs)/(M,np,bs)
        perm = list(range(1, len(logW.shape))) + [0]
        logW = logW - torch.logsumexp(logW, axis=0) # (np,bs)
        categorical = Categorical(logits=logW.permute(perm)) # (bs,np)
        
        if sample_size==None:
            #indices = categorical.sample().unsqueeze(0).repeat(self.n_particles,1) # (np,bs)
            indices = torch.repeat_interleave(categorical.sample().unsqueeze(0), logW.size(0), dim=0) #(np,bs)/(M,np,bs)
            if isinstance(X, list):
                X_resampled = []
                for item in X:
                    if len(indices.shape)==len(item.shape):
                        X_resampled.append(torch.gather(item, 0, indices)[0]) # (bs)/(np,bs)
                    elif len(indices.shape)<len(item.shape):
                        #X_resampled.append(torch.gather(item, 0, indices.unsqueeze(len(indices.shape)).repeat(1,1,item.size(2)))[0]) # (bs,Dx)/(np,bs,Dx)
                        X_resampled.append(torch.gather(item, 0, torch.repeat_interleave(indices.unsqueeze(len(indices.shape)), item.shape[-1], dim=len(indices.shape)))[0]) # (bs,Dx)/(np,bs,Dx)
            else:
                X_resampled = torch.gather(X, 0, indices.unsqueeze(len(indices.shape)).repeat(1,1,X.size(2)))[0] # (bs,Dx)/(np,bs,Dx)
        else:
            indices = categorical.sample(torch.Size([sample_size])) # (np,bs)
            if isinstance(X, list):
                X_resampled = []
                for item in X:
                    if len(indices.shape)==len(item.shape):
                        X_resampled.append(torch.gather(item, 0, indices)) # (np,bs)
                    elif len(indices.shape)<len(item.shape):
                        X_resampled.append(torch.gather(item, 0, indices.unsqueeze(2).repeat(1,1,item.size(2)))) # (np,bs,Dx)
            else:
                indices = indices.unsqueeze(2).repeat(1,1,X.size(2)) # (np,bs,Dx)
                X_resampled = torch.gather(X, 0, indices) # (np,bs,Dx)
                
        return X_resampled
    

    