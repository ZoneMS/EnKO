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


class BaseSystem(nn.Module):
    def __init__(self, x_dim, config, device):
        super(BaseSystem, self).__init__()
        
        self.x_dim = x_dim
        self.z_dim = config["network"]["Dz"]
        self.device = device
        self.config = config
        self.n_particles = config["network"]["n_particles"]
        output_dist = config["network"]["output_dist"]
        
        if output_dist=="Gauss":
            self._reparameterized_sample_emission = self._reparameterized_sample
        elif output_dist=="Laplace":
            self._reparameterized_sample_emission = self._reparameterized_sample_Laplace
        elif output_dist=="Bernoulli":
            self._reparameterized_sample_emission = self._reparameterized_sample_Bernoulli
    
    
    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mean)
    
    
    def _reparameterized_sample_Laplace(self, mean, scale):
        """using mean to sample"""
        laplace = torch.distributions.laplace.Laplace(0.0, 1.0)
        eps = laplace.sample(mean.size())
        eps = Variable(eps).to(self.device)
        return eps.mul(scale).add_(mean)
    
    
    def _reparameterized_sample_Bernoulli(self, mean, scale):
        """Bernoulli: special case of Gumbel-softmax trick"""
        # scale is redundant
        eps = torch.FloatTensor(std.size()).uniform_()
        eps = Variable(eps).to(self.device)
        return f.sigmoid(torch.log(eps) - torch.log(1-eps) + torch.log(mean) - torch.log(1-mean))
    
    