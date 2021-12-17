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

from .base import BaseModel
from .utils import get_model, get_nonlinear_fn


class OuterModel(BaseModel):
    def __init__(self, x_dim, config, device):
        super(OuterModel, self).__init__(x_dim, config, device)
        
        output_dist = config["network"]["outer_output_dist"]
        self.output_fn = get_nonlinear_fn(config["network"]["outer_output_fn"])
        self.a_dim = config["outer"]["a_dim"]
        self.only_outer_learning_epochs = config["training"]["only_outer_learning_epochs"]
        self.outer_scale = config["outer"]["outer_scale"]
        self.model_name = config["data"]["model"]
        
        if output_dist=="Gauss":
            self._log_prob_emission = self._log_prob
            self._reparameterized_sample_emission = self._reparameterized_sample
        elif output_dist=="Laplace":
            self._log_prob_emission = self._log_prob_Laplace
            self._reparameterized_sample_emission = self._reparameterized_sample_Laplace
        elif output_dist=="Bernoulli":
            self._log_prob_emission = self._log_prob_Bernoulli
        else:
            raise ValueError("input output distribution type is incorrect.")
        
