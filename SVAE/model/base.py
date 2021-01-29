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

from .system import *
from .utils import get_nonlinear_fn


class BaseModel(nn.Module):
    def __init__(self, x_dim, config, device):
        super(BaseModel, self).__init__()
        
        self.x_dim = x_dim
        self.device = device
        self.config = config
        self.z_dim = int(config["network"]["Dz"])
        self.h_dim = int(config["network"]["Dh"])
        self.n_particles = int(config["network"]["n_particles"])
        self.n_layers = int(config["network"]["n_layers"])
        self.bias = config["network"]["bias"]
        output_dist = config["network"]["output_dist"]
        output_fn = config["network"]["output_fn"]
        self.system_name = config["data"]["system"]
        self.r_dim = int(config["network"]["n_rnn_units"])
        self.rnn_mode = config["network"]["rnn"]
        self.dropout_ratio = float(config["network"]["dropout_ratio"])
        self.enc_steps = config["network"]["enc_steps"]
        if self.enc_steps is not None:
            self.enc_steps = int(self.enc_steps)
        
        if self.system_name=="EnKO":
            self.system = EnKO(x_dim, config, device)
        elif self.system_name=="FIVO":
            self.system = FIVO(x_dim, config, device)
        elif self.system_name=="IWAE":
            self.system = BaseSystem(x_dim, config, device)
        
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
            
        self.output_fn = get_nonlinear_fn(output_fn)
        
        if self.rnn_mode=="RNN":
            self.RNN = nn.RNN
        elif self.rnn_mode=="GRU":
            self.RNN = nn.GRU
        elif self.rnn_mode=="LSTM":
            self.RNN = nn.LSTM
        else:
            raise ValueError("input RNN type is incorrect.")
        
        
    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mean)
    
    
    def _reparameterized_sample2(self, mean, std, mean2, std2):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mean), eps.mul(std2).add_(mean2)
    
    
    def _reparameterized_sample_Laplace(self, mean, scale):
        """using mean to sample"""
        laplace = torch.distributions.laplace.Laplace(0.0, 1.0)
        eps = laplace.sample(mean.size())
        eps = Variable(eps).to(self.device)
        return eps.mul(scale).add_(mean)
    
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass
    
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2, dim=None, keepdim=False):
        """Using std to compute KLD"""
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element, dim=dim, keepdim=keepdim)
    
    
    def _kld_gauss_log(self, mean_1, logvar_1, mean_2, logvar_2, dim=None, keepdim=False):
        """Using std to compute KLD"""
        kld_element =  (logvar_1 - logvar_2 + 
            (torch.exp(logvar_1) + (mean_1 - mean_2).pow(2)) /
            torch.exp(logvar_2) - 1)
        return 0.5 * torch.sum(kld_element, dim=dim, keepdim=keepdim)
    
    
    def _kld_gauss_normal(self, mean, std, dim=None, keepdim=False):
        """Using std to compute KLD"""
        kld_element =  (-2 * torch.log(std) + std.pow(2) + mean.pow(2) - 1)
        return 0.5 * torch.sum(kld_element, dim=dim, keepdim=keepdim)
    
    
    def _log_prob(self, mean, std, x, dim=None, keepdim=False, logvar=False):
        """Negative log likelihood for Gaussian distribution"""
        if logvar:
            # consider std as logvar
            return - 0.5 * torch.sum((x-mean).pow(2) / torch.exp(std) \
                        + math.log(2*math.pi) + std, dim=dim, keepdim=keepdim)
        else:
            return - 0.5 * torch.sum((x-mean).pow(2) / std.pow(2) \
                        + torch.log(2*math.pi*std.pow(2)), dim=dim, keepdim=keepdim)
    
    def _log_prob_Laplace(self, mean, scale, x, dim=None, keepdim=False):
        """Negative log likelihood for Laplace distribution"""
        return - torch.sum(torch.abs(x - mean) / scale + torch.log(2*scale), dim=dim, keepdim=keepdim)

    def _log_prob_Bernoulli(self, mean, th, x, dim=None, keepdim=False):
        """Negative log likelihood for Bernoulli distribution"""
        # scale is redundant
        th = th * torch.ones(mean.size()).to(self.device)
        return - torch.sum(x * torch.log(torch.max(mean, th)) + (1-x) * torch.log(torch.max(1 - mean, th)), dim=dim, keepdim=keepdim)
    
    
    def _log_prob_normal(self, x, dim=None, keepdim=False):
        """Negative log likelihood for Gaussian distribution"""
        return - 0.5 * torch.sum(x.pow(2) + math.log(2*math.pi), dim=dim, keepdim=keepdim)
    
    
    def _get_elbo(self, log_gs, log_fs, log_qs):
        log_g = torch.logsumexp(log_gs, axis=1).sum(axis=0).mean() #(T,np,bs),(T,bs),(bs)
        log_f = torch.logsumexp(log_fs, axis=1).sum(axis=0).mean()
        log_q = torch.logsumexp(log_qs, axis=1).sum(axis=0).mean()
        log_ZSMC = torch.logsumexp(log_gs+log_fs-log_qs, axis=1).sum(axis=0).mean()
        ESS = self._calculate_ess(log_gs+log_fs-log_qs).mean() #(T,)
        return (-log_ZSMC, -log_f, -log_g, log_q, ESS)
    
    
    def _calculate_ess(self, log_W):
        log_W = log_W.permute(1,0,2) #(np,T,bs)
        log_W = log_W - torch.logsumexp(log_W, dim=0) #(np,T,bs)
        ESS = 1 / torch.sum(torch.exp(log_W)**2, dim=0) #(T,bs)
        ESS_mean = torch.mean(ESS, dim=1) #(T,)
        return ESS_mean