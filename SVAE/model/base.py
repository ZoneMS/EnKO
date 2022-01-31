import os, sys, time, math, copy
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
from .utils import get_nonlinear_fn, Identity


class BaseModel(nn.Module):
    def __init__(self, x_dim, config, device):
        super(BaseModel, self).__init__()
        
        if type(x_dim) is int:
            self.input_image_on = False
        elif type(x_dim) is tuple and len(x_dim)==3:
            self.input_image_on = True
            self.input_channels, self.height, self.width = x_dim
        else:
            raise ValueError("Input data dimension is incorrect. Type of observation dimension is {}".format(type(x_dim)))
        self.x_dim = x_dim
        self.device = device
        self.config = config
        self.z_dim = int(config["network"]["Dz"])
        self.h_dim = int(config["network"]["Dh"])
        self.n_particles = int(config["network"]["n_particles"])
        self.n_layers = int(config["network"]["n_layers"])
        self.bias = config["network"]["bias"]
        output_dist = config["network"]["output_dist"]
        self.system_name = config["data"]["system"]
        self.r_dim = int(config["network"]["n_rnn_units"])
        self.rnn_mode = config["network"]["rnn"]
        self.dropout_ratio = float(config["network"]["dropout_ratio"])
        self.loss_type = config["network"]["loss_type"]
        if self.loss_type not in ["prodsum", "sumprod", "spos"]:
            raise ValueError("input loss type is incorrect.")
        self.enc_steps = config["network"]["enc_steps"]
        if self.enc_steps is not None:
            self.enc_steps = int(self.enc_steps)
        self.kld_penalty_weight = config["network"]["kld_penalty_weight"]
        self.kld_penalty_on = self.kld_penalty_weight > 0
        self.transition_penalty_on = config["network"]["transition_penalty_on"]
        self.exclude_filtering_on = config["enko"]["exclude_filtering_on"]
        # self.thresholds_of_cosine_similarity = config["network"]["cosine_th"]
        
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
            
        self.output_fn = get_nonlinear_fn(config["network"]["output_fn"])
        self.activation_fn = get_nonlinear_fn(config["network"]["activation_fn"])
        
        if self.rnn_mode=="GRU":
            self.RNN = nn.GRU
        elif self.rnn_mode=="LSTM":
            self.RNN = nn.LSTM
        elif self.rnn_mode in ["ODE-RNN", "ODE-GRU", "ODE-LSTM"]:
            self.RNN = ODE_RNN
        elif self.rnn_mode=="GRU-ODE":
            from .ode_rnn import ODE_RNN
            self.RNN = GRU_ODE
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

    def _log_prob_Bernoulli(self, mean, scale, x, dim=None, keepdim=False, th=1e-2):
        """Negative log likelihood for Bernoulli distribution"""
        # scale is redundant
        th = th * torch.ones(mean.size()).to(self.device)
        return torch.sum(x * torch.log(torch.max(mean, th)) + (1-x) * torch.log(torch.max(1 - mean, th)), dim=dim, keepdim=keepdim)
    
    
    def _log_prob_normal(self, x, dim=None, keepdim=False):
        """Negative log likelihood for Gaussian distribution"""
        return - 0.5 * torch.sum(x.pow(2) + math.log(2*math.pi), dim=dim, keepdim=keepdim)
    
    
    def _get_elbo(self, log_gs, log_fs, log_qs, Z_prevs, klds=None, saturated_on=False):
        if self.transition_penalty_on and (not saturated_on):
            log_fs = self.x_dim / self.z_dim * log_fs
            log_qs = self.x_dim / self.z_dim * log_qs
        if self.loss_type in ["prodsum", "fivo"]:
            log_g = torch.logsumexp(log_gs, axis=1).sum(axis=0).mean() #(T,np,bs),(T,bs),(bs)
            log_f = torch.logsumexp(log_fs, axis=1).sum(axis=0).mean()
            log_q = torch.logsumexp(log_qs, axis=1).sum(axis=0).mean()
            log_ZSMC = torch.logsumexp(log_gs+log_fs-log_qs-math.log(self.n_particles), axis=1).sum(axis=0).mean()
        elif self.loss_type in ["sumprod", "iwae"]:
            log_g = torch.logsumexp(log_gs.sum(axis=0), axis=0).mean() #(T,np,bs),(np,bs),(bs)
            log_f = torch.logsumexp(log_fs.sum(axis=0), axis=0).mean()
            log_q = torch.logsumexp(log_qs.sum(axis=0), axis=0).mean()
            log_ZSMC = torch.logsumexp((log_gs+log_fs-log_qs).sum(axis=0), axis=0).mean() - math.log(self.n_particles)
        elif self.loss_type=="spos":
            log_q = torch.logsumexp(log_qs, axis=1) - math.log(self.n_particles) #(T,bs)
            log_ZSMC = (torch.logsumexp(((log_fs+log_gs).transpose(0,1)-log_q).sum(axis=1), axis=0) - math.log(self.n_particles)).mean() #(T,np,bs)->(np,T,bs)->(np,bs)->(bs,)
            log_f = torch.logsumexp(log_fs.sum(axis=0), axis=0).mean()
            log_g = torch.logsumexp(log_gs.sum(axis=0), axis=0).mean()
            log_q = log_q.sum(axis=0).mean() #(bs,)
        ESS = self._calculate_ess(log_gs+log_fs-log_qs).mean() #(T,)
        mean_cosine_similarity = self.calculate_cosine_similarity(Z_prevs).mean() #(T,nc,bs)
        if self.kld_penalty_on:
            kld = klds.sum(axis=0).mean() #(T,np,bs),(np,bs)
            #kld = torch.logsumexp(klds.sum(axis=0), axis=0).mean() #(T,np,bs),(np,bs),(bs,)
            if not saturated_on:
                log_ZSMC = log_ZSMC - self.kld_penalty_weight * kld
            return (-log_ZSMC, -log_f, -log_g, log_q, ESS, mean_cosine_similarity, kld)
        else:
            return (-log_ZSMC, -log_f, -log_g, log_q, ESS, mean_cosine_similarity)
    
    
    def _get_dropout_elbo(self, log_gs, log_fs, log_qs, dropout_on):
        # loggs,logfs,logqs:(T,np,bs), dropout_on:(T,bs)
        log_g = torch.logsumexp(log_gs, axis=1)[dropout_on==0].sum(axis=0).mean() #(T,np,bs),(T,bs),(bs)
        log_f = torch.logsumexp(log_fs, axis=1)[dropout_on==0].sum(axis=0).mean()
        log_q = torch.logsumexp(log_qs, axis=1)[dropout_on==0].sum(axis=0).mean()
        log_ZSMC = torch.logsumexp(log_gs+log_fs-log_qs, axis=1)[dropout_on==0].sum(axis=0).mean()
        return (-log_ZSMC, -log_f, -log_g, log_q)
    
    
    def _get_dropout_elbo2(self, log_gs, log_fs, log_qs, dropout_on, saturated_on):
        # loggs,logfs,logqs:(T,np,bs), dropout_on:(T,bs)
        T, _, batch_size = log_gs.shape
        log_gd = torch.logsumexp(log_gs, axis=1)[dropout_on==0].sum(axis=0).mean() #(T,np,bs),(T,bs),(bs)
        log_fd = torch.logsumexp(log_fs, axis=1)[dropout_on==0].sum(axis=0).mean()
        log_qd = torch.logsumexp(log_qs, axis=1)[dropout_on==0].sum(axis=0).mean()
        log_ZSMCd = torch.logsumexp(log_gs+log_fs-log_qs, axis=1)[dropout_on==0].sum(axis=0).mean()
        
        log_Ws = torch.zeros([T,batch_size,self.n_particles]).to(self.device)
        log_Ws[dropout_on==0] = (log_gs+log_fs-log_qs).transpose(1,2)[dropout_on==0] #(T,bs,np)
        log_Ws[dropout_on==1] = log_gs.transpose(1,2)[dropout_on==1] #(T,bs,np) 
        log_Ws = log_Ws.transpose(1,2) #(T,np,bs)
        log_g = torch.logsumexp(log_gs, axis=1).sum(axis=0).mean() #(T,np,bs),(T,bs),(bs)
        log_f = torch.logsumexp(log_fs, axis=1)[dropout_on==1].sum(axis=0).mean()
        log_q = torch.logsumexp(log_qs, axis=1)[dropout_on==1].sum(axis=0).mean()
        log_ZSMC = torch.logsumexp(log_Ws, axis=1).sum(axis=0).mean()
        ESS = self._calculate_ess(log_Ws).mean()
        
        total_loss = -(log_ZSMC+log_ZSMCd) if saturated_on else -log_ZSMC
        return (total_loss, -log_ZSMC, -log_f, -log_g, log_q, -log_ZSMCd, -log_fd, -log_gd, log_qd, ESS)
    
    
    def _calculate_ess(self, log_W):
        log_W = log_W.permute(1,0,2) #(np,T,bs)
        log_W = log_W - torch.logsumexp(log_W, dim=0) #(np,T,bs)
        ESS = 1 / torch.sum(torch.exp(log_W)**2, dim=0) #(T,bs)
        ESS_mean = torch.mean(ESS, dim=1) #(T,)
        return ESS_mean
    
    
    def calculate_cosine_similarity(self, Z):
        # Z: (T,np,bs,Dz)
        n_particles = Z.shape[1]
        combinations = torch.combinations(torch.arange(n_particles)) #(nc,2)
        Z_mean = Z.mean(1) #(T,bs,Dz)
        Z_centered = Z - Z_mean[:,np.newaxis] #(T,np,bs,Dz)
        inner_products = (Z_centered[:,combinations[:,0]] * Z_centered[:,combinations[:,1]]).sum(3) #(T,nc,bs)
        cosine_similarity = torch.where(inner_products==0, torch.ones_like(inner_products), 
                                        inner_products/(torch.norm(Z_centered[:,combinations[:,0]], dim=3)*torch.norm(Z_centered[:,combinations[:,1]], dim=3))) #(T,nc,bs)
        return cosine_similarity
    
    
    
    def _construct_dense_network(self, input_dim, hidden_dim, output_dim=None, n_layers=1, activation_fn=Identity, output_fn=Identity):
        net = [nn.Linear(input_dim, hidden_dim),
              activation_fn()]
        for i in range(n_layers):
            net += [nn.Linear(hidden_dim, hidden_dim),
                   activation_fn()]
        if output_dim is not None:
            net += [nn.Linear(hidden_dim, output_dim),
                   output_fn()]
        return nn.Sequential(*net)
    
    
    def _construct_conv_encoder_decoder(self, config, dec_input_dim, output_fn):
        filter_enc = [int(self.input_channels)] + [int(v) for v in config["conv"]["filter_enc"]]
        kernel_enc = [int(v) for v in config["conv"]["kernel_enc"]]
        stride_enc = [int(v) for v in config["conv"]["stride_enc"]]
        padding_enc = [int(v) for v in config["conv"]["padding_enc"]]
        bn_enc = config["conv"]["bn_enc"]
        filter_dec = [int(v) for v in config["conv"]["filter_dec"]] + [int(self.input_channels)]
        kernel_dec = [int(v) for v in config["conv"]["kernel_dec"]]
        stride_dec = [int(v) for v in config["conv"]["stride_dec"]]
        padding_dec = [int(v) for v in config["conv"]["padding_dec"]]
        output_padding_dec = [int(v) for v in config["conv"]["output_padding_dec"]]
        bn_dec = config["conv"]["bn_dec"]
        
        hidden_w = calculate_conv_size(self.width, kernel_enc, stride_enc, padding_enc)
        hidden_h = calculate_conv_size(self.height, kernel_enc, stride_enc, padding_enc)
        enc_dim = hidden_w * hidden_h * filter_enc[-1]
        
        # encoder
        enc = []
        for (ic,oc,k,s,p,bn) in zip(filter_enc[:-1], filter_enc[1:], kernel_enc, stride_enc, padding_enc, bn_enc):
            if bn:
                enc += [
                    nn.Conv2d(ic,oc,k,s,p),
                    nn.BatchNorm2d(oc),
                    nn.ReLU()
                ]
            else:
                enc += [
                    nn.Conv2d(ic,oc,k,s,p),
                    nn.ReLU()
                ]
        enc += [Flatten()]
        enc = nn.Sequential(*enc)
        
        dec = [nn.Linear(dec_input_dim, enc_dim),
                UnFlatten(hidden_w, hidden_h)]
        for (ic,oc,k,s,p,op,bn) in zip(filter_dec[:-2], filter_dec[1:-1], kernel_dec[:-1], stride_dec[:-1], padding_dec[:-1], output_padding_dec[:-1], bn_dec):
            if bn:
                self.dec += [
                    nn.ConvTranspose2d(ic,oc,k,s,p,op),
                    nn.BatchNorm2d(oc),
                    nn.ReLU()
                ]
            else:
                self.dec += [
                    nn.ConvTranspose2d(ic,oc,k,s,p,op),
                    nn.ReLU()
                ]
        self.dec += [
            nn.ConvTranspose2d(filter_dec[-2], filter_dec[-1], kernel_dec[-1], stride_dec[-1], padding_dec[-1]),
            output_fn()
        ]
        dec = nn.Sequential(*dec)
        
        return enc_dim, enc, dec