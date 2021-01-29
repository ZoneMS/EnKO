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


class EnKO(BaseSystem):
    def __init__(self, x_dim, config, device):
        super(EnKO, self).__init__(x_dim, config, device)

        filtering_method = config["enko"]["filtering_method"]
        inflation_method = config["enko"]["inflation_method"]
        self.inflation_factor = config["enko"]["inflation_factor"]
        
        if filtering_method=="inverse":
            self.update = self._enkf_inverse
        elif filtering_method=="inverse-ide":
            self.update = self._enkf_inverse_ide
        elif filtering_method=="diag-inverse":
            self.update = self._enkf_diagonal_inverse
        elif filtering_method=="etkf-diag":
            self.update = self._etkf_R_diagonal
        else:
            raise ValueError("input filtering method is incorrect.")
        self.filtering_method = filtering_method
            
        if inflation_method in ["multiplicative", "additive", "RTPP", "RTPS"]:
            self.inflation_method = inflation_method
        else:
            self.inflation_method = None
        
    
    
    def _enkf_inverse(self, x_t, Z_t, dec_mean_t, dec_std_t):
        x_hat_t = self._reparameterized_sample_emission(dec_mean_t, dec_std_t) #(np,bs,Dx)
        centered_Z_t = Z_t - Z_t.mean(dim=0) #(np,bs,Dz)
        centered_x_t = x_hat_t - x_hat_t.mean(dim=0) #(np,bs,Dx)
        centered_dec_mean_t = dec_mean_t - dec_mean_t.mean(dim=0) #(np,bs,Dx)
        addjust_term = self.n_particles / (self.n_particles - 1)
        Sigma_zx_t = addjust_term * (centered_Z_t.unsqueeze(3).repeat(1,1,1,self.x_dim)
                    * centered_dec_mean_t.unsqueeze(2).repeat(1,1,self.z_dim,1)).mean(axis=0) # (bs,Dz,Dx)
        # direct estimate for HVH+R
        Sigma_w_t = addjust_term * (centered_x_t.unsqueeze(3).repeat(1,1,1,self.x_dim)
                    * centered_x_t.unsqueeze(2).repeat(1,1,self.x_dim,1)).mean(axis=0) # (bs,Dx,Dx)
        
        # Kalman gain
        K_t = Sigma_zx_t @ torch.inverse(Sigma_w_t) # (bs,Dz,Dx)
        increment = (K_t @ (x_t - x_hat_t).permute(1,2,0)).permute(2,0,1) # (np,bs,Dz)
        Z_filtered_t = Z_t + increment #(np,bs,Dz)
        
        if self.inflation_method=="RTPP":
            centered_Z_filtered_t = Z_filtered_t - Z_filtered_t.mean(dim=0) #(np,bs,Dz)
            Z_filtered_t = Z_filtered_t.mean(dim=0) + self.inflation_factor * centered_Z_t + (1-self.inflation_factor) * centered_Z_filtered_t #(np,bs,Dz)
        elif self.inflation_method=="RTPS":
            sigma_z_t = torch.sqrt(addjust_term * (centered_Z_t*centered_Z_t).mean(dim=0)) #(bs,Dz)
            centered_Z_filtered_t = Z_filtered_t - Z_filtered_t.mean(dim=0) #(np,bs,Dz)
            sigma_z_filtered_t = torch.sqrt(addjust_term * (centered_Z_filtered_t*centered_Z_filtered_t).mean(dim=0)) #(bs,Dz)
            Z_filtered_t = Z_filtered_t.mean(dim=0) + (self.inflation_factor * sigma_z_t + (1-self.inflation_factor) * sigma_z_filtered_t) / sigma_z_filtered_t * centered_Z_filtered_t #(np,bs,Dz)
        return Z_filtered_t
    
    
    def _enkf_inverse_ide(self, x_t, Z_t, dec_mean_t, dec_std_t):
        x_hat_t = self._reparameterized_sample_emission(dec_mean_t, dec_std_t) #(np,bs,Dx)
        centered_Z_t = Z_t - Z_t.mean(dim=0) #(np,bs,Dz)
        centered_x_t = x_hat_t - x_hat_t.mean(dim=0) #(np,bs,Dx)
        centered_dec_mean_t = dec_mean_t - dec_mean_t.mean(dim=0) #(np,bs,Dx)
        addjust_term = self.n_particles / (self.n_particles - 1)
        Sigma_zx_t = addjust_term * (centered_Z_t.unsqueeze(3).repeat(1,1,1,self.x_dim)
                    * centered_dec_mean_t.unsqueeze(2).repeat(1,1,self.z_dim,1)).mean(axis=0) # (bs,Dz,Dx)
        # indirect estimate for HVH+R
        Sigma_w_t = torch.diag_embed(dec_std_t.mean(dim=0)) \
                    + addjust_term * (centered_dec_mean_t.unsqueeze(3).repeat(1,1,1,self.x_dim)
                                * centered_dec_mean_t.unsqueeze(2).repeat(1,1,self.x_dim,1)).mean(axis=0) # (bs,Dx,Dx)
        
        # Kalman gain
        K_t = Sigma_zx_t @ torch.inverse(Sigma_w_t) # (bs,Dz,Dx)
        increment = (K_t @ (x_t - x_hat_t).permute(1,2,0)).permute(2,0,1) # (np,bs,Dz)
        Z_filtered_t = Z_t + increment #(np,bs,Dz)
        
        if self.inflation_method=="RTPP":
            centered_Z_filtered_t = Z_filtered_t - Z_filtered_t.mean(dim=0) #(np,bs,Dz)
            Z_filtered_t = Z_filtered_t.mean(dim=0) + self.inflation_factor * centered_Z_t + (1-self.inflation_factor) * centered_Z_filtered_t #(np,bs,Dz)
        elif self.inflation_method=="RTPS":
            sigma_z_t = torch.sqrt(addjust_term * (centered_Z_t*centered_Z_t).mean(dim=0)) #(bs,Dz)
            centered_Z_filtered_t = Z_filtered_t - Z_filtered_t.mean(dim=0) #(np,bs,Dz)
            sigma_z_filtered_t = torch.sqrt(addjust_term * (centered_Z_filtered_t*centered_Z_filtered_t).mean(dim=0)) #(bs,Dz)
            Z_filtered_t = Z_filtered_t.mean(dim=0) + (self.inflation_factor * sigma_z_t + (1-self.inflation_factor) * sigma_z_filtered_t) / sigma_z_filtered_t * centered_Z_filtered_t #(np,bs,Dz)
        return Z_filtered_t
    
    
    
    def _enkf_diagonal_inverse(self, x_t, Z_t, dec_mean_t, dec_std_t):
        x_hat_t = self._reparameterized_sample_emission(dec_mean_t, dec_std_t) #(np,bs,Dx)
        centered_Z_t = Z_t - Z_t.mean(dim=0) #(np,bs,Dz)
        centered_x_t = x_hat_t - x_hat_t.mean(dim=0) #(np,bs,Dx)
        centered_dec_mean_t = dec_mean_t - dec_mean_t.mean(dim=0) #(np,bs,Dx)
        addjust_term = self.n_particles / (self.n_particles - 1)
        Sigma_zx_t = addjust_term * (centered_Z_t.unsqueeze(3).repeat(1,1,1,self.x_dim)
                    * centered_dec_mean_t.unsqueeze(2).repeat(1,1,self.z_dim,1)).mean(axis=0) # (bs,Dz,Dx)
        Sigma_w_t = addjust_term * (centered_x_t**2).mean(axis=0) # (bs,Dx)
        
        # Kalman gain
        K_t = (Sigma_zx_t.permute(1,0,2) / Sigma_w_t).permute(1,0,2) # (bs,Dz,Dx)
        increment = (K_t @ (x_t - x_hat_t).permute(1,2,0)).permute(2,0,1) # (np,bs,Dz)
        Z_filtered_t = Z_t + increment #(np,bs,Dz)
        
        if self.inflation_method=="RTPP":
            centered_Z_filtered_t = Z_filtered_t - Z_filtered_t.mean(dim=0) #(np,bs,Dz)
            Z_filtered_t = Z_filtered_t.mean(dim=0) + self.inflation_factor * centered_Z_t + (1-self.inflation_factor) * centered_Z_filtered_t #(np,bs,Dz)
        elif self.inflation_method=="RTPS":
            sigma_z_t = torch.sqrt(addjust_term * (centered_Z_t*centered_Z_t).mean(dim=0)) #(bs,Dz)
            centered_Z_filtered_t = Z_filtered_t - Z_filtered_t.mean(dim=0) #(np,bs,Dz)
            sigma_z_filtered_t = torch.sqrt(addjust_term * (centered_Z_filtered_t*centered_Z_filtered_t).mean(dim=0)) #(bs,Dz)
            Z_filtered_t = Z_filtered_t.mean(dim=0) + (self.inflation_factor * sigma_z_t + (1-self.inflation_factor) * sigma_z_filtered_t) / sigma_z_filtered_t * centered_Z_filtered_t #(np,bs,Dz)
        return Z_filtered_t
    
    
    
    def _etkf_R_diagonal(self, x_t, Z_t, dec_mean_t, dec_std_t):
        # x_t:(bs,Dx), Z_t:(np,bs,Dz), dec_mean_t:(np,bs,Dx), dec_std_t:(np,bs,Dx)
        x_hat_t = self._reparameterized_sample_emission(dec_mean_t, dec_std_t) #(np,bs,Dx)
        Z_mean_t = Z_t.mean(dim=0) #(bs,Dz)
        centered_Z_t = (Z_t - Z_mean_t).permute(1,2,0) #(np,bs,Dx)->(bs,Dz,np)
        x_hat_mean_t = x_hat_t.mean(dim=0) #(bs,Dx)
        centered_x_t = x_hat_t - x_hat_mean_t #(np,bs,Dx)
        centered_dec_mean_t = dec_mean_t - dec_mean_t.mean(dim=0) #Yb;(np,bs,Dx)
        C_t = (centered_dec_mean_t / dec_std_t.pow(2)).permute(1,0,2) #C;(bs,np,Dx)
        if self.inflation_method=="multiplicative":
            Pa_t = torch.inverse((self.n_particles - 1)*torch.eye(self.n_particles).to(self.device)/self.inflation_factor + C_t @ centered_dec_mean_t.permute(1,2,0)) #Pa;(bs,np,np)
        else:
            Pa_t = torch.inverse((self.n_particles - 1)*torch.eye(self.n_particles).to(self.device) + C_t @ centered_dec_mean_t.permute(1,2,0)) #Pa;(bs,np,np)
        Wa_t = torch.cholesky((self.n_particles - 1) * Pa_t) #Wa;(bs,np,np)
        wa_t = ((Pa_t @ C_t) @ (x_t - x_hat_mean_t).unsqueeze(2)).repeat(1,1,self.n_particles) #wa;(bs,np,Dx)@(bs,Dx,1)->(bs,np,1)->(bs,np,np)
        Z_filtered_t = Z_mean_t.unsqueeze(2).repeat(1,1,self.n_particles) + centered_Z_t @ (Wa_t + wa_t) #(bs,Dz,np)
        Z_filtered_t = Z_filtered_t.permute(2,0,1) #(np,bs,Dz)
        return Z_filtered_t
    