import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from .base import BaseModel


class SRNN(BaseModel):
    def __init__(self, x_dim, config, device):
        super(SRNN, self).__init__(x_dim, config, device)
        self.residual_on = config["network"]["residual_on"]
        
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(2*self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(2*self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(2*self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(self.h_dim, x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(self.h_dim, x_dim)
        
        #recurrence
        self.rnn_fw = nn.GRU(self.h_dim, self.h_dim, self.n_layers, self.bias)
        self.rnn_bw = nn.GRU(2*self.h_dim, self.h_dim, self.n_layers, self.bias)
        
    
    def forward(self, x, inputs):
        (log_gs, log_fs, log_qs), (Zs, Xs, hs) = self.SMC(x, inputs)
        loss_set = self._get_elbo(log_gs, log_fs, log_qs)
        return loss_set, (Zs, Xs, hs)
    
    
    def SMC(self, x, inputs):
        Zs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) #(T,np,bs,Dz)
        Xs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.x_dim), requires_grad=False).to(self.device) #(T,np,bs,Dz)
        log_fs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        log_gs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        log_qs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        phi_z_filtered_t = Variable(torch.zeros(self.n_particles, x.size(1), self.h_dim)).to(self.device) #(np,bs,Dh)
        
        phi_u = self.phi_x(inputs) # (T,bs,Dh)
        rnn_out, _ = self.rnn_fw(phi_u) #(T,bs,Dh)
        phi_x = self.phi_x(x) # (T,bs,Dh)
        rnn_bw_out, _ = self.rnn_bw(torch.flip(torch.cat([phi_x, rnn_out], 2), [0])) #(T,bs,Dh)
        rnn_bw_out = torch.flip(rnn_bw_out.view(x.size(0), x.size(1), self.h_dim), [0]) # (T,bs,Dh)
        
        # expansion for particles
        rnn_out = rnn_out.unsqueeze(1).repeat(1,self.n_particles,1,1) # (T,np,bs,Dh)
        rnn_bw_out = rnn_bw_out.unsqueeze(1).repeat(1,self.n_particles,1,1) # (T,np,bs,Dh)
        
        for t in range(x.size(0)):
            #encoder
            enc_t = self.enc(torch.cat([phi_z_filtered_t, rnn_bw_out[t]], 2)) #(np,bs,Dh)
            enc_mean_t = self.enc_mean(enc_t) #(np,bs,Dz)
            enc_std_t = self.enc_std(enc_t) #(np,bs,Dz)

            #prior
            prior_t = self.prior(torch.cat([phi_z_filtered_t, rnn_out[t]], 2)) #(np,bs,Dh)
            prior_mean_t = self.prior_mean(prior_t) #(np,bs,Dz)
            prior_std_t = self.prior_std(prior_t) #(np,bs,Dz)

            #sampling and reparameterization
            if self.residual_on:
                enc_mean_t = enc_mean_t + prior_mean_t
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t) #(np,bs,Dz)
            phi_z_t = self.phi_z(z_t) #(np,bs,Dh)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, rnn_out[t]], 2)) #(np,bs,Dh)
            dec_mean_t = self.dec_mean(dec_t) #(np,bs,Dx)
            dec_std_t = self.dec_std(dec_t) #(np,bs,Dx)
            
            #computing losses
            log_g_t = self._log_prob_emission(dec_mean_t, # (np,bs,Dx)
                                       dec_std_t, # (Dx)
                                       x[t].clone(), # (bs,Dx)
                                       dim=2
                                      ) # (np,bs)
            log_q_t = self._log_prob(enc_mean_t, enc_std_t, z_t, dim=2) #(np,bs)
            log_f_t = self._log_prob(prior_mean_t, prior_std_t, z_t, dim=2) #(np,bs)
            log_W_t = log_g_t + log_f_t - log_q_t - math.log(self.n_particles)
            
            #update
            if self.system_name=="EnKO":
                z_filtered_t = self.system.update(x[t].clone(), z_t, dec_mean_t, dec_std_t) #(np,bs,Dz)
                phi_z_filtered_t = self.phi_z(z_filtered_t) #(np,bs,Dh)
                dec_filtered_mean_t = self.dec_mean(self.dec(torch.cat([phi_z_filtered_t, rnn_out[t]], 2)) ) # (np,bs,Dx)
                #dec_filtered_std_t = self.dec_std(dec_filtered_t) #(np,bs,Dx)
            elif self.system_name=="FIVO":
                z_filtered_t, phi_z_filtered_t, dec_filtered_mean_t = self.system._resample_particles([z_t, phi_z_t, dec_mean_t], # (np,bs,Dz),(np,bs,Dx)
                                                                        log_W_t, # (np,bs)
                                                                        sample_size=self.n_particles)
            elif self.system_name=="IWAE":
                z_filtered_t, phi_z_filtered_t, dec_filtered_mean_t = z_t, phi_z_t, dec_mean_t
            
            # write
            log_gs[t] = log_g_t
            log_fs[t] = log_f_t
            log_qs[t] = log_q_t
            Zs[t] = z_filtered_t
            Xs[t] = dec_filtered_mean_t
            
        return (log_gs, log_fs, log_qs), (Zs, Xs, rnn_out)
    
    
    def calculate_mse(self, x, z_t, h_t, pred_steps=5):
        # x:(T,bs,Dx), z_t:(T,np,bs,Dz), h_t:(T,np,bs,Dh)
        T = x.size(0)
        batch_size = x.size(1)
        MSE = Variable(torch.zeros(pred_steps, batch_size, x.size(2)), requires_grad=False).to(self.device) #(ps,bs,Dx)
        TV = Variable(torch.zeros(pred_steps, batch_size, x.size(2)), requires_grad=False).to(self.device) #(ps,bs,Dx)
        h_t = h_t.unsqueeze(0) #(1,T,np,bs,Dh)
        phi_z_t = self.phi_z(z_t) #(T,np,bs,Dh)
        x_hat_t = x.unsqueeze(1).repeat(1,self.n_particles,1,1) #(T,np,bs,Dx)
        
        for t in range(pred_steps):
            #prior
            z_t = self.prior_mean(self.prior(torch.cat([h_t[-1], phi_z_t], 3))) # (T,np,bs,Dz)
            
            #for next step
            phi_u_t = self.phi_x(x_hat_t) #(T,np,bs,Dh)
            
            #recurrence
            _, h_t = self.rnn_fw(phi_u_t.reshape(T*batch_size*self.n_particles, self.h_dim).unsqueeze(0), 
                            h_t.reshape(self.n_layers, T*batch_size*self.n_particles, self.h_dim))
            h_t = h_t.reshape(self.n_layers, T, self.n_particles, batch_size, self.h_dim)
            
            #decoder
            phi_z_t = self.phi_z(z_t) #(T,np,ns,Dh)
            x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 3))) #(T,np,bs,Dx)
            
            #calcurate
            MSE[t] = ((x[t+1:] - x_hat_t[:-(t+1)].mean(axis=1))**2).mean(axis=0) #(bs,Dx)
            TV[t] = ((x[t+1:] - x[t+1:].mean(axis=0))**2).mean(axis=0) #(bs,Dx)
        
        return MSE, TV
    
    
    
    def calculate_r_squared(self, x, z_t, h_t, pred_steps=5):
        # x:(T,bs,Dx), z_t:(T,np,bs,Dz), h_t:(T,np,bs,Dh)
        T = x.size(0)
        batch_size = x.size(1)
        MSE = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        R2 = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        h_t = h_t.unsqueeze(0) #(1,T,np,bs,Dh)
        phi_z_t = self.phi_z(z_t) #(T,np,bs,Dh)
        x_hat_t = x.unsqueeze(1).repeat(1,self.n_particles,1,1) #(T,np,bs,Dx)
        
        for t in range(pred_steps):
            #prior
            z_t = self.prior_mean(self.prior(torch.cat([h_t[-1], phi_z_t], 3))) # (T,np,bs,Dz)
            
            #for next step
            phi_u_t = self.phi_x(x_hat_t) #(T,np,bs,Dh)
            
            #recurrence
            _, h_t = self.rnn_fw(phi_u_t.reshape(T*batch_size*self.n_particles, self.h_dim).unsqueeze(0), 
                            h_t.reshape(self.n_layers, T*batch_size*self.n_particles, self.h_dim))
            h_t = h_t.reshape(self.n_layers, T, self.n_particles, batch_size, self.h_dim)
            
            #decoder
            phi_z_t = self.phi_z(z_t) #(T,np,ns,Dh)
            x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 3))) #(T,np,bs,Dx)
            
            #calcurate
            MSE[t] = ((x[t+1:] - x_hat_t[:-(t+1)].mean(axis=1))**2).sum(axis=2).mean(axis=0) #(bs)
            deno_t = ((x[t+1:] - x[t+1:].mean(axis=0))**2).sum(axis=2).mean(axis=0) #(bs)
            R2[t] = 1 - MSE[t] / deno_t #(bs)
        
        return R2, MSE
    
    
    
    def prediction(self, z_t, h_t, pred_steps=5):
        # z_t:(np,bs,Dz), h_t:(np,bs,Dh)
        batch_size = z_t.size(1)
        Zs = Variable(torch.ones(pred_steps, self.n_particles, batch_size, self.z_dim), requires_grad=False).to(self.device) #(ps,np,bs,Dz)
        Xs = Variable(torch.ones(pred_steps, self.n_particles, batch_size, self.x_dim), requires_grad=False).to(self.device) #(ps,np,bs,Dx)
        h_t = h_t.unsqueeze(0) #(1,np,bs,Dh)
        phi_z_t = self.phi_z(z_t) #(T,np,bs,Dh)
        
        for t in range(pred_steps):
            #prior
            z_t = self.prior_mean(self.prior(torch.cat([h_t[-1], phi_z_t],2))) # (np,bs,Dz)
            
            #decoder
            phi_z_t = self.phi_z(z_t) #(np,ns,Dh)
            x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 2))) #(np,bs,Dx)
            
            #for next step
            phi_u_t = self.phi_x(x_hat_t) #(np,bs,Dh)
            
            #recurrence
            _, h_t = self.rnn_fw(phi_u_t.reshape(batch_size*self.n_particles, self.h_dim).unsqueeze(0), 
                            h_t.reshape(self.n_layers, batch_size*self.n_particles, self.h_dim))
            h_t = h_t.reshape(self.n_layers, self.n_particles, batch_size, self.h_dim)
            
            #write
            Zs[t] = z_t
            Xs[t] = x_hat_t
        
        return (Xs, Zs)
    
    
    def get_next_Z(self, z_t, h_t=None):
        # z_t:(np,bs,Dz), h_t:(np,bs,Dh)
        if h_t is None:
            h_t = torch.zeros(z_t.size(0), z_t.size(1), self.h_dim).to(self.device) #(np,bs,Dh)
        phi_z_t = self.phi_z(z_t) #(np,bs,Dh)
        
        #prior
        z_t = self.prior_mean(self.prior(torch.cat([h_t, phi_z_t],2))) # (np,bs,Dz)
        return z_t
