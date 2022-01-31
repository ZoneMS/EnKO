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

from .base import BaseModel


class VRNN(BaseModel):
    def __init__(self, x_dim, config, device):
        super(VRNN, self).__init__(x_dim, config, device)
        self.init_inference = config["network"]["init_inference"]
        
        if self.dropout_ratio==0 and True:
            self.phi_x = nn.Sequential(
                nn.Linear(self.x_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
            self.phi_z = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())

            self.enc = nn.Sequential(
                nn.Linear(self.h_dim + self.r_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())

            self.prior = nn.Sequential(
                nn.Linear(self.r_dim, self.h_dim),
                nn.ReLU())

            self.dec = nn.Sequential(
                nn.Linear(self.h_dim + self.r_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
        else:
            self.phi_x = nn.Sequential(
                nn.Linear(self.x_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())
            self.phi_z = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())

            self.enc = nn.Sequential(
                nn.Linear(self.h_dim + self.r_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())

            self.prior = nn.Sequential(
                nn.Linear(self.r_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())

            self.dec = nn.Sequential(
                nn.Linear(self.h_dim + self.r_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())
        
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())
        
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        self.dec_std = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(self.h_dim, self.x_dim)
        
        #recurrence
        if self.rnn_mode in ["RNN", "GRU", "LSTM"]:
            self.rnn = self.RNN(self.h_dim + self.h_dim, self.r_dim, self.n_layers, self.bias, dropout=self.dropout_ratio)
        elif self.rnn_mode in ["ODE-RNN", "ODE-GRU", "ODE-LSTM"]:
            _, mode = self.rnn_mode.split("-")
            self.rnn = self.RNN(mode, device, self.h_dim + self.h_dim, self.r_dim, int(config["ode"]["n_ode_units"]), self.n_layers, int(config["ode"]["n_ode_layers"]), config["ode"]["ode_activation_fn"], float(config["ode"]["ode_dt"]), self.bias, dropout=self.dropout_ratio)
        
        if self.init_inference:
            self.rnn_h0_fw = nn.GRU(self.x_dim, self.h_dim, self.n_layers, dropout=self.dropout_ratio)
            self.rnn_h0_bw = nn.GRU(self.x_dim, self.h_dim, self.n_layers, dropout=self.dropout_ratio)
            self.init_h0 = nn.Linear(2*self.h_dim, self.r_dim)
        
    
    def forward(self, x, saturated_on=False, t=None):
        (log_gs, log_fs, log_qs), (x_hat, x_prevs, Z_fils, Z_prevs, hs) = self.SMC(x, t)
        loss_set = self._get_elbo(log_gs, log_fs, log_qs, Z_prevs, None)
        return loss_set, (Z_prevs, Z_fils, x_hat, hs)
    
    
    def SMC(self, x, time=None):
        inputs = x.unsqueeze(1).repeat(1,self.n_particles,1,1) #(T,np,bs,Dx)
        Zs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) #(T,np,bs,Dz)
        Xs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.x_dim), requires_grad=False).to(self.device) #(T,np,bs,Dx)
        Z_prevs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) #(T,np,bs,Dz)
        X_prevs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.x_dim), requires_grad=False).to(self.device) #(T,np,bs,Dx)
        #klds = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        log_gs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        log_fs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        log_qs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        hs = Variable(torch.zeros( self.n_layers, x.size(0), self.n_particles, x.size(1), self.r_dim)).to(self.device) #(nl,T,np,bs,Dr)
        
        if self.init_inference:
            if self.enc_steps is None:
                enc_steps = x.size(0)
            else:
                enc_steps = self.enc_steps
            _, h0_fw = self.rnn_h0_fw(x[:enc_steps]) #(1,bs,Dh)
            _, h0_bw = self.rnn_h0_bw(torch.flip(x[:enc_steps], [0])) #(1,bs,Dh)
            h0 = torch.cat([h0_fw[0], h0_bw[0]],dim=1) #(bs,2*Dh)
            h = self.init_h0(h0).unsqueeze(0).repeat(self.n_particles,1,1).unsqueeze(0) #(1,np,bs,Dr)
        else:
            h = Variable(torch.zeros(self.n_layers, self.n_particles, x.size(1), self.r_dim)).to(self.device) #(nl,np,bs,Dr)
        hp = h.clone()
        
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(inputs[t]) #(np,bs,Dh)

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 2)) #(np,bs,Dh)
            enc_mean_t = self.enc_mean(enc_t) #(np,bs,Dz)
            enc_std_t = self.enc_std(enc_t) #(np,bs,Dz)

            #prior
            if self.system_name=="EnKO" and self.exclude_filtering_on:
                prior_t = self.prior(hp[-1]) #(np,bs,Dh)
            else:
                prior_t = self.prior(h[-1]) #(np,bs,Dh)
            prior_mean_t = self.prior_mean(prior_t) #(np,bs,Dz)
            prior_std_t = self.prior_std(prior_t) #(np,bs,Dz)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t) #(np,bs,Dz)
            phi_z_t = self.phi_z(z_t) #(np,bs,Dh)
            log_q_t = self._log_prob(enc_mean_t, enc_std_t, z_t, dim=2) #(np,bs)
            log_f_t = self._log_prob(prior_mean_t, prior_std_t, z_t, dim=2) #(np,bs)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 2)) #(np,bs,Dh)
            dec_mean_t = self.dec_mean(dec_t) #(np,bs,Dx)
            dec_std_t = self.dec_std(dec_t) #(np,bs,Dx)
            x_t = self._reparameterized_sample(dec_mean_t, dec_std_t) #(np,bs,Dx)
            
            # log likelihood
            log_g_t = self._log_prob_emission(dec_mean_t, # (np,bs,Dx)
                                       dec_std_t, # (np,bs,Dx)
                                       x[t].clone(), # (bs,Dx)
                                       dim=2
                                      ) # (np,bs)
            log_W_t = log_g_t + log_f_t - log_q_t - math.log(self.n_particles)
            
            #update
            if self.system_name=="EnKO":
                z_filtered_t = self.system.update(x[t].clone(), z_t, dec_mean_t, dec_std_t) #(np,bs,Dz)
                phi_z_filtered_t = self.phi_z(z_t) #(np,bs,Dh)
                dec_filtered_t = self.dec(torch.cat([phi_z_filtered_t, h[-1]], 2)) #(np,bs,Dh)
                dec_filtered_mean_t = self.dec_mean(dec_filtered_t) #(np,bs,Dx)
                #dec_filtered_std_t = self.dec_std(dec_filtered_t) #(np,bs,Dx)
            elif self.system_name=="FIVO":
                z_filtered_t, phi_z_filtered_t, dec_filtered_mean_t = self.system._resample_particles([z_t, phi_z_t, dec_mean_t], # (np,bs,Dz),(np,bs,Dx)
                                                                        log_W_t, # (np,bs)
                                                                        sample_size=self.n_particles)
            elif self.system_name=="IWAE":
                z_filtered_t, phi_z_filtered_t, dec_filtered_mean_t = z_t, phi_z_t, dec_mean_t
            #kld_t = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t, dim=2) #(np,bs)

            #recurrence
            if time is None:
                if self.system_name=="EnKO" and self.exclude_filtering_on:
                    _, hp = self.rnn(torch.cat([phi_x_t, phi_z_t], 2).reshape(x.size(1)*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                                h.reshape(self.n_layers, x.size(1)*self.n_particles, self.r_dim)) #(nl,bs*np,Dr)
                    hp = hp.reshape(self.n_layers, self.n_particles, x.size(1), self.r_dim) #(nl,np,bs,Dr)
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_filtered_t], 2).reshape(x.size(1)*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                            h.reshape(self.n_layers, x.size(1)*self.n_particles, self.r_dim)) #(nl,bs*np,Dr)
            else:
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_filtered_t], 2).reshape(x.size(1)*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                            h.reshape(self.n_layers, x.size(1)*self.n_particles, self.r_dim),
                            time[t+1]-time[t]) #(nl,bs*np,Dr)
            h = h.reshape(self.n_layers, self.n_particles, x.size(1), self.r_dim) #(nl,np,bs,Dr)
            
            # write
            log_gs[t] = log_g_t
            #klds[t] = kld_t
            log_fs[t] = log_f_t
            log_qs[t] = log_q_t
            Zs[t] = z_filtered_t
            Xs[t] = dec_filtered_mean_t
            Z_prevs[t] = z_t
            X_prevs[t] = dec_mean_t
            hs[:,t] = hp if self.system_name=="EnKO" and self.exclude_filtering_on else h
            
        return (log_gs, log_fs, log_qs), (Xs, X_prevs, Zs, Z_prevs, hs)
    
    
    def calculate_mse(self, x, h_t, pred_steps=5, time=None):
        # x:(T,bs,Dx), h_t:(nl,T,np,bs,Dh)
        T = x.size(0)
        batch_size = x.size(1)
        MSE = np.zeros([pred_steps, batch_size, x.size(2)])
        TV = np.zeros([pred_steps, batch_size, x.size(2)])
#         MSE = Variable(torch.zeros(pred_steps, batch_size, x.size(2)), requires_grad=False).to(self.device) #(ps,bs,Dx)
#         TV = Variable(torch.zeros(pred_steps, batch_size, x.size(2)), requires_grad=False).to(self.device) #(ps,bs,Dx)
        if time is not None:
            dt = time[1:] - time[:-1] #(T,bs)
        
        for t in range(pred_steps):
            #prior
            z_t = self.prior_mean(self.prior(h_t[-1])) # (T,np,bs,Dz)
            
            #decoder
            phi_z_t = self.phi_z(z_t) #(T,np,ns,Dh)
            x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 3))) #(T,np,bs,Dx)
            
            #for next step
            phi_x_t = self.phi_x(x_hat_t) #(T,np,bs,Dh)
            
            #recurrence
            if time is None:
                _, h_t = self.rnn(torch.cat([phi_x_t, phi_z_t], 3).reshape(T*batch_size*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                            h_t.reshape(self.n_layers, T*batch_size*self.n_particles, self.r_dim))
            else:
                dt = torch.roll(dt,-1,0)
                _, h_t = self.rnn(torch.cat([phi_x_t, phi_z_t], 3).reshape(T*batch_size*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                            h_t.reshape(self.n_layers, T*batch_size*self.n_particles, self.r_dim),
                            dt)
            h_t = h_t.reshape(self.n_layers, T, self.n_particles, batch_size, self.r_dim)
            
            #calcurate
            MSE[t] = ((x[t+1:] - x_hat_t[:-(t+1)].mean(axis=1))**2).mean(axis=0).data.cpu().numpy() #(bs,Dx)
            TV[t] = ((x[t+1:] - x[t+1:].mean(axis=0))**2).mean(axis=0).data.cpu().numpy() #(bs,Dx)
        
        return MSE, TV
    
    
    def init_running(self, x):
        # x:(T,bs,Dx)
        Zs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) #(T,np,bs,Dz)
        Xs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1), self.x_dim), requires_grad=False).to(self.device) #(T,np,bs,Dx)
        
        if self.init_inference:
            if self.enc_steps is None:
                enc_steps = x.size(0)
            else:
                enc_steps = self.enc_steps
            _, h0_fw = self.rnn_h0_fw(x[:enc_steps]) #(1,bs,Dh)
            _, h0_bw = self.rnn_h0_bw(torch.flip(x[:enc_steps], [0])) #(1,bs,Dh)
            h0 = torch.cat([h0_fw[0], h0_bw[0]],dim=1) #(bs,2*Dh)
            h = self.init_h0(h0).unsqueeze(0).repeat(self.n_particles,1,1).unsqueeze(0) #(1,np,bs,Dr)
        else:
            h = Variable(torch.zeros(self.n_layers, self.n_particles, x.size(1), self.r_dim)).to(self.device) #(nl,np,bs,Dr)
        
        phi_x_t = self.phi_x(x[0].clone()).repeat(self.n_particles,1,1) #(np,bs,Dh)
        for t in range(x.size(0)):
            #prior
            prior_t = self.prior(h[-1]) #(np,bs,Dh)
            prior_mean_t = self.prior_mean(prior_t) #(np,bs,Dz)
            prior_std_t = self.prior_std(prior_t) #(np,bs,Dz)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t) #(np,bs,Dz)
            phi_z_t = self.phi_z(z_t) #(np,bs,Dh)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 2)) #(np,bs,Dh)
            dec_mean_t = self.dec_mean(dec_t) #(np,bs,Dx)
            
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 2).reshape(x.size(1)*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                            h.reshape(self.n_layers, x.size(1)*self.n_particles, self.r_dim)) #(nl,bs*np,Dr)
            h = h.reshape(self.n_layers, self.n_particles, x.size(1), self.r_dim) #(nl,np,bs,Dr)
            phi_x_t = self.phi_x(dec_mean_t) #(np,bs,Dh)
            
            Zs[t] = z_t
            Xs[t] = dec_mean_t
        
        return Zs, Xs
 
    
    
    def calculate_r_squared(self, x, h_t, pred_steps=5):
        # x:(T,bs,Dx), h_t:(nl,T,np,bs,Dh)
        T = x.size(0)
        batch_size = x.size(1)
        MSE = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        R2 = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        
        for t in range(pred_steps):
            #prior
            z_t = self.prior_mean(self.prior(h_t[-1])) # (T,np,bs,Dz)
            
            #decoder
            phi_z_t = self.phi_z(z_t) #(T,np,ns,Dh)
            x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 3))) #(T,np,bs,Dx)
            
            #for next step
            phi_x_t = self.phi_x(x_hat_t) #(T,np,bs,Dh)
            
            #recurrence
            _, h_t = self.rnn(torch.cat([phi_x_t, phi_z_t], 3).reshape(T*batch_size*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                            h_t.reshape(self.n_layers, T*batch_size*self.n_particles, self.h_dim))
            h_t = h_t.reshape(self.n_layers, T, self.n_particles, batch_size, self.h_dim)
            
            #calcurate
            MSE[t] = ((x[t+1:] - x_hat_t[:-(t+1)].mean(axis=1))**2).sum(axis=2).mean(axis=0) #(bs)
            deno_t = ((x[t+1:] - x[t+1:].mean(axis=0))**2).sum(axis=2).mean(axis=0) #(bs)
            R2[t] = 1 - MSE[t] / deno_t #(bs)
        
        return R2, MSE
    
    
    def get_next_Z(self, z_t, h_t=None):
        if h_t is None:
            h_t = Variable(torch.zeros(self.n_layers, z_t.size(0), z_t.size(1), self.r_dim)).to(self.device) #(nl,bs,bs,Dr)
        
        #decoder
        phi_z_t = self.phi_z(z_t) #(bs,bs,Dh)
        x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 2))) #(bs,bs,Dx)

        #for next step
        phi_x_t = self.phi_x(x_hat_t) #(bs,bs,Dh)

        #recurrence
        _, h_t = self.rnn(torch.cat([phi_x_t, phi_z_t], 2).reshape(z_t.size(0)*z_t.size(1), 2*self.h_dim).unsqueeze(0))
        h_t = h_t.reshape(self.n_layers, z_t.size(0), z_t.size(1), self.h_dim)
        return self.prior_mean(self.prior(h_t[-1])) # (bs,bs,Dz)
    
    
    def get_X_from_Z(self, z_t, h_t=None):
        if h_t is None:
            h_t = Variable(torch.zeros(self.n_layers, z_t.size(0), z_t.size(1), self.r_dim)).to(self.device) #(nl,bs,bs,Dr)
        
        #decoder
        phi_z_t = self.phi_z(z_t) #(bs,bs,Dh)
        x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 2))) #(bs,bs,Dx)
        return x_hat_t
    
    
    def prediction(self, h_t, pred_steps=5):
        # h_t:(nl,np,bs,Dh)
        batch_size = h_t.size(2)
        Zs = Variable(torch.ones(pred_steps, self.n_particles, batch_size, self.z_dim), requires_grad=False).to(self.device) #(ps,np,bs,Dz)
        Xs = Variable(torch.ones(pred_steps, self.n_particles, batch_size, self.x_dim), requires_grad=False).to(self.device) #(ps,np,bs,Dx)
        
        for t in range(pred_steps):
            #prior
            z_t = self.prior_mean(self.prior(h_t[-1])) # (np,bs,Dz)
            
            #decoder
            phi_z_t = self.phi_z(z_t) #(np,ns,Dh)
            x_hat_t = self.dec_mean(self.dec(torch.cat([phi_z_t, h_t[-1]], 2))) #(np,bs,Dx)
            
            #for next step
            phi_x_t = self.phi_x(x_hat_t) #(np,bs,Dh)
            
            #recurrence
            _, h_t = self.rnn(torch.cat([phi_x_t, phi_z_t], 2).reshape(batch_size*self.n_particles, 2*self.h_dim).unsqueeze(0), 
                            h_t.reshape(self.n_layers, batch_size*self.n_particles, self.r_dim))
            h_t = h_t.reshape(self.n_layers, self.n_particles, batch_size, self.r_dim)
            
            #write
            Zs[t] = z_t
            Xs[t] = x_hat_t
        
        return (Xs, Zs)
        
        
