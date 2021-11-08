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

from .base import BaseModel
from .utils import get_gpu_info


class SVO(BaseModel):
    def __init__(self, x_dim, config, device):
        super(SVO, self).__init__(x_dim, config, device)
        
        self.sigma_init = config["network"]["sigma_init"]
        self.sigma_min = config["network"]["sigma_min"]
        self.sigma_train = config["network"]["sigma_train"]
        
        ## network of vae
        if self.dropout_ratio==0 and False: #-201219,200304-09
            self.q0_tran = nn.Sequential(
                nn.Linear(2*self.h_dim, self.h_dim),
                nn.ReLU())

            self.f_tran = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())

            self.q_tran = nn.Sequential(
                nn.Linear(2*self.h_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())

            self.g_tran = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
        else:
            self.q0_tran = nn.Sequential(
                nn.Linear(2*self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())

            self.f_tran = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())

            self.q_tran = nn.Sequential(
                nn.Linear(2*self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())

            self.g_tran = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.Dropout(self.dropout_ratio),
                nn.ReLU())
        
        self.q0_mu = nn.Linear(self.h_dim, self.z_dim)
#         self.q0_mu = nn.Sequential(
#             nn.Linear(self.h_dim, self.z_dim),
#             nn.Dropout(self.dropout_ratio)
#         ) #201224-210109
        self.f_mu = nn.Linear(self.h_dim, self.z_dim)
        self.q_mu = nn.Linear(self.h_dim, self.z_dim)
        self.g_mu = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            self.output_fn()
        )
#         self.g_mu = nn.Linear(self.h_dim, self.x_dim)
        
        self.q0_sigma = nn.Parameter(self.sigma_init * torch.ones(self.z_dim), requires_grad=self.sigma_train)
        self.f_sigma = nn.Parameter(self.sigma_init * torch.ones(self.z_dim), requires_grad=self.sigma_train)
        self.q_sigma = nn.Parameter(self.sigma_init * torch.ones(self.z_dim), requires_grad=self.sigma_train)
        self.g_sigma = nn.Parameter(self.sigma_init * torch.ones(self.x_dim), requires_grad=self.sigma_train)
        
        # birnn
        self.x_smoother = nn.LSTM(self.x_dim, self.h_dim, self.n_layers, bidirectional=True, dropout=self.dropout_ratio)
        #self.Z0_smoother = nn.LSTM(self.x_dim, self.h_dim, self.n_layers, bidirectional=True)
        self.Z0_smoother_fw = nn.LSTM(self.x_dim, self.h_dim, self.n_layers, dropout=self.dropout_ratio)
        self.Z0_smoother_bw = nn.LSTM(self.x_dim, self.h_dim, self.n_layers, dropout=self.dropout_ratio)
        
        self.smooth_obs = True
        self.dropout_elbo = False
        self.global_context_on = False
            
    
    
    def forward(self, x, saturated_on=False):
        (_, log_gs, log_fs, log_qs, klds), (x_hat, x_prevs, Z_fils, Z_prevs), (_, _, gc) = self.SMC(x)
        loss_set = self._get_elbo(log_gs, log_fs, log_qs, klds, saturated_on)
        return loss_set, (Z_fils, x_hat, gc)
#         return loss_set, (Z_fils, x_prevs, gc)
        
        
    def SMC(self, x):#(T,bs,Dx)
        x_hat = Variable(torch.zeros(x.size(0), self.n_particles, x.size(1), self.x_dim), requires_grad=False).to(self.device) # (T,np,bs,Dx)
        x_prevs = Variable(torch.zeros(x.size(0), self.n_particles, x.size(1), self.x_dim), requires_grad=False).to(self.device) # (T,np,bs,Dx)
        Zs_ta = Variable(torch.zeros(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) # (T,np,bs,Dz)
        Zs_filtered_ta = Variable(torch.zeros(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) # (T,np,bs,Dz)
        log_gs = Variable(torch.zeros(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device) # (T,np,bs)
        log_fs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        log_qs = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        log_Ws = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        if self.kld_penalty_on:
            klds = Variable(torch.ones(x.size(0), self.n_particles, x.size(1)), requires_grad=False).to(self.device).div_(self.n_particles) #(T,np,bs)
        else:
            klds = None
        
        ### initial calculation
        if self.enc_steps is None:
            enc_steps = x.size(0)
        else:
            enc_steps = self.enc_steps
            
        if self.smooth_obs:
            preprocessed_obs, _ = self.x_smoother(x[:enc_steps]) # (T,bs,Dx)->(T,bs,2*Dh)
            _, (preprocessed_Z0_fw, _) = self.Z0_smoother_fw(x[:enc_steps]) # (T,bs,Dx)->(1,bs,Dh)
            _, (preprocessed_Z0_bw, _) = self.Z0_smoother_bw(torch.flip(x[:enc_steps], [0])) # (T,bs,Dx)->(1,bs,Dh)
            preprocessed_Z0 = torch.cat([preprocessed_Z0_fw[0], preprocessed_Z0_bw[0]],dim=1) # (bs,2*Dh)
        else:
            preprocessed_obs = x #(T,bs,Dx)
            preprocessed_Z0 = x[0].clone() #(bs,Dx)
        
        if self.global_context_on:
            _, (global_h, _) = self.g_smoother(x[:enc_steps]) # (T,bs,Dx)->(2,bs,Dh)
            global_h = global_h.transpose(0,1).reshape(x.size(1), 2*self.h_dim) #(bs,2*Dh)
            gc = self.g_net(global_h) #(bs,Dg)
            if self.g_random_on:
                gc = self._reparameterized_sample(gc[:,:self.g_dim], f.softplus(gc[:,self.g_dim:])) #(bs,Dg)
        else:
            gc = None
        
        
        #---------------- t=0 ----------------#
        if self.global_context_on:
            prior_t = self.q0_mu(self.q0_tran(torch.cat([preprocessed_Z0, gc], 1))) #(bs,Dz)
            proposal_t = self.q_mu(self.q_tran(torch.cat([preprocessed_obs[0].clone(), gc], 1))) #(bs,Dz)
        else:
            prior_t = self.q0_mu(self.q0_tran(preprocessed_Z0)) #(bs,Dz)
            proposal_t = self.q_mu(self.q_tran(preprocessed_obs[0].clone())) #(bs,Dz)
        Z_t, q_t_log_prob, f_t_log_prob, kld_t = self._sample_from_2_dist(prior_t,
                                                       proposal_t,
                                                       self.q0_sigma,
                                                       self.q_sigma,
                                                       0,
                                                       self.n_particles)
        dec_mean_t, Z_filtered_t, dec_filtered_mean_t, g_t_log_prob, log_W_t = self._one_ahead_calculation(x[0].clone(), Z_t, f_t_log_prob, q_t_log_prob)

        x_hat[0] = dec_filtered_mean_t
        x_prevs[0] = dec_mean_t
        Zs_ta[0] = Z_t
        Zs_filtered_ta[0] = Z_filtered_t
        log_gs[0] = g_t_log_prob
        log_fs[0] = f_t_log_prob
        log_qs[0] = q_t_log_prob
        log_Ws[0] = log_W_t
        if self.kld_penalty_on:
            klds[0] = kld_t
        
        
        #---------------- t=1, ..., T-1 ----------------#
        for t in range(1, x.size(0)):
            if self.global_context_on:
                prior_t = self.f_mu(self.f_tran(torch.cat([Z_filtered_t, gc.repeat(self.n_particles,1,1)], 2))) #(np,bs,Dz)
                proposal_t = self.q_mu(self.q_tran(torch.cat([preprocessed_obs[t].clone(), gc], 1))) #(bs,Dz)
            else:
                prior_t = self.f_mu(self.f_tran(Z_filtered_t)) #(np,bs,Dz)
                proposal_t = self.q_mu(self.q_tran(preprocessed_obs[t].clone())) #(bs,Dz)
            Z_t, q_t_log_prob, f_t_log_prob, kld_t = self._sample_from_2_dist(prior_t,
                                                                       proposal_t,
                                                                       self.f_sigma,
                                                                       self.q_sigma,
                                                                       t,
                                                                       None)
            if self.system_name=="EnKO" and self.exclude_filtering_on:
                if self.global_context_on:
                    f_t_log_prob = self._log_prob(self.f_mu(self.f_tran(torch.cat([Zs_ta[t-1].clone(), gc.repeat(self.n_particles,1,1)], 2))), 
                                              torch.sqrt(torch.max(f.softplus(self.f_sigma), self.sigma_min*torch.ones(self.f_sigma.shape).to(self.device))), 
                                              Z_t, dim=-1) #(np,bs) 
                else:
                    f_t_log_prob = self._log_prob(self.f_mu(self.f_tran(Zs_ta[t-1].clone())), 
                                              torch.sqrt(torch.max(f.softplus(self.f_sigma), self.sigma_min*torch.ones(self.f_sigma.shape).to(self.device))), 
                                              Z_t, dim=-1) #(np,bs) 
            dec_mean_t, Z_filtered_t, dec_filtered_mean_t, g_t_log_prob, log_W_t = self._one_ahead_calculation(x[t].clone(), Z_t, f_t_log_prob, q_t_log_prob)
            
            x_hat[t] = dec_filtered_mean_t # (np,bs,Dx)
            x_prevs[t] = dec_mean_t
            Zs_ta[t] = Z_t
            Zs_filtered_ta[t] = Z_filtered_t
            log_gs[t] = g_t_log_prob
            log_fs[t] = f_t_log_prob
            log_qs[t] = q_t_log_prob
            log_Ws[t] = log_W_t
            if self.kld_penalty_on:
                klds[t] = kld_t
    
        return (log_Ws, log_gs, log_fs, log_qs, klds), (x_hat, x_prevs, Zs_filtered_ta, Zs_ta), (preprocessed_obs, preprocessed_Z0, gc)
            

        
            
    def _sample_from_2_dist(self, mu1, mu2, sigma1, sigma2, t, sample_size=None):
        """sample from 2 distribution"""
        sig_inv1 = 1 / torch.max(f.softplus(sigma1), self.sigma_min*torch.ones(sigma1.shape).to(self.device)) # (Dx)
        sig_inv2 = 1 / torch.max(f.softplus(sigma2), self.sigma_min*torch.ones(sigma2.shape).to(self.device)) # (Dx)
        
        combined_cov = 1 / (sig_inv1 + sig_inv2) # (Dx)
        combined_mean = combined_cov * (mu1 * sig_inv1 + mu2 * sig_inv2) # (bs,Dx) or (np,bs,Dx)
        
        kld_t = None
        if sample_size == None:
            X = self._reparameterized_sample(combined_mean, torch.sqrt(combined_cov).repeat(mu1.size(0),mu1.size(1),1)) # (np,bs,Dx)
            if self.kld_penalty_on:
                kld_t = self._kld_gauss(combined_mean, torch.sqrt(combined_cov), mu1, torch.sqrt(1/sig_inv1), dim=2) #(np,bs)
        else:
            #X = self._reparameterized_sample(combined_mean.unsqueeze(0).repeat(sample_size,1,1), torch.sqrt(combined_cov).repeat(sample_size,mu1.size(0),1)) #(np,bs,Dx)
            combined_mean_ex = torch.repeat_interleave(combined_mean.unsqueeze(0), sample_size, dim=0)
            X = self._reparameterized_sample(combined_mean_ex, torch.sqrt(combined_cov)*torch.ones_like(combined_mean_ex)) #(np,bs,Dx)/(M,np,bs,Dx)
            if self.kld_penalty_on:
                kld_t = self._kld_gauss(combined_mean.unsqueeze(0).repeat(sample_size,1,1), torch.sqrt(combined_cov), mu1, torch.sqrt(1/sig_inv1), dim=2) #(np,bs)
            
        if self.dropout_elbo:
            dropout_on = torch.bernoulli(self.dropout_elbo_ratio*torch.ones(mu1.shape[-2])).to(self.device) #(bs)
            if sample_size == None:
                X_prior = self._reparameterized_sample(mu1, torch.sqrt(1/sig_inv1).repeat(mu1.size(0),mu1.size(1),1)) # (np,bs,Dx)
            else:
                mu1_ex = torch.repeat_interleave(mu1.unsqueeze(0), sample_size, dim=0) #(np,bs,Dx)/(M,np,bs,Dx)
                X_prior = self._reparameterized_sample(mu1_ex, torch.sqrt(1/sig_inv1)*torch.ones_like(mu1_ex)) #(np,bs,Dx)/(M,np,bs,Dx)
            self.dropout_on[t] = dropout_on
            dropout_on = dropout_on.unsqueeze(1).repeat(1,sigma1.size(0)) #(bs,Dx)
            X = (1-dropout_on)*X + dropout_on*X_prior
        
        q_t_log_prob = self._log_prob(combined_mean, torch.sqrt(combined_cov), X, dim=-1) #(np,bs)
        f_t_log_prob = self._log_prob(mu1, torch.sqrt(1/sig_inv1), X, dim=-1) #(np,bs)
        return X, q_t_log_prob, f_t_log_prob, kld_t
    
    
    
    def _one_ahead_calculation(self, x_t, Z_t, f_t_log_prob, q_t_log_prob):
        dec_mean_t = self.g_mu(self.g_tran(Z_t)) #(np,bs,Dx)
        dec_std_t = torch.max(f.softplus(self.g_sigma), self.sigma_min*torch.ones_like(self.g_sigma).to(self.device)) #(Dx)
        dec_std_t = dec_std_t.repeat(self.n_particles,x_t.size(0),1) #(np,bs,Dx)
        
        g_t_log_prob = self._log_prob_emission(dec_mean_t, # (np,bs,Dx)
                                   dec_std_t, # (Dx)
                                   x_t, # (bs,Dx)
                                   dim=2
                                  ) # (np,bs)
        log_W_t = f_t_log_prob + g_t_log_prob - q_t_log_prob - math.log(self.n_particles) # (np,bs)

        # Z_t, Z_filtered_t: (np,bs,Dx)
        if self.system_name=="EnKO":
            Z_filtered_t = self.system.update(x_t, Z_t, dec_mean_t, dec_std_t)
            dec_filtered_mean_t = self.g_mu(self.g_tran(Z_filtered_t))
        elif self.system_name=="FIVO":
            Z_filtered_t, dec_filtered_mean_t = self.system._resample_particles([Z_t, dec_mean_t], # (np,bs,Dz),(np,bs,Dx)
                                                                    log_W_t, # (np,bs)
                                                                    sample_size=self.n_particles)
        elif self.system_name=="IWAE":
            Z_filtered_t, dec_filtered_mean_t = Z_t, dec_mean_t
        
        return dec_mean_t, Z_filtered_t, dec_filtered_mean_t, g_t_log_prob, log_W_t
    
    
    
    def calculate_mse(self, x, Z_t, pred_steps=5):
        # x:(T,bs,Dx), Z_t:(T,np,bs,Dz)
        #MSE = Variable(torch.zeros(pred_steps, x.size(1), x.size(2)), requires_grad=False).to(self.device) #(ps,bs,Dx)
        #TV = Variable(torch.zeros(pred_steps, x.size(1), x.size(2)), requires_grad=False).to(self.device) #(ps,bs,Dx)
        MSE = np.zeros([pred_steps, x.size(1), x.size(2)]) #(ps,bs,Dx)
        TV = np.zeros([pred_steps, x.size(1), x.size(2)]) #(ps,bs,Dx)
        
        for t in range(pred_steps):
            #prior
            Z_t = self.f_mu(self.f_tran(Z_t.detach())) # (T,np,bs,Dz)
            
            #decoder
            x_hat_t = self.g_mu(self.g_tran(Z_t.detach())) #(T,np,bs,Dx)
            
            #calcurate
            MSE[t] = ((x[t+1:] - x_hat_t[:-(t+1)].mean(axis=1))**2).mean(axis=0).data.cpu().numpy() #(bs,Dx)
            TV[t] = ((x[t+1:] - x[t+1:].mean(axis=0))**2).mean(axis=0).data.cpu().numpy() #(bs,Dx)
        
        return MSE, TV
    
    
    def calculate_r_squared(self, x, Z_t, pred_steps=5):
        # x:(T,bs,Dx), Z_t:(T,np,bs,Dz)
        MSE = Variable(torch.zeros(pred_steps, x.size(1)), requires_grad=False).to(self.device) #(ps,bs)
        R2 = Variable(torch.zeros(pred_steps, x.size(1)), requires_grad=False).to(self.device) #(ps,bs)
        
        for t in range(pred_steps):
            #prior
            Z_t = self.f_mu(self.f_tran(Z_t)) # (T,np,bs,Dz)
            
            #decoder
            x_hat_t = self.g_mu(self.g_tran(Z_t)) #(T,np,bs,Dx)
            
            #calcurate
            MSE[t] = ((x[t+1:] - x_hat_t[:-(t+1)].mean(axis=1))**2).sum(axis=2).mean(axis=0) #(bs)
            deno_t = ((x[t+1:] - x[t+1:].mean(axis=0))**2).sum(axis=2).mean(axis=0) #(bs)
            R2[t] = 1 - MSE[t] / deno_t #(bs)
        
        return R2, MSE
    
    
    
    def prediction(self, Z_t, pred_steps=5):
        # Z_t:(T,np,bs,Dz)/(np,bs,Dz)
        if len(Z_t.size())==3:
            batch_size = Z_t.size(1)
            Zs = Variable(torch.ones(pred_steps, self.n_particles, batch_size, self.z_dim), requires_grad=False).to(self.device) #(ps,np,bs,Dz)
            Xs = Variable(torch.ones(pred_steps, self.n_particles, batch_size, self.x_dim), requires_grad=False).to(self.device) #(ps,np,bs,Dx)
        elif len(Z_t.size())==4:
            batch_size = Z_t.size(2)
            Zs = Variable(torch.ones(pred_steps, Z_t.size(0), self.n_particles, batch_size, self.z_dim), requires_grad=False).to(self.device) #(ps,T,np,bs,Dz)
            Xs = Variable(torch.ones(pred_steps, Z_t.size(0), self.n_particles, batch_size, self.x_dim), requires_grad=False).to(self.device) #(ps,T,np,bs,Dx)
        
        for t in range(pred_steps):
            #prior
            Z_t = self.f_mu(self.f_tran(Z_t)) #(np,bs,Dz)
            
            #decoder
            x_hat_t = self.g_mu(self.g_tran(Z_t)) #(np,bs,Dx)
            
            #write
            Zs[t] = Z_t
            Xs[t] = x_hat_t
        
        return (Xs, Zs)
    
    
    def init_running(self, x):
        #(T,bs,Dx)
        Zs = Variable(torch.zeros(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) # (T,np,bs,Dz)
        
        ### initial calculation
        if self.enc_steps is None:
            enc_steps = x.size(0)
        else:
            enc_steps = self.enc_steps
            
        if self.smooth_obs:
            _, (preprocessed_Z0_fw, _) = self.Z0_smoother_fw(x[:enc_steps]) # (T,bs,Dx)->(1,bs,Dh)
            _, (preprocessed_Z0_bw, _) = self.Z0_smoother_bw(torch.flip(x[:enc_steps], [0])) # (T,bs,Dx)->(1,bs,Dh)
            preprocessed_Z0 = torch.cat([preprocessed_Z0_fw[0], preprocessed_Z0_bw[0]],dim=1) # (bs,2*Dh)
        else:
            preprocessed_Z0 = x[0].clone() #(bs,Dx)
        if self.global_context_on:
            _, (global_h, _) = self.g_smoother(x[:enc_steps]) # (T,bs,Dx)->(2,bs,Dh)
            global_h = global_h.transpose(0,1).reshape(x.size(1), 2*self.h_dim) #(bs,2*Dh)
            gc = self.g_net(global_h) #(bs,Dg)
            if self.g_random_on:
                gc = gc[:,:self.g_dim] #(bs,Dg)
        
        if self.global_context_on:
            prior_t = self.q0_mu(self.q0_tran(torch.cat([preprocessed_Z0, gc], 1))) #(bs,Dz)
        else:
            prior_t = self.q0_mu(self.q0_tran(preprocessed_Z0)) #(bs,Dz)
        prior_std = torch.sqrt(torch.max(f.softplus(self.q0_sigma), self.sigma_min*torch.ones(self.q0_sigma.shape).to(self.device))) #(Dz)
        Z_t = self._reparameterized_sample(prior_t.unsqueeze(0).repeat(self.n_particles,1,1), prior_std.repeat(self.n_particles,x.size(1),1)) #(np,bs,Dz)
        prior_std = torch.sqrt(torch.max(f.softplus(self.f_sigma), self.sigma_min*torch.ones(self.f_sigma.shape).to(self.device))) #(Dz)
        Zs[0] = Z_t
        
        for t in range(1, x.size(0)):
            if self.global_context_on:
                prior_t = self.f_mu(self.f_tran(torch.cat([Z_t, gc.repeat(self.n_particles,1,1)], 2))) #(np,bs,Dz)
            else:
                prior_t = self.f_mu(self.f_tran(Z_t)) #(np,bs,Dz)
            Z_t = self._reparameterized_sample(prior_t, prior_std.repeat(self.n_particles,x.size(1),1)) #(np,bs,Dz)
            Zs[t] = Z_t
        
        Xs = self.g_mu(self.g_tran(Zs)) #(T,np,bs,Dx)
        return Zs, Xs
    
    
    def running_only_q(self, x):
        Zs = Variable(torch.zeros(x.size(0), self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) # (T,np,bs,Dz)
        
        ### initial calculation
        if self.enc_steps is None:
            enc_steps = x.size(0)
        else:
            enc_steps = self.enc_steps
            
        if self.smooth_obs:
            preprocessed_obs, _ = self.x_smoother(x[:enc_steps]) # (T,bs,Dx)->(T,bs,2*Dh)
            _, (preprocessed_Z0_fw, _) = self.Z0_smoother_fw(x[:enc_steps]) # (T,bs,Dx)->(1,bs,Dh)
            _, (preprocessed_Z0_bw, _) = self.Z0_smoother_bw(torch.flip(x[:enc_steps], [0])) # (T,bs,Dx)->(1,bs,Dh)
            preprocessed_Z0 = torch.cat([preprocessed_Z0_fw[0], preprocessed_Z0_bw[0]],dim=1) # (bs,2*Dh)
        else:
            preprocessed_Z0 = x[0].clone() #(bs,Dx)
        if self.global_context_on:
            _, (global_h, _) = self.g_smoother(x[:enc_steps]) # (T,bs,Dx)->(2,bs,Dh)
            global_h = global_h.transpose(0,1).reshape(x.size(1), 2*self.h_dim) #(bs,2*Dh)
            gc = self.g_net(global_h) #(bs,Dg)
            if self.g_random_on:
                gc = gc[:,:self.g_dim] #(bs,Dg)
        
        if self.global_context_on:
            #prior_t = self.q0_mu(self.q0_tran(torch.cat([preprocessed_Z0, gc], 1))) #(bs,Dz)
            proposal_t = self.q_mu(self.q_tran(torch.cat([preprocessed_obs[0].clone(), gc], 1))) #(bs,Dz)
        else:
            #prior_t = self.q0_mu(self.q0_tran(preprocessed_Z0)) #(bs,Dz)
            proposal_t = self.q_mu(self.q_tran(preprocessed_obs[0].clone())) #(bs,Dz)
        #prior_std = torch.sqrt(torch.max(f.softplus(self.q0_sigma), self.sigma_min*torch.ones(self.q0_sigma.shape).to(self.device))) #(Dz)
        proposal_std = torch.sqrt(torch.max(f.softplus(self.q_sigma), self.sigma_min*torch.ones(self.q_sigma.shape).to(self.device))) #(Dz)
        Z_t = self._reparameterized_sample(proposal_t.unsqueeze(0).repeat(self.n_particles,1,1), proposal_std.repeat(self.n_particles,x.size(1),1)) #(np,bs,Dz)
        Zs[0] = Z_t
        
        for t in range(1, x.size(0)):
            if self.global_context_on:
                proposal_t = self.q_mu(self.q_tran(torch.cat([preprocessed_obs[t].clone(), gc], 1))) #(bs,Dz)
            else:
                proposal_t = self.q_mu(self.q_tran(preprocessed_obs[t].clone())) #(bs,Dz)
            Z_t = self._reparameterized_sample(proposal_t, proposal_std.repeat(self.n_particles,x.size(1),1)) #(np,bs,Dz)
            Zs[t] = Z_t
        
        Xs = self.g_mu(self.g_tran(Zs)) #(T,np,bs,Dx)
        return Zs, Xs
            
    
    
    def get_next_Z(self, Z_t, H_t=None): #(bs,bs,Dz)
        return self.f_mu(self.f_tran(Z_t)) #(bs,bs,Dz)
    
    
    def get_X_from_Z(self, Z_t, H_t=None):
        return self.g_mu(self.g_tran(Z_t)) #(np,bs,Dx)

    
    def _sample_from_dist(self, mu, sigma, sample_size=None):
        """sample from 2 distribution"""
        sig = torch.max(f.softplus(sigma), self.sigma_min*torch.ones(sigma.shape).to(self.device)) # (Dx)
        
        if sample_size == None:
            X = self._reparameterized_sample(mu, torch.sqrt(sig).repeat(mu.size(0),mu.size(1),1)) # (np,bs,Dx)
        else:
            X = self._reparameterized_sample(mu.unsqueeze(0).repeat(sample_size,1,1), torch.sqrt(sig).repeat(sample_size,mu.size(0),1)) #(np,bs,Dx)
        
        q_t_log_prob = self._log_prob(mu, torch.sqrt(sig), X, dim=2) #(np,bs)
        return X, q_t_log_prob