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


class PSVO(SVO):
    def __init__(self, x_dim, config, device):
        super(PSVO, self).__init__(x_dim, config, device)
        self.M = config["network"]["n_bw_particles"]
        
        ## network
        self.BSim_qT_tran = nn.Sequential(
            nn.Linear(2*self.h_dim, self.h_dim),
            nn.ReLU())
        self.BSim_qT_mu = nn.Linear(self.h_dim, self.z_dim)
        self.BSim_qT_sigma = nn.Parameter(self.sigma_init * torch.ones(self.z_dim), requires_grad=self.sigma_train)
        
        self.f_inv_tran = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU())
        self.f_inv_mu = nn.Linear(self.h_dim, self.z_dim)
        self.f_inv_sigma = nn.Parameter(self.sigma_init * torch.ones(self.z_dim), requires_grad=self.sigma_train)
        
        self.BSim_q_tran = nn.Sequential(
            nn.Linear(2*self.h_dim, self.h_dim),
            nn.ReLU())
        self.BSim_q_mu = nn.Linear(self.h_dim, self.z_dim)
        self.BSim_q_sigma = nn.Parameter(self.sigma_init * torch.ones(self.z_dim), requires_grad=self.sigma_train)
        

    
    
    def forward(self, x):
        (log_Ws, log_gs, log_fs, log_qs), (_, _, _, Z_prevs), (preprocessed_obs, preprocessed_Z0) = self.SMC(x)
        (bw_log_Omegas, f_log_probs, g_log_probs), (bw_X, bw_Z) = self.backward_simulation(Z_prevs, log_Ws, preprocessed_obs, preprocessed_Z0, x)
        
        ## loss calculation
        joint = (f_log_probs + g_log_probs).sum(axis=0) # (np,bs)
        proposal = bw_log_Omegas.sum(axis=0) # (np,bs)
        log_ZSMC = torch.logsumexp(joint - proposal, axis=0) - math.log(self.n_particles) # (bs)
        log_ZSMC = log_ZSMC.mean()
        
        log_fw = torch.logsumexp(log_Ws, axis=1).sum(axis=0).mean() # (T,np,bs) -> (T,bs) -> (bs)
        ESS = self._calculate_ess(log_Ws).mean() #(T,)
        return (-log_ZSMC, -log_fw, ESS), (bw_Z, bw_X, _)
        
        
        
    def backward_simulation(self, Z_prevs, log_Ws, preprocessed_obs, preprocessed_Z0, x):
        T = x.size(0)
        bw_Xs_ta = Variable(torch.empty(T, self.n_particles, x.size(1), self.x_dim), requires_grad=False).to(self.device) # (T,np,bs,Dy)
        bw_Zs_ta = Variable(torch.empty(T, self.n_particles, x.size(1), self.z_dim), requires_grad=False).to(self.device) # (T,np,bs,Dx)
        bw_log_Omegas_ta = Variable(torch.empty(T, self.n_particles, x.size(1)), requires_grad=False).to(self.device) # (T,np,bs)
        f_log_probs_ta = Variable(torch.empty(T, self.n_particles, x.size(1)), requires_grad=False).to(self.device) # (T,np,bs)
        g_log_probs_ta = Variable(torch.empty(T, self.n_particles, x.size(1)), requires_grad=False).to(self.device) # (T,np,bs)
        bw_q_log_probs_ta = Variable(torch.empty(T, self.n_particles, x.size(1)), requires_grad=False).to(self.device) # (T,np,bs)
        
        
        #---------------- t=T-1 ----------------#
        proposal_t = self.BSim_qT_mu(self.BSim_qT_tran(preprocessed_obs[-1].clone())) #(bs,Dz)
        proposal_sigma_t = torch.max(f.softplus(self.BSim_qT_sigma), self.sigma_min*torch.ones_like(self.BSim_q_sigma).to(self.device)) #(Dz)
        bw_Z_t = self._reparameterized_sample(proposal_t.unsqueeze(0).unsqueeze(0).repeat(self.M,self.n_particles,1,1), torch.sqrt(proposal_sigma_t).view(1,1,1,self.z_dim).repeat(self.M,self.n_particles,x.size(1),1)) #(M,np,bs,Dz)
        bw_q_log_prob = self._log_prob(proposal_t,
                                        proposal_sigma_t,
                                        bw_Z_t,
                                        dim=3) #(M,np,bs)
        
        bw_Z_t_tiled = bw_Z_t.unsqueeze(2).repeat(1,1,self.n_particles,1,1) #(M,np,np,bs,Dz)
        f_tm1_log_prob = self._log_prob(self.f_mu(self.f_tran(Z_prevs[T-2].clone())), #(np,bs,Dz)
                                        torch.max(f.softplus(self.f_sigma), self.sigma_min*torch.ones_like(self.f_sigma).to(self.device)), #(Dz)
                                         bw_Z_t_tiled, #(M,np,np,bs,Dz)
                                         dim=4
        ) # (M,np,np,bs)[m,k,j,b]
        g_t_log_prob = self._log_prob(self.g_mu(self.g_tran(bw_Z_t)), #(M,np,bs,Dx)
                                        torch.max(f.softplus(self.g_sigma), self.sigma_min*torch.ones_like(self.g_sigma).to(self.device)), # (Dx)
                                         x[-1], # (bs,Dx)
                                         dim=3
                                        ) # (M,np,bs)

        log_W_tm1 = log_Ws[T-2].clone() - torch.logsumexp(log_Ws[T-2].clone(), axis=0) # (np,bs)
        log_W_t = torch.logsumexp(f_tm1_log_prob + log_W_tm1, axis=2) # (M,np,bs) summation for j
        
        bw_log_omega_t = log_W_t + g_t_log_prob - bw_q_log_prob # (M,np,bs)
        bw_log_omega_t = bw_log_omega_t - torch.logsumexp(bw_log_omega_t, axis=0) # (M,np,bs)
        
        bw_Z_t, bw_log_omega_t, g_t_log_prob, bw_q_log_prob = \
            self.system._resample_particles([bw_Z_t, bw_log_omega_t, g_t_log_prob, bw_q_log_prob], # (M,np,bs,Dz) or (M,np,bs)
                           bw_log_omega_t, # (M,np,bs)
                           sample_size=None) # (np,bs,Dz) or (np,bs)
        
        bw_log_Omega_t = bw_log_omega_t + bw_q_log_prob + math.log(self.M) # (np,bs)
        
        bw_Zs_ta[T-1] = bw_Z_t
        g_log_probs_ta[T-1] = g_t_log_prob
        bw_q_log_probs_ta[T-1] = bw_q_log_prob
        bw_log_Omegas_ta[T-1] = bw_log_Omega_t
        bw_Xs_ta[T-1] = self.g_mu(self.g_tran(bw_Z_t)) # (np,bs,Dx)
        
        
        #---------------- t=T-2 to 1 ----------------#
        for t in reversed(range(1,T-1)):
            # proposal q(x_t | x_{t+1}, y_{1:T})
            prior_t = self.f_inv_mu(self.f_inv_tran(bw_Z_t))
            proposal_t = self.BSim_q_mu(self.BSim_q_tran(preprocessed_obs[t].clone()))
            bw_Z_t, bw_q_log_prob, _ = self._sample_from_2_dist(prior_t,
                                                                proposal_t,
                                                                self.f_inv_sigma,
                                                                self.BSim_q_sigma,
                                                                t,
                                                                self.M) # (M,np,bs,Dz),(M,np,bs)
            # f(x_{t+1} | x_t)
            f_t_log_prob = self._log_prob(self.f_mu(self.f_tran(bw_Z_t)), # (M,np,bs,Dz)
                                           torch.max(f.softplus(self.f_sigma), self.sigma_min*torch.ones_like(self.f_sigma).to(self.device)), #(Dx)
                                           bw_Zs_ta[t+1].clone(), # (np,bs,Dz)
                                          dim=3
                                          ) #(M,np,bs)
            
            # p(x_t | y_{1:t}) is propotional to \int p(x_t-1 | y_{1:t-1}) * f(x_t | x_t-1) dx_t-1 * g(y_t | x_t)
            bw_Z_t_tiled = bw_Z_t.unsqueeze(2).repeat(1,1,self.n_particles,1,1) #(M,np,np,bs,Dz)
            f_tm1_log_prob = self._log_prob(self.f_mu(self.f_tran(Z_prevs[t-1].clone())), #(np,bs,Dz)
                                            torch.max(f.softplus(self.f_sigma), self.sigma_min*torch.ones_like(self.f_sigma).to(self.device)), #(Dz)
                                             bw_Z_t_tiled, #(M,np,np,bs,Dz)
                                            dim=4
                                            ) # (M,np,np,bs)[m,k,j,b]
            g_t_log_prob = self._log_prob(self.g_mu(self.g_tran(bw_Z_t)), #(M,np,bs,Dx)
                                            torch.max(f.softplus(self.g_sigma), self.sigma_min*torch.ones_like(self.g_sigma).to(self.device)), # (Dx)
                                             x[t], # (bs,Dx)
                                          dim=3
                                            ) # (M,np,bs)
            
            log_W_tm1 = log_Ws[t-1].clone() - torch.logsumexp(log_Ws[t-1].clone(), axis=0) # (np,bs)
            log_W_t = torch.logsumexp(f_tm1_log_prob + log_W_tm1, axis=2) # (M,np,bs) summation for j
            
            #print(log_W_t.shape, f_t_log_prob.shape, g_t_log_prob.shape, bw_q_log_prob.shape)
            bw_log_omega_t = log_W_t + f_t_log_prob + g_t_log_prob - bw_q_log_prob # (M,np,bs)
            bw_log_omega_t = bw_log_omega_t - torch.logsumexp(bw_log_omega_t, axis=0) # (M,np,bs)

            bw_Z_t, bw_log_omega_t, f_t_log_prob, g_t_log_prob, bw_q_log_prob = \
                self.system._resample_particles([bw_Z_t, bw_log_omega_t, f_t_log_prob, g_t_log_prob, bw_q_log_prob], # (M,np,bs,Dx) or (M,np,bs)
                               bw_log_omega_t, # (M,np,bs)
                               sample_size=None) # (np,bs,Dz) or (np,bs)

            bw_log_Omega_t = bw_log_omega_t + bw_q_log_prob + math.log(self.M) # (np,bs)

            bw_Zs_ta[t] = bw_Z_t
            f_log_probs_ta[t+1] = f_t_log_prob
            g_log_probs_ta[t] = g_t_log_prob
            bw_q_log_probs_ta[t] = bw_q_log_prob
            bw_log_Omegas_ta[t] = bw_log_Omega_t
            bw_Xs_ta[t] = self.g_mu(self.g_tran(bw_Z_t)) # (np,bs,Dx)
            
        
        #---------------- t=0 ----------------#
        prior_t = self.f_inv_mu(self.f_inv_tran(bw_Z_t))
        proposal_t = self.BSim_q_mu(self.BSim_q_tran(preprocessed_obs[0].clone()))
        bw_Z_t, bw_q_log_prob, _ = self._sample_from_2_dist(prior_t,
                                                            proposal_t,
                                                            self.f_inv_sigma,
                                                            self.BSim_q_sigma,
                                                            0,
                                                            self.M) # (M,np,bs,Dz),(M,np,bs)
        f_t_log_prob = self._log_prob(self.f_mu(self.f_tran(bw_Z_t)), # (M,np,bs,Dx)
                                        torch.max(f.softplus(self.f_sigma), self.sigma_min*torch.ones_like(self.f_sigma).to(self.device)), #(Dx)
                                        bw_Zs_ta[1].clone(), # (np,bs,Dx)
                                      dim=3
                                        ) #(M,np,bs)
        g_t_log_prob = self._log_prob(self.g_mu(self.g_tran(bw_Z_t)), #(M,np,bs,Dx)
                                        torch.max(f.softplus(self.g_sigma), self.sigma_min*torch.ones_like(self.g_sigma).to(self.device)), # (Dx)
                                        x[0], # (bs,Dx)
                                      dim=3
                                        ) # (M,np,bs)
        f_init_log_prob = self._log_prob(self.q0_mu(self.q0_tran(preprocessed_Z0)), # (bs,Dz)
                                            torch.max(f.softplus(self.q0_sigma), self.sigma_min*torch.ones_like(self.q0_sigma).to(self.device)), #(Dz)
                                            bw_Z_t, # (M,np,bs,Dx)
                                         dim=3
                                            ) #(M,np,bs)

        log_W_t = f_init_log_prob
        bw_log_omega_t = log_W_t + f_t_log_prob + g_t_log_prob - bw_q_log_prob # (M,np,bs)
        
        bw_Z_t, bw_log_omega_t, f_t_log_prob, f_init_log_prob, g_t_log_prob, bw_q_log_prob = \
            self.system._resample_particles([bw_Z_t, bw_log_omega_t, f_t_log_prob, f_init_log_prob, g_t_log_prob, bw_q_log_prob], # (M,np,bs,Dz) or (M,np,bs)
                           bw_log_omega_t, # (M,np,bs)
                           sample_size=None) # (np,bs,Dz) or (np,bs)
        
        bw_log_Omega_t = bw_log_omega_t + bw_q_log_prob + math.log(self.M) # (np,bs)
        
        bw_Zs_ta[0] = bw_Z_t
        f_log_probs_ta[1] = f_t_log_prob
        f_log_probs_ta[0] = f_init_log_prob
        g_log_probs_ta[0] = g_t_log_prob
        bw_q_log_probs_ta[0] = bw_q_log_prob
        bw_log_Omegas_ta[0] = bw_log_Omega_t
        bw_Xs_ta[0] = self.g_mu(self.g_tran(bw_Z_t)) # (np,bs,Dx)
    
        return (bw_log_Omegas_ta, f_log_probs_ta, g_log_probs_ta), (bw_Xs_ta, bw_Zs_ta)