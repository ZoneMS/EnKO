import os, sys, time, math, copy
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

from .utils import calculate_conv_size, Flatten, UnFlatten, get_nonlinear_fn, get_model
from .conv import Conv



class StyleConv(Conv):
    def __init__(self, x_dim, config, device):
        super(StyleConv, self).__init__(x_dim, config, device)
        
        self.style_dim = int(config["outer"]["style_dim"])
        self.style_weight = config["outer"]["style_weight"]
        assert self.style_dim < self.a_dim
        
        ## construct internal vae
        self.model = get_model(self.a_dim - self.style_dim, config, device)
        
        ## construct encoder
        self.style_enc = copy.deepcopy(self.enc) #(c,h,w)->(enc_dim)
        self.enc_mean = nn.Linear(self.enc_dim, self.a_dim - self.style_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.enc_dim, self.a_dim - self.style_dim),
            nn.Softplus()
        )
        self.style_enc_mean = nn.Linear(self.enc_dim, self.style_dim)
        self.style_enc_std = nn.Sequential(
            nn.Linear(self.enc_dim, self.style_dim),
            nn.Softplus()
        )
            
    
    
    def forward(self, x, epoch, train_on=True, *args):
        # x:(T,bs,c,h,w)
        T, batch_size, c, h, w = x.shape
        
        ## Encoder
        enc = self.enc(x.reshape(T*batch_size,c,h,w))
        enc = enc.reshape(T,batch_size,-1) #(T,bs,enc_dim)
        enc_mean = self.enc_mean(enc) #(T,bs,Dt)
        enc_std = self.enc_std(enc) #(T,bs,Dt)
        a_tran = self._reparameterized_sample(enc_mean, enc_std) #(T,bs,Dt)
        kld_tran = self._kld_gauss_normal(enc_mean, enc_std, dim=0).mean(dim=0).sum() #(bs,Dt)->(Dt)
        
        style_enc = self.style_enc(x.reshape(T*batch_size,c,h,w)).reshape(T,batch_size,-1) #(T,bs,enc_dim)
        style_enc_mean = self.style_enc_mean(enc) #(T,bs,Ds)
        style_enc_std = self.style_enc_std(enc) #(T,bs,Ds)
        global_style_enc_mean = style_enc_mean.mean(0) #(bs,Ds)
        global_style_enc_std = style_enc_std.mean(0) #(bs,Ds)
        a_style = self._reparameterized_sample(global_style_enc_mean, global_style_enc_std) #(bs,Ds)
        kld_style = self._kld_gauss_normal(global_style_enc_mean, global_style_enc_std, dim=1).mean() #(bs,Ds)->(bs)
        kld_average = self._kld_gauss(global_style_enc_mean, global_style_enc_std, style_enc_mean, style_enc_std, dim=0).mean(dim=0).sum() #(bs,Ds)->(Ds,)
        
        ## Decoder
        a = torch.cat([a_tran, a_style.unsqueeze(0).repeat(T,1,1)], dim=2) #(T,bs,Da)
        dec = self.dec(a.reshape(T*batch_size,self.a_dim)) #(T*bs,c,h,w)
        dec = dec.reshape(T,batch_size,c,h,w) #(T,bs,c,h,w)
        negative_log_likelihood = - self._log_prob_emission(dec, None, x, dim=0).mean(dim=0).sum() #(bs,c,h,w)->(c,h,w)
        vae_loss = kld_tran + kld_style + self.style_weight * kld_average + negative_log_likelihood
            
        loss_set, var_set = self.model(a_tran, *args)
        z, a_hat_tran, H = var_set
        a_hat = torch.cat([a_hat_tran, a_style.unsqueeze(0).unsqueeze(0).repeat(T,self.n_particles,1,1)], dim=3) #(T,np,bs,Ds)
        x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,Dx)
        x_hat = x_hat.reshape(T,self.n_particles,batch_size,c,h,w) #(T,np,bs,c,h,w)
        
        if epoch <= self.only_outer_learning_epochs and train_on:
            total_loss = self.outer_scale * vae_loss
        else:
            total_loss = self.outer_scale * vae_loss + loss_set[0]
        
        return [total_loss, vae_loss, negative_log_likelihood, kld_tran, kld_style, kld_average] + list(loss_set), (z, a_hat, a_style, H, x_hat)
    
    
    
    def calculate_predictive_metrics(self, x, Z_t, a_style, H_t, pred_steps=10, evaluation_metrics=["MSE", "FIP"]):
        # x:(T,bs,c,h,w), Z_t:(T,np,bs,Dz), a_style:(bs,Ds)
        T, batch_size, c, h, w = x.shape
        x_res = x.reshape(T,batch_size,c*h*w) #(T,bs,c*h*w)
        MSE = np.zeros([pred_steps, batch_size])
        R2 = np.zeros([pred_steps, batch_size])
        NIP = np.zeros([pred_steps, batch_size])
        
        for t in range(pred_steps):
            Z_t = self.model.get_next_Z(Z_t, H_t) #(T,np,bs,Dz)
            a_hat_tran = self.model.get_X_from_Z(Z_t, H_t) #(T,np,bs,Da)
            a_hat = torch.cat([a_hat_tran, a_style.unsqueeze(0).unsqueeze(0).repeat(T,self.n_particles,1,1)], dim=3) #(T,np,bs,Ds)
            x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,c,h,w)
            x_hat = x_hat.reshape(T,self.n_particles,batch_size,-1).mean(1) #(T,bs,Dx)
            if "MSE" in evaluation_metrics:
                MSE[t] = (x_res[t+1:] - x_hat[:-(t+1)]).pow(2).sum(2).mean(0).data.cpu().numpy() #(bs,)
                R2[t] = (1 - ((x_res[t+1:] - x_hat[:-(t+1)])**2).sum(2).mean(0) / ((x_res[t+1:] - x_res[t+1:].mean(0))**2).sum(2).mean(0)).data.cpu().numpy() #(bs,)
            if "FIP" in evaluation_metrics:
                NIP[t] = (torch.abs(x_res[t+1:] - x_hat[:-(t+1)]) > 0.5).to(torch.float32).mean(0).sum(1).data.cpu().numpy() #(bs,)
        FIP = NIP / (c*h*w)
        
        results = []
        if "MSE" in evaluation_metrics:
            results += [MSE, R2]
        if "FIP" in evaluation_metrics:
            results += [NIP, FIP]
        return np.array(results)
    
    
    def init_running(self, x):
        # x:(T,bs,c,h,w)
        T, batch_size, c, h, w = x.shape
        
        enc = self.enc(x.reshape(T*batch_size,c,h,w))
        enc = enc.reshape(T,batch_size,-1) #(T,bs,enc_dim)
        enc_mean = self.enc_mean(enc) #(T,bs,Da)
        enc_std = self.enc_std(enc) #(T,bs,Da)
        a_tran = self._reparameterized_sample(enc_mean, enc_std) #(T,bs,Da)
        
        style_enc = self.style_enc(x.reshape(T*batch_size,c,h,w)).reshape(T,batch_size,-1) #(T,bs,enc_dim)
        style_enc_mean = self.style_enc_mean(enc) #(T,bs,Ds)
        style_enc_std = self.style_enc_std(enc) #(T,bs,Ds)
        global_style_enc_mean = style_enc_mean.mean(0) #(bs,Ds)
        global_style_enc_std = style_enc_std.mean(0) #(bs,Ds)
        a_style = self._reparameterized_sample(global_style_enc_mean, global_style_enc_std) #(bs,Ds)
        
        z, a_hat_tran = self.model.init_running(a_tran) #(T,np,bs,Dz),(T,np,bs,Da)
        a_hat = torch.cat([a_hat_tran, a_style.unsqueeze(0).unsqueeze(0).repeat(T,self.n_particles,1,1)], dim=3) #(T,np,bs,Ds)
        x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,Dx)
        x_hat = x_hat.reshape(T,self.n_particles,batch_size,c,h,w) #(T,np,bs,c,h,w)
        return z, a_hat, x_hat
    
    
    def running_only_q(self, x):
        # x:(T,bs,c,h,w)
        T, batch_size, c, h, w = x.shape
        
        enc = self.enc(x.reshape(T*batch_size,c,h,w))
        enc = enc.reshape(T,batch_size,-1) #(T,bs,enc_dim)
        enc_mean = self.enc_mean(enc) #(T,bs,Da)
        enc_std = self.enc_std(enc) #(T,bs,Da)
        a = self._reparameterized_sample(enc_mean, enc_std) #(T,bs,Da)
        
        z, a_hat = self.model.running_only_q(a) #(T,np,bs,Dz),(T,np,bs,Da)
        x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,Dx)
        x_hat = x_hat.reshape(T,self.n_particles,batch_size,c,h,w) #(T,np,bs,c,h,w)
        return z, a_hat, x_hat
    
    
    
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
        
        return (Zs, Xs)
    