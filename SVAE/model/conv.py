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

from .utils import calculate_conv_size, Flatten, UnFlatten, get_nonlinear_fn, get_model
from .outer import OuterModel




class Conv(OuterModel):
    def __init__(self, x_dim, config, device):
        super(Conv, self).__init__(x_dim, config, device)
        
        input_channels, height, width = x_dim
        filter_enc = [int(input_channels)] + [int(v) for v in config["conv"]["filter_enc"]]
        kernel_enc = [int(v) for v in config["conv"]["kernel_enc"]]
        stride_enc = [int(v) for v in config["conv"]["stride_enc"]]
        padding_enc = [int(v) for v in config["conv"]["padding_enc"]]
        bn_enc = config["conv"]["bn_enc"]
        filter_dec = [int(v) for v in config["conv"]["filter_dec"]] + [int(input_channels)]
        kernel_dec = [int(v) for v in config["conv"]["kernel_dec"]]
        stride_dec = [int(v) for v in config["conv"]["stride_dec"]]
        padding_dec = [int(v) for v in config["conv"]["padding_dec"]]
        output_padding_dec = [int(v) for v in config["conv"]["output_padding_dec"]]
        bn_dec = config["conv"]["bn_dec"]
        
        hidden_w = calculate_conv_size(width, kernel_enc, stride_enc, padding_enc)
        hidden_h = calculate_conv_size(height, kernel_enc, stride_enc, padding_enc)
        self.enc_dim = hidden_w * hidden_h * filter_enc[-1]
        
        ## construct internal vae
        self.model = get_model(self.a_dim, config, device)
        
        ## construct encoder
        #self.enc = nn.ModuleList()
        self.enc = []
        for (ic,oc,k,s,p,bn) in zip(filter_enc[:-1], filter_enc[1:], kernel_enc, stride_enc, padding_enc, bn_enc):
            if bn:
                self.enc += [
                    nn.Conv2d(ic,oc,k,s,p),
                    nn.BatchNorm2d(oc),
                    nn.ReLU()
                ]
            else:
                self.enc += [
                    nn.Conv2d(ic,oc,k,s,p),
                    nn.ReLU()
                ]
        self.enc.append(Flatten())
        self.enc = nn.Sequential(*self.enc)
        self.enc_mean = nn.Linear(self.enc_dim, self.a_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.enc_dim, self.a_dim),
            nn.Softplus()
        )
        
        ## construct decoder
        self.dec = [nn.Linear(self.a_dim, self.enc_dim),
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
            self.output_fn()
        ]
        self.dec = nn.Sequential(*self.dec)
        
            
    
    
    def forward(self, x, epoch, train_on=True, *args):
        # x:(T,bs,c,h,w)
        T, batch_size, c, h, w = x.shape
        
        ## Encoder
#         enc = x.reshape(T*batch_size,c,h,w)
#         for l in self.enc:
#             print(enc.shape)
#             enc = l(enc) #(T*bs,cc,ch,cw)
        enc = self.enc(x.reshape(T*batch_size,c,h,w))
        enc = enc.reshape(T,batch_size,-1) #(T,bs,enc_dim)
        enc_mean = self.enc_mean(enc) #(T,bs,Da)
        enc_std = self.enc_std(enc) #(T,bs,Da)
        #print("enc", enc)
        a = self._reparameterized_sample(enc_mean, enc_std) #(T,bs,Da)
        kl_divergence = self._kld_gauss_normal(enc_mean, enc_std, dim=0).mean(dim=0).sum() #(bs,Da)->(Da)
        
        ## Decoder
        dec = self.dec(a.reshape(T*batch_size,self.a_dim)) #(T*bs,c,h,w)
        dec = dec.reshape(T,batch_size,c,h,w) #(T,bs,c,h,w)
        negative_log_likelihood = - self._log_prob_emission(dec, None, x, dim=0).mean(dim=0).sum() #(bs,c,h,w)->(c,h,w)
        vae_loss = kl_divergence + negative_log_likelihood
            
        if epoch <= self.only_outer_learning_epochs and train_on:
            if True:
                loss_set, var_set = self.model(a, *args)
                z_prevs, z_fils, a_hat, H = var_set
                x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,Dx)
                x_hat = x_hat.reshape(T,self.n_particles,batch_size,c,h,w) #(T,np,bs,c,h,w)
            else:
                loss_set = torch.zeros(10)
                z_prevs, z_fils, a_hat, H, x_hat = None, None, None, None
            total_loss = self.outer_scale * vae_loss
        else:
            ## inner model loss
            loss_set, var_set = self.model(a, *args)
            z_prevs, z_fils, a_hat, H = var_set
            x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,Dx)
            x_hat = x_hat.reshape(T,self.n_particles,batch_size,c,h,w) #(T,np,bs,c,h,w)
            total_loss = self.outer_scale * vae_loss + loss_set[0]
        
        return [total_loss, vae_loss, negative_log_likelihood, kl_divergence] + list(loss_set), (z_prevs, z_fils, a_hat, H, x_hat)
    
    
    
    def calculate_predictive_metrics(self, x, Z_t, H_t, pred_steps=10, evaluation_metrics=["MSE", "FIP"]):
        # x:(T,bs,c,h,w), Z_t:(T,np,bs,Dz)
        T, batch_size, c, h, w = x.shape
        x_res = x.reshape(T,batch_size,c*h*w) #(T,bs,c*h*w)
        #MSE = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        #R2 = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        MSE = np.zeros([pred_steps, batch_size])
        R2 = np.zeros([pred_steps, batch_size])
        NIP = np.zeros([pred_steps, batch_size])
        
        for t in range(pred_steps):
            Z_t = self.model.get_next_Z(Z_t, H_t) #(T,np,bs,Dz)
            a_hat = self.model.get_X_from_Z(Z_t, H_t) #(T,np,bs,Da)
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
    
    
    def calculate_mse(self, x, Z_t, pred_steps=5):
        # x:(T,bs,c,h,w), Z_t:(T,np,bs,Dz)
        T, batch_size, c, h, w = x.shape
        x_res = x.reshape(T,batch_size,c*h*w) #(T,bs,c*h*w)
        #MSE = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        #R2 = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        MSE = np.zeros([pred_steps, batch_size])
        R2 = np.zeros([pred_steps, batch_size])

        if False:
            (a_hat, _) = self.model.prediction(Z_t, pred_steps) #(ps,T,np,bs,Da)
            #x_hat = a_hat.reshape(pred_steps*T*self.n_particles*batch_size,self.a_dim) #(ps*T*np*bs,Da)
            x_hat = self.dec(a_hat.reshape(pred_steps*T*self.n_particles*batch_size,self.a_dim)) #(ps*T*np*bs,Dx)
            x_hat = x_hat.reshape(pred_steps,T,self.n_particles,batch_size,-1) #(ps,T,np,bs,Dx)

            for t in range(pred_steps):
                #calcurate
                MSE[t] = ((x_res[t+1:] - x_hat[t,:-(t+1)].mean(axis=1))**2).mean(axis=0) #(bs,Dx)
                TV[t] = ((x_res[t+1:] - x_res[t+1:].mean(axis=0))**2).mean(axis=0) #(bs,Dx)
        else:
            for t in range(pred_steps):
                Z_t = self.model.get_next_Z(Z_t) #(T,np,bs,Dz)
                a_hat = self.model.get_X_from_Z(Z_t) #(T,np,bs,Da)
                x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,c,h,w)
                x_hat = x_hat.reshape(T,self.n_particles,batch_size,-1).mean(1) #(T,bs,Dx)
                MSE[t] = (x_res[t+1:] - x_hat[:-(t+1)]).pow(2).sum(2).mean(0).data.cpu().numpy() #(bs,)
                R2[t] = (1 - ((x_res[t+1:] - x_hat[:-(t+1)])**2).sum(2).mean(0) / ((x_res[t+1:] - x_res[t+1:].mean(0))**2).sum(2).mean(0)).data.cpu().numpy() #(bs,)
        
        return MSE, R2
    
    
    def calculate_nip(self, x, Z_t, pred_steps=5):
        # x:(T,bs,c,h,w), Z_t:(T,np,bs,Dz)
        T, batch_size, c, h, w = x.shape
        x_res = x.reshape(T,batch_size,c*h*w) #(T,bs,Dx)
        NIP = Variable(torch.zeros(pred_steps, batch_size), requires_grad=False).to(self.device) #(ps,bs)
        
        if False:
            (a_hat, _) = self.model.prediction(Z_t, pred_steps) #(ps,T,np,bs,Da)
            x_hat = self.dec(a_hat.reshape(pred_steps*T*self.n_particles*batch_size,self.a_dim)) #(ps*T*np*bs,Dx)
            x_hat = x_hat.reshape(pred_steps,T,self.n_particles,batch_size,-1) #(ps,T,np,bs,Dx)

            for t in range(pred_steps):
                #calcurate
                NIP[t] = (torch.abs(x_res[t+1:] - x_hat[t,:-(t+1)].mean(axis=1)) > 0.5).to(torch.float32).mean(0).sum(1) #(bs,)
        else:
            for t in range(pred_steps):
                Z_t = self.model.get_next_Z(Z_t) #(T,np,bs,Dz)
                a_hat = self.model.get_X_from_Z(Z_t) #(T,np,bs,Da)
                x_hat = self.dec(a_hat.reshape(T*self.n_particles*batch_size,self.a_dim)) #(T*np*bs,Dx)
                x_hat = x_hat.reshape(T,self.n_particles,batch_size,-1).mean(1) #(T,bs,Dx)
                NIP[t] = (torch.abs(x_res[t+1:] - x_hat[:-(t+1)]) > 0.5).to(torch.float32).mean(0).sum(1) #(bs,)
        
        return NIP
    
    
    
    def calculate_r_squared(self, x, Z_t, pred_steps=5):
        # x:(T,bs,Dx), Z_t:(T,np,bs,Dz)
        R2 = Variable(torch.zeros(pred_steps, x.size(1)), requires_grad=False).to(self.device) #(ps,bs)
        
        for t in range(pred_steps):
            #prior
            Z_t = self.f_mu(self.f_tran(Z_t)) # (T,np,bs,Dz)
            
            #decoder
            x_hat_t = self.g_mu(self.g_tran(Z_t)) #(T,np,bs,Dx)
            
            #calcurate
            MSE_t = ((x[t+1:] - x_hat_t[:-(t+1)].mean(axis=1))**2).sum(axis=2).sum(axis=0) #(bs)
            deno_t = ((x[t+1:] - x[t+1:].mean(axis=0))**2).sum(axis=2).sum(axis=0) #(bs)
            R2[t] = 1 - MSE_t / deno_t #(bs)
        
        return R2
    
    
    
    def init_running(self, x):
        # x:(T,bs,c,h,w)
        T, batch_size, c, h, w = x.shape
        
        enc = self.enc(x.reshape(T*batch_size,c,h,w))
        enc = enc.reshape(T,batch_size,-1) #(T,bs,enc_dim)
        enc_mean = self.enc_mean(enc) #(T,bs,Da)
        enc_std = self.enc_std(enc) #(T,bs,Da)
        a = self._reparameterized_sample(enc_mean, enc_std) #(T,bs,Da)
        
        z, a_hat = self.model.init_running(a) #(T,np,bs,Dz),(T,np,bs,Da)
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
    