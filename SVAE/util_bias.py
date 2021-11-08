import os, sys, json, copy
import time, shutil
import subprocess
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy import special

import torch
import torch.utils
from torch.autograd import Variable



def reparametrized_sampling(params, dist="Gauss"):
    if dist in ["Gauss", "gauss", "Normal", "normal"]:
        # 0:mu, 1:std>0
        eps = Variable(torch.FloatTensor(params[0].size()).normal_())
        return eps.mul(params[1]).add_(params[0])
    elif dist in ["Cauchy", "cauchy"]:
        # 0:mu, 1:gamma>0
        eps = Variable(torch.FloatTensor(params[0].size()).cauchy_())
        return eps.mul(params[1]).add_(params[0])
    elif "student-t" in dist.lower():
        # 0:loc, 1:scale>0
        degree = int(dist.split("-t")[1])
        eps = torch.distributions.studentT.StudentT(degree).sample(params[0].size())
        return eps.mul(params[1]).add_(params[0])
    
    
def random_sampling(params, dist="Gauss", size=None):
    if dist in ["Gauss", "gauss", "Normal", "normal"]:
        # 0:mu, 1:std>0
        return np.random.normal(params[0], params[1], size)
    elif dist in ["Cauchy", "cauchy"]:
        # 0:mu, 1:gamma>0
        return params[1] * np.random.standard_cauchy(size) + params[0]
    elif "student-t" in dist.lower():
        # 0:loc, 1:scale>0
        degree = int(dist.split("-t")[1])
        return params[1] * np.random.standard_t(degree, size) + params[0]
    
    
def expectation(params, dist="Gauss"):
    if dist in ["Gauss", "gauss", "Normal", "normal", "Cauchy", "cauchy"] or "student-t" in dist.lower():
        # use median instead of mean in cauchy
        return params[0]
    

def log_prob(z, params, dist="Gauss", dim=-1):
    if dist in ["Gauss", "gauss", "Normal", "normal"]:
        # 0:mu, 1:std>0
        #print(z.shape, params[0].shape, params[1].shape)
        return -0.5 * ((z - params[0]).pow(2) / params[1].pow(2) + torch.log(2 * math.pi * params[1].pow(2))).sum(dim)
    elif dist in ["Cauchy", "cauchy"]:
        # 0:mu, 1:gamma>0
        return (torch.log(params[1]) - math.log(math.pi) - torch.log((z - params[0]).pow(2) + params[1].pow(2))).sum(dim)
    elif "student-t" in dist.lower():
        # 0:loc, 1:scale>0
        degree = int(dist.split("-t")[1])
        y = (z - params[0]) / params[1]
        Z = (torch.log(params[1]) +
             0.5 * math.log(degree) +
             0.5 * math.log(math.pi) +
             math.lgamma(0.5 * degree) -
             math.lgamma(0.5 * (degree + 1.)))
        return (-0.5 * (degree + 1.) * torch.log1p(y.pow(2) / degree) - Z).sum(dim)
#         return (torch.log(1 + ((z - params[0]) / params[1]).pow(2) / degree).pow(-0.5 * (degree + 1))
#                 - torch.log(torch.abs(params[1]) * math.sqrt(degree * math.pi) * math.gamma(0.5 * degree) 
#                             / math.gamma(0.5 * (degree + 1)))).sum(dim)
    
    


def generate_obs(sparams, tparams, T, n_samples, dists, funcs):
    # sparams 0,1:f1, 2:f, 3:g, 4,5:q1, 6:q
    # tparams 0:f, 1:g, 2:q
    # dists 0:f1, 1:f, 2:g, 3:q1, 4:q
    # funcs 0:f, 1:g, 2:q
    Dx = len(sparams[3])
    Dz = len(sparams[0])
    x = np.zeros((T, n_samples, Dx))
    
    z_t = random_sampling(sparams[0:2], dists[0], (n_samples, Dz)) #(ns,Dz)
    x[0] = random_sampling([funcs[1](z_t, tparams[1]), sparams[3]], dists[2]) #(ns,Dx)
    for t in range(1,T):
        z_t = random_sampling([funcs[0](z_t, tparams[0]), sparams[2]], dists[1]) #(ns,Dx)
        x[t] = random_sampling([funcs[1](z_t, tparams[1]), sparams[3]], dists[2]) #(ns,Dx)
    return x


def enkf(z_t, x_t, sparam, tparam, dist, func):
    n_particles, _, Dz = z_t.shape
    Dx = x_t.shape[1]
    dec_mean_t = expectation([func(z_t, tparam), sparam], dist) #(np,bs,Dx)
    x_hat_t = reparametrized_sampling([func(z_t, tparam), sparam], dist) #(np,bs,Dx)
    centered_z_t = z_t - z_t.mean(dim=0) #(np,bs,Dz)
    centered_x_t = x_hat_t - x_hat_t.mean(dim=0) #(np,bs,Dx)
    centered_x_mean_t = dec_mean_t - dec_mean_t.mean(dim=0) #(np,bs,Dx)
    addjust_term = n_particles / (n_particles - 1)
    Sigma_zx_t = addjust_term * (centered_z_t.unsqueeze(3).repeat(1,1,1,Dx)
                * centered_x_mean_t.unsqueeze(2).repeat(1,1,Dz,1)).mean(axis=0) # (bs,Dz,Dx)
    # direct estimate for HVH+R
    Sigma_w_t = addjust_term * (centered_x_t.unsqueeze(3).repeat(1,1,1,Dx)
                * centered_x_t.unsqueeze(2).repeat(1,1,Dx,1)).mean(axis=0) # (bs,Dx,Dx)
    # Kalman gain
    K_t = Sigma_zx_t @ torch.inverse(Sigma_w_t) # (bs,Dz,Dx)
    increment = (K_t @ (x_t - x_hat_t).permute(1,2,0)).permute(2,0,1) # (np,bs,Dz)
    z_filtered_t = z_t + increment #(np,bs,Dz)
    return z_filtered_t


def resample_particles(X, log_W_t, with_resampling_gradient_on=False):
    # X:(np,bs,Dx), log_W_t:(np,bs)
    n_particles = X.size(0)
    perm = list(range(1, len(log_W_t.shape))) + [0]
    log_W_t = log_W_t - torch.logsumexp(log_W_t, axis=0) # (np,bs)
    categorical = torch.distributions.categorical.Categorical(logits=log_W_t.permute(perm)) # (bs,np)
    
    indices = categorical.sample(torch.Size([n_particles])) # (np,bs)
    if isinstance(X, list):
        X_resampled = []
        for item in X:
            if len(indices.shape)==len(item.shape):
                X_resampled.append(torch.gather(item, 0, indices)) # (np,bs)
            elif len(indices.shape)<len(item.shape):
                X_resampled.append(torch.gather(item, 0, indices.unsqueeze(2).repeat(1,1,item.size(2)))) # (np,bs,Dx)
    else:
        ex_indices = indices.unsqueeze(2).repeat(1,1,X.size(2)) # (np,bs,Dx)
        X_resampled = torch.gather(X, 0, ex_indices) # (np,bs,Dx)
        
    if with_resampling_gradient_on:
        discrete_loss = torch.nn.functional.nll_loss(log_W_t.permute(perm).repeat(n_particles,1), indices.reshape(-1), 
                                                          reduce=False).reshape(n_particles, -1) #(np*bs,np),(np*bs)->(np*bs)
        return X_resampled, discrete_loss
    else:
        return X_resampled



def compute_loss(log_Ws, loss_type="iwae"):
    # log_Ws [T,np,bs]
    n_particles = log_Ws.shape[1]
    if loss_type in ["iwae", "enko", "simple"]:
        loss = torch.logsumexp(log_Ws.sum(axis=0), axis=0).mean() - math.log(n_particles)
    elif loss_type in ["fivo"]:
        loss = torch.logsumexp(log_Ws - math.log(n_particles), axis=1).sum(axis=0).mean()
    elif loss_type in ["fivor"]:
        loss = log_Ws.sum([0,1]).mean()
    return loss


def compute_ssm(x, sparams, tparams, n_particles, dists, funcs, system="enko"):
    # x [T,bs,Dx]
    # sparams 0,1:f1, 2:f, 3:g, 4,5:q1, 6:q
    # tparams 0:f, 1:g, 2:q
    # dists 0:f1, 1:f, 2:g, 3:q1, 4:q
    # funcs 0:f, 1:g, 2:q
    T, batch_size, Dx = x.shape
    Dz = len(sparams[0])
    log_Ws = Variable(torch.zeros(T, n_particles, batch_size), requires_grad=False)
    log_Ds = Variable(torch.zeros(T, n_particles, batch_size), requires_grad=False)
    
    if system == "simple":
        z_t = reparametrized_sampling([sparams[0], sparams[1]*torch.ones(n_particles,batch_size,Dz)], dists[0]) #(np,bs,Dz)
        log_Ws[0] = log_prob(x[0], [funcs[1](z_t, tparams[1]), sparams[3]], dists[2])
    elif system in ["fivo", "iwae", "enko"]:
        z_t = reparametrized_sampling([sparams[4], sparams[5]*torch.ones(n_particles,batch_size,Dz)], dists[3]) #(np,bs,Dz)
        log_g_t = log_prob(x[0], [funcs[1](z_t, tparams[1]), sparams[3]], dists[2])
        log_f_t = log_prob(z_t, sparams[:2], dists[0])
        log_q_t = log_prob(z_t, sparams[4:6], dists[3])
        log_Ws[0] = log_f_t + log_g_t - log_q_t
    
    for t in range(1,T):
        if system == "simple":
            z_t = reparametrized_sampling([funcs[0](z_t, tparams[0]), sparams[2]], dists[1])
            log_Ws[t] = log_prob(x[t], [funcs[1](z_t, tparams[1]), sparams[3]], dists[2])
        else:
            if system == "enko":
                z_prev_t = z_t
                z_filtered_t = enkf(z_t, x[t-1], sparams[3], tparams[1], dists[2], funcs[1])
            elif system == "fivo":
                z_filtered_t, log_Ds[t] = resample_particles(z_t, log_Ws[t-1].clone(), True)
            elif system == "iwae":
                z_filtered_t = z_t
            z_t = reparametrized_sampling([funcs[2](z_filtered_t, tparams[2]), sparams[6]], dists[4])

            log_g_t = log_prob(x[t], [funcs[1](z_t, tparams[1]), sparams[3]], dists[2])
            if system == "enko":
                log_f_t = log_prob(z_t, [funcs[0](z_prev_t, tparams[0]), sparams[2]], dists[1])
            elif system in ["fivo", "iwae"]:
                log_f_t = log_prob(z_t, [funcs[0](z_filtered_t, tparams[0]), sparams[2]], dists[1])
            log_q_t = log_prob(z_t, [funcs[2](z_filtered_t, tparams[2]), sparams[6]], dists[4])
            log_Ws[t] = log_f_t + log_g_t - log_q_t
#             print("t={}: logf:({:4e},{:4e}), log_g_t:({:4e},{:4e}), log_q_t:({:4e},{:4e})".format(t, 
#                                 log_f_t.min(), log_f_t.max(), log_g_t.min(), log_g_t.max(), log_q_t.min(), log_q_t.max()))
#             print("z_t:({:4e},{:4e}), z_ft:({:4e},{:4e})".format(t, z_t.min(), z_t.max(), z_filtered_t.min(), z_filtered_t.max()))
        
    if system == "fivo":
        return [compute_loss(log_Ws, system), compute_loss(log_Ds, "fivor")]
    else:
        return compute_loss(log_Ws, system)
    
    
    
    
def simulation(tparams, sparams, dists, funcs, n_simulation=100, n_particles=16, T=100, n_samples=10, system_list=["fivo", "enko", "iwae", "simple"], config=None):
    if config is not None:
        n_simulation = config["n_simulation"]
        n_particles = config["n_particles"]
        T = config["n_timesteps"]
        n_samples = config["n_samples"]
    
    total_system_list = sum([["fivo", "fivor"] if s=="fivo" else [s] for s in system_list], [])
    
    tgrads = [np.zeros([len(system_list)+1, n_simulation, *tparam.shape]) for tparam in tparams] # 3,(n_system,n_sim,*) 
    sgrads = [np.zeros([len(system_list)+1, n_simulation, *sparam.shape]) for sparam in sparams] # 7,(n_system,n_sim,*)
    losses = np.zeros([len(system_list)+1, n_simulation])
    
    x = generate_obs(sparams, tparams, T, n_samples, dists, funcs)
    tparams = [Variable(torch.from_numpy(tparam), requires_grad=True) for tparam in tparams]
    sparams = [Variable(torch.from_numpy(sparam), requires_grad=True) for sparam in sparams]
    
    count = 0
    for i, system in enumerate(system_list):
        for j in range(n_simulation):
            #print("\r calculation {} system {}/{}".format(system, j+1, n_simulation), end="")
            if system == "fivo":
                loss = compute_ssm(Variable(torch.from_numpy(x), requires_grad=False), sparams, tparams, 
                                   n_particles, dists, funcs, system)
                losses[count,j] = loss[0].item()
                loss[0].backward()
                for k in range(len(sgrads)):
                    sgrads[k][count,j] = sparams[k].grad.data.numpy()
                    sparams[k].grad.zero_()
                for k in range(len(tgrads)):
                    tgrads[k][count,j] = tparams[k].grad.data.numpy()
                    tparams[k].grad.zero_()

                loss = compute_ssm(Variable(torch.from_numpy(x), requires_grad=False), sparams, tparams, 
                                   n_particles, dists, funcs, system)
                losses[count+1,j] = loss[0].item()
                loss[1].backward()
                for k in range(len(sgrads)):
                    sgrads[k][count+1,j] = sparams[k].grad.data.numpy() * loss[0].item()
                    sparams[k].grad.zero_()
                for k in range(len(tgrads)):
                    tgrads[k][count+1,j] = tparams[k].grad.data.numpy() * loss[0].item()
                    tparams[k].grad.zero_()
            else:
                loss = compute_ssm(Variable(torch.from_numpy(x), requires_grad=False), sparams, tparams, 
                                   n_particles, dists, funcs, system)
                losses[count,j] = loss.item()
                loss.backward()
                for k in range(len(sgrads)):
                    sgrads[k][count,j] = sparams[k].grad.data.numpy()
                    sparams[k].grad.zero_()
                for k in range(len(tgrads)):
                    tgrads[k][count,j] = tparams[k].grad.data.numpy()
                    tparams[k].grad.zero_()
        if system=="fivo":
            count += 2
        else:
            count += 1
        print("\n")
    
    return tgrads, sgrads, losses



### Transforms
def linear_transform(x, A):
    # x:(*,Dx), A:(Da,Dx) -> (*,Da)
    return x @ A.T


def square_tanh_transform(x, A):
    # x:(*,Dx), A:(Dx,(Dx+1)^2) -> (*,Dx)
    Dx = x.shape[-1]
    if type(x)==np.ndarray:
        x_ex = np.concatenate([x, np.ones(x.shape[:-1])[...,np.newaxis]], -1)
        x_sq = x_ex.repeat(Dx+1, axis=-1) * np.tile(x_ex, [1 for i in range(x.ndim - 1)] + [Dx+1])
        return np.tanh(x_sq @ A.T)
    elif type(x)==torch.Tensor:
        x_ex = torch.cat([x, torch.ones(x.shape[:-1]).unsqueeze(-1)], -1)
        x_sq = torch.repeat_interleave(x_ex, Dx+1, dim=-1) \
                * x_ex.repeat([1 for i in range(x.ndim - 1)] + [Dx+1])
        return torch.tanh(x_sq @ A.T)
    else:
        raise ValueError("input type must be numpy.ndarray or torch.Tensor, but input is {}.".format(type(x)))


def get_transform(name="linear"):
    if name=="linear":
        return linear_transform
    elif name=="square-tanh":
        return square_tanh_transform
    
    
    
def extract_parameter_features(tgrads, name="f-linear"):
    if name in ["f-linear", "q-linear"]:
        Dz = tgrads.shape[2]
        region_list = [np.eye(Dz, dtype=bool), ~np.eye(Dz, dtype=bool)]
        rname_list = ["{}-diag-std".format(name[0]), "{}-offdiag-std".format(name[0])]
    elif name in ["f-square-tanh", "q-square-tanh"]:
        Dz = tgrads.shape[2]
        region_list = [np.concatenate([np.zeros([Dz, (Dz+1)*Dz], dtype=bool), np.eye(Dz, dtype=bool), np.zeros([Dz,1], dtype=bool)], -1),
                      np.concatenate([np.zeros([Dz, (Dz+1)*Dz], dtype=bool), ~np.eye(Dz, dtype=bool), np.zeros([Dz,1], dtype=bool)], -1),
                      np.concatenate([np.zeros([Dz, (Dz+2)*Dz], dtype=bool), np.ones([Dz,1], dtype=bool)], -1),
                      np.concatenate([np.ones([Dz, (Dz+1)*Dz], dtype=bool), np.zeros([Dz,Dz+1], dtype=bool)], -1)]
        rname_list = ["{}-diag-std".format(name[0]), "{}-offdiag-std".format(name[0]),
                     "{}-bias-std".format(name[0]), "{}-second-std".format(name[0])]
    elif name=="g-linear":
        _, _, Dx, Dz = tgrads.shape
        region_list = [np.ones([Dx, Dz], dtype=bool)]
        rname_list = ["g-std"]
    elif name=="g-square-tanh":
        Dx = tgrads.shape[2]
        Dz = int(math.sqrt(tgrads.shape[3]) - 1)
        region_list = [np.concatenate([np.zeros([Dx, (Dz+1)*Dz], dtype=bool), np.ones([Dx, Dz], dtype=bool), np.zeros([Dx, 1], dtype=bool)], -1),
                      np.concatenate([np.zeros([Dx, (Dz+2)*Dz], dtype=bool), np.ones([Dx, 1], dtype=bool)], -1),
                      np.concatenate([np.ones([Dx, (Dz+2)*Dz], dtype=bool), np.zeros([Dx, 1], dtype=bool)], -1)]
        rname_list = ["g-first-std", "g-bias-std", "g-second-std"]
        
    results = []
    for region in region_list:
        results.append(tgrads[:,:,region].std(axis=1).mean(axis=1))
    
    return results, rname_list

        

def compute_bias_and_variance(Dz, Dx, transforms=None, dists=None, init_std=1e-1, std=1e-2,
                              offdiag_scale=0.1, confounder_scale=0.05, shared_evolution_on=True, seed=1, config=None):
    if config is not None:
        seed = config["seed"]
        init_std = config["init_std"]
        std = config["std"]
        transforms = list(config["transforms"].values()) # f,g,q
        dists = list(config["distributions"].values()) # f1,f,g,q1,q
        offdiag_scale = config["offdiag_scale"]
        confounder_scale = config["confounder_scale"]
        shared_evolution_on = config["shared_evolution_on"]
        
    funcs = [get_transform(name) for name in transforms] # f,g,q
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # sparams 0,1:f1, 2:f, 3:g, 4,5:q1, 6:q
    sparams = [np.zeros(Dz), init_std*np.ones(Dz), std*np.ones(Dz), std*np.ones(Dx),
               np.zeros(Dz), init_std*np.ones(Dz), std*np.ones(Dz)]
    
    if transforms[0]=="linear":
        f_params = np.eye(Dz) + offdiag_scale * (np.random.rand(Dz, Dz) - 0.5)
    elif transforms[0]=="square-tanh":
        f_params = np.zeros([Dz,(Dz+1)**2])
        f_params[:,Dz*(Dz+1):-1] = np.eye(Dz) + offdiag_scale * (np.random.rand(Dz, Dz) - 0.5)
        f_params[:,:Dz*(Dz+1)] = confounder_scale * (np.random.rand(Dz, Dz*(Dz+1)) - 0.5)
        
    if transforms[1]=="linear":
        g_params = 1. * (np.random.rand(Dx, Dz) - 0.5)
    elif transforms[1]=="square-tanh":
        g_params = np.zeros([Dx,(Dz+1)**2])
        g_params[:,Dz*(Dz+1):-1] = offdiag_scale * (np.random.rand(Dx, Dz) - 0.5)
        g_params[:,:Dz*(Dz+1)] = confounder_scale * (np.random.rand(Dx, Dz*(Dz+1)) - 0.5)
        
    if transforms[2]==transforms[0] and shared_evolution_on:
        q_params = copy.deepcopy(f_params)
    elif transforms[2]=="linear":
        q_params = np.eye(Dz) + offdiag_scale * (np.random.rand(Dz, Dz) - 0.5)
    elif transforms[2]=="square-tanh":
        q_params = np.zeros([Dz,(Dz+1)**2])
        q_params[:,Dz*(Dz+1):-1] = np.eye(Dz) + offdiag_scale * (np.random.rand(Dz, Dz) - 0.5)
        q_params[:,:Dz*(Dz+1)] = confounder_scale * (np.random.rand(Dz, Dz*(Dz+1)) - 0.5)
        
    tparams = [f_params, g_params, q_params]
    
    # get results for losses and gradients
    # tgrads:(3,n_system,n_sim,*), sgrads:(7,n_system,n_sim,*), losses:(n_system,n_sim)
    tgrads, sgrads, losses = simulation(tparams, sparams, dists, funcs, config=config)
    
    rname_list = ["mean-bias", "loss-std"]
    # (n_results,n_system)
    results = [losses[-1].mean() - losses.mean(axis=1), losses.std(axis=1)]
    
    for i, pname in enumerate(["f", "g", "q"]):
        temp_results, temp_rname_list = extract_parameter_features(tgrads[i], "{}-{}".format(pname, transforms[i]))
        results += temp_results
        rname_list += temp_rname_list
    
    for i, pname in enumerate(["f1-loc", "f1-scale", "f-scale", "g-scale", "q1-loc", "q1-scale", "q-scale"]):
        results.append(sgrads[i].std(axis=1).mean(axis=1))
        rname_list.append("{}-std".format(pname))
        
    return np.array(results), rname_list
