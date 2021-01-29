import os, sys, json
import time, shutil
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from ODE.model import FitzHughNagumo
from ODE.scheme import RungeKuttaScheme



def load_data(config):
    data_name = config["data"]["data_name"] # Allen, FHN, Lorenz
    
    if data_name=="Allen":
        obs_train = np.load("../data/allen/normalized_train.npy") #(ns,T,Dx)=(30,1000,1)
        obs_valid = np.load("../data/allen/normalized_test.npy") #(ns,T,Dx)=(10,1000,1)
        Dx = obs_train.shape[2]
        obs_test = obs_valid.copy()
    elif data_name in ["FHN", "Lorenz"]:
        if data_name=="FHN":
            obs = np.load("../data/FHN/FHN_rk_obs0_ns400_dt001_T3000_ds15_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(400,200,1)
        elif data_name=="Lorenz":
            obs = np.load("../data/Lorenz/Lorentz_rk_obs_ns100_dt001_T750_ds3_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(100,250,3)
        n_sample = len(obs)
        train_sp = int(n_sample*config["train"]["train_rate"]) # train separating point
        valid_sp = int(n_sample*config["train"]["valid_rate"]) + train_sp # validation separating point
        obs_train = obs[:train_sp]
        obs_valid = obs[train_sp:valid_sp]
        obs_test = obs[valid_sp:]
        Dx = obs.shape[2]
    return obs_train, obs_valid, obs_test, Dx



def transform_data(obs_train, obs_valid, config):
    scaling = config["data"]["scaling"]
    scaling_factor = config["data"]["scaling_factor"]
    
    if scaling=="min-max":
        obs_max = obs_train.max(axis=0).max(axis=0)
        obs_min = obs_train.min(axis=0).min(axis=0)
        svec = obs_max - obs_min
        obs_train = (obs_train - obs_min) / svec
        obs_valid = (obs_valid - obs_min) / svec
        #obs_test = (obs_test - obs_min) / (obs_max - obs_min)
    elif scaling=="abs-div":
        svec = np.absolute(obs_train).max(axis=0).max(axis=0)
        obs_train = obs_train / svec
        obs_valid = obs_valid / svec
    elif scaling=="th-abs-div":
        abs_max = np.absolute(obs_train).max(axis=0).max(axis=0)
        div = np.where(1/abs_max < scaling_factor, 1/abs_max, 1)
        obs_train = obs_train * div
        obs_valid = obs_valid * div
        svec = 1 / div
    elif scaling=="standard":
        obs_mean = obs_train.reshape(-1,Dx).mean(axis=0)
        svec = np.std(obs_train.reshape(-1,Dx), axis=0)
        obs_train = (obs_train - obs_mean) / svec
        obs_valid = (obs_valid - obs_mean) / svec
        #obs_test = (obs_test - obs_mean) / svec
    else:
        svec = 1
    
    return obs_train, obs_valid, svec


def get_dataset(config, obs_train, obs_valid):
    batch_size = config["train"]["batch_size"]
    model_name = config["data"]["model"]
    num_workers = config["train"]["num_workers"]
    
    train_tensor = torch.utils.data.TensorDataset(torch.from_numpy(obs_train),
                                           torch.from_numpy(np.insert(obs_train,0,0,axis=1)[:,:-1]))
    valid_tensor = torch.utils.data.TensorDataset(torch.from_numpy(obs_valid),
                                           torch.from_numpy(np.insert(obs_valid,0,0,axis=1)[:,:-1]))
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader
        



def get_model(config, Dx, device):
    loss_name_list_dict = {"VRNN":["-logSMC", "-logf", "-logg", "logq", "ESS"],
                           "SRNN":["-logSMC", "-logf", "-logg", "logq", "ESS"],
                           "AESMC":["-logSMC", "-logf", "-logg", "logq", "ESS"],
                           "SVO":["-logSMC", "-logf", "-logg", "logq", "ESS"],
                           "PSVO":["-logSMC", "-logfw", "ESS"]}
    model_name = config["data"]["model"]
    system = config["data"]["system"]
    if system in ["FIVO", "EnKO"]:
        loss_name_list_dict["NODE"] = ["-logSMC"]
    
    if model_name=="VRNN":
        from model.vrnn import VRNN
        model = VRNN(Dx, config, device).to(device)
    elif model_name=="SRNN":
        from model.srnn import SRNN
        model = SRNN(Dx, config, device).to(device)
    elif model_name=="AESMC":
        from model.aesmc import AESMC
        model = AESMC(Dx, config, device).to(device)
    elif model_name=="SVO":
        from model.svo import SVO
        model = SVO(Dx, config, device).to(device)
    elif model_name=="PSVO":
        from model.psvo import PSVO
        model = PSVO(Dx, config, device).to(device)
    loss_name_list = loss_name_list_dict[model_name]
 
    return model, loss_name_list




def plot_loss(epoch, train_loss, valid_loss, loss_name_list, result_dir):
    n_losses = len(loss_name_list)
    
    # plot loss figure
    if n_losses==1:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
    elif n_losses<=3:
        fig, ax = plt.subplots(1, n_losses, figsize=(5*n_losses, 5))
    elif n_losses==4:
        fig, ax = plt.subplots(2, 2, figsize=(10,10))
    elif n_losses>=5 and n_losses<=9:
        vert = (n_losses-1)//3+1
        fig, ax = plt.subplots(vert, 3, figsize=(15,5*vert))
    elif n_losses>=10 and n_losses<=16:
        vert = (n_losses-1)//4+1
        fig, ax = plt.subplots(vert, 4, figsize=(20,5*vert))
    
    for i in range(len(fig.axes) - n_losses):
        fig.axes[-(i+1)].axis("off")
            
    for i, loss_name in enumerate(loss_name_list):
        fig.axes[i].plot(range(epoch), train_loss[:epoch,i],
                          label="train")
        fig.axes[i].plot(range(epoch), valid_loss[:epoch,i],
                      label="test")
        fig.axes[i].set_xlabel("epoch")
        fig.axes[i].set_ylabel(loss_name)
        fig.axes[i].legend()
            
    fig.savefig(os.path.join(result_dir, "loss.png"), 
                bbox_inches="tight")
    
    for i in range(n_losses):
        fig.axes[i].set_ylim(train_loss[:,i].min(), train_loss[0,i]+abs(train_loss[0,i]))
    fig.savefig(os.path.join(result_dir, "lim_loss.png"), 
                bbox_inches="tight")
    
    for i in range(n_losses):
        if np.all(train_loss[:epoch,i]>0):
            fig.axes[i].set_yscale("log")
    fig.savefig(os.path.join(result_dir, "log_loss.png"), 
                bbox_inches="tight")
    
    

def plot_predictive_result(epoch, result, pred_steps, result_dir, rname_list, add_name=""):
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    for i, rname in enumerate(rname_list):
        i0 = i//2; i1 = i%2
        ax[i0,i1].plot(np.arange(1,pred_steps+1), result[i])
        ax[i0,i1].set_xlabel("timestep")
        ax[i0,i1].set_ylabel(rname)
    fig.savefig(os.path.join(result_dir, "pred_evals_{}{}.png".format(add_name, epoch)), 
                bbox_inches="tight")
    
    
    
def predictive_plot(config, device, result_dir, model, obs_test, pred_start=100, pred_steps=20, data_num=5):
    horizontal = True
    model_name = config["data"]["model"]
    model.eval()
    Dx = obs_test.shape[2]
    data = Variable(torch.from_numpy(obs_test[:data_num]))
    if model_name in ["SVO", "PSVO", "AESMC"]:
        _, (Z, X, _) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
        (x_hat, _) = model.prediction(Z[pred_start], pred_steps)
    elif model_name in ["VRNN"]:
        _, (_, X, H) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
        (x_hat, _) = model.prediction(H[:,pred_start], pred_steps)
    elif model_name in ["SRNN"]:
        covariate = Variable(torch.from_numpy(np.insert(obs_test[:data_num],0,0,axis=1)[:,:-1]))
        _, (Z, X, H) = model(data.transpose(0,1).to(device), covariate.transpose(0,1).to(device)) #(T,np,bs,Dz)
        (x_hat, _) = model.prediction(Z[pred_start], H[pred_start], pred_steps)
    X = X.detach().cpu().numpy().mean(axis=1) #(T,bs,Dx)
    x_hat = x_hat.detach().cpu().numpy() #(ps,np,bs,Dx)
    x_hat_mean = x_hat.mean(axis=1) #(ps,bs,Dx)
    x_hat_std = x_hat.std(axis=1) #(ps,bs,Dx)
    
    (vert, hori) = (Dx, data_num) if horizontal else (data_num, Dx)
    fig, ax = plt.subplots(vert, hori, figsize=(5*hori, 5*vert))
    for i in range(data_num):
        for j in range(Dx):
            #(iv, ih) = (j, i) if horizontal else (i, j)
            ivh = hori*j+i if horizontal else hori*i+j
            fig.axes[ivh].plot(obs_test[i,:pred_start+pred_steps,j], label="obs", c="k")
            fig.axes[ivh].plot(X[:pred_start,i,j], c="b")
            fig.axes[ivh].fill_between(range(pred_start, pred_start+pred_steps), x_hat_mean[:,i,j]-2*x_hat_std[:,i,j], x_hat_mean[:,i,j]+2*x_hat_std[:,i,j], color="b", alpha=0.1)
            fig.axes[ivh].fill_between(range(pred_start, pred_start+pred_steps), x_hat_mean[:,i,j]-x_hat_std[:,i,j], x_hat_mean[:,i,j]+x_hat_std[:,i,j], color="b", alpha=0.2)
            fig.axes[ivh].plot(range(pred_start, pred_start+pred_steps), x_hat_mean[:,i,j], c="b")
    
    fig.savefig(os.path.join(result_dir, "predictive_plot{}{}.pdf".format("" if horizontal else "_vert", pred_start)), bbox_inches="tight")
    
    
    
def fhn_quiver_plot(config, device, result_dir, model, epoch, obs_test, data_num=10, n_lattice=15):
    # Z:(T,bs,Dz)
    horizontal = True
    seed = config["train"]["seed"]
    model_name = config["data"]["model"]
    true = np.load("../data/FHN/FHN_rk_true_ns400_dt001_T3000_ds15_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(400,200,2)
    n_sample = len(true)
    train_sp = int(n_sample*config["train"]["train_rate"]) # train separating point
    valid_sp = int(n_sample*config["train"]["valid_rate"]) + train_sp # validation separating point
    model.eval()
    
    data = Variable(torch.from_numpy(obs_test[:data_num]))
    if model_name=="SRNN":
        covariate = Variable(torch.from_numpy(np.insert(obs_test[:data_num],0,0,axis=1)[:,:-1]))
        _, (Z, _, _) = model(data.transpose(0,1).to(device), covariate.transpose(0,1).to(device)) #(T,np,bs,Dz)
    else:
        _, (Z, _, _) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
    Z = Z.detach().cpu().numpy().mean(axis=1) #(T,bs,Dz)
    
    fig, ax = plt.subplots(1,2,figsize=(10,5)) if horizontal else plt.subplots(2,1,figsize=(5,10)) 
    for i in range(data_num):
        ax[0].plot(true[valid_sp+i,:,0], true[valid_sp+i,:,1])
        ax[0].scatter(true[valid_sp+i,0,0], true[valid_sp+i,0,1])
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    z1_range, z2_range = ax[0].get_xlim(), ax[0].get_ylim()
    z1_coords = np.linspace(z1_range[0], z1_range[1], num=n_lattice) #(nl)
    z2_coords = np.linspace(z2_range[0], z2_range[1], num=n_lattice) #(nl)
    Zs = np.stack(np.meshgrid(z1_coords, z2_coords), axis=-1) #(2,nl,nl)->(nl,nl,2)
    a = 0.7; b = 0.8; c = 0.08
    def I(t):
        return 1
    dt = 0.01; ds = 15
    fhn = FitzHughNagumo(a*b, a*c, 1, I)
    rk = RungeKuttaScheme(dt, ds+1, fhn, seed=seed)
    Zs_p1 = np.zeros_like(Zs)
    for i in range(n_lattice):
        for j in range(n_lattice):
            Zs_p1[i,j] = rk.perfect_simulation(Zs[i,j])[-1]
    scale = int(5 / 3 * max(abs(z1_range[0]) + abs(z1_range[1]), abs(z2_range[0]) + abs(z2_range[1])))
    
    ax[0].quiver(Zs[:,:,0], Zs[:,:,1], Zs_p1[:,:,0] - Zs[:,:,0], Zs_p1[:,:,1] - Zs[:,:,1], scale=scale)
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    
    
    for i in range(data_num):
        ax[1].plot(Z[:,i,0], Z[:,i,1])
        ax[1].scatter(Z[0,i,0], Z[0,i,1])
    
    z1_range, z2_range = ax[1].get_xlim(), ax[1].get_ylim()
    z1_coords = np.linspace(z1_range[0], z1_range[1], num=n_lattice) #(nl)
    z2_coords = np.linspace(z2_range[0], z2_range[1], num=n_lattice) #(nl)
    Zs = np.stack(np.meshgrid(z1_coords, z2_coords), axis=-1).astype("float32") #(2,nl,nl)->(nl,nl,2)
    Zs_p1 = model.get_next_Z(Variable(torch.from_numpy(Zs)).to(device)) #(nl,nl,Dz)
    Zs_p1 = Zs_p1.detach().cpu().numpy() #(nl,nl,Dz)
    scale = int(5 / 3 * max(abs(z1_range[0]) + abs(z1_range[1]), abs(z2_range[0]) + abs(z2_range[1])))
    
    ax[1].quiver(Zs[:,:,0], Zs[:,:,1], Zs_p1[:,:,0] - Zs[:,:,0], Zs_p1[:,:,1] - Zs[:,:,1], scale=scale)
    ax[1].set_xlabel("z1")
    ax[1].set_ylabel("z2")
    
    fig.savefig(os.path.join(result_dir, "quiver_plot{}{}.pdf".format("" if horizontal else "_vert", epoch)), bbox_inches="tight")
    
    
    
def lorenz_traj_plot(config, device, result_dir, model, epoch, obs_test, data_num=10):
    # comparison between input and output regarding tratin data
    horizontal = True
    model.eval()
    model_name = config["data"]["model"]
    true = np.load("../data/Lorenz/Lorentz_rk_true_ns100_dt001_T750_ds3_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(100,250,3)
    n_sample = len(true)
    train_sp = int(n_sample*config["train"]["train_rate"]) # train separating point
    valid_sp = int(n_sample*config["train"]["valid_rate"]) + train_sp # validation separating point

    data = Variable(torch.from_numpy(obs_test[:data_num]))
    if model_name=="SRNN":
        covariate = Variable(torch.from_numpy(np.insert(obs_test[:data_num],0,0,axis=1)[:,:-1]))
        _, (Z, _, _) = model(data.transpose(0,1).to(device), covariate.transpose(0,1).to(device)) #(T,np,bs,Dz)
    else:
        _, (Z, _, _) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
    Z = Z.detach().cpu().numpy().mean(axis=1) #(T,bs,Dz)
    
    fig = plt.figure(figsize=(9,5)) if horizontal else plt.figure(figsize=(5,9)) 
    ax0 = fig.add_subplot(1,2,1,projection="3d") if horizontal else fig.add_subplot(2,1,1,projection="3d")
    for i in range(data_num):
        ax0.plot(true[valid_sp+i,:,0], true[valid_sp+i,:,1], true[valid_sp+i,:,2])
        ax0.scatter(true[valid_sp+i,0,0], true[valid_sp+i,0,1], true[valid_sp+i,0,2])
    ax0.grid(False)
    print(true.max(axis=0).max(axis=0), true.min(axis=0).min(axis=0))
    ax0.set_xlim(-20,20)
    ax0.set_ylim(-35,35)
    ax0.set_zlim(-10,55)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_zticks([])
    ax0.axis("off")
    ax0.set_position([0,0,0.5,0.5]) if horizontal else ax0.set_position([0,0.27,0.5,0.5]) 

    ax1 = fig.add_subplot(1,2,2,projection="3d") if horizontal else fig.add_subplot(2,1,2,projection="3d") 
    for i in range(data_num):
        ax1.plot(Z[:,i,0], Z[:,i,1], Z[:,i,2])
        ax1.scatter(Z[0,i,0], Z[0,i,1], Z[0,i,2])
    ax1.grid(False)
    Z_min = Z.min(axis=0).min(axis=0)
    Z_max = Z.max(axis=0).max(axis=0)
    Z_range = -5*np.ones(3) #np.minimum(0.1 * (Z_max - Z_min), 5)
    print(Z_min, Z_max)
    ax1.set_xlim(Z_min[0]-Z_range[0], Z_max[0]+Z_range[0])
    ax1.set_ylim(Z_min[1]-Z_range[1], Z_max[1]+Z_range[1])
    ax1.set_zlim(Z_min[2]-Z_range[2], Z_max[2]+Z_range[2])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.axis("off")
    ax1.set_position([0.27,0,0.5,0.5]) if horizontal else ax1.set_position([0,0,0.5,0.5])
    ax1.view_init(30,30) # elevation and azimuth
    
    fig.savefig(os.path.join(result_dir, "traj_plot{}{}.pdf".format("" if horizontal else "_vert", epoch)), bbox_inches="tight")
    
    
    
def allen_traj_plot(config, device, result_dir, model, epoch, obs_test):
    # comparison between input and output regarding tratin data
    model.eval()
    model_name = config["data"]["model"]
    tab_color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    gray_color_list = ["k", "dimgray", "darkgray", "lightgray"]
    n_sample = obs_test.shape[0]
    test_max = np.load("../data/allen/test_max2.npy") #(ns)=(10)

    data = Variable(torch.from_numpy(obs_test))
    if model_name=="SRNN":
        covariate = Variable(torch.from_numpy(np.insert(obs_test,0,0,axis=1)[:,:-1]))
        _, (Z, X, _) = model(data.transpose(0,1).to(device), covariate.transpose(0,1).to(device)) #(T,np,bs,Dz)
    else:
        _, (Z, X, _) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
    Z = Z.detach().cpu().numpy().mean(axis=1) #(T,bs,Dz)
    X = X.detach().cpu().numpy().mean(axis=1) #(T,bs,Dx)
    
    vert = n_sample//5
    fig, ax = plt.subplots(vert,5,figsize=(15,2*vert))
    for i in range(n_sample):
#         fig.axes[i].plot(test_max[i]*obs_test[i,:,0], label="observation", c="k")
#         fig.axes[i].plot(test_max[i]*X[:,i,0], label="inference", c="gray", ls="--")
        fig.axes[i].plot(test_max[i]*obs_test[i,:,0], label="observation")
        fig.axes[i].plot(test_max[i]*X[:,i,0], label="inference")
        imax = test_max[i]*max(X[:,i,0].max(), obs_test[i,:,0].max())
        imin = test_max[i]*min(X[:,i,0].min(), obs_test[i,:,0].min())
        fig.axes[i].text(0,imax-(imax-imin)/10,i+1,c=tab_color_list[i],bbox=dict(facecolor="none", edgecolor=tab_color_list[i]))
    fig.axes[-1].legend(loc="upper right", bbox_to_anchor=(-1.2,-0.15), ncol=2)
    np.save(os.path.join(result_dir, "trajX{}.npy".format(epoch)), X)
    fig.savefig(os.path.join(result_dir, "traj_plot{}.pdf".format(epoch)), bbox_inches="tight")
    
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1,projection="3d")
    for i in range(n_sample):
        ax1.plot(Z[:,i,0], Z[:,i,1], Z[:,i,2])
        ax1.scatter(Z[0,i,0], Z[0,i,1], Z[0,i,2], label=i+1)
    ax1.grid(False)
    Z_min = Z.min(axis=0).min(axis=0)
    Z_max = Z.max(axis=0).max(axis=0)
    Z_range = 0*np.ones(3) #np.minimum(0.1 * (Z_max - Z_min), 5)
    print(Z_min, Z_max)
    ax1.set_xlim(Z_min[0]-Z_range[0], Z_max[0]+Z_range[0])
    ax1.set_ylim(Z_min[1]-Z_range[1], Z_max[1]+Z_range[1])
    ax1.set_zlim(Z_min[2]-Z_range[2], Z_max[2]+Z_range[2])
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax1.set_zticks([])
    #ax1.axis("off")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_position([0,0,1,1])
    ax1.view_init(30,30) # elevation and azimuth
    ax1.legend(loc="upper right", ncol=2)
    np.save(os.path.join(result_dir, "trajZ{}.npy".format(epoch)), Z)
    fig.savefig(os.path.join(result_dir, "latent_traj_plot{}.pdf".format(epoch)), bbox_inches="tight")
    