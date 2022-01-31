import os, sys, json, copy
import time, shutil
import subprocess
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim

from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

sys.path.append("../")
from ODE.model import FitzHughNagumo
from ODE.scheme import RungeKuttaScheme



def load_data(config):
    start_time = time.time()
    data_name = config["data"]["data_name"] # Allen, TEPCO, FHN, Lorenz
    train_num = int(config["train"]["train_num"])
    valid_num = int(config["train"]["valid_num"])
    time_array, time_train, time_valid, time_test = None, None, None, None
    
    if data_name=="Mocap":
        obs = np.load("../data/Mocap/mocap35.npy").astype("float32") #(ns,T,Dx)=(23,300,50)
        obs_train = obs[:train_num]
        obs_valid = obs[train_num:train_num+valid_num]
        obs_test = obs[train_num+valid_num:]
        Dx = obs.shape[2]
    elif data_name=="rmnist":
        obs = np.load("../data/rmnist/rot-mnist-3s-015.npy").astype("float32").reshape(1042,16,1,28,28) #(ns,T,Dx)=(1042,16,784)
        obs_train = obs[:train_num]
        obs_valid = obs[train_num:train_num+valid_num]
        obs_test = obs[train_num+valid_num:]
        Dx = obs.shape[2:]
    elif data_name in ["FHN", "Lorenz"]:
        st_point = config["data"]["st_point"]
        if data_name=="FHN":
            obs = np.load("../data/FHN/FHN_rk_obs0_ns400_dt001_T3000_ds15_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(400,200,1)
        elif data_name=="Lorenz":
            obs = np.load("../data/Lorenz/Lorentz_rk_obs_ns100_dt001_T750_ds3_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(100,250,3)
        n_sample = len(obs)
        obs = obs[:,st_point:]
        train_sp = int(n_sample*config["train"]["train_rate"]) # train separating point
        valid_sp = int(n_sample*config["train"]["valid_rate"]) + train_sp # validation separating point
        obs_train = obs[:train_sp]
        obs_valid = obs[train_sp:valid_sp]
        obs_test = obs[valid_sp:]
        if time_array is not None:
            dt = float(config["data"]["dt"])
            time_array = dt*np.insert(time_array,0,0,axis=1).astype("float32")
            time_array = time_array[:,st_point:]
            time_train = time_array[:train_sp]
            time_valid = time_array[train_sp:valid_sp]
            time_test = time_array[valid_sp:]
        Dx = obs.shape[2]
        
    print("load data:{} sec".format(time.time() - start_time))
    return obs_train, obs_valid, obs_test, time_train, time_valid, time_test, Dx



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
        Dx = obs_train.shape[2]
        obs_mean = obs_train.reshape(-1,Dx).mean(axis=0)
        svec = np.std(obs_train.reshape(-1,Dx), axis=0)
        obs_train = (obs_train - obs_mean) / svec
        obs_valid = (obs_valid - obs_mean) / svec
        #obs_test = (obs_test - obs_mean) / svec
    else:
        svec = 1
    
    return obs_train, obs_valid, svec


def get_dataset(config, obs_train, obs_valid, time_train, time_valid):
    batch_size = config["train"]["batch_size"]
    model_name = config["data"]["model"]
    num_workers = 0#config["train"]["num_workers"]
    #rnn_name = config["network"]["rnn"]
    
    if time_train is None:
        train_tensor = torch.utils.data.TensorDataset(torch.from_numpy(obs_train),
                                               torch.from_numpy(np.insert(obs_train,0,0,axis=1)[:,:-1]))
        valid_tensor = torch.utils.data.TensorDataset(torch.from_numpy(obs_valid),
                                               torch.from_numpy(np.insert(obs_valid,0,0,axis=1)[:,:-1]))
    else:
        train_tensor = torch.utils.data.TensorDataset(torch.from_numpy(obs_train),
                                               torch.from_numpy(time_train))
        valid_tensor = torch.utils.data.TensorDataset(torch.from_numpy(obs_valid),
                                               torch.from_numpy(time_valid))
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_tensor, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader
        



def get_model(config, Dx, device):
    loss_name_list_dict = {"VRNN":["loss", "-logf", "-logg", "logq", "ESS", "MCS"],
                           "AESMC":["loss", "-logf", "-logg", "logq", "ESS", "MCS"],
                           "SVO":["loss", "-logf", "-logg", "logq", "ESS", "MCS"]}
    outer_model = config["data"]["outer_model"]
    model_name = config["data"]["model"]
    system = config["data"]["system"]
    kld_penalty_on = config["network"]["kld_penalty_weight"] > 0
    
    if system in ["FIVO", "EnKO"]:
        loss_name_list_dict["NODE"] = ["loss"]
    
    if outer_model=="Conv":
        from model.conv import Conv
        loss_name_list = ["loss", "VAE_Loss", "NLL", "KLD"] + ["inner {}".format(loss_name) for loss_name in loss_name_list_dict[model_name]]
        if kld_penalty_on:
            loss_name_list += ["inner kld"]
        model = Conv(Dx, config, device).to(device)
    elif outer_model=="StyleConv":
        from model.style_conv import StyleConv
        loss_name_list = ["loss", "VAE_Loss", "NLL", "KLD-model", "KLD-style", "KLD-average"] + ["inner {}".format(loss_name) for loss_name in loss_name_list_dict[model_name]]
        if kld_penalty_on:
            loss_name_list += ["inner kld"]
        model = StyleConv(Dx, config, device).to(device)
    else:
        if model_name=="VRNN":
            from model.vrnn import VRNN
            model = VRNN(Dx, config, device).to(device)
        elif model_name=="AESMC":
            from model.aesmc import AESMC
            model = AESMC(Dx, config, device).to(device)
        elif model_name=="SVO":
            from model.svo import SVO
            model = SVO(Dx, config, device).to(device)
        loss_name_list = loss_name_list_dict[model_name]
        if kld_penalty_on:
            loss_name_list += ["kld"]
 
    return model, loss_name_list




def generate_valid_function(model, valid_loader, obs_valid, device, loss_name_list, model_name, pred_steps, display_steps, evaluation_metrics, total_evaluation_metrics):
    def valid(epoch, saturated_on):
        """uses test data to evaluate likelihood of the model"""
        start_time = time.time()
        model.eval()
        loss = np.zeros(n_losses)
        pred_evals = np.zeros((len(total_evaluation_metrics), pred_steps))
        
        for (data, covariate) in valid_loader:                                            
            data = Variable(data)
            data = data.transpose(0,1).to(device)
            covariate = Variable(covariate) # (bs,T,Dx)/(bs,T)
            covariate = covariate.transpose(0,1).to(device) # (T,bs,Dx)/(T,bs)

            if outer_model is None:
                if model_name=="SRNN":
                    _loss, (Z, _, H)  = model(data, covariate)
                elif model_name in ["SVO", "SVOD", "SVOp", "SVO-II", "MINN-SVO", "PSVO", "AESMC", "AESMCp", "NODE"]:
                    _loss, (Z, _, _)  = model(data, saturated_on)
                elif model_name in ["SVOg"]:
                    _loss, (Z, _, H)  = model(data, saturated_on)
                elif model_name in ["VRNN", "MINN"]:
                    if time_train is not None:
                        _loss, (_, _, Z)  = model(data, saturated_on, covariate)
                    else:
                        _loss, (_, _, Z)  = model(data, saturated_on)
                elif model_name=="ODE2VAE":
                    _loss, _ = model(data, len(train_loader), epoch)
                elif model_name=="KVAE":
                    _loss, (_, _, Z, _, _, _) = model(data, epoch)
                else:
                    _loss, _  = model(data)
            else:
                if model_name in ["SVO", "SVOp", "SVO-II", "MINN-SVO", "PSVO", "AESMC", "AESMCp"]:
                    _loss, (Z, _, _, _)  = model(data, epoch, saturated_on)
                elif model_name in ["VRNN", "MINN"]:
                    _loss, (_, _, Z, _)  = model(data, epoch)
                else:
                    _loss, _  = model(data, epoch)
            
            for i in range(n_losses):
                loss[i] += _loss[i].item()
            
            eval_count = 0
            if "MSE" in evaluation_metrics:
                _TV = None
                if model_name in ["SRNN", "SVOg"]:
                    _MSE, _TV = model.calculate_mse(data, Z, H, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
                elif model_name=="ODE2VAE":
                    _MSE = model.calculate_mse(data, pred_steps) #(ps,bs,Dx)
                elif model_name=="KVAE":
                    _MSE = model.calculate_mse(data, Z, pred_steps) #(ps,bs,Dx)
                elif time_train is None:
                    _MSE, _TV = model.calculate_mse(data, Z, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
                else:
                    _MSE, _TV = model.calculate_mse(data, Z, pred_steps, covariate) #(ps,bs,Dx),(ps,bs,Dx)
                _MSE = _MSE.data.cpu().numpy()
                MSE = _MSE.sum(axis=2) #(ps,bs)
                sMSE = (_MSE*svec*svec).sum(axis=2) #(ps,bs)
                if _TV is None:
                    TV = sTV = 1
                else:
                    _TV = _TV.data.cpu().numpy()
                    TV = _TV.sum(axis=2) #(ps,bs)
                    sTV = (_TV*svec*svec).sum(axis=2) #(ps,bs)
                pred_evals[0] += MSE.sum(axis=1)
                pred_evals[1] += sMSE.sum(axis=1)
                pred_evals[2] += (1 - MSE/TV).sum(axis=1) #R2
                pred_evals[3] += (1 - sMSE/sTV).sum(axis=1) #scaling R2
                eval_count += 4
                
            if "FIP" in evaluation_metrics:
                FIP = model.calculate_fip(data, Z, pred_steps) #(ps,bs)
                pred_evals[eval_count] += FIP.sum(1).data.cpu().numpy() #(ps,)

        loss /= len(valid_loader)
        pred_evals /= len(obs_valid)
        
        for i, loss_name in enumerate(loss_name_list):
            experiment.log_metric("valid {}".format(loss_name), loss[i], step=epoch)
        for i, eval_name in enumerate(total_evaluation_metrics):
            for j in range(pred_steps):
                experiment.log_metric("{}-step {}".format(j+1, eval_name), pred_evals[i,j], step=epoch)
        
        if epoch%display_steps==0:
            print_contents = "====> Test set loss:"
            for i, loss_name in enumerate(loss_name_list):
                print_contents += " {} = {:.4f}".format(loss_name,
                                                        loss[i])
            print(print_contents)
        return loss, pred_evals
    return valid





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
    
    

def plot_predictive_result(result, pred_steps, result_dir, rname_list, experiment=None, add_name=None, epoch=None):
    if len(rname_list)==1:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
    elif len(rname_list)<=3:
        fig, ax = plt.subplots(1, len(rname_list), figsize=(5*len(rname_list), 5))
    elif len(rname_list)==4:
        fig, ax = plt.subplots(2, 2, figsize=(10,10))
    elif len(rname_list)>=5 and len(rname_list)<=9:
        vert = (len(rname_list)-1)//3+1
        fig, ax = plt.subplots(vert, 3, figsize=(15,5*vert))
    elif len(rname_list)>=10 and len(rname_list)<=16:
        vert = (len(rname_list)-1)//4+1
        fig, ax = plt.subplots(vert, 4, figsize=(20,5*vert))
        
    for i in range(len(fig.axes) - len(rname_list)):
        fig.axes[-(i+1)].axis("off")
        
    for i, rname in enumerate(rname_list):
        fig.axes[i].plot(np.arange(1,pred_steps+1), result[i])
        fig.axes[i].set_xlabel("timestep")
        fig.axes[i].set_ylabel(rname)
            
    fig.savefig(os.path.join(result_dir, "pred_evals{}{}.pdf".format("" if add_name is None else "_{}".format(add_name), "" if epoch is None else epoch)), 
                bbox_inches="tight")
    if experiment is not None:
        experiment.log_figure("{}predictive evaluations".format("" if add_name is None else "{}_".format(add_name)), fig)
        
        
        
def plot_cosine_similarity(cosine_similarity, result_dir, experiment=None, add_name=None):
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(np.linspace(0,1,np.prod(cosine_similarity.shape)), np.sort(cosine_similarity.flatten()))
    ax.set_xlabel("percentage")
    ax.set_ylabel("cosine similarity")
    fig.savefig(os.path.join(result_dir, "cosine_similarity{}.pdf".format("" if add_name is None else "_{}".format(add_name))), bbox_inches="tight")
    if experiment is not None:
        experiment.log_figure("{}cosine similarity".format("" if add_name is None else "{} ".format(add_name)), fig)
        

    
    
def predictive_plot(config, device, result_dir, model, obs_test, pred_start=100, pred_steps=20, data_num=5, experiment=None, add_name=None):
    horizontal = True
    model_name = config["data"]["model"]
    add_name = "" if add_name is None else "{}_".format(add_name)
    model.eval()
    Dx = obs_test.shape[2]
    data = Variable(torch.from_numpy(obs_test[:data_num]))
    if model_name in ["SVO", "SVOp", "SVO-II", "MINN-SVO", "PSVO", "AESMC", "NODE"]:
        _, (_, Z, X, _) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
        (x_hat, _) = model.prediction(Z[pred_start], pred_steps)
    elif model_name in ["VRNN", "MINN"]:
        _, (_, _, X, H) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
        (x_hat, _) = model.prediction(H[:,pred_start], pred_steps)
    elif model_name in ["SRNN"]:
        covariate = Variable(torch.from_numpy(np.insert(obs_test[:data_num],0,0,axis=1)[:,:-1]))
        _, (_, Z, X, H) = model(data.transpose(0,1).to(device), covariate.transpose(0,1).to(device)) #(T,np,bs,Dz)
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
    
    fig.savefig(os.path.join(result_dir, "{}predictive_plot{}{}.pdf".format(add_name, "" if horizontal else "_vert", pred_start)), bbox_inches="tight")
    if experiment is not None:
        experiment.log_figure("{}predictive_plot".format(add_name), fig)
    
    
    
def fhn_quiver_plot(config, device, result_dir, model, obs_test, experiment=None, add_name=None, epoch="", data_num=10, n_lattice=15, horizontal=True, separate_fig_on=False):
    # Z:(T,bs,Dz)
    add_name = "" if add_name is None else "{}_".format(add_name)
    model_name = config["data"]["model"]
    seed = config["train"]["seed"]
    true = np.load("../data/FHN/FHN_rk_true_ns400_dt001_T3000_ds15_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(400,200,2)
    n_sample = len(true)
    train_sp = int(n_sample*config["train"]["train_rate"]) # train separating point
    valid_sp = int(n_sample*config["train"]["valid_rate"]) + train_sp # validation separating point
    model.eval()
    
    data = Variable(torch.from_numpy(obs_test[:data_num]))
    if model_name=="SRNN":
        covariate = Variable(torch.from_numpy(np.insert(obs_test[:data_num],0,0,axis=1)[:,:-1]))
        _, (_, Z, _, _) = model(data.transpose(0,1).to(device), covariate.transpose(0,1).to(device)) #(T,np,bs,Dz)
    else:
        _, (_, Z, _, _) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
    Z = Z.detach().cpu().numpy().mean(axis=1) #(T,bs,Dz)
    
    if separate_fig_on:
        figs, ax = [], []
        for i in range(2):
            fig, ax1 = plt.subplots(1,1,figsize=(5,5))
            figs.append(fig)
            ax.append(ax1)
    else:
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
    
    if separate_fig_on:
        for fig, fname in zip(figs, ["orig", "recon"]):
            fig.savefig(os.path.join(result_dir, "{}quiver_plot{}_{}{}.pdf".format(add_name, "" if horizontal else "_vert", fname, epoch)), bbox_inches="tight")
            if experiment is not None:
                experiment.log_figure("{}quiver_plot_{}".format(add_name, fname), fig)
    else:
        fig.savefig(os.path.join(result_dir, "{}quiver_plot{}{}.pdf".format(add_name, "" if horizontal else "_vert", epoch)), bbox_inches="tight")
        if experiment is not None:
            experiment.log_figure("{}quiver_plot".format(add_name), fig)
    
    
    
def lorenz_traj_plot(config, device, result_dir, model, obs_test, experiment=None, add_name=None, epoch="", data_num=10, horizontal=True, plot_axes=True, separate_fig_on=True):
    # comparison between input and output regarding tratin data
    add_name = "" if add_name is None else "{}_".format(add_name)
    model.eval()
    model_name = config["data"]["model"]
    true = np.load("../data/Lorenz/Lorentz_rk_true_ns100_dt001_T750_ds3_ssd0_osd01.npy").astype("float32") # (ns,T,Dx)=(100,250,3)
    n_sample = len(true)
    train_sp = int(n_sample*config["train"]["train_rate"]) # train separating point
    valid_sp = int(n_sample*config["train"]["valid_rate"]) + train_sp # validation separating point

    data = Variable(torch.from_numpy(obs_test[:data_num]))
    if model_name=="SRNN":
        covariate = Variable(torch.from_numpy(np.insert(obs_test[:data_num],0,0,axis=1)[:,:-1]))
        _, (_, Z, _, _) = model(data.transpose(0,1).to(device), covariate.transpose(0,1).to(device)) #(T,np,bs,Dz)
    else:
        _, (_, Z, _, _) = model(data.transpose(0,1).to(device)) #(T,np,bs,Dz)
    Z = Z.detach().cpu().numpy().mean(axis=1) #(T,bs,Dz)
    
    if separate_fig_on:
        figs = [plt.figure(figsize=(5,5)), plt.figure(figsize=(5,5))]
        ax0 = figs[0].add_subplot(1,1,1,projection="3d")
        ax1 = figs[1].add_subplot(1,1,1,projection="3d")
    else:
        fig = plt.figure(figsize=(9,5)) if horizontal else plt.figure(figsize=(5,9)) 
        ax0 = fig.add_subplot(1,2,1,projection="3d") if horizontal else fig.add_subplot(2,1,1,projection="3d")
        ax1 = fig.add_subplot(1,2,2,projection="3d") if horizontal else fig.add_subplot(2,1,2,projection="3d") 
    for i in range(data_num):
        ax0.plot(true[valid_sp+i,:,0], true[valid_sp+i,:,1], true[valid_sp+i,:,2])
        ax0.scatter(true[valid_sp+i,0,0], true[valid_sp+i,0,1], true[valid_sp+i,0,2])
    ax0.grid(False)
    print(true.max(axis=0).max(axis=0), true.min(axis=0).min(axis=0))
    if not plot_axes:
        ax0.set_xlim(-20,20)
        ax0.set_ylim(-35,35)
        ax0.set_zlim(-10,55)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_zticks([])
        ax0.axis("off")
    if separate_fig_on:
        ax0.set_position([0,0,1,1])
    else:
        ax0.set_position([0,0,0.5,0.5]) if horizontal else ax0.set_position([0,0.27,0.5,0.5]) 

    for i in range(data_num):
        ax1.plot(Z[:,i,0], Z[:,i,1], Z[:,i,2])
        ax1.scatter(Z[0,i,0], Z[0,i,1], Z[0,i,2])
    ax1.grid(False)
    if not plot_axes:
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
    if separate_fig_on:
        ax1.set_position([0,0,1,1])
    else:
        ax1.set_position([0.27,0,0.5,0.5]) if horizontal else ax1.set_position([0,0,0.5,0.5])
    ax1.view_init(60,-120) # elevation and azimuth. default (30,30)
    
    if separate_fig_on:
        for fig, fname in zip(figs, ["orig", "recon"]):
            fig.savefig(os.path.join(result_dir, "{}traj_plot{}{}_{}{}.pdf".format(add_name, "" if horizontal else "_vert", "_ax" if plot_axes else "", fname, epoch)), bbox_inches="tight")
            if experiment is not None:
                experiment.log_figure("{}trajectory_plot_{}".format(add_name, fname), fig)
    else:
        fig.savefig(os.path.join(result_dir, "{}traj_plot{}{}{}.pdf".format(add_name, "" if horizontal else "_vert", "_ax" if plot_axes else "", epoch)), bbox_inches="tight")
        if experiment is not None:
            experiment.log_figure("{}trajectory_plot".format(add_name), fig)        
        
        
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=("index", "uuid", "name", "timestamp", "memory.total", "memory.free", "memory.used", "utilization.gpu", "utilization.memory"), no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]