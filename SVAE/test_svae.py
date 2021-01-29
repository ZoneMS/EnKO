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

from utils import load_data, transform_data, get_dataset, get_model, plot_predictive_result, fhn_quiver_plot, lorenz_traj_plot, allen_traj_plot, predictive_plot


def test():
    if True:
        name = "210129_Lorenz_gs/sEnKOmSRNN"
        name = "../results/{}".format(name)
        test_main(name)
    else:
        gs_name = "210129_FHN_gs"
        files = os.listdir(os.path.join("../results", gs_name))
        files_dir = [os.path.join("../results", gs_name, f) for f in files if os.path.isdir(os.path.join("../results", gs_name, f))]
        for f in files_dir:
            if os.path.exists(os.path.join(f, "train_loss.npy")):
                test_main(f)

def test_main(result_dir, epoch=None):
    pred_start = 50
    predictive_steps = 150
    data_num = 5
    
    f = open(os.path.join(result_dir, "config.json"), "r")
    config = json.load(f)
    f.close()
    
    print(result_dir)
    if epoch is None:
        save_freq = config["print"]["save_freq"]
        train_loss = (np.load(os.path.join(result_dir, "train_loss.npy"))[:,0])
        epoch = (np.count_nonzero(1 - np.isnan(train_loss))//save_freq)*save_freq
    
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("load config from {}".format(result_dir))
    
    model_name = config["data"]["model"] # VRNN, AESMC, SRNN, SVO, PSVO (only FIVO)
    data_name = config["data"]["data_name"] # Lorenz, FHN, Allen
    
    ## Set seed and cuda
    print("Set seed {}.".format(config["train"]["seed"]))
    cuda = torch.cuda.is_available()
    if cuda:
        print("cuda is available")
        device = torch.device('cuda', config["train"]["gpu"])
    else:
        print("cuda is not available")
        device = torch.device("cpu")
    torch.manual_seed(config["train"]["seed"])
    if cuda:
        torch.cuda.manual_seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])
    
#     cuda = False
#     device = torch.device('cpu')
#     np.random.seed(config["train"]["seed"])
#     torch.autograd.set_detect_anomaly(True)
    
    
    ## Read data and set parameters
    obs_train, _, obs_test, Dx = load_data(config)
    obs_train, obs_test, svec = transform_data(obs_train, obs_test, config)
    _, test_loader = get_dataset(config, obs_train, obs_test)
    
    model, loss_name_list = get_model(config, Dx, device)
    model.load_state_dict(torch.load(os.path.join(result_dir, "state_dict_{}.pth".format(epoch))))
    n_losses = len(loss_name_list)
    pred_steps = config["training"]["pred_steps"]
    Dz = config["network"]["Dz"]
    n_particles = config["network"]["n_particles"]
    
    
    def test(epoch):
        """uses test data to evaluate likelihood of the model"""
        model.eval()
        loss = np.zeros(n_losses)
        pred_evals = np.zeros((4, pred_steps))

        for (data, covariate) in test_loader:                                            
            data = Variable(data)
            data = data.transpose(0,1).to(device)
            covariate = Variable(covariate) # (bs,T,Dx)/(bs,T)
            covariate = covariate.transpose(0,1).to(device) # (T,bs,Dx)/(T,bs)

            if model_name=="SRNN":
                _loss, (Z, _, H)  = model(data, covariate)
            elif model_name in ["SVO", "PSVO", "AESMC"]:
                _loss, (Z, _, _)  = model(data)
            elif model_name in ["VRNN"]:
                _loss, (_, _, Z)  = model(data)
            else:
                _loss, _  = model(data)
            
            for i in range(n_losses):
                loss[i] += _loss[i].item()
            
            if model_name=="SRNN":
                _MSE, _TV = model.calculate_mse(data, Z, H, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
            else:
                _MSE, _TV = model.calculate_mse(data, Z, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
            _MSE, _TV = _MSE.data.cpu().numpy(), _TV.data.cpu().numpy()
            MSE, TV = _MSE.sum(axis=2), _TV.sum(axis=2) #(ps,bs),(ps,bs)
            sMSE, sTV = (_MSE*svec*svec).sum(axis=2), (_TV*svec*svec).sum(axis=2) #(ps,bs),(ps,bs)
            pred_evals[0] += MSE.sum(axis=1)
            pred_evals[1] += sMSE.sum(axis=1)
            pred_evals[2] += (1 - MSE/TV).sum(axis=1) #R2
            pred_evals[3] += (1 - sMSE/sTV).sum(axis=1) #scaling R2

        loss /= len(test_loader)
        pred_evals /= len(obs_test)
        
        print_contents = "====> Test set loss:"
        for i, loss_name in enumerate(loss_name_list):
            print_contents += " {} = {:.4f}".format(loss_name,
                                                    loss[i])
        print(print_contents)
        return loss, pred_evals
    
    
    ## Train model
    def execute():
        # training + testing
        test_loss, pred_evals = test(epoch)
        np.save(os.path.join(result_dir, "test_loss{}.npy".format(epoch)), test_loss)
        np.save(os.path.join(result_dir, "pred_evals_test{}.npy".format(epoch)), pred_evals)
        plot_predictive_result(epoch, pred_evals, pred_steps, result_dir, ["MSE", "sMSE", "R2", "sR2"], "test")
        
        if data_name in ["FHN", "Lorenz", "Allen"]:
            predictive_plot(config, device, result_dir, model, obs_test, pred_start, predictive_steps, data_num)
        
        if data_name=="FHN":
            fhn_quiver_plot(config, device, result_dir, model, epoch, obs_test)
        elif data_name=="Lorenz":
            lorenz_traj_plot(config, device, result_dir, model, epoch, obs_test)
        elif data_name=="Allen":
            allen_traj_plot(config, device, result_dir, model, epoch, obs_test)
        
        
    
    ## Excute
    # measure for culabas rutine error
    execute()

    
if __name__ == "__main__":
    test()