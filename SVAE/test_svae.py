import os, sys, json
import time, shutil, glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import load_data, transform_data, get_dataset, get_model, plot_predictive_result, fhn_quiver_plot, lorenz_traj_plot, allen_traj_plot, predictive_plot, rot_mnist_plot, reconstructed_amc_plot


def test():
    if True:
        name = "211027_EnKO_StyleConv-SVO_rmnist"
        name = "../results/{}".format(name)
        test_main(name, valid_on=True, gpu_id=0)
        test_main(name, gpu_id=0)
    else:
        gs_name = "211027_StyleConv_SVO_rmnist_gs"
        files = os.listdir(os.path.join("../results", gs_name))
        files_dir = [os.path.join("../results", gs_name, f) for f in files if os.path.isdir(os.path.join("../results", gs_name, f))]
        for f in files_dir:
            if len(glob.glob(os.path.join(f, "state_dict*.pth")))!=0:
                test_main(f, valid_on=True, gpu_id=0)
                test_main(f, valid_on=False, gpu_id=0)

                
def test_main(result_dir, experiment=None, epoch=None, pred_steps=None, valid_on=False, gpu_id=None):
    f = open(os.path.join(result_dir, "config.json"), "r")
    config = json.load(f)
    f.close()
    
    pred_start = config["test"]["pred_start"]
    predictive_steps = config["test"]["predictive_steps"]
    data_num = config["test"]["data_num"]
    
    print(result_dir)
#     if epoch is None:
#         save_freq = config["print"]["save_freq"]
#         train_loss = (np.load(os.path.join(result_dir, "train_loss.npy"))[:,0])
#         epoch = (np.count_nonzero(1 - np.isnan(train_loss))//save_freq)*save_freq
    
    if False:
        f = open("config.json", "r")
        ref_config = json.load(f)
        f.close()
        
        for top_key in ref_config.keys():
            for key in ref_config[top_key].keys():
                if key not in config[top_key]:
                    config[top_key][key] = ref_config[top_key][key]
#     with open(os.path.join(result_dir, "config.json"), "w") as f:
#         json.dump(config, f, indent=4, ensure_ascii=False)
    print("load config from {}".format(result_dir))
    
    outer_model = config["data"]["outer_model"]
    model_name = config["data"]["model"]
    data_name = config["data"]["data_name"]
    
    ## Set seed and cuda
    if gpu_id is None:
        gpu_id = config["train"]["gpu"]
        
    print("Set seed {}.".format(config["train"]["seed"]))
    cuda = torch.cuda.is_available()
    if cuda:
        print("cuda is available")
        device = torch.device('cuda', gpu_id)
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
    max_epoch = config["train"]["epoch"] if epoch is None or epoch=="new" else epoch
    epoch_name = "" if epoch is None else str(epoch)
    metrics_total_dict = {"MSE":["MSE", "sMSE", "R2", "sR2"], "FIP":["NIP", "FIP"]}
    if outer_model in ["Conv", "StyleConv"]:
        metrics_total_dict["MSE"] = ["MSE", "R2"]
    evaluation_metrics = config["test"]["evaluation_metrics"]
    total_evaluation_metrics = sum([metrics_total_dict[i] for i in evaluation_metrics], [])
    
    obs_train, obs_valid, obs_test, time_train, time_valid, time_test, Dx = load_data(config)
    if valid_on:
        obs_train, obs_test, svec = transform_data(obs_train, obs_valid, config)
        _, test_loader = get_dataset(config, obs_train, obs_test, time_train, time_valid)
    else:
        obs_train, obs_test, svec = transform_data(obs_train, obs_test, config)
        _, test_loader = get_dataset(config, obs_train, obs_test, time_train, time_test)
    
    model, loss_name_list = get_model(config, Dx, device)
    model.load_state_dict(torch.load(os.path.join(result_dir, "state_dict{}.pth".format("" if epoch is None else "_{}".format(epoch)))))
    n_losses = len(loss_name_list)
    if pred_steps is None:
        #pred_steps = config["training"]["pred_steps"]
        pred_steps = int(obs_train.shape[1]) - 1
    Dz = config["network"]["Dz"]
    n_particles = config["network"]["n_particles"]
    saturated_on = True
    
    
    def test():
        """uses test data to evaluate likelihood of the model"""
        start_time = time.time()
        model.eval()
        loss = np.zeros(n_losses)
        pred_evals = np.zeros((len(total_evaluation_metrics), pred_steps))
        
        for (data, covariate) in test_loader:
            data = Variable(data)
            data = data.transpose(0,1).to(device)
            covariate = Variable(covariate) # (bs,T,Dx)/(bs,T)
            covariate = covariate.transpose(0,1).to(device) # (T,bs,Dx)/(T,bs)

            H = None
            if outer_model is None:
                if model_name=="SRNN":
                    _loss, (Z, _, H)  = model(data, covariate)
                elif model_name in ["SVO", "AESMC"]:
                    _loss, (Z, _, _)  = model(data, saturated_on)
                elif model_name in ["VRNN"]:
                    if time_train is not None:
                        _loss, (_, _, Z)  = model(data, saturated_on, covariate)
                    else:
                        _loss, (_, _, Z)  = model(data, saturated_on)
                else:
                    _loss, _  = model(data)
            elif outer_model=="Conv":
                if model_name in ["SVO", "AESMC"]:
                    _loss, (Z, _, _, _)  = model(data, max_epoch, False)
                elif model_name in ["VRNN"]:
                    _loss, (_, _, Z, _)  = model(data, max_epoch)
                else:
                    _loss, _  = model(data, max_epoch, max_epoch)
            elif outer_model=="StyleConv":
                if model_name in ["SVO", "AESMC"]:
                    _loss, (Z, _, a_style, _, _)  = model(data, max_epoch, False, False)
                elif model_name in ["VRNN"]:
                    _loss, (_, _, a_style, Z, _)  = model(data, max_epoch, False)
            
            for i in range(n_losses):
                loss[i] += _loss[i].item()
            
            eval_count = 0
            if outer_model in ["Conv"]:
                _metrics = model.calculate_predictive_metrics(data, Z, H, pred_steps, evaluation_metrics) #(nm,ps,bs)
                pred_evals += _metrics.sum(2) #(nm,ps)
            elif outer_model == "StyleConv":
                _metrics = model.calculate_predictive_metrics(data, Z, a_style, H, pred_steps, evaluation_metrics) #(nm,ps,bs)
                pred_evals += _metrics.sum(2) #(nm,ps)
            else:
                if "MSE" in evaluation_metrics:
                    _TV = None
                    if time_train is None:
                        _MSE, _TV = model.calculate_mse(data, Z, pred_steps) #(ps,bs,Dx),(ps,bs,Dx)
                    else:
                        _MSE, _TV = model.calculate_mse(data, Z, pred_steps, covariate) #(ps,bs,Dx),(ps,bs,Dx)
                    
                    #_MSE = _MSE.data.cpu().numpy()
                    MSE = _MSE.sum(axis=2) #(ps,bs)
                    sMSE = (_MSE*svec*svec).sum(axis=2) #(ps,bs)
                    if _TV is None:
                        TV = sTV = 1
                    else:
                        #_TV = _TV.data.cpu().numpy()
                        TV = _TV.sum(axis=2) #(ps,bs)
                    sTV = (_TV*svec*svec).sum(axis=2) #(ps,bs)
                    pred_evals[0] += MSE.sum(axis=1)
                    pred_evals[1] += sMSE.sum(axis=1)
                    pred_evals[2] += (1 - MSE/TV).sum(axis=1) #R2
                    pred_evals[3] += (1 - sMSE/sTV).sum(axis=1) #scaling R2
                    eval_count += 4

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
        test_loss, pred_evals = test()
        np.save(os.path.join(result_dir, "{}_loss{}.npy".format("valid" if valid_on else "test", epoch_name)), test_loss)
        np.save(os.path.join(result_dir, "pred_evals_{}{}.npy".format("valid" if valid_on else "test", epoch_name)), pred_evals)
        plot_predictive_result(pred_evals, pred_steps, result_dir, total_evaluation_metrics, experiment, "{}{}".format("valid" if valid_on else "test", epoch_name))
        
        if data_name in ["FHN", "Lorenz", "Allen"]:
            predictive_plot(config, device, result_dir, model, obs_test, pred_start, predictive_steps, data_num, experiment, "valid" if valid_on else "test")
        
        if data_name=="FHN":
            fhn_quiver_plot(config, device, result_dir, model, obs_test, experiment, "valid" if valid_on else "test")
        elif data_name=="Lorenz":
            lorenz_traj_plot(config, device, result_dir, model, obs_test, experiment, "valid" if valid_on else "test")
        elif data_name=="Allen":
            allen_traj_plot(config, device, result_dir, model, obs_test, experiment)
        
        
    
    ## Excute
    # measure for culabas rutine error
    execute()

    
if __name__ == "__main__":
    test()